// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/resource.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#include <cstring>
#include <type_traits>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Vulkan {

template <typename T>
static bool SameBits(const T& a, const T& b) {
    static_assert(std::is_trivially_copyable_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

static TextureKey MakeTextureKey(const AmdGpu::Image& tsharp, const Shader::ImageResource& desc,
                                 u32 binding) {
    TextureKey key{};
    key.tsharp = tsharp;
    key.is_depth = desc.is_depth;
    key.is_array = desc.is_array;
    key.is_written = desc.is_written;
    key.mip_level = desc.mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex
                        ? desc.constant_mip_index
                    : desc.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex
                        ? binding
                        : ~0U;
    return key;
}

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()},
      submit_draw_interval{EmulatorSettings.GetSubmitDrawInterval()} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);

    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
        buffer_cache.OnSubmit();
        DebugState.frame_submits.Add(1);
    });
}

Rasterizer::~Rasterizer() = default;

namespace {

/// Drops the stream hints of a draw when the draw is over.
struct StreamHintsScope {
    VideoCore::BufferCache& buffer_cache;

    ~StreamHintsScope() {
        buffer_cache.ClearStreamHints();
    }
};

enum class DrawFilter { None, EliminateFastClear, FmaskDecompress, Resolve, NoPrimitive, Copy };

struct DrawFilterResult {
    DrawFilter filter;
    bool depth_copy;
    bool stencil_copy;
};

/// What a draw with these registers does in place of drawing, if anything.
DrawFilterResult GetDrawFilter(const AmdGpu::Regs& regs) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return {DrawFilter::EliminateFastClear};
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return {DrawFilter::FmaskDecompress};
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return {DrawFilter::Resolve};
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return {DrawFilter::NoPrimitive};
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return {DrawFilter::Copy, depth_copy, stencil_copy};
    }
    return {DrawFilter::None};
}

} // Anonymous namespace

Rasterizer::TextureLookup& Rasterizer::FindTextureLookup(const TextureKey& key, const u64 hash) {
    const u32 set_index = hash % TextureLookupSets;
    auto& set = texture_lookup_sets[set_index];
    TextureLookup* const lookups = &texture_lookups[set_index * TextureLookupWays];
    const u64 now = ++texture_lookup_clock;
    u32 oldest = 0;
    for (u32 way = 0; way < TextureLookupWays; ++way) {
        if (set.hashes[way] == hash && SameBits(lookups[way].key, key)) {
            set.last_uses[way] = now;
            return lookups[way];
        }
        if (set.last_uses[way] < set.last_uses[oldest]) {
            oldest = way;
        }
    }
    AmdGpu::cp_perf.tex_evictions += set.last_uses[oldest] != 0;
    set.hashes[oldest] = hash;
    set.last_uses[oldest] = now;
    return lookups[oldest];
}

void Rasterizer::PrefetchDrawsAhead(const DrawAhead& taken, const DrawAhead* next,
                                    const DrawAhead* second, const DrawAhead* third) {
    using VideoCore::PrefetchRead;
    const auto slots = [this](const DrawAhead& ahead) {
        if (!ahead.lookup.pipeline) {
            return;
        }
        buffer_cache.PrefetchStreamCopies(ahead.stream_hints);
        for (const DrawAhead::ImageKey& image : ahead.image_keys) {
            if (image.key.tsharp.Address() == 0) {
                continue;
            }
            const u32 set_index = image.hash % TextureLookupSets;
            const auto& set = texture_lookup_sets[set_index];
            for (u32 way = 0; way < TextureLookupWays; ++way) {
                if (set.hashes[way] == image.hash) {
                    PrefetchRead(&texture_lookups[set_index * TextureLookupWays + way],
                                 sizeof(TextureLookup));
                    break;
                }
            }
        }
    };
    if (&taken != slots_ahead) {
        slots(taken);
    }
    slots_ahead = next;
    if (next) {
        slots(*next);
    }
    if (third) {
        static constexpr std::size_t VertexInputsBytes = 2 * VideoCore::CacheLineSize;
        const auto* const entry = reinterpret_cast<const u8*>(third);
        PrefetchRead(entry, reinterpret_cast<const u8*>(&third->has_vertex_inputs) + 1 - entry);
        const auto inputs = [](const auto& list) {
            PrefetchRead(&list, VertexInputsBytes);
            PrefetchRead(reinterpret_cast<const u8*>(&list + 1) - 1);
        };
        inputs(third->attributes);
        inputs(third->bindings);
        inputs(third->guest_buffers);
        inputs(third->vertex_ranges);
    }
    if (second && second->lookup.pipeline) {
        static constexpr std::size_t MaxFlatBytes = 4 * VideoCore::CacheLineSize;
        const auto array = [](const auto& list, std::size_t max_bytes = ~std::size_t{}) {
            PrefetchRead(list.data(), std::min(list.size() * sizeof(list[0]), max_bytes));
        };
        array(second->stream_hints);
        array(second->buffer_sharps);
        array(second->image_keys);
        array(second->lookup.flats, MaxFlatBytes);
    }
}

bool Rasterizer::NeedsPipeline(const AmdGpu::Regs& regs) {
    return GetDrawFilter(regs).filter == DrawFilter::None;
}

bool Rasterizer::FilterDraw() {
    const auto [filter, depth_copy, stencil_copy] = GetDrawFilter(liverpool->regs);
    switch (filter) {
    case DrawFilter::None:
        return true;
    case DrawFilter::EliminateFastClear:
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    case DrawFilter::FmaskDecompress:
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        EliminateFastClear();
        return false;
    case DrawFilter::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    case DrawFilter::NoPrimitive:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    case DrawFilter::Copy:
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }
    UNREACHABLE();
}

const GraphicsPipeline* Rasterizer::LookupPipeline(const DrawIndirectParams& params) {
    const DrawAhead* const ahead = std::exchange(draw_ahead, nullptr);
    vertex_ahead = nullptr;
    if (ahead && ahead->lookup.pipeline) {
        if (const u64 tick = scheduler.CurrentTick(); tick != stream_ahead_tick) {
            // Everything recorded before the tick changed has run. Since then, only the last
            // draw prepared ahead and this one can have recorded reads of their hints.
            stream_ahead_tick = tick;
            buffer_cache.ReleaseStreamAhead(stream_ahead_position);
        }
        stream_ahead_position = ahead->stream_position;
        if (LoadStageUserDataAhead(*ahead)) {
            ++AmdGpu::cp_perf.la_lookups_used;
            buffer_cache.SetStreamHints(ahead->stream_hints, ahead->binding_write_epoch);
            vertex_ahead = ahead;
            return ahead->lookup.pipeline;
        }
        ++AmdGpu::cp_perf.la_lookups_rejected;
    }
    const auto lk = LockPipelineLookups();
    const GraphicsPipeline* const pipeline =
        pipeline_cache.GetGraphicsPipeline(liverpool->regs, AmdGpu::cp_perf, params, &own_lookup);
    if (pipeline) {
        LoadStageUserData(own_lookup);
    }
    return pipeline;
}

void Rasterizer::LookAhead(DrawAhead& ahead, const AmdGpu::Regs& regs, bool is_indexed,
                           u32 index_offset, u64 run, AmdGpu::CpPerf& perf) {
    ahead.lookup.pipeline = nullptr;
    if (!NeedsPipeline(regs) || !pipeline_cache.RecordsLookupReads(regs)) {
        return;
    }
    ahead.gpu_write_epoch = memory->GpuWriteEpoch();
    {
        std::scoped_lock lk{pipeline_cache.LookupMutex()};
        pipeline_cache.GetGraphicsPipeline(regs, perf, {}, &ahead.lookup);
    }
    if (!ahead.lookup.pipeline) {
        return;
    }
    ahead.run = run;
    ahead.step_rate_0 = regs.vgt_instance_step_rate_0;
    ahead.step_rate_1 = regs.vgt_instance_step_rate_1;
    ahead.index_address = 0;
    ahead.index_size = 0;
    if (is_indexed) {
        const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
        const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
        const VAddr base_address = regs.index_base_address.Address<VAddr>();
        if (base_address != 0) {
            const u32 num_indices = std::min(regs.num_indices, regs.max_index_size);
            ahead.index_address = base_address + u64(index_offset) * index_size;
            ahead.index_size = u64(num_indices) * index_size;
        }
    }
}

void Rasterizer::BindAhead(DrawAhead& ahead, AmdGpu::CpPerf& perf) {
    using Result = VideoCore::BufferCache::StreamAheadResult;
    const u64 stream_t0 = AmdGpu::PerfNs();
    const PipelineLookup& lookup = ahead.lookup;
    ahead.stream_hints.clear();
    ahead.binding_write_epoch = memory->GpuWriteEpoch();
    ahead.stream_position = buffer_cache.StreamAheadPosition();
    // The draw looks its hints up by address and size, so a binding this misses or gets wrong
    // only makes the draw read guest memory itself.
    const auto read = [&](VAddr address, u64 size) {
        if (std::ranges::any_of(ahead.stream_hints, [&](const auto& hint) {
                return hint.device_addr == address && hint.size == size;
            })) {
            return;
        }
        VideoCore::BufferCache::StreamHint hint;
        switch (buffer_cache.ReadStreamAhead(address, size, ahead.run, ahead.binding_write_epoch,
                                             hint)) {
        case Result::Known:
            ++perf.sa_ahead_known;
            ahead.stream_hints.push_back(hint);
            break;
        case Result::Same:
            ++perf.sa_ahead_same;
            ahead.stream_hints.push_back(hint);
            break;
        case Result::Copied:
            ++perf.sa_ahead_copied;
            perf.sa_ahead_bytes += size;
            ahead.stream_hints.push_back(hint);
            break;
        case Result::Full:
            ++perf.sa_ahead_full;
            break;
        case Result::None:
            break;
        }
    };

    // In the order the draw binds them: the buffers of each stage, the vertex buffers, the
    // index buffer.
    ahead.stage_sharps.clear();
    ahead.buffer_sharps.clear();
    ahead.image_keys.clear();
    for (u32 i = 0; i < lookup.num_stages; ++i) {
        const PipelineLookup::Stage& stage = lookup.stages[i];
        const Shader::StageUserDataView user_data{
            stage.pgm_base, {lookup.flats.data() + stage.flat_begin, stage.flat_size}};
        ahead.stage_sharps.push_back({
            .info = stage.info,
            .first_buffer = static_cast<u32>(ahead.buffer_sharps.size()),
            .first_image = static_cast<u32>(ahead.image_keys.size()),
            .num_buffers = static_cast<u32>(stage.info->buffers.size()),
            .num_images = static_cast<u32>(stage.info->images.size()),
        });
        const DrawAhead::StageSharps& stage_sharps = ahead.stage_sharps.back();
        for (u32 buffer = 0; buffer < stage_sharps.num_buffers; ++buffer) {
            const Shader::BufferResource& desc = stage.info->buffers[buffer];
            if (desc.IsSpecial()) {
                continue;
            }
            const auto vsharp = desc.GetSharp(user_data);
            ahead.buffer_sharps.push_back(vsharp);
            if (!desc.is_written) {
                read(vsharp.base_address, vsharp.GetSize());
            }
        }
        for (u32 image = 0; image < stage_sharps.num_images; ++image) {
            const Shader::ImageResource& desc = stage.info->images[image];
            DrawAhead::ImageKey& image_key = ahead.image_keys.emplace_back();
            image_key.key = MakeTextureKey(desc.GetSharp(user_data), desc, 0);
            image_key.hash = XXH3_64bits(&image_key.key, sizeof(image_key.key));
        }
    }

    // The V#s the lookup recorded are those of the fetch shader's attributes, in their order.
    const auto& sharps = lookup.vs_reads.sharps;
    ahead.attributes.clear();
    ahead.bindings.clear();
    ahead.guest_buffers.clear();
    ahead.vertex_ranges.clear();
    ahead.has_vertex_inputs =
        sharps.size() == lookup.pipeline->GetFetchShader().attributes.size() &&
        sharps.size() <= MaxVertexBufferCount;
    if (ahead.has_vertex_inputs && !sharps.empty()) {
        VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
        lookup.pipeline->GetVertexInputs(ahead.attributes, ahead.bindings, divisors,
                                         ahead.guest_buffers, ahead.step_rate_0, ahead.step_rate_1,
                                         nullptr, sharps);
    }
    VertexInputs<std::pair<VAddr, VAddr>> ranges;
    for (const AmdGpu::Buffer& buffer : sharps) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0 && ranges.size() < ranges.capacity()) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }
    if (!ranges.empty()) {
        std::ranges::sort(ranges);
        ahead.vertex_ranges.push_back(ranges[0]);
        for (const auto& [range_base, range_end] : ranges) {
            auto& [base_address, end_address] = ahead.vertex_ranges.back();
            if (end_address < range_base) {
                ahead.vertex_ranges.emplace_back(range_base, range_end);
            } else {
                end_address = std::max(end_address, range_end);
            }
        }
        for (const auto& [base_address, end_address] : ahead.vertex_ranges) {
            read(base_address, end_address - base_address);
        }
    }

    if (ahead.index_address != 0) {
        read(ahead.index_address, ahead.index_size);
    }
    perf.la_stream_ns += AmdGpu::PerfNs() - stream_t0;
}

std::unique_lock<std::mutex> Rasterizer::LockPipelineLookups() {
    std::unique_lock lk{pipeline_cache.LookupMutex(), std::try_to_lock};
    while (!lk.owns_lock()) {
        liverpool->RunPendingCommands();
        std::this_thread::yield();
        lk.try_lock();
    }
    return lk;
}

void Rasterizer::LoadStageUserData(const PipelineLookup& lookup) {
    const auto& regs = liverpool->regs;
    for (u32 i = 0; i < lookup.num_stages; ++i) {
        const PipelineLookup::Stage& stage = lookup.stages[i];
        auto& user_data = stage_user_data[u32(stage.info->sw_stage)];
        user_data.pgm_base = stage.pgm_base;
        user_data.user_data = regs.ProgramForStage(stage.hw_stage)->user_data;
        // The lookup stays as it is while its draw runs.
        user_data.flattened_ud_buf = {lookup.flats.data() + stage.flat_begin, stage.flat_size};
    }
}

bool Rasterizer::LoadStageUserDataAhead(const DrawAhead& ahead) {
    const PipelineLookup& lookup = ahead.lookup;
    const bool is_check = (++lookups_ahead & (LookupCheckInterval - 1)) == 0;
    if (!is_check && memory->IsFreeOfGpuWrites(ahead.gpu_write_epoch)) {
        LoadStageUserData(lookup);
        return true;
    }
    ++AmdGpu::cp_perf.la_lookups_checked;
    const auto& regs = liverpool->regs;
    for (u32 i = 0; i < lookup.num_stages; ++i) {
        const PipelineLookup::Stage& stage = lookup.stages[i];
        const Shader::Info& info = *stage.info;
        const auto* const pgm = regs.ProgramForStage(stage.hw_stage);
        auto& user_data = stage_user_data[u32(info.sw_stage)];
        const std::span lookup_flat{lookup.flats.data() + stage.flat_begin, stage.flat_size};
        // The registers the lookup was made from are the ones of this draw. What it read from
        // guest memory is read again: the shader, and the user data with all it points to.
        if (!pgm || !pgm->Address<u32*>()) {
            return false;
        }
        const auto params = AmdGpu::GetParams(*pgm);
        if (params.hash != info.pgm_hash || params.Base() != stage.pgm_base) {
            return false;
        }
        user_data.Refresh(info, params);
        if (!std::ranges::equal(user_data.flattened_ud_buf, lookup_flat)) {
            return false;
        }
        if (info.sw_stage != Shader::SwStage::Vertex || !info.has_fetch_shader) {
            continue;
        }
        // The fetch shader, and the V#s it loads.
        const auto& [fetch_code, sharps] = lookup.vs_reads;
        const u32* const code =
            Shader::Gcn::GetFetchShaderCode(user_data.user_data, info.fetch_shader_sgpr_base);
        if (!std::equal(fetch_code.begin(), fetch_code.end(), code)) {
            return false;
        }
        const auto& attributes = lookup.pipeline->GetFetchShader().attributes;
        if (attributes.size() != sharps.size()) {
            return false;
        }
        for (u32 attrib = 0; attrib < attributes.size(); ++attrib) {
            if (attributes[attrib].GetSharp(user_data) != sharps[attrib]) {
                return false;
            }
        }
    }
    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask =
            regs.color_target_mask.GetMask(cb) & regs.color_shader_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        auto& lookup = cb_lookups[cb];
        const u64 generation = texture_cache.RegistryGeneration();
        const u64 swaps = texture_cache.SwapGeneration();
        if (lookup.generation == generation && scheduler.CurrentTick() <= lookup.valid_until &&
            SameBits(lookup.col_buf, col_buf) && lookup.hint.raw == hint.raw &&
            texture_cache.IsSwapCurrent(lookup.swaps, lookup.binding.desc.info.guest_address,
                                        lookup.binding.desc.info.guest_size)) {
            desc = lookup.binding.desc;
            image_id = lookup.binding.image_id;
            texture_cache.TouchFoundImage(image_id);
        } else {
            std::construct_at(&desc, col_buf, hint);
            image_id = texture_cache.FindImage(desc);
            lookup.valid_until = texture_cache.FoundImageValidUntil(image_id, desc);
            const bool reusable = texture_cache.RegistryGeneration() == generation &&
                                  texture_cache.SwapGeneration() == swaps &&
                                  lookup.valid_until != 0;
            lookup.generation = reusable ? generation : 0;
            lookup.swaps = swaps;
            lookup.col_buf = col_buf;
            lookup.hint = hint;
            lookup.binding = {image_id, desc};
        }
        bound_images.emplace_back(image_id);
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        auto& lookup = db_lookups[regs.depth_control.depth_write_enable];
        const u64 generation = texture_cache.RegistryGeneration();
        const u64 swaps = texture_cache.SwapGeneration();
        if (lookup.generation == generation && scheduler.CurrentTick() <= lookup.valid_until &&
            SameBits(lookup.depth_buffer, regs.depth_buffer) &&
            SameBits(lookup.depth_view, regs.depth_view) && lookup.htile_address == htile_address &&
            lookup.hint.raw == hint.raw &&
            texture_cache.IsSwapCurrent(lookup.swaps, lookup.binding.desc.info.guest_address,
                                        lookup.binding.desc.info.guest_size)) {
            desc = lookup.binding.desc;
            image_id = lookup.binding.image_id;
            texture_cache.TouchFoundImage(image_id);
        } else {
            auto& perf = AmdGpu::cp_perf;
            perf.db_expired +=
                lookup.generation == generation && scheduler.CurrentTick() > lookup.valid_until;
            ++perf.db_misses[lookup.generation == 0                              ? 0
                             : lookup.generation != generation                   ? 1
                             : !SameBits(lookup.depth_buffer, regs.depth_buffer) ? 2
                             : !SameBits(lookup.depth_view, regs.depth_view)     ? 3
                             : lookup.htile_address != htile_address || lookup.hint.raw != hint.raw
                                 ? 4
                                 : 5];
            std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                              htile_address, hint);
            image_id = texture_cache.FindImage(desc);
            const bool is_perfect = texture_cache.IsPerfectMatch(image_id, desc);
            lookup.valid_until = texture_cache.FoundImageValidUntil(image_id, desc);
            const bool reusable = texture_cache.RegistryGeneration() == generation &&
                                  texture_cache.SwapGeneration() == swaps &&
                                  lookup.valid_until != 0;
            if (!reusable) {
                ++perf.db_unreusable[texture_cache.RegistryGeneration() != generation ? 0
                                     : texture_cache.SwapGeneration() != swaps        ? 1
                                                                                      : 2];
                if (!is_perfect && AmdGpu::TakePerfLine(AmdGpu::perf_db_site_budget)) {
                    LOG_INFO(Render,
                             "PERF db_site hint={}x{} pitch={} {} | cache id={} {} | region{}",
                             hint.width, hint.height, desc.info.pitch,
                             VideoCore::TextureCache::DescribeImage(desc.info), image_id.index,
                             VideoCore::TextureCache::DescribeImage(
                                 texture_cache.GetImage(image_id).info),
                             texture_cache.DescribeRegion(desc.info));
                }
            }
            lookup.generation = reusable ? generation : 0;
            lookup.swaps = swaps;
            lookup.depth_buffer = regs.depth_buffer;
            lookup.depth_view = regs.depth_view;
            lookup.htile_address = htile_address;
            lookup.hint = hint;
            lookup.binding = {image_id, desc};
        }
        bound_images.emplace_back(image_id);
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.image_id = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs,
                                          const Shader::StageUserData& info,
                                          const Shader::Gcn::FetchShaderData& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (!fetch_shader.Empty()) {
        if (vertex_offset == 0 && fetch_shader.vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader.vertex_offset_sgpr];
        }
        if (fetch_shader.instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader.instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    const bool is_sampled =
        AmdGpu::PerfDrawTimings && (perf_vk_counter++ & (AmdGpu::PerfVkSampleRate - 1)) == 0;
    AmdGpu::perf_draw_scale = is_sampled ? AmdGpu::PerfVkSampleRate : 0;
    SCOPE_EXIT {
        AmdGpu::perf_vk_sample = false;
        AmdGpu::perf_draw_scale = 1;
    };
    const AmdGpu::CpDrawPerfScope perf_scope{AmdGpu::cp_perf.draw_ns};
    ++AmdGpu::cp_perf.draws;

    const u64 pre_t0 = AmdGpu::PerfInDrawNs();
    FlushPeriodically();
    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }
    AmdGpu::cp_perf.draw_pre_ns += AmdGpu::PerfInDrawNs() - pre_t0;

    const auto& regs = liverpool->regs;
    const StreamHintsScope hints_scope{buffer_cache};
    const GraphicsPipeline* pipeline;
    {
        const AmdGpu::CpDrawPerfScope perf_pipeline{AmdGpu::cp_perf.draw_pipeline_ns};
        pipeline = LookupPipeline({});
    }
    if (!pipeline) {
        return;
    }
    if (pipeline != perf_last_pipeline) {
        ++AmdGpu::cp_perf.pipe_changes;
        perf_last_pipeline = pipeline;
    }
    AmdGpu::perf_vk_sample = is_sampled;

    const u64 bind_t0 = AmdGpu::PerfInDrawNs();
    PrepareRenderState(pipeline);
    const u64 bind_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_prep_ns += bind_t1 - bind_t0;
    if (!BindResources(pipeline)) {
        return;
    }
    const u64 bind_t2 = AmdGpu::PerfInDrawNs();
    const auto state = BeginRendering(pipeline);
    const u64 bind_t3 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_begin_ns += bind_t3 - bind_t2;
    AmdGpu::cp_perf.draw_bind_ns += bind_t3 - bind_t0;

    scheduler.DeferRecording();
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        const u64 index_t0 = AmdGpu::PerfInDrawNs();
        BindIndexBuffer(index_offset);
        AmdGpu::cp_perf.index_ns += AmdGpu::PerfInDrawNs() - index_t0;
    }
    const u64 vb_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_vb_ns += vb_t1 - bind_t3;

    if (needs_barrier) {
        ++AmdGpu::cp_perf.draw_barriers;
        runtime.FlushBarriers();
    }
    const u64 barrier_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_barrier_ns += barrier_t1 - vb_t1;

    pipeline->BindResources(set_writes, push_data);
    const u64 desc_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_desc_ns += desc_t1 - barrier_t1;
    UpdateDynamicState(pipeline, is_indexed);
    const u64 dyn_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_dyn_ns += dyn_t1 - desc_t1;
    scheduler.BeginRendering(state);
    const u64 rp_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_rp_ns += rp_t1 - dyn_t1;

    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] =
        GetDrawOffsets(regs, stage_user_data[u32(Shader::SwStage::Vertex)], fetch_shader);

    {
        const AmdGpu::VkCallPerf vk_perf{AmdGpu::cp_perf.vk_draw_ns};
        scheduler.Record([handle = pipeline->Handle(), is_indexed, num_indices = regs.num_indices,
                          num_instances = regs.num_instances.NumInstances(),
                          vertex_offset = vertex_offset,
                          instance_offset = instance_offset](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);

            if (is_indexed) {
                cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(vertex_offset),
                                   instance_offset);
            } else {
                cmdbuf.draw(num_indices, num_instances, vertex_offset, instance_offset);
            }
        });
        // Whatever waits for the recorder thread hands the rest over.
        if ((++draws_recorded & (PublishDrawInterval - 1)) == 0) {
            scheduler.PublishRecording();
        }
    }
    DebugState.IncDrawCall();
    AmdGpu::cp_perf.cmd_record_ns += AmdGpu::PerfInDrawNs() - rp_t1;

    ResetBindings(false);
    AmdGpu::perf_vk_sample = false;
    AmdGpu::cp_perf.draw_cmd_ns += AmdGpu::PerfInDrawNs() - rp_t1;
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;
    const bool is_sampled =
        AmdGpu::PerfDrawTimings && (perf_vk_counter++ & (AmdGpu::PerfVkSampleRate - 1)) == 0;
    AmdGpu::perf_draw_scale = is_sampled ? AmdGpu::PerfVkSampleRate : 0;
    SCOPE_EXIT {
        AmdGpu::perf_draw_scale = 1;
    };
    const AmdGpu::CpDrawPerfScope perf_scope{AmdGpu::cp_perf.draw_ns};
    ++AmdGpu::cp_perf.draws;

    FlushPeriodically();
    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
        .is_indirect = true,
    };
    const StreamHintsScope hints_scope{buffer_cache};
    const GraphicsPipeline* pipeline;
    {
        const AmdGpu::CpDrawPerfScope perf_pipeline{AmdGpu::cp_perf.draw_pipeline_ns};
        pipeline = LookupPipeline(params);
    }
    if (!pipeline) {
        return;
    }

    const u64 bind_t0 = AmdGpu::PerfInDrawNs();
    PrepareRenderState(pipeline);
    const u64 bind_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_prep_ns += bind_t1 - bind_t0;
    if (!BindResources(pipeline)) {
        return;
    }
    const u64 bind_t2 = AmdGpu::PerfInDrawNs();
    const auto state = BeginRendering(pipeline);
    const u64 bind_t3 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.draw_begin_ns += bind_t3 - bind_t2;
    AmdGpu::cp_perf.draw_bind_ns += bind_t3 - bind_t0;

    scheduler.DeferRecording();
    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer(0, true);
    }

    const auto size = stride * max_count;
    const auto [buffer, base] = buffer_cache.ObtainBuffer(arg_address + offset, size);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
        bound_buffers.emplace_back(count_buffer, count_offset, 4,
                                   vk::AccessFlagBits2::eIndirectCommandRead);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    pipeline->BindResources(set_writes, push_data);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    ASSERT((is_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand)) ==
           stride);
    scheduler.Record([handle = pipeline->Handle(), is_indexed, buffer = buffer->Handle(),
                      base = base, has_count = count_address != 0,
                      count_buffer = count_address != 0 ? count_buffer->Handle() : vk::Buffer{},
                      count_offset = count_address != 0 ? count_offset : 0, max_count,
                      stride](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);

        if (is_indexed) {
            if (has_count) {
                cmdbuf.drawIndexedIndirectCount(buffer, base, count_buffer, count_offset, max_count,
                                                stride);
            } else {
                cmdbuf.drawIndexedIndirect(buffer, base, max_count, stride);
            }
        } else {
            if (has_count) {
                cmdbuf.drawIndirectCount(buffer, base, count_buffer, count_offset, max_count,
                                         stride);
            } else {
                cmdbuf.drawIndirect(buffer, base, max_count, stride);
            }
        }
    });
    scheduler.PublishRecording();
    DebugState.IncDrawCall();

    ResetBindings(false);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    const AmdGpu::CpPerfScope perf_scope{AmdGpu::cp_perf.dispatch_ns};
    ++AmdGpu::cp_perf.dispatches;

    FlushPeriodically(true);
    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline;
    {
        const AmdGpu::CpPerfScope perf_pipeline{AmdGpu::cp_perf.dispatch_pipeline_ns};
        const auto lk = LockPipelineLookups();
        pipeline = pipeline_cache.GetComputePipeline(liverpool->regs, cs_program, AmdGpu::cp_perf);
        if (pipeline) {
            stage_user_data[u32(Shader::SwStage::Compute)].Load(
                pipeline->GetStage(Shader::SwStage::Compute));
        }
    }
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    const u64 hle_t0 = AmdGpu::PerfNs();
    const bool is_hle = ExecuteShaderHLE(cs, stage_user_data[u32(Shader::SwStage::Compute)],
                                         liverpool->regs, cs_program, *this);
    const u64 bind_t0 = AmdGpu::PerfNs();
    AmdGpu::cp_perf.dispatch_hle_ns += bind_t0 - hle_t0;
    if (is_hle) {
        ++AmdGpu::cp_perf.dispatch_hles;
        return;
    }

    const bool is_bound = BindResources(pipeline);
    AmdGpu::cp_perf.dispatch_bind_ns += AmdGpu::PerfNs() - bind_t0;
    if (!is_bound) {
        ++AmdGpu::cp_perf.dispatch_skips;
        return;
    }

    scheduler.DeferRecording();
    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering(Scheduler::RenderEnd::Dispatch);
    pipeline->BindResources(set_writes, push_data);

    scheduler.Record([handle = pipeline->Handle(), dim_x = cs_program.dim_x,
                      dim_y = cs_program.dim_y,
                      dim_z = cs_program.dim_z](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(dim_x, dim_y, dim_z);
    });
    scheduler.PublishRecording();
    DebugState.IncDispatch();

    ResetBindings(true);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    const AmdGpu::CpPerfScope perf_scope{AmdGpu::cp_perf.dispatch_ns};
    ++AmdGpu::cp_perf.dispatches;

    FlushPeriodically(true);
    scheduler.PopPendingOperations();

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline;
    {
        const AmdGpu::CpPerfScope perf_pipeline{AmdGpu::cp_perf.dispatch_pipeline_ns};
        const auto lk = LockPipelineLookups();
        pipeline = pipeline_cache.GetComputePipeline(liverpool->regs, cs_program, AmdGpu::cp_perf);
        if (pipeline) {
            stage_user_data[u32(Shader::SwStage::Compute)].Load(
                pipeline->GetStage(Shader::SwStage::Compute));
        }
    }
    if (!pipeline) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);
    bound_buffers.emplace_back(buffer, base, size, vk::AccessFlagBits2::eIndirectCommandRead);

    scheduler.DeferRecording();
    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering(Scheduler::RenderEnd::Dispatch);
    pipeline->BindResources(set_writes, push_data);

    scheduler.Record([handle = pipeline->Handle(), buffer = buffer->Handle(),
                      base = base](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatchIndirect(buffer, base);
    });
    scheduler.PublishRecording();
    DebugState.IncDispatch();

    ResetBindings(true);
}

void Rasterizer::FlushPeriodically(bool is_dispatch) {
    if (submit_draw_interval == 0) {
        return;
    }
    if (const u64 tick = scheduler.CurrentTick(); tick != last_submit_tick) {
        last_submit_tick = tick;
        draws_since_submit = 0;
    }
    ++draws_since_submit;
    // Auto: GPU writes to memory the game's threads are known to touch wait in the open command
    // buffer. They are submitted at the first draw behind the dispatches that made them, so
    // that the GPU runs them and they can be downloaded before those threads get there.
    const bool is_auto = submit_draw_interval == SubmitIntervalAuto;
    const bool is_early = is_auto && !is_dispatch && draws_since_submit >= AutoSubmitMinDraws &&
                          buffer_cache.IsEagerWritePending();
    // Auto: the GPU has run everything submitted, so what is recorded is all it could work on.
    // A GPU that is behind gets its work at the largest distance, and so does every GPU while
    // the GPU thread has no time to spare: a submit takes its time.
    bool is_idle = false;
    if (is_auto && has_submit_slack && !is_early && draws_since_submit >= AutoSubmitIdleDraws &&
        draws_since_submit < AutoSubmitMaxDraws && draws_since_submit % AutoSubmitMinDraws == 0) {
        const u64 check_t0 = AmdGpu::PerfNs();
        is_idle = scheduler.IsFree(scheduler.CurrentTick() - 1);
        ++AmdGpu::cp_perf.auto_checks;
        AmdGpu::cp_perf.auto_check_ns += AmdGpu::PerfNs() - check_t0;
    }
    const bool is_due = is_early || is_idle ||
                        draws_since_submit >= (is_auto ? AutoSubmitMaxDraws : submit_draw_interval);
    if (is_due) {
        const AmdGpu::CpPerfScope perf_flush{AmdGpu::cp_perf.flush_ns};
        ++AmdGpu::cp_perf.flushes;
        AmdGpu::cp_perf.early_flushes += is_early;
        AmdGpu::cp_perf.idle_flushes += is_idle;
        DebugState.frame_inner_submits.Add(1);
        DebugState.frame_inner_submit_draws.Add(draws_since_submit);
        scheduler.Flush();
    }
}

u64 Rasterizer::Flush() {
    const AmdGpu::CpPerfScope perf_flush{AmdGpu::cp_perf.flush_ns};
    ++AmdGpu::cp_perf.flushes;
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::OnSubmit() {
    buffer_cache.TickFrame();
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::OnFence() {
    const AmdGpu::CpPerfScope perf_fence{AmdGpu::cp_perf.fence_ns};
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // The lookup of a draw holds while the draw runs, and with it what was made of its user
    // data. One draw in SharpsAheadCheckInterval makes all of it again and compares.
    sharps_ahead = pipeline->IsCompute() ? nullptr : vertex_ahead;
    is_sharps_check = sharps_ahead && (++sharps_ahead_uses & (SharpsAheadCheckInterval - 1)) == 0;
    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                          stage->samplers.size());
        const u64 buffers_t0 = AmdGpu::PerfDrawNs();
        BindBuffers(*stage, binding, push_data);
        const u64 textures_t0 = AmdGpu::PerfDrawNs();
        BindTextures(*stage, binding);
        const u64 textures_t1 = AmdGpu::PerfDrawNs();
        AmdGpu::cp_perf.bind_buffers_ns += textures_t0 - buffers_t0;
        AmdGpu::cp_perf.bind_textures_ns += textures_t1 - textures_t0;
        if (pipeline->IsCompute()) {
            AmdGpu::cp_perf.dispatch_bindbuf_ns += textures_t0 - buffers_t0;
            AmdGpu::cp_perf.dispatch_bindtex_ns += textures_t1 - textures_t0;
        }
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        const AmdGpu::CpPerfScope perf_dma{AmdGpu::cp_perf.dma_sync_ns};
        buffer_cache.SynchronizeDmaBuffers();
    }

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline) {
    const u64 vi_t0 = AmdGpu::PerfInDrawNs();
    const auto& regs = liverpool->regs;
    VertexInputs<vk::VertexInputAttributeDescription2EXT> own_attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> own_bindings;
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<AmdGpu::Buffer> own_guest_buffers;
    // The lookup the pipeline came from holds for this draw, and with it the V#s it read.
    const DrawAhead* ahead = std::exchange(vertex_ahead, nullptr);
    if (ahead && (!ahead->has_vertex_inputs || !instance.IsVertexInputDynamicState() ||
                  ahead->step_rate_0 != regs.vgt_instance_step_rate_0 ||
                  ahead->step_rate_1 != regs.vgt_instance_step_rate_1)) {
        ahead = nullptr;
    }
    const bool is_check = ahead && (++vertex_ahead_uses & (VertexAheadCheckInterval - 1)) == 0;
    if (!ahead || is_check) {
        pipeline->GetVertexInputs(own_attributes, own_bindings, divisors, own_guest_buffers,
                                  regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1,
                                  &stage_user_data[u32(Shader::SwStage::Vertex)]);
    }
    if (is_check) {
        ++AmdGpu::cp_perf.va_checks;
        if (!std::ranges::equal(own_attributes, ahead->attributes) ||
            !std::ranges::equal(own_bindings, ahead->bindings) ||
            !std::ranges::equal(own_guest_buffers, ahead->guest_buffers)) {
            ++AmdGpu::cp_perf.va_diffs;
            ahead = nullptr;
        }
    }
    AmdGpu::cp_perf.va_used += ahead != nullptr;
    const auto& attributes = ahead ? ahead->attributes : own_attributes;
    const auto& bindings = ahead ? ahead->bindings : own_bindings;
    const auto& guest_buffers = ahead ? ahead->guest_buffers : own_guest_buffers;

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        const AmdGpu::VkCallPerf vk_perf{AmdGpu::cp_perf.vk_vi_ns};
        const std::size_t bindings_size =
            bindings.size() * sizeof(vk::VertexInputBindingDescription2EXT);
        const std::size_t attributes_size =
            attributes.size() * sizeof(vk::VertexInputAttributeDescription2EXT);
        scheduler.Record(
            bindings_size + attributes_size,
            [&](u8* data) {
                std::memcpy(data, bindings.data(), bindings_size);
                std::memcpy(data + bindings_size, attributes.data(), attributes_size);
            },
            [num_bindings = static_cast<u32>(bindings.size()),
             num_attributes = static_cast<u32>(attributes.size())](vk::CommandBuffer cmdbuf,
                                                                   const u8* data) {
                const auto* const binding_descs =
                    reinterpret_cast<const vk::VertexInputBindingDescription2EXT*>(data);
                const auto* const attribute_descs =
                    reinterpret_cast<const vk::VertexInputAttributeDescription2EXT*>(binding_descs +
                                                                                     num_bindings);
                cmdbuf.setVertexInputEXT(num_bindings, binding_descs, num_attributes,
                                         attribute_descs);
            });
    }
    const u64 vi_t1 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.vi_inputs_ns += vi_t1 - vi_t0;

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    VertexInputs<BufferRange> ranges_merged{};
    if (ahead) {
        for (const auto& [base_address, end_address] : ahead->vertex_ranges) {
            ranges_merged.emplace_back(base_address, end_address);
        }
    } else {
        for (const auto& buffer : guest_buffers) {
            if (buffer.base_address != 0 && buffer.GetSize() > 0) {
                ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
            }
        }
    }

    // Merge connecting ranges together
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) = buffer_cache.ObtainBuffer(range.base_address, size);
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
        bound_buffers.emplace_back(range.buffer, range.offset, size,
                                   vk::AccessFlagBits2::eVertexAttributeRead);
    }

    const u64 vi_t2 = AmdGpu::PerfInDrawNs();
    AmdGpu::cp_perf.vi_obtain_ns += vi_t2 - vi_t1;

    // Bind vertex buffers
    VertexInputs<vk::Buffer> host_buffers;
    VertexInputs<vk::DeviceSize> host_offsets;
    VertexInputs<vk::DeviceSize> host_sizes;
    VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->buffer->Handle());
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    // The handles, offsets, sizes and strides follow each other in the recorded data.
    static_assert(sizeof(vk::Buffer) == sizeof(vk::DeviceSize));
    const auto num_buffers = static_cast<u32>(guest_buffers.size());
    const std::size_t array_size = num_buffers * sizeof(vk::DeviceSize);
    scheduler.Record(
        array_size * 4,
        [&](u8* data) {
            std::memcpy(data, host_buffers.data(), array_size);
            std::memcpy(data + array_size, host_offsets.data(), array_size);
            std::memcpy(data + array_size * 2, host_sizes.data(), array_size);
            std::memcpy(data + array_size * 3, host_strides.data(), array_size);
        },
        [num_buffers, is_dynamic = instance.IsVertexInputDynamicState()](vk::CommandBuffer cmdbuf,
                                                                         const u8* data) {
            const auto* const buffers = reinterpret_cast<const vk::Buffer*>(data);
            const auto* const offsets = reinterpret_cast<const vk::DeviceSize*>(data) + num_buffers;
            if (is_dynamic) {
                cmdbuf.bindVertexBuffers(0, num_buffers, buffers, offsets);
            } else {
                cmdbuf.bindVertexBuffers2(0, num_buffers, buffers, offsets, offsets + num_buffers,
                                          offsets + num_buffers * 2);
            }
        });
    AmdGpu::cp_perf.vi_bind_ns += AmdGpu::PerfInDrawNs() - vi_t2;
}

void Rasterizer::BindIndexBuffer(u32 index_offset, bool is_indirect) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr base_address = regs.index_base_address.Address<VAddr>();
    const VAddr index_address = base_address + u64(index_offset) * index_size;

    // Bind index buffer.
    const u32 num_indices =
        is_indirect ? regs.max_index_size : std::min(regs.num_indices, regs.max_index_size);
    const u64 requested_size = u64(num_indices) * index_size;
    const u64 mapped_size =
        base_address != 0 ? memory->ClampRangeSize(index_address, requested_size) : 0;
    u32 index_buffer_size = static_cast<u32>(Common::AlignDown(mapped_size, index_size));
    vk::Buffer index_buffer{};
    u64 buffer_offset{};
    if (index_buffer_size != 0) {
        const auto [buffer, offset] = buffer_cache.ObtainBuffer(index_address, index_buffer_size);
        needs_barrier |= runtime.IsBufferAccessed(buffer, offset, index_buffer_size);
        bound_buffers.emplace_back(buffer, offset, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = buffer->Handle();
        buffer_offset = offset;
    } else if (!instance.IsMaintenance6Supported()) {
        const auto* null_buffer = buffer_cache.GetNullIndexBuffer();
        index_buffer_size = static_cast<u32>(null_buffer->SizeBytes());
        needs_barrier |= runtime.IsBufferAccessed(null_buffer, 0, index_buffer_size);
        bound_buffers.emplace_back(null_buffer, 0, index_buffer_size,
                                   vk::AccessFlagBits2::eIndexRead);
        index_buffer = null_buffer->Handle();
    }

    const AmdGpu::VkCallPerf vk_perf{AmdGpu::cp_perf.vk_index_ns};
    scheduler.Record([index_buffer, buffer_offset, index_buffer_size, index_type,
                      has_size = instance.IsMaintenance5Supported()](vk::CommandBuffer cmdbuf) {
        if (has_size) {
            cmdbuf.bindIndexBuffer2KHR(index_buffer, buffer_offset, index_buffer_size, index_type);
        } else {
            cmdbuf.bindIndexBuffer(index_buffer, buffer_offset, index_type);
        }
    });
}

void Rasterizer::NoteSharpsDiff(const Shader::Info& stage, std::string_view kind, u32 index) {
    ++AmdGpu::cp_perf.ra_diffs;
    if (sharps_logged_diffs < 16) {
        ++sharps_logged_diffs;
        LOG_ERROR(Render, "PERF ra_diff stage={:#x} {}={}", stage.pgm_hash, kind, index);
    }
}

const DrawAhead::StageSharps* Rasterizer::FindSharpsAhead(const Shader::Info& stage) const {
    if (!sharps_ahead) {
        return nullptr;
    }
    for (const DrawAhead::StageSharps& sharps : sharps_ahead->stage_sharps) {
        if (sharps.info == &stage && sharps.num_buffers == stage.buffers.size() &&
            sharps.num_images == stage.images.size()) {
            return &sharps;
        }
    }
    return nullptr;
}

void Rasterizer::ResetBindings(bool is_compute) {
    const u64 reset_t0 = AmdGpu::PerfDrawNs();
    AmdGpu::cp_perf.bound_images += static_cast<u32>(bound_images.size());
    AmdGpu::cp_perf.bound_buffers += static_cast<u32>(bound_buffers.size());
    for (auto& image_id : bound_images) {
        auto& image = texture_cache.GetImage(image_id);
        if (image.binding.is_target || image.binding.is_storage) {
            image.InvalidateSubrects();
        }
        image.binding = {};
    }
    const u64 reset_t1 = AmdGpu::PerfDrawNs();
    AmdGpu::cp_perf.reset_images_ns += reset_t1 - reset_t0;
    // Only the host writes the stream buffer, so no hazard check ever looks at its read
    // ranges; reads of it only add their stage to the next barrier.
    const VideoCore::Buffer* const stream_buffer = &buffer_cache.GetStreamBuffer();
    for (const auto [buffer, offset, size, src_access] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        if (buffer == stream_buffer && !(src_access & vk::AccessFlagBits2::eShaderWrite)) {
            runtime.AccessStage(dst_stage);
            continue;
        }
        runtime.AccessBuffer(buffer, offset, size, dst_stage, src_access);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
    AmdGpu::cp_perf.reset_buffers_ns += AmdGpu::PerfDrawNs() - reset_t1;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    const auto& user_data = stage_user_data[u32(Shader::SwStage::Compute)];

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(user_data).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(user_data).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const auto& user_data = stage_user_data[u32(Shader::SwStage::Compute)];
    const AmdGpu::Buffer buf0 = desc0.GetSharp(user_data);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(user_data);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() < 2 || info.buffers.size() > 3 ||
        !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }
    if (info.buffers.size() == 3 && info.buffers[2].buffer_type != Shader::BufferType::Flatbuf) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const auto& user_data = stage_user_data[u32(Shader::SwStage::Compute)];
    const AmdGpu::Buffer buf0 = desc0.GetSharp(user_data);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(user_data);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid and number of component must match
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth ||
        AmdGpu::NumComponents(buf1.GetDataFmt()) != vk::componentCount(image1.info.pixel_format)) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const auto& user_data = stage_user_data[u32(stage.sw_stage)];
    const u64 alignment = instance.StorageMinAlignment();
    const DrawAhead::StageSharps* const ahead = FindSharpsAhead(stage);
    u32 ahead_buffer = ahead ? ahead->first_buffer : 0;
    AmdGpu::cp_perf.ra_stages += ahead != nullptr;
    for (const auto& desc : stage.buffers) {
        const auto src_access =
            vk::AccessFlagBits2::eShaderRead |
            (desc.is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone);
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                needs_barrier |=
                    runtime.IsBufferAccessed(gds_buf, 0, gds_buf->SizeBytes(), desc.is_written);
                bound_buffers.emplace_back(gds_buf, 0, gds_buf->SizeBytes(), src_access);
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = user_data.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(user_data.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = liverpool->GetCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            AmdGpu::Buffer vsharp;
            if (ahead) {
                vsharp = sharps_ahead->buffer_sharps[ahead_buffer++];
                if (is_sharps_check) {
                    ++AmdGpu::cp_perf.ra_checks;
                    if (const auto own = desc.GetSharp(user_data); !SameBits(vsharp, own)) {
                        NoteSharpsDiff(stage, "buffer", ahead_buffer - 1 - ahead->first_buffer);
                        vsharp = own;
                    }
                }
            } else {
                vsharp = desc.GetSharp(user_data);
            }
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                if (size != vsharp.GetSize()) {
                    LOG_DEBUG(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                VideoCore::ObtainBufferFlags flags = {};
                if (desc.is_written) {
                    flags |= VideoCore::ObtainBufferFlags::IsWritten;
                }
                if (desc.is_formatted) {
                    flags |= VideoCore::ObtainBufferFlags::IsTexelBuffer;
                }
                const u64 obtain_t0 = desc.is_written ? AmdGpu::PerfNs() : 0;
                const auto [buffer, offset] =
                    buffer_cache.ObtainBuffer(vsharp.base_address, size, flags);
                if (desc.is_written) {
                    AmdGpu::cp_perf.ow_obtain_ns += AmdGpu::PerfNs() - obtain_t0;
                }
                // The limit is a power of two.
                const u64 offset_aligned = Common::AlignDownPow2(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(buffer->Handle(), offset_aligned, size + adjust);
                bound_buffers.emplace_back(buffer, offset, size, src_access);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    const u64 inval_t0 = AmdGpu::PerfNs();
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
                    AmdGpu::cp_perf.ow_inval_ns += AmdGpu::PerfNs() - inval_t0;
                }
                needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
            }
        }

        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eStorageBuffer;
        set_write.pBufferInfo = &buffer_infos.back();
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    const bool is_timed = AmdGpu::perf_vk_sample;
    const u64 first_t0 = is_timed ? AmdGpu::PerfNs() : 0;
    const bool is_checked = AmdGpu::perf_bind_check && stage.sw_stage != Shader::SwStage::Compute;
    if (is_checked) {
        checked_image_has_key.fill(false);
    }
    const u32 check_cmds_before = AmdGpu::cp_perf.rec_cmds;
    const u64 check_generation = texture_cache.RegistryGeneration();
    u32 check_images = 0;
    u32 check_same = 0;
    u32 check_equal = 0;
    const auto& user_data = stage_user_data[u32(stage.sw_stage)];
    const u32 first_image_idx = image_infos.size();
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    const DrawAhead::StageSharps* const ahead = FindSharpsAhead(stage);
    u32 ahead_image = ahead ? ahead->first_image : 0;
    u32 num_images{};
    for (const auto& image_desc : stage.images) {
        const DrawAhead::ImageKey* ahead_key =
            ahead ? &sharps_ahead->image_keys[ahead_image++] : nullptr;
        if (ahead_key && is_sharps_check) {
            ++AmdGpu::cp_perf.ra_checks;
            const TextureKey own = MakeTextureKey(image_desc.GetSharp(user_data), image_desc, 0);
            if (!SameBits(ahead_key->key, own) ||
                ahead_key->hash != XXH3_64bits(&own, sizeof(own))) {
                NoteSharpsDiff(stage, "image", ahead_image - 1 - ahead->first_image);
                ahead_key = nullptr;
            }
        }
        const auto tsharp = ahead_key ? ahead_key->key.tsharp : image_desc.GetSharp(user_data);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !magic_enum::enum_contains(data_fmt) || !magic_enum::enum_contains(num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            image_bindings[num_images++].image_id = {};
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(tsharp);

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] = image_bindings[num_images++];
            ++AmdGpu::cp_perf.tex_lookups;
            const bool is_ahead = ahead_key && i == 0;
            const TextureKey key =
                is_ahead ? ahead_key->key : MakeTextureKey(tsharp, image_desc, static_cast<u32>(i));
            const u64 hash = is_ahead ? ahead_key->hash : XXH3_64bits(&key, sizeof(key));
            if (const u32 slot = num_images - 1; is_checked && slot < MaxCheckedSlots) {
                checked_image_keys[slot] = key;
                checked_image_has_key[slot] = true;
            }
            auto& lookup = FindTextureLookup(key, hash);
            const u64 generation = texture_cache.RegistryGeneration();
            const u64 swaps = texture_cache.SwapGeneration();
            if (lookup.generation == generation && scheduler.CurrentTick() <= lookup.valid_until &&
                SameBits(lookup.key, key) &&
                texture_cache.IsSwapCurrent(lookup.swaps, lookup.binding.desc.info.guest_address,
                                            lookup.binding.desc.info.guest_size)) {
                ++AmdGpu::cp_perf.tex_hits;
                desc = lookup.binding.desc;
                image_id = lookup.binding.image_id;
                texture_cache.TouchFoundImage(image_id);
            } else {
                if (!SameBits(lookup.key, key)) {
                    ++AmdGpu::cp_perf.tex_miss_key;
                } else if (lookup.generation != 0) {
                    ++AmdGpu::cp_perf.tex_miss_generation;
                } else {
                    ++AmdGpu::cp_perf.tex_miss_unreusable;
                }
                std::construct_at(&desc, tsharp, image_desc);

                if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                    ASSERT(num_bindings == 1);
                    desc.view_info.range.base.level += image_desc.constant_mip_index;
                    desc.view_info.range.extent.levels = 1;
                } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    desc.view_info.range.base.level += i;
                    desc.view_info.range.extent.levels = 1;
                }

                image_id = texture_cache.FindImage(desc);
                lookup.valid_until = texture_cache.FoundImageValidUntil(image_id, desc);
                const bool reusable = texture_cache.RegistryGeneration() == generation &&
                                      texture_cache.SwapGeneration() == swaps &&
                                      lookup.valid_until != 0;
                lookup.generation = reusable ? generation : 0;
                lookup.swaps = swaps;
                lookup.key = key;
                lookup.binding = {image_id, desc};
            }
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (desc.IsSubrectTexture(image->info)) {
                continue;
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    const u64 second_t0 = is_timed ? AmdGpu::PerfNs() : 0;
    // Second pass to re-bind images that were updated after binding
    for (u32 i = 0; i < num_images; ++i) {
        auto& [image_id, desc] = image_bindings[i];
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind ||
                (desc.IsSubrectTexture(old_image.info) &&
                 False(old_image.flags & VideoCore::ImageFlagBits::Registered))) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            if (desc.IsSubrectTexture(image.info)) {
                texture_cache.UpdateImage(image_id);
                auto& view = runtime.FindSubrectView(image, desc.info, desc.view_info);
                image_infos.emplace_back(VK_NULL_HANDLE, *view.image_view,
                                         vk::ImageLayout::eShaderReadOnlyOptimal);
                continue;
            }
            auto& image_view = texture_cache.FindTexture(image_id, desc);
            image.binding.is_storage |= is_storage;
            const auto binding = image.binding;

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
                if (instance.IsAttachmentFeedbackLoopLayoutSupported() && image.binding.is_target) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                        vk::PipelineStageFlagBits2::eAllGraphics, vk::AccessFlagBits2::eShaderRead);
                } else {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                }
            } else {
                if (is_storage) {
                    needs_barrier |= runtime.Transit(
                        &image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    needs_barrier |= runtime.Transit(
                        &image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                        vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (is_checked && i < MaxCheckedSlots && checked_image_has_key[i]) {
                auto& last = checked_images[u32(stage.sw_stage)][i];
                const auto& info = image_infos.back();
                const vk::ImageView view = info.imageView;
                const u64 run = liverpool->LookaheadRun();
                ++check_images;
                if (last.info == &stage && last.run == run &&
                    SameBits(last.key, checked_image_keys[i])) {
                    ++check_same;
                    check_equal += last.view == view && last.layout == info.imageLayout;
                }
                last = {&stage, run, checked_image_keys[i], view, info.imageLayout};
            }
        }
    }
    if (is_checked && check_images != 0) {
        auto& perf = AmdGpu::cp_perf;
        perf.bc_images += check_images;
        perf.bc_images_same += check_same;
        perf.bc_images_equal += check_equal;
        ++perf.bc_image_stages;
        if (check_same == check_images) {
            ++perf.bc_image_stages_same;
            perf.bc_image_stages_quiet += check_equal == check_images &&
                                          perf.rec_cmds == check_cmds_before &&
                                          texture_cache.RegistryGeneration() == check_generation;
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = array_size;
        set_write.descriptorType =
            is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
        set_write.pImageInfo = &image_infos[image_info_idx];

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    const u64 samplers_t0 = is_timed ? AmdGpu::PerfNs() : 0;
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(user_data);
        if (!ssharp.Valid() || (ssharp.border_color_type.Value() == AmdGpu::BorderColor::Custom &&
                                liverpool->regs.ta_bc_base.Address() == 0)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid S# max_aniso={}, filter_mode={}, mip_filter={}, "
                        "border_color_type={}, border_color_base={:#x}",
                        static_cast<u32>(ssharp.max_aniso.Value()),
                        static_cast<u32>(ssharp.filter_mode.Value()),
                        static_cast<u32>(ssharp.mip_filter.Value()),
                        static_cast<u32>(ssharp.border_color_type.Value()),
                        liverpool->regs.ta_bc_base.Address());
            ssharp = AmdGpu::Sampler{};
        }
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base, sampler.is_depth);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[set_write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
    if (is_timed) {
        auto& perf = AmdGpu::cp_perf;
        perf.tx_first_ns += (second_t0 - first_t0) * AmdGpu::PerfVkSampleRate;
        perf.tx_second_ns += (samplers_t0 - second_t0) * AmdGpu::PerfVkSampleRate;
        perf.tx_samplers_ns += (AmdGpu::PerfNs() - samplers_t0) * AmdGpu::PerfVkSampleRate;
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state;
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    bool has_attachment = false;
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        has_attachment = true;
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            runtime.FlushBarriers();
            needs_barrier |=
                runtime.Transit(image,
                                instance.IsAttachmentFeedbackLoopLayoutSupported()
                                    ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                    : vk::ImageLayout::eGeneral,
                                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                vk::AccessFlagBits2::eColorAttachmentWrite |
                                    vk::AccessFlagBits2::eColorAttachmentRead);
            attachment_feedback_loop = true;
        } else {
            needs_barrier |= runtime.Transit(image, vk::ImageLayout::eColorAttachmentOptimal,
                                             vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                             vk::AccessFlagBits2::eColorAttachmentWrite |
                                                 vk::AccessFlagBits2::eColorAttachmentRead,
                                             desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.image_id; image_id) {
        has_attachment = true;
        auto& desc = db_desc.desc;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        needs_barrier |= runtime.Transit(&image, new_layout,
                                         vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                             vk::PipelineStageFlagBits2::eLateFragmentTests,
                                         vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                             vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                                         desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};
        attachment.is_clear = 0;

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    // Without attachments the area stays at the max framebuffer size. Draws cannot pass the
    // scissors, so bound it by them.
    if (!has_attachment) {
        state.num_layers = 1;
        const AmdGpu::Scissor scissor = regs.GetCombinedScissor();
        if (scissor.bottom_right_x > 0 && scissor.bottom_right_y > 0) {
            state.width = std::min<u32>(state.width, scissor.bottom_right_x);
            state.height = std::min<u32>(state.height, scissor.bottom_right_y);
        }
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    const auto write_id = texture_cache.FindImage(write_desc);
    auto& write_image = texture_cache.GetImage(write_id);
    // The copy destination is never bound as a target, so its stencil plane is linked here.
    texture_cache.AssociateStencil(write_id, write_desc.info);

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    const AmdGpu::CpPerfScope perf_copy{AmdGpu::cp_perf.copy_ns};
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsWritten);
    }();
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    const AmdGpu::CpPerfScope perf_copy{AmdGpu::cp_perf.copy_ns};
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsTexelBuffer);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes,
                                         VideoCore::ObtainBufferFlags::IsWritten |
                                             VideoCore::ObtainBufferFlags::IsTexelBuffer);
    }();
    const vk::BufferCopy copy = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::AccessBesideGpuData(VAddr addr, u32 size, bool is_write, u8* data) {
    if (!IsMapped(addr, size) || !buffer_cache.AccessBesideGpuData(addr, size, is_write, data)) {
        return false;
    }
    if (is_write) {
        // Like InvalidateMemory does for a write that faults.
        texture_cache.InvalidateMemory(addr, size);
    }
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size, false, false, assume_locks);
    return true;
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateRegsDynamicState(const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    auto& dynamic_state = scheduler.GetDynamicState();
    // The values UpdateRegsDynamicState sets depend on config, context and uconfig registers
    // alone, and nothing else sets them: they hold while no such register changed.
    const u64 generation = liverpool->TakeStateRegsGeneration();
    if (generation != dynamic_regs_generation || is_indexed != dynamic_is_indexed) {
        UpdateRegsDynamicState(is_indexed);
        dynamic_regs_generation = generation;
        dynamic_is_indexed = is_indexed;
    } else {
        ++AmdGpu::cp_perf.ds_skips;
        if (dynamic_skips++ % DynamicCheckInterval == 0) {
            // A setter marks a value dirty only when it changes it.
            using DirtyBits = std::array<u8, sizeof(dynamic_state.dirty_state)>;
            DirtyBits was_dirty;
            std::memcpy(&was_dirty, &dynamic_state.dirty_state, sizeof(was_dirty));
            std::memset(&dynamic_state.dirty_state, 0, sizeof(dynamic_state.dirty_state));
            UpdateRegsDynamicState(is_indexed);
            DirtyBits changed;
            std::memcpy(&changed, &dynamic_state.dirty_state, sizeof(changed));
            ++AmdGpu::cp_perf.ds_checks;
            if (changed != DirtyBits{}) {
                ++AmdGpu::cp_perf.ds_diffs;
                if (dynamic_logged_diffs < 32) {
                    ++dynamic_logged_diffs;
                    u32 bits{};
                    std::memcpy(&bits, &changed, std::min(sizeof(bits), sizeof(changed)));
                    LOG_ERROR(Render_Vulkan, "PERF ds_diff dirty={:#x} generation={}", bits,
                              generation);
                }
            }
            for (std::size_t i = 0; i < changed.size(); ++i) {
                changed[i] |= was_dirty[i];
            }
            std::memcpy(&dynamic_state.dirty_state, &changed, sizeof(changed));
        }
    }
    UpdateColorBlendingState(pipeline);

    const AmdGpu::VkCallPerf vk_perf{AmdGpu::cp_perf.vk_dyn_ns};
    dynamic_state.Commit(instance, scheduler);
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const AmdGpu::Scissor scsr = regs.GetCombinedScissor();

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x =
                std::min(vp_scsr.bottom_right_x, s16(regs.viewport_scissors[i].bottom_right_x));
            vp_scsr.bottom_right_y =
                std::min(vp_scsr.bottom_right_y, s16(regs.viewport_scissors[i].bottom_right_y));
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan
