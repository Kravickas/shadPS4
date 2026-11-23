// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

/// What the lookahead threads prepare for a draw they reach before the GPU thread does. The
/// lookahead thread makes the lookup and takes the registers below, the binding thread the rest.
struct DrawAhead {
    PipelineLookup lookup;
    /// Count of the GPU's writes to guest memory before the lookup read any of it.
    u64 gpu_write_epoch{};
    u64 run{};
    /// The bytes the index buffer binding of the draw covers, if it has one.
    VAddr index_address{};
    u64 index_size{};
    u32 step_rate_0{};
    u32 step_rate_1{};

    /// Count of the GPU's writes to guest memory before the bytes of the bindings were read.
    u64 binding_write_epoch{};
    /// Position of the buffer cache's reads ahead before those of this draw.
    u64 stream_position{};
    /// The bytes of the draw's read-only bindings the stream buffer could serve.
    std::vector<VideoCore::BufferCache::StreamHint> stream_hints;
    /// The vertex inputs of the draw, from the V#s the lookup read, and the ranges of its vertex
    /// buffers merged and in address order.
    bool has_vertex_inputs{};
    VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    VertexInputs<std::pair<VAddr, VAddr>> vertex_ranges;
};

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// Performs a faulting CPU access that touches no GPU data directly on guest memory.
    /// Returns false if the fault has to be handled by InvalidateMemory or ReadMemory.
    bool AccessBesideGpuData(VAddr addr, u32 size, bool is_write, u8* data);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();
    void OnFence();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    /// For the lookahead thread: looks the pipeline of a draw up from the registers it will run
    /// with. perf gets the probe timings.
    /// run names the run of draws the draw belongs to.
    void LookAhead(DrawAhead& ahead, const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset,
                   u64 run, AmdGpu::CpPerf& perf);

    /// For the binding thread: reads the bytes of the bindings of a draw whose pipeline was
    /// looked up ahead and makes its vertex inputs.
    void BindAhead(DrawAhead& ahead, AmdGpu::CpPerf& perf);

    /// Hands over what the lookahead thread prepared for the next draw, if anything.
    void SetDrawAhead(const DrawAhead* ahead) {
        draw_ahead = ahead;
    }

    /// Whether a draw with these registers gets as far as looking its pipeline up.
    static bool NeedsPipeline(const AmdGpu::Regs& regs);

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    /// Returns the pipeline of the draw and loads its stages' user data, using the lookup made
    /// ahead when its inputs still hold.
    const GraphicsPipeline* LookupPipeline(const DrawIndirectParams& params);

    /// Takes the lock of the pipeline cache's lookups, running the commands of other threads
    /// while the lookahead thread holds it.
    std::unique_lock<std::mutex> LockPipelineLookups();

    /// Loads the user data of the stages of the GPU thread's own lookup.
    void LoadStageUserData(const PipelineLookup& lookup);

    /// Loads the user data of the stages of a lookup made ahead, if the lookup holds for the
    /// draw. Without GPU writes to guest memory since, what the lookup read is what the draw
    /// reads; else, and for one lookup in LookupCheckInterval, all of it is read again.
    bool LoadStageUserDataAhead(const DrawAhead& ahead);

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0, bool is_indirect = false);

    void ResetBindings(bool is_compute);

    void FlushPeriodically();

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    std::array<Shader::StageUserData, MaxShaderStages> stage_user_data;
    PipelineLookup own_lookup;
    const DrawAhead* draw_ahead{};
    /// The lookahead result the pipeline of the draw came from, for its vertex inputs.
    const DrawAhead* vertex_ahead{};
    static constexpr u32 VertexAheadCheckInterval = 64;
    u32 vertex_ahead_uses{};
    static constexpr u32 LookupCheckInterval = 64;
    u32 lookups_ahead{};
    /// For releasing the buffer cache's reads ahead: the tick the last draw prepared ahead
    /// started in, and where the reads stood before those of that draw.
    u64 stream_ahead_tick{};
    u64 stream_ahead_position{};
    const bool host_markers_enabled;
    const bool guest_markers_enabled;
    const u32 submit_draw_interval;
    u64 last_submit_tick{};
    u32 draws_since_submit{};
    const GraphicsPipeline* perf_last_pipeline{};
    u32 perf_vk_counter{};

    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    std::array<ImageBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    ImageBinding db_desc;

    /// Registers and result of the last render target lookup that found a perfect match. The
    /// same registers give the same image while the texture cache registry generation is equal
    /// and no image in its range was swapped with a parked alias.
    /// A lookup holds while generation and swaps are those of the texture cache and the
    /// scheduler tick is not past valid_until.
    struct ColorTargetLookup {
        u64 generation{};
        u64 swaps{};
        u64 valid_until{};
        AmdGpu::ColorBuffer col_buf;
        AmdGpu::CbDbExtent hint;
        ImageBinding binding;
    };
    struct DepthTargetLookup {
        u64 generation{};
        u64 swaps{};
        u64 valid_until{};
        AmdGpu::DepthBuffer depth_buffer;
        AmdGpu::DepthView depth_view;
        VAddr htile_address;
        AmdGpu::CbDbExtent hint;
        ImageBinding binding;
    };
    std::array<ColorTargetLookup, AmdGpu::NUM_COLOR_BUFFERS> cb_lookups{};
    /// Of the depth control register the lookup only reads depth_write_enable, which is the
    /// index here.
    std::array<DepthTargetLookup, 2> db_lookups{};

    /// Inputs of a texture lookup: the T# and the shader resource fields ImageDesc reads.
    struct TextureKey {
        AmdGpu::Image tsharp;
        u32 is_depth;
        u32 is_array;
        u32 is_written;
        u32 mip_level; ///< Mip the binding is limited to by the storage fallback, or ~0U
    };
    /// Last texture lookup that found a perfect match, per hash slot of its key. The same key
    /// gives the same image while the texture cache registry generation is equal and no image
    /// in its range was swapped with a parked alias.
    struct TextureLookup {
        u64 generation{};
        u64 swaps{};
        u64 valid_until{};
        TextureKey key;
        ImageBinding binding;
    };
    std::array<TextureLookup, 4096> texture_lookups{};

    /// Probe: what the image slots of each graphics stage resolved to in the last draw.
    struct CheckedImage {
        const Shader::Info* info;
        u64 run;
        TextureKey key;
        vk::ImageView view;
        vk::ImageLayout layout;
    };
    static constexpr u32 MaxCheckedSlots = 32;
    std::array<std::array<CheckedImage, MaxCheckedSlots>, MaxShaderStages> checked_images{};
    std::array<TextureKey, MaxCheckedSlots> checked_image_keys{};
    std::array<bool, MaxCheckedSlots> checked_image_has_key{};

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        vk::AccessFlags2 src_access;
    };
    std::vector<BoundBuffer> bound_buffers;
    std::vector<VideoCore::ImageId> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    bool attachment_feedback_loop{};
    bool needs_barrier{};
};

} // namespace Vulkan
