// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
struct ComputeProgram;
struct CpPerf;
union Regs;
} // namespace AmdGpu

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
    }
};

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    bool is_indirect;
};

/// A graphics pipeline lookup together with what it read from guest memory, so that a lookup
/// made ahead of its draw can be checked against that memory when the draw runs.
struct PipelineLookup {
    struct Stage {
        const Shader::Info* info;
        u32 hw_stage; ///< Index of the program registers the stage was looked up from
        VAddr pgm_base;
        u32 flat_begin; ///< Position of the stage's flattened user data in flats
        u32 flat_size;
    };

    const GraphicsPipeline* pipeline{};
    u32 num_stages{};
    std::array<Stage, MaxShaderStages> stages;
    std::vector<u32> flats;
    Shader::VsInputsRead vs_reads;
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    /// Lookups come from the GPU thread and from its lookahead thread; whoever makes one holds
    /// this. The lookahead thread can wait for the GPU thread while it holds it (a readback of
    /// memory its lookup reads), so the GPU thread must not block on it.
    std::mutex& LookupMutex() {
        return lookup_mutex;
    }

    /// Whether a PipelineLookup holds all that a lookup with these registers reads from guest
    /// memory.
    bool RecordsLookupReads(const AmdGpu::Regs& regs) const;

    /// Looks the pipeline of a draw up from the given registers. perf gets the probe timings.
    /// With lookup, also records what the lookup read from guest memory.
    const GraphicsPipeline* GetGraphicsPipeline(const AmdGpu::Regs& regs, AmdGpu::CpPerf& perf,
                                                const DrawIndirectParams params = {},
                                                PipelineLookup* lookup = nullptr);

    const ComputePipeline* GetComputePipeline(const AmdGpu::Regs& regs,
                                              const AmdGpu::ComputeProgram& cs_program,
                                              AmdGpu::CpPerf& perf);

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    std::mutex lookup_mutex;
    /// State and probes of the lookup in progress.
    const AmdGpu::Regs* lookup_regs{};
    const AmdGpu::ComputeProgram* lookup_cs_program{};
    AmdGpu::CpPerf* lookup_perf{};
    /// Where the lookup in progress leaves what its vertex stage read, if anywhere.
    Shader::VsInputsRead* lookup_vs_reads{};
    /// Program registers each stage of the lookup in progress came from.
    std::array<u32, MaxShaderStages> lookup_hw_stages{};
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    DrawIndirectParams draw_indirect_params{};
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    Shader::Gcn::FetchShaderData* fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    struct PerfLastStage {
        u64 hash;
        std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data;
    };
    std::array<PerfLastStage, MaxShaderStages> perf_last_stages{};
    static constexpr u32 PerfSampleRate = 8;
    u32 perf_sample_counter{};
    bool perf_sample{};
};

} // namespace Vulkan
