// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include <unordered_map>
#include <vector>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {
class TextureCache;
class XfbCapture;
} // namespace VideoCore

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
} // namespace Vulkan

namespace Vulkan::HostPasses {

// Rasterizes per-vertex motion between two frames of transform feedback captures into a motion
// vector target, using the scene depth with an EQUAL test so only visible surfaces write.
class XfbVelocityPass {
public:
    XfbVelocityPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                    VideoCore::TextureCache& texture_cache);
    ~XfbVelocityPass();

    void Render(const VideoCore::XfbCapture& capture, vk::Extent2D output_extent);

    // Single-sample motion (RG16F, pixels) and depth (R32F) of the last rendered frame, both in
    // eGeneral. Empty until a frame was rendered.
    struct Outputs {
        vk::Image motion;
        vk::ImageView motion_view;
        vk::Image depth;
        vk::ImageView depth_view;
        vk::Extent2D size;
    };
    std::optional<Outputs> FrameOutputs() const;

    [[nodiscard]] vk::Extent2D Size() const noexcept {
        return size;
    }

private:
    struct PipelineKey {
        vk::Format depth_format;
        vk::SampleCountFlagBits samples;
        bool depth_clamp;
        bool depth_clip;
        bool negative_one_to_one;

        bool operator==(const PipelineKey&) const = default;
    };

    struct PipelineKeyHash {
        size_t operator()(const PipelineKey& key) const noexcept;
    };

    // Mirrors the push constant block of xfb_velocity.vert and xfb_velocity.frag.
    struct PushConstants {
        u32 cur_offset;
        u32 prev_offset;
        u32 prev_max_vertices;
        u32 has_prev;
        std::array<float, 2> viewport_size;
        std::array<float, 2> viewport_offset;
        std::array<float, 2> depth_range;
        u32 blended;
        u32 negative_one_to_one;
    };
    static_assert(sizeof(PushConstants) == 48);

    // Mirrors Region in xfb_camera_sums.comp.
    struct RegionEntry {
        PushConstants constants;
        u32 counter_offset;
        std::array<u32, 3> pad;
    };
    static_assert(sizeof(RegionEntry) == 64);

    vk::Pipeline GetPipeline(const PipelineKey& key);
    void ResizeTargets(u32 width, u32 height, vk::SampleCountFlagBits samples);
    void Resolve(vk::CommandBuffer cmdbuf);
    void CopyDepth(vk::CommandBuffer cmdbuf, vk::ImageView depth_view, vk::ImageLayout layout);
    void CreateCameraPipelines();
    void SolveCamera(vk::CommandBuffer cmdbuf, const VideoCore::XfbCapture& capture, u32 count);

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::TextureCache& texture_cache;

    vk::ShaderModule vertex_module;
    vk::ShaderModule fragment_module;
    vk::UniqueDescriptorSetLayout desc_set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    std::unordered_map<PipelineKey, vk::UniquePipeline, PipelineKeyHash> pipelines;

    vk::ShaderModule resolve_module;
    vk::UniqueDescriptorSetLayout resolve_set_layout;
    vk::UniquePipelineLayout resolve_layout;
    vk::UniquePipeline resolve_pipeline;

    vk::ShaderModule camera_sums_module;
    vk::ShaderModule camera_hypotheses_module;
    vk::ShaderModule camera_solve_module;
    vk::UniqueDescriptorSetLayout camera_sums_set_layout;
    vk::UniqueDescriptorSetLayout camera_hypotheses_set_layout;
    vk::UniqueDescriptorSetLayout camera_solve_set_layout;
    vk::UniquePipelineLayout camera_sums_layout;
    vk::UniquePipelineLayout camera_hypotheses_layout;
    vk::UniquePipelineLayout camera_solve_layout;
    vk::UniquePipeline camera_sums_pipeline;
    vk::UniquePipeline camera_hypotheses_pipeline;
    vk::UniquePipeline camera_solve_pipeline;
    bool camera_supported{};

    // One entry per velocity draw, in issue order. Read by the camera solve and by capture tools.
    VideoCore::Buffer region_table;
    // Per-region normal equations for the camera solve, in doubles.
    VideoCore::Buffer camera_sums;
    // Candidate transforms fitted to random region samples, with their inlier counts.
    VideoCore::Buffer camera_hypotheses;
    // Solved clip to previous clip transform and its statistics.
    VideoCore::Buffer camera_result;
    std::vector<RegionEntry> entries;

    vk::Extent2D size{};
    VideoCore::UniqueImage motion_image;
    vk::UniqueImageView motion_view;
    VideoCore::UniqueImage mask_image;
    vk::UniqueImageView mask_view;
    // Multisampled render targets when the scene depth is multisampled, resolved into the above.
    VideoCore::UniqueImage motion_ms_image;
    vk::UniqueImageView motion_ms_view;
    VideoCore::UniqueImage mask_ms_image;
    vk::UniqueImageView mask_ms_view;
    VideoCore::UniqueImage depth_out;
    vk::UniqueImageView depth_out_view;
    vk::ShaderModule depth_copy_module;
    vk::ShaderModule depth_copy_ms_module;
    vk::UniqueDescriptorSetLayout depth_copy_set_layout;
    vk::UniquePipelineLayout depth_copy_layout;
    vk::UniquePipeline depth_copy_pipeline;
    vk::UniquePipeline depth_copy_ms_pipeline;
    bool outputs_valid{};
    vk::SampleCountFlagBits samples{vk::SampleCountFlagBits::e1};
    bool targets_initialized{};

    u32 frames{};
    u64 window_draws{};
    u64 window_matched{};
};

} // namespace Vulkan::HostPasses
