// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_map>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {
class TextureCache;
class XfbCapture;
} // namespace VideoCore

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace Vulkan::HostPasses {

// Rasterizes per-vertex motion between two frames of transform feedback captures into a motion
// vector target, using the scene depth with an EQUAL test so only visible surfaces write.
class XfbVelocityPass {
public:
    XfbVelocityPass(const Instance& instance, Scheduler& scheduler,
                    VideoCore::TextureCache& texture_cache);
    ~XfbVelocityPass();

    void Render(const VideoCore::XfbCapture& capture);

    [[nodiscard]] vk::Extent2D Size() const noexcept {
        return size;
    }

private:
    struct PipelineKey {
        vk::Format depth_format;
        bool depth_clamp;
        bool depth_clip;
        bool negative_one_to_one;

        bool operator==(const PipelineKey&) const = default;
    };

    struct PipelineKeyHash {
        size_t operator()(const PipelineKey& key) const noexcept;
    };

    struct PushConstants {
        u32 cur_offset;
        u32 prev_offset;
        u32 prev_max_vertices;
        u32 has_prev;
        std::array<float, 2> viewport_size;
    };

    vk::Pipeline GetPipeline(const PipelineKey& key);
    void ResizeTargets(u32 width, u32 height);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;

    vk::ShaderModule vertex_module;
    vk::ShaderModule fragment_module;
    vk::UniqueDescriptorSetLayout desc_set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    std::unordered_map<PipelineKey, vk::UniquePipeline, PipelineKeyHash> pipelines;

    vk::Extent2D size{};
    VideoCore::UniqueImage motion_image;
    vk::UniqueImageView motion_view;
    VideoCore::UniqueImage mask_image;
    vk::UniqueImageView mask_view;
    bool targets_initialized{};

    u32 frames{};
    u64 window_draws{};
    u64 window_matched{};
};

} // namespace Vulkan::HostPasses
