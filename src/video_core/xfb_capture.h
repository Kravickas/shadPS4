// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <span>
#include <vector>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/texture_cache/types.h"

namespace Vulkan {
class Instance;
} // namespace Vulkan

namespace VideoCore {

// Within 1% to absorb rounding of odd internal resolutions.
inline bool XfbMatchesAspect(u32 width, u32 height, vk::Extent2D output) {
    if (output.width == 0 || output.height == 0) {
        return false;
    }
    const u64 lhs = u64(width) * output.height;
    const u64 rhs = u64(height) * output.width;
    const u64 diff = lhs > rhs ? lhs - rhs : rhs - lhs;
    return diff * 100 <= std::max(lhs, rhs);
}

// Rasterization state of a captured draw, replayed by the velocity pass so coverage matches.
struct XfbDrawState {
    vk::Viewport viewport;
    vk::Rect2D scissor;
    vk::CullModeFlags cull_mode;
    vk::FrontFace front_face;
    float depth_bias_constant;
    float depth_bias_clamp;
    float depth_bias_slope;
    bool depth_bias_enabled;
    bool depth_clamp;
    bool depth_clip;
    bool negative_one_to_one;
};

struct XfbRegion {
    u64 key;
    u32 offset;
    u32 max_vertices;
    u32 counter_offset;
    ImageId depth_id;
    vk::Image depth_image;
    vk::ImageView depth_view;
    vk::Format depth_format;
    bool has_stencil;
    vk::SampleCountFlagBits samples;
    // Owns its pixels but its vectors are not trusted.
    bool blended;
    u32 width;
    u32 height;
    XfbDrawState state;
};

// Captures clip-space positions of scene draws through transform feedback. Two buffer sets
// alternate per frame so the previous frame's capture stays readable while the current one is
// written.
class XfbCapture {
public:
    explicit XfbCapture(const Vulkan::Instance& instance);

    bool Begin(vk::CommandBuffer cmdbuf, const XfbRegion& region);
    void End(vk::CommandBuffer cmdbuf);
    void EndFrame(vk::CommandBuffer cmdbuf);

    [[nodiscard]] std::span<const XfbRegion> CurrentRegions() const noexcept {
        return regions;
    }

    [[nodiscard]] std::span<const XfbRegion> PreviousRegions() const noexcept {
        return prev_regions;
    }

    [[nodiscard]] vk::Buffer CurrentBuffer() const noexcept {
        return (parity == 0 ? buffer_a : buffer_b).buffer;
    }

    [[nodiscard]] vk::Buffer PreviousBuffer() const noexcept {
        return (parity == 0 ? buffer_b : buffer_a).buffer;
    }

    [[nodiscard]] vk::Buffer CurrentCounters() const noexcept {
        return (parity == 0 ? counters_a : counters_b).buffer;
    }

    static constexpr u64 BufferSize = 512_MB;
    static constexpr u32 MaxRegions = 65536;

private:
    // Transform feedback usage is outside VideoCore::Buffer's fixed usage flags.
    UniqueBuffer buffer_a;
    UniqueBuffer buffer_b;
    UniqueBuffer counters_a;
    UniqueBuffer counters_b;
    u32 parity{};
    u64 write_offset{};
    std::vector<XfbRegion> regions;
    std::vector<XfbRegion> prev_regions;
    u32 frames{};
    u32 overflow_draws{};
    u64 window_draws{};
    u64 window_vertices{};
};

} // namespace VideoCore
