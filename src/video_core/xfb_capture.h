// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/texture_cache/types.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

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
    vk::ImageView depth_view;
    vk::Format depth_format;
    bool has_stencil;
    u32 width;
    u32 height;
    XfbDrawState state;
};

// Captures clip-space positions of scene draws through transform feedback. Two buffer sets
// alternate per frame so the previous frame's capture stays readable while the current one is
// written.
class XfbCapture {
public:
    explicit XfbCapture(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler);

    bool Begin(vk::CommandBuffer cmdbuf, const XfbRegion& region);
    void End(vk::CommandBuffer cmdbuf);
    void EndFrame(vk::CommandBuffer cmdbuf);

    [[nodiscard]] std::span<const XfbRegion> CurrentRegions() const noexcept {
        return regions;
    }

    [[nodiscard]] std::span<const XfbRegion> PreviousRegions() const noexcept {
        return prev_regions;
    }

    [[nodiscard]] const Buffer& CurrentBuffer() const noexcept {
        return parity == 0 ? buffer_a : buffer_b;
    }

    [[nodiscard]] const Buffer& PreviousBuffer() const noexcept {
        return parity == 0 ? buffer_b : buffer_a;
    }

    [[nodiscard]] const Buffer& CurrentCounters() const noexcept {
        return parity == 0 ? counters_a : counters_b;
    }

    static constexpr u64 BufferSize = 256_MB;
    static constexpr u32 MaxRegions = 65536;

private:
    Buffer& Current() noexcept {
        return parity == 0 ? buffer_a : buffer_b;
    }

    Buffer& Counters() noexcept {
        return parity == 0 ? counters_a : counters_b;
    }

    Buffer buffer_a;
    Buffer buffer_b;
    Buffer counters_a;
    Buffer counters_b;
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
