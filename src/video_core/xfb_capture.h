// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

// Captures clip-space positions of scene draws through transform feedback. Two buffers alternate
// per frame so the previous frame's capture stays readable while the current one is written.
class XfbCapture {
public:
    struct Region {
        u64 key;
        u32 offset;
        u32 max_vertices;
    };

    explicit XfbCapture(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler);

    bool Begin(vk::CommandBuffer cmdbuf, u64 key, u32 max_vertices);
    void End(vk::CommandBuffer cmdbuf);
    void EndFrame(vk::CommandBuffer cmdbuf);

    [[nodiscard]] std::span<const Region> PreviousRegions() const noexcept {
        return prev_regions;
    }

    [[nodiscard]] const Buffer& PreviousBuffer() const noexcept {
        return parity == 0 ? buffer_b : buffer_a;
    }

private:
    static constexpr u64 BufferSize = 128_MB;

    Buffer& Current() noexcept {
        return parity == 0 ? buffer_a : buffer_b;
    }

    Buffer buffer_a;
    Buffer buffer_b;
    u32 parity{};
    u64 write_offset{};
    std::vector<Region> regions;
    std::vector<Region> prev_regions;
    u32 frames{};
    u32 overflow_draws{};
    u64 window_draws{};
    u64 window_vertices{};
};

} // namespace VideoCore
