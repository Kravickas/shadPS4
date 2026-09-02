// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "shader_recompiler/xfb_layout.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/xfb_capture.h"

namespace VideoCore {

// Frames between summary lines.
constexpr u32 ReportInterval = 300;

constexpr vk::BufferUsageFlags CaptureUsage = vk::BufferUsageFlagBits::eTransformFeedbackBufferEXT |
                                              vk::BufferUsageFlagBits::eStorageBuffer |
                                              vk::BufferUsageFlagBits::eTransferSrc;

XfbCapture::XfbCapture(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler)
    : buffer_a{instance, scheduler, MemoryUsage::DeviceLocal, 0, CaptureUsage, BufferSize},
      buffer_b{instance, scheduler, MemoryUsage::DeviceLocal, 0, CaptureUsage, BufferSize} {
    Vulkan::SetObjectName(instance.GetDevice(), buffer_a.Handle(), "XFB Capture A");
    Vulkan::SetObjectName(instance.GetDevice(), buffer_b.Handle(), "XFB Capture B");
}

bool XfbCapture::Begin(vk::CommandBuffer cmdbuf, u64 key, u32 max_vertices) {
    const u64 bytes = u64(max_vertices) * Shader::XfbVertexStride;
    if (bytes == 0 || write_offset + bytes > BufferSize) {
        ++overflow_draws;
        return false;
    }
    const vk::Buffer handle = Current().Handle();
    const vk::DeviceSize offset = write_offset;
    const vk::DeviceSize size = bytes;
    cmdbuf.bindTransformFeedbackBuffersEXT(0, 1, &handle, &offset, &size);
    cmdbuf.beginTransformFeedbackEXT(0, 0, nullptr, nullptr);
    regions.push_back(Region{key, static_cast<u32>(write_offset), max_vertices});
    write_offset += bytes;
    ++window_draws;
    window_vertices += max_vertices;
    return true;
}

void XfbCapture::End(vk::CommandBuffer cmdbuf) {
    cmdbuf.endTransformFeedbackEXT(0, 0, nullptr, nullptr);
}

void XfbCapture::EndFrame(vk::CommandBuffer cmdbuf) {
    // The buffer about to be reused was last written two frames ago; order those writes before
    // the new ones.
    const vk::BufferMemoryBarrier2 barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
        .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
        .dstAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
        .buffer = PreviousBuffer().Handle(),
        .offset = 0,
        .size = vk::WholeSize,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &barrier,
    });

    prev_regions.swap(regions);
    regions.clear();
    parity ^= 1;
    write_offset = 0;

    if (++frames >= ReportInterval) {
        LOG_INFO(Render_Vulkan,
                 "xfb capture: frames={} draws/frame={:.1f} vertices/frame={:.0f} "
                 "overflow_draws={}",
                 frames, double(window_draws) / double(frames),
                 double(window_vertices) / double(frames), overflow_draws);
        frames = 0;
        window_draws = 0;
        window_vertices = 0;
        overflow_draws = 0;
    }
}

} // namespace VideoCore
