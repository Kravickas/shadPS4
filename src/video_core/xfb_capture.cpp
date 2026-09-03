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

constexpr vk::BufferUsageFlags CounterUsage =
    vk::BufferUsageFlagBits::eTransformFeedbackCounterBufferEXT |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferSrc;

constexpr u64 CounterBufferSize = u64(XfbCapture::MaxRegions) * sizeof(u32);

XfbCapture::XfbCapture(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler)
    : buffer_a{instance, scheduler, MemoryUsage::DeviceLocal, 0, CaptureUsage, BufferSize},
      buffer_b{instance, scheduler, MemoryUsage::DeviceLocal, 0, CaptureUsage, BufferSize},
      counters_a{instance, scheduler, MemoryUsage::DeviceLocal, 0, CounterUsage, CounterBufferSize},
      counters_b{instance, scheduler,    MemoryUsage::DeviceLocal,
                 0,        CounterUsage, CounterBufferSize} {
    Vulkan::SetObjectName(instance.GetDevice(), buffer_a.Handle(), "XFB Capture A");
    Vulkan::SetObjectName(instance.GetDevice(), buffer_b.Handle(), "XFB Capture B");
    Vulkan::SetObjectName(instance.GetDevice(), counters_a.Handle(), "XFB Counters A");
    Vulkan::SetObjectName(instance.GetDevice(), counters_b.Handle(), "XFB Counters B");
}

bool XfbCapture::Begin(vk::CommandBuffer cmdbuf, const XfbRegion& region) {
    const u64 bytes = u64(region.max_vertices) * Shader::XfbVertexStride;
    if (bytes == 0 || write_offset + bytes > BufferSize || regions.size() >= MaxRegions) {
        ++overflow_draws;
        return false;
    }
    const vk::Buffer handle = Current().Handle();
    const vk::DeviceSize offset = write_offset;
    const vk::DeviceSize size = bytes;
    cmdbuf.bindTransformFeedbackBuffersEXT(0, 1, &handle, &offset, &size);
    cmdbuf.beginTransformFeedbackEXT(0, vk::ArrayProxy<const vk::Buffer>{},
                                     vk::ArrayProxy<const vk::DeviceSize>{});

    XfbRegion stored = region;
    stored.offset = static_cast<u32>(write_offset);
    stored.counter_offset = static_cast<u32>(regions.size() * sizeof(u32));
    regions.push_back(stored);

    write_offset += bytes;
    ++window_draws;
    window_vertices += region.max_vertices;
    return true;
}

void XfbCapture::End(vk::CommandBuffer cmdbuf) {
    // Writes the byte offset where capture stopped; the velocity pass derives the vertex count
    // from it with vkCmdDrawIndirectByteCountEXT.
    const vk::Buffer counter = Counters().Handle();
    const vk::DeviceSize counter_offset = regions.back().counter_offset;
    cmdbuf.endTransformFeedbackEXT(0, 1, &counter, &counter_offset);
}

void XfbCapture::EndFrame(vk::CommandBuffer cmdbuf) {
    // The buffers about to be reused were written two frames ago and read by the velocity pass
    // last frame.
    const std::array barriers = {
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT |
                            vk::PipelineStageFlagBits2::eVertexShader,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT |
                             vk::AccessFlagBits2::eShaderStorageRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .dstAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
            .buffer = PreviousBuffer().Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        },
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT |
                            vk::PipelineStageFlagBits2::eDrawIndirect,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT |
                             vk::AccessFlagBits2::eTransformFeedbackCounterReadEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .dstAccessMask = vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT,
            .buffer = (parity == 0 ? counters_b : counters_a).Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pBufferMemoryBarriers = barriers.data(),
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
