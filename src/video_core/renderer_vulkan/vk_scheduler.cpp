// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/sampling_profiler.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#if _MSC_VER
#include <intrin.h>
#elif defined(ARCH_X86_64)
#include <xmmintrin.h>
#endif

namespace Vulkan {

namespace {

void ThreadPause() {
#if defined(ARCH_X86_64)
    _mm_pause();
#elif _MSC_VER
    __yield();
#else
    asm("yield");
#endif
}

} // Anonymous namespace

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::SubmitOutOfBand(vk::CommandBuffer cmdbuf, vk::Fence fence, u64 wait_tick) const {
    // The submission signalling wait_tick makes the GPU writes before it available, the wait
    // makes them visible to the copies.
    const vk::Semaphore timeline = work_semaphore.Handle();
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_tick,
    };
    const bool own_queue = instance.HasReadbackQueue();
    const vk::SubmitInfo submit_info = {
        .pNext = own_queue ? &timeline_si : nullptr,
        .waitSemaphoreCount = own_queue ? 1U : 0U,
        .pWaitSemaphores = &timeline,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmdbuf,
    };
    // Both queues are externally synchronized; without a readback queue it is the graphics one.
    std::scoped_lock lk{submit_mutex};
    const auto submit_result = instance.GetReadbackQueue().submit(submit_info, fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;
    ++AmdGpu::cp_perf.render_passes;

    Record([state = new_state](vk::CommandBuffer cmdbuf) {
        std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
        for (u32 i = 0; i < state.num_color_attachments; ++i) {
            const auto& cb = state.color_attachments[i];
            color_attachments[i] = vk::RenderingAttachmentInfo{
                .imageView = cb.image_view,
                .imageLayout = cb.image_layout,
                .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue =
                    vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
            };
        }

        const auto& db = state.depth_stencil_attachment;
        const vk::RenderingAttachmentInfo depth_attachment = {
            .imageView = db.image_view,
            .imageLayout = db.image_layout,
            .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.depthStencil =
                                             vk::ClearDepthStencilValue{
                                                 .depth = std::bit_cast<float>(db.clear_value[0])}},
        };
        const vk::RenderingAttachmentInfo stencil_attachment = {
            .imageView = db.image_view,
            .imageLayout = db.image_layout,
            .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                vk::ClearValue{.depthStencil =
                                   vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
        };

        const vk::RenderingInfo rendering_info = {
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = {state.width, state.height},
                },
            .layerCount = state.num_layers,
            .colorAttachmentCount = state.num_color_attachments,
            .pColorAttachments = color_attachments.data(),
            .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
            .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
        };

        cmdbuf.beginRendering(rendering_info);
    });
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
}

void Scheduler::DeferRecording() {
    if (is_deferred) {
        return;
    }
    if (!recorder_thread.joinable()) {
        record_ring = std::make_unique_for_overwrite<u8[]>(RecordCapacity);
        recorder_thread = std::jthread(std::bind_front(&Scheduler::RecorderThread, this));
    }
    record_cmdbuf = sessions.back().primary;
    recording_thread = std::this_thread::get_id();
    is_deferred = true;
}

void Scheduler::PublishRecording() {
    if (record_published == record_head) {
        return;
    }
    record_published = record_head;
    ++AmdGpu::cp_perf.rec_publishes;
    // Sequentially consistent with the recorder thread, which sets recorder_sleeping and then
    // reads record_write: one of the two sees the other's store.
    record_write.store(record_head);
    if (recorder_sleeping.load()) {
        {
            std::scoped_lock lk{recorder_mutex};
        }
        recorder_cv.notify_one();
        ++AmdGpu::cp_perf.rec_wakes;
    }
}

void Scheduler::SyncRecording() {
    if (!is_deferred) {
        return;
    }
    ASSERT_MSG(std::this_thread::get_id() == recording_thread,
               "Command buffer taken by another thread while its recording is deferred");
    const u64 sync_t0 = AmdGpu::PerfNs();
    PublishRecording();
    while (record_tail.load(std::memory_order_acquire) != record_head) {
        ThreadPause();
    }
    record_tail_seen = record_head;
    is_deferred = false;
    ++AmdGpu::cp_perf.rec_syncs;
    AmdGpu::cp_perf.rec_sync_ns += AmdGpu::PerfNs() - sync_t0;
}

Scheduler::RecordEntry* Scheduler::AllocateRecord(std::size_t payload_size) {
    ASSERT_MSG(std::this_thread::get_id() == recording_thread,
               "Command recorded by another thread while recording is deferred");
    const u64 size = Common::AlignUp(sizeof(RecordEntry) + payload_size, RecordAlignment);
    u64 offset = record_head % RecordCapacity;
    if (offset + size > RecordCapacity) {
        // The entry does not fit before the end of the ring: an entry without a command covers
        // the rest of it.
        const u64 rest = RecordCapacity - offset;
        WaitRecordSpace(rest);
        ::new (record_ring.get() + offset) RecordEntry{.invoke = nullptr, .size = rest};
        record_head += rest;
        offset = 0;
    }
    WaitRecordSpace(size);
    record_head += size;
    ++AmdGpu::cp_perf.rec_cmds;
    AmdGpu::cp_perf.rec_bytes += size;
    return ::new (record_ring.get() + offset) RecordEntry{.invoke = nullptr, .size = size};
}

void Scheduler::WaitRecordSpace(u64 size) {
    if (record_head + size - record_tail_seen <= RecordCapacity) {
        return;
    }
    record_tail_seen = record_tail.load(std::memory_order_acquire);
    if (record_head + size - record_tail_seen <= RecordCapacity) {
        return;
    }
    ++AmdGpu::cp_perf.rec_space_waits;
    PublishRecording();
    do {
        ThreadPause();
        record_tail_seen = record_tail.load(std::memory_order_acquire);
    } while (record_head + size - record_tail_seen > RecordCapacity);
}

void Scheduler::RecorderThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandRecorder");
    Common::SamplingProfiler::RegisterCurrentThread("GpuCommandRecorder");

    u64 tail = 0;
    u32 idle_spins = 0;
    while (!stoken.stop_requested()) {
        const u64 write = record_write.load(std::memory_order_acquire);
        if (tail == write) {
            if (++idle_spins < RecorderIdleSpins) {
                ThreadPause();
                continue;
            }
            idle_spins = 0;
            std::unique_lock lk{recorder_mutex};
            recorder_sleeping.store(true);
            recorder_cv.wait(lk, stoken, [&] { return record_write.load() != tail; });
            recorder_sleeping.store(false);
            AmdGpu::recorder_perf.sleeps.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        idle_spins = 0;

        const u64 run_t0 = AmdGpu::PerfNs();
        const vk::CommandBuffer cmdbuf = record_cmdbuf;
        do {
            auto* const entry = std::launder(
                reinterpret_cast<RecordEntry*>(record_ring.get() + tail % RecordCapacity));
            if (entry->invoke) {
                entry->invoke(entry + 1, cmdbuf);
            }
            tail += entry->size;
        } while (tail != write);
        record_tail.store(tail, std::memory_order_release);
        AmdGpu::recorder_perf.run_ns.fetch_add(AmdGpu::PerfNs() - run_t0,
                                               std::memory_order_relaxed);
    }
}

vk::CommandBuffer Scheduler::UploadCommandBuffer() {
    auto& upload_cmdbuf = sessions.back().upload;
    if (upload_cmdbuf) {
        return upload_cmdbuf;
    }
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    upload_cmdbuf = command_pool.Commit();
    Check(upload_cmdbuf.begin(begin_info));
    return upload_cmdbuf;
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    work_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    if (pending_ops_empty.load(std::memory_order_relaxed)) {
        return;
    }
    std::unique_lock lk(pending_ops_mutex);
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
    pending_ops_empty.store(pending_ops.empty(), std::memory_order_relaxed);
}

void Scheduler::BeginSession() {
    EndSession();

    auto& session = sessions.emplace_back();

    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    session.primary = command_pool.Commit();
    Check(session.primary.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::EndSession() {
    if (sessions.empty()) {
        return;
    }

    if (on_session) {
        on_session();
    }

    const auto& session = sessions.back();
    if (session.upload) {
        Check(session.upload.end());
    }

    EndRendering();
    SyncRecording();
    Check(session.primary.end());
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndSession();

    std::vector<vk::CommandBuffer> cmd_buffers;
    cmd_buffers.reserve(sessions.size() * 2);

    for (const auto& session : sessions) {
        if (session.upload) {
            cmd_buffers.push_back(session.upload);
        }
        cmd_buffers.push_back(session.primary);
    }
    sessions.clear();

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    std::array<vk::PipelineStageFlags, std::tuple_size_v<decltype(info.wait_semas)>>
        wait_stage_masks;
    wait_stage_masks.fill(vk::PipelineStageFlagBits::eAllCommands);

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = static_cast<u32>(cmd_buffers.size()),
        .pCommandBuffers = cmd_buffers.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    work_semaphore.Refresh();
    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, Scheduler& scheduler) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        scheduler.Record(
            viewports.size() * sizeof(vk::Viewport),
            [this](u8* data) {
                std::memcpy(data, viewports.data(), viewports.size() * sizeof(vk::Viewport));
            },
            [count = static_cast<u32>(viewports.size())](vk::CommandBuffer cmdbuf, const u8* data) {
                cmdbuf.setViewportWithCount(count, reinterpret_cast<const vk::Viewport*>(data));
            });
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        scheduler.Record(
            scissors.size() * sizeof(vk::Rect2D),
            [this](u8* data) {
                std::memcpy(data, scissors.data(), scissors.size() * sizeof(vk::Rect2D));
            },
            [count = static_cast<u32>(scissors.size())](vk::CommandBuffer cmdbuf, const u8* data) {
                cmdbuf.setScissorWithCount(count, reinterpret_cast<const vk::Rect2D*>(data));
            });
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        scheduler.Record([enabled = depth_test_enabled](vk::CommandBuffer cmdbuf) {
            cmdbuf.setDepthTestEnable(enabled);
        });
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        scheduler.Record([enabled = depth_write_enabled](vk::CommandBuffer cmdbuf) {
            cmdbuf.setDepthWriteEnable(enabled);
        });
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        scheduler.Record(
            [op = depth_compare_op](vk::CommandBuffer cmdbuf) { cmdbuf.setDepthCompareOp(op); });
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            scheduler.Record([enabled = depth_bounds_test_enabled](vk::CommandBuffer cmdbuf) {
                cmdbuf.setDepthBoundsTestEnable(enabled);
            });
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            scheduler.Record([min = depth_bounds_min, max = depth_bounds_max](
                                 vk::CommandBuffer cmdbuf) { cmdbuf.setDepthBounds(min, max); });
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        scheduler.Record([enabled = depth_bias_enabled](vk::CommandBuffer cmdbuf) {
            cmdbuf.setDepthBiasEnable(enabled);
        });
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        scheduler.Record([constant = depth_bias_constant, clamp = depth_bias_clamp,
                          slope = depth_bias_slope](vk::CommandBuffer cmdbuf) {
            cmdbuf.setDepthBias(constant, clamp, slope);
        });
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        scheduler.Record([enabled = stencil_test_enabled](vk::CommandBuffer cmdbuf) {
            cmdbuf.setStencilTestEnable(enabled);
        });
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            scheduler.Record([ops = stencil_front_ops](vk::CommandBuffer cmdbuf) {
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, ops.fail_op,
                                    ops.pass_op, ops.depth_fail_op, ops.compare_op);
            });
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                scheduler.Record([ops = stencil_front_ops](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, ops.fail_op, ops.pass_op,
                                        ops.depth_fail_op, ops.compare_op);
                });
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                scheduler.Record([ops = stencil_back_ops](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, ops.fail_op, ops.pass_op,
                                        ops.depth_fail_op, ops.compare_op);
                });
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            scheduler.Record([value = stencil_front_reference](vk::CommandBuffer cmdbuf) {
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, value);
            });
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                scheduler.Record([value = stencil_front_reference](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront, value);
                });
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                scheduler.Record([value = stencil_back_reference](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, value);
                });
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            scheduler.Record([value = stencil_front_write_mask](vk::CommandBuffer cmdbuf) {
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, value);
            });
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                scheduler.Record([value = stencil_front_write_mask](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront, value);
                });
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                scheduler.Record([value = stencil_back_write_mask](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, value);
                });
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            scheduler.Record([value = stencil_front_compare_mask](vk::CommandBuffer cmdbuf) {
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, value);
            });
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                scheduler.Record([value = stencil_front_compare_mask](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront, value);
                });
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                scheduler.Record([value = stencil_back_compare_mask](vk::CommandBuffer cmdbuf) {
                    cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack, value);
                });
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        scheduler.Record([enabled = primitive_restart_enable](vk::CommandBuffer cmdbuf) {
            cmdbuf.setPrimitiveRestartEnable(enabled);
        });
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        scheduler.Record([enabled = rasterizer_discard_enable](vk::CommandBuffer cmdbuf) {
            cmdbuf.setRasterizerDiscardEnable(enabled);
        });
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        scheduler.Record(
            [mode = cull_mode](vk::CommandBuffer cmdbuf) { cmdbuf.setCullMode(mode); });
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        scheduler.Record(
            [face = front_face](vk::CommandBuffer cmdbuf) { cmdbuf.setFrontFace(face); });
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        scheduler.Record([constants = blend_constants](vk::CommandBuffer cmdbuf) {
            cmdbuf.setBlendConstants(constants.data());
        });
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            scheduler.Record([masks = color_write_masks](vk::CommandBuffer cmdbuf) {
                cmdbuf.setColorWriteMaskEXT(0, masks);
            });
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        scheduler.Record(
            [width = line_width](vk::CommandBuffer cmdbuf) { cmdbuf.setLineWidth(width); });
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        scheduler.Record([enabled = feedback_loop_enabled](vk::CommandBuffer cmdbuf) {
            cmdbuf.setAttachmentFeedbackLoopEnableEXT(enabled ? vk::ImageAspectFlagBits::eColor
                                                              : vk::ImageAspectFlagBits::eNone);
        });
    }
}

} // namespace Vulkan
