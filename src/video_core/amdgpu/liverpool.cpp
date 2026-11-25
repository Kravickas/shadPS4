// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>

#include "common/arch.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "common/sampling_profiler.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/page_manager.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#if _MSC_VER
#include <intrin.h>
#elif defined(ARCH_X86_64)
#include <xmmintrin.h>
#endif

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
    while (!ce_submits.empty()) {
        ce_submits.front().destroy();
        ce_submits.pop();
    }
    for (auto& queue : mapped_queues) {
        while (!queue.submits.empty()) {
            queue.submits.front().destroy();
            queue.submits.pop();
        }
    }
}

void Liverpool::ProcessCeQueue() {
    auto& queue = mapped_queues[GfxQueueId];
    Task::Handle task{};
    {
        std::scoped_lock lock{queue.m_access};
        if (ce_submits.empty()) {
            return;
        }
        task = ce_submits.front();
    }
    task.resume();
    if (task.done()) {
        task.destroy();
        std::scoped_lock lock{queue.m_access};
        ce_submits.pop();
        --num_submits;
        std::scoped_lock lock2{submit_mutex};
        submit_cv.notify_all();
    }
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        const CpPerfScope perf_command{cp_perf.commands_ns};
        ++cp_perf.commands;
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    Common::SamplingProfiler::RegisterCurrentThread("GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
#ifdef __linux__
    gpu_tid = gettid();
#endif

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            const auto has_work = [this] { return num_commands || num_submits || submit_done; };
            lookahead_gpu_idle.store(!has_work(), std::memory_order_relaxed);
            Common::CondvarWait(submit_cv, lk, stoken, has_work);
        }
        lookahead_gpu_idle.store(false, std::memory_order_relaxed);
        if (stoken.stop_requested()) {
            break;
        }

        VideoCore::StartCapture();

        const u64 cycle_t0 = PerfNs();
        cp_perf = {};
        const auto watch_t0 = VideoCore::PageManager::WatchCountsOfThread();
        perf_bind_check = (perf_cycles++ & (PerfBindCheckRate - 1)) == 0;
        curr_qid = -1;

        while ((num_submits || num_commands) && !stoken.stop_requested()) {
            ProcessCommands();
            ProcessCeQueue();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            task.resume();

            if (task.done()) {
                task.destroy();
                ++cp_perf.tasks;

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }

        const u64 flush_t0 = PerfNs();
        const bool was_submit_done = submit_done;
        if (submit_done) {
            VideoCore::EndCapture();
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }
        const u64 cycle_t1 = PerfNs();
        DebugState.frame_gpu_thread_ns.Add(cycle_t1 - cycle_t0);
        // Submits inside a frame take this thread's time. The share of the last cycles it
        // waited for work says whether it has that time.
        if (last_cycle_end != 0) {
            const s64 idle_ns = std::min<s64>(cycle_t0 - last_cycle_end, MaxIdleSampleNs);
            busy_avg_ns += (static_cast<s64>(cycle_t1 - cycle_t0) - busy_avg_ns) / CycleSmoothing;
            idle_avg_ns += (idle_ns - idle_avg_ns) / CycleSmoothing;
        }
        last_cycle_end = cycle_t1;
        has_submit_slack = idle_avg_ns * 100 >= (idle_avg_ns + busy_avg_ns) * MinIdlePercent;
        if (rasterizer) {
            rasterizer->SetSubmitSlack(has_submit_slack);
        }
        DebugState.frame_drain_ns.Add(cp_perf.rb_finish_ns + cp_perf.img_finish_ns);
        // The profile is of gameplay: samples are taken while the last cycle had its draw count.
        const bool is_gameplay = cp_perf.draws >= PerfProfiledDraws;
        Common::SamplingProfiler::SetEnabled(is_gameplay);
        const auto watch_t1 = VideoCore::PageManager::WatchCountsOfThread();
        cp_perf.pm_pages = watch_t1.pages - watch_t0.pages;
        cp_perf.pm_protects = watch_t1.protects - watch_t0.protects;
        cp_perf.pm_image_pages = watch_t1.image_pages - watch_t0.image_pages;
        cp_perf.pm_image_protects = watch_t1.image_protects - watch_t0.image_protects;
        LOG_INFO(
            Render,
            "PERF cp t0={} t1={} tasks={} submit_done={} flush_us={} vo_block_us={} "
            "vo_blocks={} vo_spin_us={} mem_spin_us={} mem_spins={} asc_spin_us={} "
            "draws={} draw_us={} draw_pipeline_us={} draw_bind_us={} dispatches={} "
            "dispatch_us={} dispatch_pipeline_us={} copy_us={} fence_us={} eop_us={} "
            "flushes={} host_flush_us={} rb_downloads={} rb_kb={} rb_finish_us={} "
            "img_downloads={} img_finish_us={} rbw_n={} rbw_us={} rbw_max_us={} "
            "draw_prep_us={} draw_begin_us={} bind_buffers_us={} bind_textures_us={} "
            "dma_sync_us={} eos={} eos_us={} commands={} commands_us={} draw_pre_us={} "
            "draw_vb_us={} draw_barrier_us={} draw_desc_us={} draw_dyn_us={} draw_rp_us={} "
            "draw_cmd_us={} draw_barriers={} render_passes={} zpass_dumps={} obtains={} "
            "obtain_stream={} obtain_stream_kb={} obtain_written={} obtain_uploads={} "
            "obtain_upload_kb={} obtain_texel={} obtain_texel_image={} obtain_stream_reused={} "
            "obtain_stream_reused_kb={} pipe_key_us={} pipe_stages_us={} pipe_changes={} "
            "stg_params_us={} stg_runtime_us={} stg_lookup_us={} stg_flatbuf_us={} "
            "stg_spec_us={} stg_find_us={} stg_calls={} stg_perms_scanned={} "
            "vi_inputs_us={} "
            "vi_obtain_us={} vi_bind_us={} index_us={} cmd_record_us={} reset_images_us={} "
            "reset_buffers_us={} bound_buffers={} bound_images={} tex_lookups={} tex_hits={} "
            "tex_miss_key={} tex_miss_generation={} tex_miss_unreusable={} "
            "tex_registry_changes={} tex_find_images={} tex_reg_new={} tex_reg_depth={} "
            "tex_reg_expand={} tex_reg_merge={} tex_reg_stencil={} tex_free_unmap={} "
            "tex_free_pool={} tex_free_tile={} tex_free_chance={} tex_free_target={} "
            "tex_free_mip={} tex_free_wrong={} tex_free_gc={} pm_pages={} pm_protects={} "
            "pm_image_pages={} pm_image_protects={} tex_alias_swaps={} tex_find_texture={} "
            "tex_find_storage={} tex_find_target={} tex_find_depth={} tex_find_vo={} up_16k={} "
            "up_64k={} up_256k={} up_big={} up_16k_kb={} up_64k_kb={} up_256k_kb={} up_big_kb={} "
            "ob_16k={} ob_64k={} ob_256k={} ob_big={} db_miss_unreusable={} db_miss_generation={} "
            "db_miss_buffer={} db_miss_view={} db_miss_htile_hint={} db_miss_swap={} "
            "db_un_registry={} db_un_swap={} db_un_imperfect={} resident_hits={} gf_writes={} "
            "gf_write_us={} gf_write_next_page={} gf_reads={} gf_read_us={} la_solo_runs={} "
            "la_bind_us={} la_bind_skips={} va_used={} va_checks={} va_diffs={} "
            "eager_n={} eager_windows={} eager_kb={} eager_gpu_us={} eager_us={} "
            "emulated_us={} db_expired={} eager_unmarked={} eager_skip_tick={} "
            "eager_skip_flight={} eager_known={} eager_new={} early_flushes={} "
            "ra_stages={} ra_checks={} ra_diffs={} la_run_waits={} la_run_wait_us={} "
            "la_woken_runs={} la_woken_wait_us={} la_ahead_1={} la_ahead_2={} la_ahead_3={} "
            "idle_flushes={} auto_checks={} auto_check_us={} gpu_idle_permille={} "
            "submit_slack={} bb_total={} bb_same={} bb_same_tick={} bs_total={} bs_same_tick={} "
            "bs_one_diff={} tb_total={} tb_same={} tb_same_tick={} ts_total={} ts_same_tick={} "
            "ts_one_diff={} "
            "emulated={} stg_same_inputs={} vk_vi_us={} vk_index_us={} vk_desc_us={} "
            "vk_dyn_us={} vk_draw_us={} rec_cmds={} rec_kb={} rec_publishes={} rec_wakes={} "
            "rec_syncs={} rec_sync_us={} rec_space_waits={} rec_run_us={} rec_sleeps={} "
            "sc_total_us={} sc_compare_us={} sc_guest_us={} sc_write_us={} rec_stream_write_us={} "
            "run_n={} run_d1={} run_d2={} run_d4={} run_d8={} run_d16={} run_d32={} "
            "state_packets={} state_words={} la_runs={} la_results={} la_waits={} la_wait_us={} "
            "la_sync_waits={} la_sync_us={} la_copies={} la_compares={} la_mismatches={} "
            "la_early={} la_scan_us={} la_sleeps={} la_lookups_used={} la_lookups_rejected={} "
            "la_lookup_us={} la_lookups_checked={} la_stream_us={} sa_ahead_known={} "
            "sa_ahead_same={} "
            "sa_ahead_copied={} sa_ahead_kb={} sa_ahead_full={} sa_used={} sa_stale={} "
            "sa_checks={} sa_diffs={} tx_first_us={} tx_second_us={} tx_samplers_us={} "
            "dispatch_hle_us={} dispatch_bind_us={} dispatch_hles={} dispatch_skips={} "
            "dispatch_bindbuf_us={} dispatch_bindtex_us={} ow_obtain_us={} ow_inval_us={} "
            "ow_kb={} ow_dirty_us={} ow_sync_us={} ow_lock_us={} ow_walk_us={} ow_mark_us={} "
            "ow_upload_us={} ow_uploads={} ow_upload_kb={} ow_watch_pages={} ow_protects={} "
            "bc_on={} bc_images={} bc_images_same={} "
            "bc_images_equal={} bc_image_stages={} bc_image_stages_same={} "
            "bc_image_stages_quiet={} bc_draws={} bc_draws_state_same={}",
            cycle_t0, cycle_t1, cp_perf.tasks, was_submit_done, (cycle_t1 - flush_t0) / 1000,
            cp_perf.vo_block_ns / 1000, cp_perf.vo_blocks, cp_perf.vo_spin_ns / 1000,
            cp_perf.mem_spin_ns / 1000, cp_perf.mem_spins, cp_perf.asc_spin_ns / 1000,
            cp_perf.draws, cp_perf.draw_ns / 1000, cp_perf.draw_pipeline_ns / 1000,
            cp_perf.draw_bind_ns / 1000, cp_perf.dispatches, cp_perf.dispatch_ns / 1000,
            cp_perf.dispatch_pipeline_ns / 1000, cp_perf.copy_ns / 1000, cp_perf.fence_ns / 1000,
            cp_perf.eop_ns / 1000, cp_perf.flushes, cp_perf.flush_ns / 1000, cp_perf.rb_downloads,
            cp_perf.rb_bytes / 1024, cp_perf.rb_finish_ns / 1000, cp_perf.img_downloads,
            cp_perf.img_finish_ns / 1000,
            readback_wait.count.exchange(0, std::memory_order_relaxed),
            readback_wait.wait_ns.exchange(0, std::memory_order_relaxed) / 1000,
            readback_wait.max_ns.exchange(0, std::memory_order_relaxed) / 1000,
            cp_perf.draw_prep_ns / 1000, cp_perf.draw_begin_ns / 1000,
            cp_perf.bind_buffers_ns / 1000, cp_perf.bind_textures_ns / 1000,
            cp_perf.dma_sync_ns / 1000, cp_perf.eos, cp_perf.eos_ns / 1000, cp_perf.commands,
            cp_perf.commands_ns / 1000, cp_perf.draw_pre_ns / 1000, cp_perf.draw_vb_ns / 1000,
            cp_perf.draw_barrier_ns / 1000, cp_perf.draw_desc_ns / 1000, cp_perf.draw_dyn_ns / 1000,
            cp_perf.draw_rp_ns / 1000, cp_perf.draw_cmd_ns / 1000, cp_perf.draw_barriers,
            cp_perf.render_passes, cp_perf.zpass_dumps, cp_perf.obtains, cp_perf.obtain_stream,
            cp_perf.obtain_stream_bytes / 1024, cp_perf.obtain_written, cp_perf.obtain_uploads,
            cp_perf.obtain_upload_bytes / 1024, cp_perf.obtain_texel, cp_perf.obtain_texel_image,
            cp_perf.obtain_stream_reused, cp_perf.obtain_stream_reused_bytes / 1024,
            cp_perf.pipe_key_ns / 1000, cp_perf.pipe_stages_ns / 1000, cp_perf.pipe_changes,
            cp_perf.stg_params_ns / 1000, cp_perf.stg_runtime_ns / 1000,
            cp_perf.stg_lookup_ns / 1000, cp_perf.stg_flatbuf_ns / 1000, cp_perf.stg_spec_ns / 1000,
            cp_perf.stg_find_ns / 1000, cp_perf.stg_calls, cp_perf.stg_perms_scanned,
            cp_perf.vi_inputs_ns / 1000, cp_perf.vi_obtain_ns / 1000, cp_perf.vi_bind_ns / 1000,
            cp_perf.index_ns / 1000, cp_perf.cmd_record_ns / 1000, cp_perf.reset_images_ns / 1000,
            cp_perf.reset_buffers_ns / 1000, cp_perf.bound_buffers, cp_perf.bound_images,
            cp_perf.tex_lookups, cp_perf.tex_hits, cp_perf.tex_miss_key,
            cp_perf.tex_miss_generation, cp_perf.tex_miss_unreusable, cp_perf.tex_registry_changes,
            cp_perf.tex_find_images, cp_perf.tex_sites[0], cp_perf.tex_sites[1],
            cp_perf.tex_sites[2], cp_perf.tex_sites[3], cp_perf.tex_sites[4], cp_perf.tex_sites[5],
            cp_perf.tex_sites[6], cp_perf.tex_sites[7], cp_perf.tex_sites[8], cp_perf.tex_sites[9],
            cp_perf.tex_sites[10], cp_perf.tex_sites[11], cp_perf.tex_sites[12], cp_perf.pm_pages,
            cp_perf.pm_protects, cp_perf.pm_image_pages, cp_perf.pm_image_protects,
            cp_perf.tex_alias_swaps, cp_perf.tex_find_types[0], cp_perf.tex_find_types[1],
            cp_perf.tex_find_types[2], cp_perf.tex_find_types[3], cp_perf.tex_find_types[4],
            cp_perf.upload_sizes[0], cp_perf.upload_sizes[1], cp_perf.upload_sizes[2],
            cp_perf.upload_sizes[3], cp_perf.upload_size_bytes[0] / 1024,
            cp_perf.upload_size_bytes[1] / 1024, cp_perf.upload_size_bytes[2] / 1024,
            cp_perf.upload_size_bytes[3] / 1024, cp_perf.cached_obtain_sizes[0],
            cp_perf.cached_obtain_sizes[1], cp_perf.cached_obtain_sizes[2],
            cp_perf.cached_obtain_sizes[3], cp_perf.db_misses[0], cp_perf.db_misses[1],
            cp_perf.db_misses[2], cp_perf.db_misses[3], cp_perf.db_misses[4], cp_perf.db_misses[5],
            cp_perf.db_unreusable[0], cp_perf.db_unreusable[1], cp_perf.db_unreusable[2],
            cp_perf.resident_hits, guest_faults.writes.exchange(0, std::memory_order_relaxed),
            guest_faults.write_ns.exchange(0, std::memory_order_relaxed) / 1000,
            guest_faults.next_page_writes.exchange(0, std::memory_order_relaxed),
            guest_faults.reads.exchange(0, std::memory_order_relaxed),
            guest_faults.read_ns.exchange(0, std::memory_order_relaxed) / 1000,
            cp_perf.la_solo_runs,
            lookahead_perf.stream_ns.exchange(0, std::memory_order_relaxed) / 1000,
            lookahead_perf.stream_skips.exchange(0, std::memory_order_relaxed), cp_perf.va_used,
            cp_perf.va_checks, cp_perf.va_diffs,
            eager_readbacks.downloads.exchange(0, std::memory_order_relaxed),
            eager_readbacks.windows.exchange(0, std::memory_order_relaxed),
            eager_readbacks.bytes.exchange(0, std::memory_order_relaxed) / 1024,
            eager_readbacks.gpu_ns.exchange(0, std::memory_order_relaxed) / 1000,
            eager_readbacks.total_ns.exchange(0, std::memory_order_relaxed) / 1000,
            readback_wait.emulated_ns.exchange(0, std::memory_order_relaxed) / 1000,
            cp_perf.db_expired, eager_readbacks.unmarked.exchange(0, std::memory_order_relaxed),
            eager_readbacks.skip_tick.exchange(0, std::memory_order_relaxed),
            eager_readbacks.skip_flight.exchange(0, std::memory_order_relaxed),
            eager_readbacks.known_faults.exchange(0, std::memory_order_relaxed),
            eager_readbacks.new_faults.exchange(0, std::memory_order_relaxed),
            cp_perf.early_flushes, cp_perf.ra_stages, cp_perf.ra_checks, cp_perf.ra_diffs,
            cp_perf.la_run_waits, cp_perf.la_run_wait_ns / 1000, cp_perf.la_woken_runs,
            cp_perf.la_woken_wait_ns / 1000, cp_perf.la_finished_ahead[0],
            cp_perf.la_finished_ahead[1], cp_perf.la_finished_ahead[2], cp_perf.idle_flushes,
            cp_perf.auto_checks, cp_perf.auto_check_ns / 1000,
            idle_avg_ns * 1000 / std::max<s64>(idle_avg_ns + busy_avg_ns, 1),
            static_cast<u32>(has_submit_slack), cp_perf.bb_total, cp_perf.bb_same,
            cp_perf.bb_same_tick, cp_perf.bs_total, cp_perf.bs_same_tick, cp_perf.bs_one_diff,
            cp_perf.tb_total, cp_perf.tb_same, cp_perf.tb_same_tick, cp_perf.ts_total,
            cp_perf.ts_same_tick, cp_perf.ts_one_diff,
            readback_wait.emulated.exchange(0, std::memory_order_relaxed), cp_perf.stg_same_inputs,
            cp_perf.vk_vi_ns / 1000, cp_perf.vk_index_ns / 1000, cp_perf.vk_desc_ns / 1000,
            cp_perf.vk_dyn_ns / 1000, cp_perf.vk_draw_ns / 1000, cp_perf.rec_cmds,
            cp_perf.rec_bytes / 1024, cp_perf.rec_publishes, cp_perf.rec_wakes, cp_perf.rec_syncs,
            cp_perf.rec_sync_ns / 1000, cp_perf.rec_space_waits,
            recorder_perf.run_ns.exchange(0, std::memory_order_relaxed) / 1000,
            recorder_perf.sleeps.exchange(0, std::memory_order_relaxed), cp_perf.sc_total_ns / 1000,
            cp_perf.sc_compare_ns / 1000, cp_perf.sc_guest_ns / 1000, cp_perf.sc_write_ns / 1000,
            recorder_perf.stream_write_ns.exchange(0, std::memory_order_relaxed) / 1000,
            cp_perf.run_n, cp_perf.run_draws[0], cp_perf.run_draws[1], cp_perf.run_draws[2],
            cp_perf.run_draws[3], cp_perf.run_draws[4], cp_perf.run_draws[5], cp_perf.state_packets,
            cp_perf.state_words, cp_perf.la_runs, cp_perf.la_results, cp_perf.la_waits,
            cp_perf.la_wait_ns / 1000, cp_perf.la_sync_waits, cp_perf.la_sync_ns / 1000,
            cp_perf.la_copies, cp_perf.la_compares, cp_perf.la_mismatches, cp_perf.la_early,
            lookahead_perf.scan_ns.exchange(0, std::memory_order_relaxed) / 1000,
            lookahead_perf.sleeps.exchange(0, std::memory_order_relaxed), cp_perf.la_lookups_used,
            cp_perf.la_lookups_rejected, cp_perf.la_lookup_ns / 1000, cp_perf.la_lookups_checked,
            cp_perf.la_stream_ns / 1000, cp_perf.sa_ahead_known, cp_perf.sa_ahead_same,
            cp_perf.sa_ahead_copied, cp_perf.sa_ahead_bytes / 1024, cp_perf.sa_ahead_full,
            cp_perf.sa_used, cp_perf.sa_stale, cp_perf.sa_checks, cp_perf.sa_diffs,
            cp_perf.tx_first_ns / 1000, cp_perf.tx_second_ns / 1000, cp_perf.tx_samplers_ns / 1000,
            cp_perf.dispatch_hle_ns / 1000, cp_perf.dispatch_bind_ns / 1000, cp_perf.dispatch_hles,
            cp_perf.dispatch_skips, cp_perf.dispatch_bindbuf_ns / 1000,
            cp_perf.dispatch_bindtex_ns / 1000, cp_perf.ow_obtain_ns / 1000,
            cp_perf.ow_inval_ns / 1000, cp_perf.ow_bytes / 1024, cp_perf.ow_dirty_ns / 1000,
            cp_perf.ow_sync_ns / 1000, cp_perf.ow_lock_ns / 1000, cp_perf.ow_walk_ns / 1000,
            cp_perf.ow_mark_ns / 1000, cp_perf.ow_upload_ns / 1000, cp_perf.ow_uploads,
            cp_perf.ow_upload_bytes / 1024, cp_perf.ow_watch_pages, cp_perf.ow_protects,
            perf_bind_check ? 1 : 0, cp_perf.bc_images, cp_perf.bc_images_same,
            cp_perf.bc_images_equal, cp_perf.bc_image_stages, cp_perf.bc_image_stages_same,
            cp_perf.bc_image_stages_quiet, cp_perf.bc_draws, cp_perf.bc_draws_state_same);
        if (++perf_run_log_cycles == 2048) {
            perf_run_log_cycles = 0;
            if (is_gameplay && perf_tex_site_refills < PerfTexSiteRefills) {
                ++perf_tex_site_refills;
                perf_tex_site_budget.store(PerfTexSiteLines, std::memory_order_relaxed);
                perf_db_site_budget.store(PerfDbSiteLines, std::memory_order_relaxed);
                perf_upload_site_budget.store(PerfUploadSiteLines, std::memory_order_relaxed);
                perf_eager_fault_budget.store(PerfEagerFaultLines, std::memory_order_relaxed);
            }
            std::string enders;
            for (u32 op = 0; op < perf_run_enders.size(); ++op) {
                if (perf_run_enders[op] != 0) {
                    enders += fmt::format(" {:#x}={}", op, perf_run_enders[op]);
                }
            }
            perf_run_enders.fill(0);
            LOG_INFO(Render, "PERF run_enders{}", enders);
        }

        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb) {
    FIBER_ENTER(ccb_task_name);

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type == 2) {
            ccb = NextPacket(ccb, 1);
            continue;
        }
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        // Read the size before running the packet: once it signals the guest, the guest may
        // already reuse the buffer.
        const u32 packet_words = header->type3.NumWords() + 1;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            memcpy(dump_const->Address<void*>(),
                   cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while (cblock.ce_count >= cblock.de_count &&
                   (cblock.ce_count - cblock.de_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task =
                ProcessCeUpdate({indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
            SCOPE_EXIT {
                task.handle.destroy();
            };
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const auto* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_CE();
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, packet_words);
    }

    FIBER_EXIT;
}

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

template <typename Packet>
constexpr std::size_t PacketWords = sizeof(Packet) / sizeof(u32);

} // Anonymous namespace

namespace {

// Values of Liverpool::PacketClass.
constexpr u8 PacketState = 0;
constexpr u8 PacketDraw = 1;
constexpr u8 PacketOther = 2;
constexpr u8 PacketDependsOnPayload = 0xff;

/// Class of the packets of each opcode, unless it depends on their payload.
constexpr std::array<u8, 256> PacketClasses = [] {
    using enum PM4ItOpcode;
    std::array<u8, 256> classes{};
    classes.fill(PacketOther);
    for (const PM4ItOpcode opcode : {Nop, EventWrite, DmaData}) {
        classes[static_cast<u32>(opcode)] = PacketDependsOnPayload;
    }
    for (const PM4ItOpcode opcode :
         {ContextControl, ClearState, SetConfigReg, SetContextReg, SetShReg, SetUconfigReg,
          IndexType, NumInstances, IndexBase, IndexBufferSize, SetBase, AcquireMem, PfpSyncMe}) {
        classes[static_cast<u32>(opcode)] = PacketState;
    }
    for (const PM4ItOpcode opcode :
         {DrawIndex2, DrawIndexOffset2, DrawIndexAuto, DrawIndirect, DrawIndirectCountMulti,
          DrawIndirectMulti, DrawIndexIndirect, DrawIndexIndirectMulti, DrawIndexIndirectCountMulti,
          DispatchDirect, DispatchIndirect}) {
        classes[static_cast<u32>(opcode)] = PacketDraw;
    }
    return classes;
}();

} // Anonymous namespace

Liverpool::PacketClass Liverpool::ClassifyPacket(const PM4Header* header,
                                                 bool guest_markers_enabled) {
    static_assert(static_cast<u8>(PacketClass::State) == PacketState &&
                  static_cast<u8>(PacketClass::Draw) == PacketDraw &&
                  static_cast<u8>(PacketClass::Other) == PacketOther);
    const u8 fixed_class = PacketClasses[static_cast<u32>(header->type3.opcode.Value()) & 0xff];
    if (fixed_class != PacketDependsOnPayload) [[likely]] {
        return static_cast<PacketClass>(fixed_class);
    }
    switch (header->type3.opcode) {
    case PM4ItOpcode::Nop: {
        const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
        if (nop->header.count.Value() == 0) {
            return PacketClass::State;
        }
        switch (nop->data_block[0]) {
        case PM4CmdNop::PayloadType::PatchedFlip:
            return PacketClass::Other;
        case PM4CmdNop::PayloadType::DebugMarkerPush:
        case PM4CmdNop::PayloadType::DebugColorMarkerPush:
        case PM4CmdNop::PayloadType::DebugMarkerPop:
            return guest_markers_enabled ? PacketClass::Other : PacketClass::State;
        default:
            return PacketClass::State;
        }
    }
    case PM4ItOpcode::EventWrite: {
        const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
        const bool dumps_counters = event->event_index.Value() == EventIndex::ZpassDone &&
                                    event->event_type.Value() == EventType::PixelPipeStatDump;
        return dumps_counters ? PacketClass::Other : PacketClass::State;
    }
    case PM4ItOpcode::DmaData: {
        const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
        return dma_data->dst_addr_lo == 0x3022C ? PacketClass::State : PacketClass::Other;
    }
    default:
        return PacketClass::Other;
    }
}

bool Liverpool::ChangesStateRegs(const PM4Header* header) const {
    u32 base;
    switch (header->type3.opcode) {
    case PM4ItOpcode::ClearState:
        return true;
    case PM4ItOpcode::SetConfigReg:
        base = Regs::ConfigRegWordOffset;
        break;
    case PM4ItOpcode::SetContextReg:
        base = Regs::ContextRegWordOffset;
        break;
    case PM4ItOpcode::SetUconfigReg:
        base = Regs::UconfigRegWordOffset;
        break;
    default:
        return false;
    }
    const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
    return std::memcmp(&regs.reg_array[base + set_data->reg_offset], header + 2,
                       (header->type3.NumWords() - 1) * sizeof(u32)) != 0;
}

void Liverpool::ApplyPacketState(const GfxState& state, const PM4Header* header) {
    Regs& regs = state.regs;
    const u32 count = header->type3.NumWords();
    switch (header->type3.opcode) {
    case PM4ItOpcode::ClearState: {
        regs.SetDefaults();
        break;
    }
    case PM4ItOpcode::SetConfigReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
        const auto* payload = reinterpret_cast<const u32*>(header + 2);
        std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));
        break;
    }
    case PM4ItOpcode::SetContextReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
        const auto* payload = reinterpret_cast<const u32*>(header + 2);

        std::memcpy(&regs.reg_array[reg_addr], payload, (count - 1) * sizeof(u32));

        // In the case of HW, render target memory has alignment as color block operates on
        // tiles. There is no information of actual resource extents stored in CB context
        // regs, so any deduction of it from slices/pitch will lead to a larger surface created.
        // The same applies to the depth targets. Fortunately, the guest always sends
        // a trailing NOP packet right after the context regs setup, so we can use the heuristic
        // below and extract the hint to determine actual resource dims.

        switch (reg_addr) {
        case ContextRegs::CbColor0Base:
        case ContextRegs::CbColor1Base:
        case ContextRegs::CbColor2Base:
        case ContextRegs::CbColor3Base:
        case ContextRegs::CbColor4Base:
        case ContextRegs::CbColor5Base:
        case ContextRegs::CbColor6Base:
        case ContextRegs::CbColor7Base: {
            const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                    (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
            ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

            const auto nop_offset = header->type3.count;
            if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                           "NOP hint is missing in CB setup sequence");
                state.last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
            } else {
                state.last_cb_extent[col_buf_id].raw = 0;
            }
            break;
        }
        case ContextRegs::CbColor0Cmask:
        case ContextRegs::CbColor1Cmask:
        case ContextRegs::CbColor2Cmask:
        case ContextRegs::CbColor3Cmask:
        case ContextRegs::CbColor4Cmask:
        case ContextRegs::CbColor5Cmask:
        case ContextRegs::CbColor6Cmask:
        case ContextRegs::CbColor7Cmask: {
            const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Cmask) /
                                    (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
            ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

            const auto nop_offset = header->type3.count;
            if (nop_offset == 0x04) {
                ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                           "NOP hint is missing in CB setup sequence");
                state.last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
            }
            break;
        }
        case ContextRegs::DbZInfo: {
            if (header->type3.count == 8) {
                ASSERT_MSG(payload[20] == 0xc0001000, "NOP hint is missing in DB setup sequence");
                state.last_db_extent.raw = payload[21];
            } else {
                state.last_db_extent.raw = 0;
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    case PM4ItOpcode::SetShReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        const auto set_size = (count - 1) * sizeof(u32);

        if (set_data->reg_offset >= 0x200 &&
            set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
            ASSERT(set_size <= sizeof(ComputeProgram));
            auto* addr = reinterpret_cast<u32*>(&state.cs_state) + (set_data->reg_offset - 0x200);
            std::memcpy(addr, header + 2, set_size);
        } else {
            std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset], header + 2,
                        set_size);
        }
        break;
    }
    case PM4ItOpcode::SetUconfigReg: {
        const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
        std::memcpy(&regs.reg_array[Regs::UconfigRegWordOffset + set_data->reg_offset], header + 2,
                    (count - 1) * sizeof(u32));
        break;
    }
    case PM4ItOpcode::IndexType: {
        const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
        regs.index_buffer_type.raw = index_type->raw;
        break;
    }
    case PM4ItOpcode::NumInstances: {
        const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
        regs.num_instances.num_instances = num_instances->num_instances;
        break;
    }
    case PM4ItOpcode::IndexBase: {
        const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
        regs.index_base_address.base_addr_lo = index_base->addr_lo;
        regs.index_base_address.base_addr_hi = index_base->addr_hi;
        break;
    }
    case PM4ItOpcode::IndexBufferSize: {
        const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
        state.index_buffer_num_indices = index_size->num_indices;
        break;
    }
    case PM4ItOpcode::SetBase: {
        const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
        ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
        state.indirect_args_addr = set_base->Address<u64>();
        break;
    }
    case PM4ItOpcode::DrawIndex2: {
        const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
        regs.max_index_size = draw_index->max_size;
        regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
        regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
        regs.num_indices = draw_index->index_count;
        regs.draw_initiator = draw_index->draw_initiator;
        break;
    }
    case PM4ItOpcode::DrawIndexOffset2: {
        const auto* draw_index_off = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
        regs.max_index_size = draw_index_off->max_size;
        regs.num_indices = draw_index_off->index_count;
        regs.draw_initiator = draw_index_off->draw_initiator;
        break;
    }
    case PM4ItOpcode::DrawIndexAuto: {
        const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
        regs.num_indices = draw_index->index_count;
        regs.draw_initiator = draw_index->draw_initiator;
        break;
    }
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti: {
        regs.max_index_size = state.index_buffer_num_indices;
        break;
    }
    case PM4ItOpcode::DispatchDirect: {
        const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
        state.cs_state.dim_x = dispatch_direct->dim_x;
        state.cs_state.dim_y = dispatch_direct->dim_y;
        state.cs_state.dim_z = dispatch_direct->dim_z;
        state.cs_state.dispatch_initiator = dispatch_direct->dispatch_initiator;
        break;
    }
    case PM4ItOpcode::EventWrite: {
        const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
        if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
            // TODO: handle proper synchronization, for now signal that update is done
            // immediately
            regs.cp_strmout_cntl.offset_update_done = 1;
        }
        break;
    }
    default:
        break;
    }
}

bool Liverpool::IsSafePacket(const PM4Header* header, std::size_t words_available) {
    if (words_available == 0 || header->type != 3) {
        return false;
    }
    const std::size_t count = header->type3.NumWords();
    const std::size_t packet_words = count + 1;
    if (packet_words > words_available) {
        return false;
    }
    const auto fits_regs = [&](std::size_t reg_addr) {
        return packet_words >= PacketWords<PM4CmdSetData> && reg_addr + count - 1 <= Regs::NumRegs;
    };
    const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
    switch (header->type3.opcode) {
    case PM4ItOpcode::SetConfigReg:
        return fits_regs(Regs::ConfigRegWordOffset + set_data->reg_offset);
    case PM4ItOpcode::SetUconfigReg:
        return fits_regs(Regs::UconfigRegWordOffset + set_data->reg_offset);
    case PM4ItOpcode::SetShReg: {
        if (packet_words < PacketWords<PM4CmdSetData>) {
            return false;
        }
        const std::size_t reg_offset = set_data->reg_offset;
        if (reg_offset >= 0x200 && reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
            return (reg_offset - 0x200 + count - 1) * sizeof(u32) <= sizeof(ComputeProgram);
        }
        return fits_regs(Regs::ShRegWordOffset + reg_offset);
    }
    case PM4ItOpcode::SetContextReg: {
        const std::size_t reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
        if (!fits_regs(reg_addr)) {
            return false;
        }
        // The extent hint is read from the NOP packet that follows.
        const auto* payload = reinterpret_cast<const u32*>(header + 2);
        const std::size_t nop_offset = header->type3.count;
        const auto has_hint = [&](std::size_t hint_offset) {
            return words_available >= hint_offset + 4 && payload[hint_offset] == 0xc0001000;
        };
        switch (reg_addr) {
        case ContextRegs::CbColor0Base:
        case ContextRegs::CbColor1Base:
        case ContextRegs::CbColor2Base:
        case ContextRegs::CbColor3Base:
        case ContextRegs::CbColor4Base:
        case ContextRegs::CbColor5Base:
        case ContextRegs::CbColor6Base:
        case ContextRegs::CbColor7Base:
            return (nop_offset != 0x0e && nop_offset != 0x0d && nop_offset != 0x0b) ||
                   has_hint(nop_offset);
        case ContextRegs::CbColor0Cmask:
        case ContextRegs::CbColor1Cmask:
        case ContextRegs::CbColor2Cmask:
        case ContextRegs::CbColor3Cmask:
        case ContextRegs::CbColor4Cmask:
        case ContextRegs::CbColor5Cmask:
        case ContextRegs::CbColor6Cmask:
        case ContextRegs::CbColor7Cmask:
            return nop_offset != 0x04 || has_hint(nop_offset);
        case ContextRegs::DbZInfo:
            return nop_offset != 8 || has_hint(20);
        default:
            return true;
        }
    }
    case PM4ItOpcode::Nop:
        return packet_words >= 2;
    case PM4ItOpcode::EventWrite:
        return packet_words >= PacketWords<PM4CmdEventWrite>;
    case PM4ItOpcode::DmaData:
        return packet_words >= PacketWords<PM4DmaData>;
    case PM4ItOpcode::IndexType:
        return packet_words >= PacketWords<PM4CmdDrawIndexType>;
    case PM4ItOpcode::NumInstances:
        return packet_words >= PacketWords<PM4CmdDrawNumInstances>;
    case PM4ItOpcode::IndexBase:
        return packet_words >= PacketWords<PM4CmdDrawIndexBase>;
    case PM4ItOpcode::IndexBufferSize:
        return packet_words >= PacketWords<PM4CmdDrawIndexBufferSize>;
    case PM4ItOpcode::SetBase:
        return packet_words >= PacketWords<PM4CmdSetBase> &&
               reinterpret_cast<const PM4CmdSetBase*>(header)->base_index ==
                   PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable;
    case PM4ItOpcode::DrawIndex2:
        return packet_words >= PacketWords<PM4CmdDrawIndex2>;
    case PM4ItOpcode::DrawIndexOffset2:
        return packet_words >= PacketWords<PM4CmdDrawIndexOffset2>;
    case PM4ItOpcode::DrawIndexAuto:
        return packet_words >= PacketWords<PM4CmdDrawIndexAuto>;
    case PM4ItOpcode::DispatchDirect:
        return packet_words >= PacketWords<PM4CmdDispatchDirect>;
    default:
        return true;
    }
}

bool Liverpool::HasLookaheadDraw(std::span<const u32> dcb) const {
    // The walk of the lookahead thread.
    const u32* packet = dcb.data();
    const u32* const end = packet + dcb.size();
    while (packet < end) {
        const auto* header = reinterpret_cast<const PM4Header*>(packet);
        if (header->type == 2) {
            ++packet;
            continue;
        }
        if (!IsSafePacket(header, end - packet)) {
            break;
        }
        const PacketClass packet_class = ClassifyPacket(header, guest_markers_enabled);
        if (packet_class == PacketClass::Other) {
            break;
        }
        if (packet_class == PacketClass::Draw) {
            switch (header->type3.opcode) {
            case PM4ItOpcode::DrawIndex2:
            case PM4ItOpcode::DrawIndexOffset2:
            case PM4ItOpcode::DrawIndexAuto:
                return true;
            default:
                break;
            }
        }
        packet += header->type3.NumWords() + 1;
    }
    return false;
}

void Liverpool::ApplySoloPacketState(const PM4Header* header, std::size_t words_available) {
    if (lookahead_needs_copy) {
        // The lookahead state is copied before its next run.
        return;
    }
    if (!IsSafePacket(header, words_available)) {
        lookahead_needs_copy = true;
        return;
    }
    ApplyPacketState(lookahead_state->Ref(), header);
}

void Liverpool::StartLookaheadRun(std::span<const u32> dcb) {
    if (!lookahead_thread.joinable()) {
        lookahead_state = std::make_unique<GfxStateCopy>();
        lookahead_draws = std::make_unique<Vulkan::DrawAhead[]>(NumLookaheadResults);
        lookahead_cp_perf = std::make_unique<CpPerf>();
        lookahead_binding_perf = std::make_unique<CpPerf>();
        lookahead_thread = std::jthread{std::bind_front(&Liverpool::LookaheadThread, this)};
        lookahead_binding_thread =
            std::jthread{std::bind_front(&Liverpool::LookaheadBindingThread, this)};
    }
    // The lookahead thread is stopped here, so its state has to be the real one.
    if (!lookahead_needs_copy && lookahead_compare_countdown-- == 0) {
        lookahead_compare_countdown = LookaheadCompareInterval - 1;
        ++cp_perf.la_compares;
        if (!CompareLookaheadState()) {
            ++cp_perf.la_mismatches;
            lookahead_needs_copy = true;
        }
    }
    if (lookahead_needs_copy) {
        GfxStateCopy& copy = *lookahead_state;
        std::memcpy(&copy.regs, &regs, sizeof(regs));
        std::memcpy(&copy.cs_state, &mapped_queues[GfxQueueId].cs_state, sizeof(copy.cs_state));
        copy.last_cb_extent = last_cb_extent;
        copy.last_db_extent = last_db_extent;
        copy.index_buffer_num_indices = index_buffer_num_indices;
        copy.indirect_args_addr = indirect_args_addr;
        lookahead_needs_copy = false;
        ++cp_perf.la_copies;
    }
    lookahead_run_begin = dcb.data();
    lookahead_run_end = dcb.data() + dcb.size();
    is_lookahead_running = true;
    ++cp_perf.la_runs;
    // Sequentially consistent with the two threads, which count themselves in
    // lookahead_sleepers and then read lookahead_started_run: one side sees the other's store.
    lookahead_started_run.store(++lookahead_run);
    lookahead_run_first = lookahead_taken;
    lookahead_run_woke = lookahead_sleepers.load() != 0;
    if (lookahead_run_woke) {
        {
            std::scoped_lock lk{lookahead_mutex};
        }
        lookahead_cv.notify_all();
        ++cp_perf.la_woken_runs;
    }
}

void Liverpool::WaitLookaheadStopped() {
    const auto drop_results = [this] {
        // Results nobody takes must not keep the thread from stopping.
        const u64 produced = lookahead_produced.load(std::memory_order_acquire);
        if (produced != lookahead_taken) {
            lookahead_taken = produced;
            lookahead_consumed.store(produced, std::memory_order_release);
        }
    };
    if (lookahead_stopped_run.load(std::memory_order_acquire) != lookahead_run) {
        const u64 sync_t0 = PerfNs();
        ++cp_perf.la_sync_waits;
        do {
            drop_results();
            PauseForLookahead();
        } while (lookahead_stopped_run.load(std::memory_order_acquire) != lookahead_run);
        cp_perf.la_sync_ns += PerfNs() - sync_t0;
    }
    drop_results();

    // The threads are stopped, so the probes of their work can be taken.
    cp_perf.la_lookup_ns += lookahead_cp_perf->pipe_key_ns;
    *lookahead_cp_perf = {};
    CpPerf& ahead = *lookahead_binding_perf;
    cp_perf.la_stream_ns += ahead.la_stream_ns;
    cp_perf.sa_ahead_known += ahead.sa_ahead_known;
    cp_perf.sa_ahead_same += ahead.sa_ahead_same;
    cp_perf.sa_ahead_copied += ahead.sa_ahead_copied;
    cp_perf.sa_ahead_bytes += ahead.sa_ahead_bytes;
    cp_perf.sa_ahead_full += ahead.sa_ahead_full;
    ahead = {};
}

void Liverpool::PauseForLookahead() {
    if (num_commands) {
        ProcessCommands();
    }
    ThreadPause();
}

void Liverpool::EndLookaheadRun(const u32* packet) {
    is_lookahead_solo = false;
    if (!is_lookahead_running) {
        return;
    }
    is_lookahead_running = false;
    const u64 taken = lookahead_taken;
    WaitLookaheadStopped();
    if (lookahead_stop == packet && lookahead_taken == taken) {
        return;
    }
    // The lookahead thread did not stop at this packet, so its state is not the real one.
    if (lookahead_taken == taken && lookahead_stop < packet) {
        ++cp_perf.la_early;
    } else {
        ++cp_perf.la_mismatches;
        if (lookahead_logged_mismatches < 32) {
            ++lookahead_logged_mismatches;
            LOG_ERROR(Render, "PERF lookahead stopped at {} with {} results left, run ends at {}",
                      fmt::ptr(lookahead_stop), lookahead_taken - taken, fmt::ptr(packet));
        }
    }
    lookahead_needs_copy = true;
}

void Liverpool::TakeLookaheadResult(std::span<const u32> dcb) {
    for (u32 attempt = 0; attempt < 2; ++attempt) {
        u64 ready = lookahead_ready.load(std::memory_order_acquire);
        if (ready == lookahead_taken) {
            const u64 wait_t0 = PerfNs();
            ++cp_perf.la_waits;
            while (lookahead_stopped_run.load(std::memory_order_acquire) != lookahead_run &&
                   lookahead_ready.load(std::memory_order_acquire) == lookahead_taken) {
                PauseForLookahead();
            }
            ready = lookahead_ready.load(std::memory_order_acquire);
            const u64 wait_ns = PerfNs() - wait_t0;
            cp_perf.la_wait_ns += wait_ns;
            if (lookahead_taken == lookahead_run_first) {
                ++cp_perf.la_run_waits;
                cp_perf.la_run_wait_ns += wait_ns;
                cp_perf.la_woken_wait_ns += lookahead_run_woke ? wait_ns : 0;
            }
        }
        if (ready != lookahead_taken) {
            const LookaheadResult& result =
                lookahead_results[lookahead_taken % NumLookaheadResults];
            if (result.packet == dcb.data()) {
                const Vulkan::DrawAhead& taken =
                    lookahead_draws[lookahead_taken % NumLookaheadResults];
                rasterizer->SetDrawAhead(&taken);
                lookahead_consumed.store(++lookahead_taken, std::memory_order_release);
                ++cp_perf.la_results;
                // Finished results are not written again before they are taken.
                const auto finished = [&](u64 index) -> const Vulkan::DrawAhead* {
                    return index < ready ? &lookahead_draws[index % NumLookaheadResults] : nullptr;
                };
                rasterizer->PrefetchDrawsAhead(taken, finished(lookahead_taken),
                                               finished(lookahead_taken + 1),
                                               finished(lookahead_taken + 2));
                for (u32 distance = 0; distance < cp_perf.la_finished_ahead.size(); ++distance) {
                    cp_perf.la_finished_ahead[distance] += lookahead_taken + distance < ready;
                }
                return;
            }
            ++cp_perf.la_mismatches;
            if (lookahead_logged_mismatches < 32) {
                ++lookahead_logged_mismatches;
                LOG_ERROR(Render, "PERF lookahead result is for {}, the draw is at {}",
                          fmt::ptr(result.packet), fmt::ptr(dcb.data()));
            }
        } else {
            ++cp_perf.la_early;
        }
        // The lookahead thread is not at this draw: restart it here with the real state.
        rasterizer->SetDrawAhead(nullptr);
        is_lookahead_running = false;
        WaitLookaheadStopped();
        lookahead_needs_copy = true;
        if (attempt == 0) {
            StartLookaheadRun(dcb);
        }
    }
}

bool Liverpool::CompareLookaheadState() {
    const GfxStateCopy& copy = *lookahead_state;
    const bool is_regs_equal = std::memcmp(&copy.regs, &regs, sizeof(regs)) == 0;
    const bool is_cs_equal = std::memcmp(&copy.cs_state, &mapped_queues[GfxQueueId].cs_state,
                                         sizeof(copy.cs_state)) == 0;
    const bool is_rest_equal = std::memcmp(copy.last_cb_extent.data(), last_cb_extent.data(),
                                           sizeof(last_cb_extent)) == 0 &&
                               copy.last_db_extent.raw == last_db_extent.raw &&
                               copy.index_buffer_num_indices == index_buffer_num_indices &&
                               copy.indirect_args_addr == indirect_args_addr;
    if (is_regs_equal && is_cs_equal && is_rest_equal) {
        return true;
    }
    if (lookahead_logged_mismatches < 32) {
        ++lookahead_logged_mismatches;
        u32 reg = 0;
        while (reg < Regs::NumRegs && copy.regs.reg_array[reg] == regs.reg_array[reg]) {
            ++reg;
        }
        const bool has_reg = reg < Regs::NumRegs;
        LOG_ERROR(Render,
                  "PERF lookahead state differs: reg {:#x} real {:#x} lookahead {:#x}, compute "
                  "state equal {}, rest equal {}",
                  has_reg ? reg : 0, has_reg ? regs.reg_array[reg] : 0,
                  has_reg ? copy.regs.reg_array[reg] : 0, is_cs_equal, is_rest_equal);
    }
    return false;
}

void Liverpool::LookaheadThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuLookahead");
    Common::SamplingProfiler::RegisterCurrentThread("GpuLookahead");

    const GfxState state = lookahead_state->Ref();
    u64 run = 0;
    u32 idle_spins = 0;
    while (!stoken.stop_requested()) {
        const u64 started = lookahead_started_run.load(std::memory_order_acquire);
        if (started == run) {
            if (SpinsForLookaheadRun(++idle_spins)) {
                ThreadPause();
                continue;
            }
            idle_spins = 0;
            SleepUntilLookaheadRun(stoken, run);
            lookahead_perf.sleeps.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        idle_spins = 0;
        run = started;

        const u64 scan_t0 = PerfNs();
        const u32* packet = lookahead_run_begin;
        const u32* const end = lookahead_run_end;
        u64 produced = lookahead_produced.load(std::memory_order_relaxed);
        while (packet < end) {
            const auto* header = reinterpret_cast<const PM4Header*>(packet);
            if (header->type == 2) {
                ++packet;
                continue;
            }
            if (!IsSafePacket(header, end - packet)) {
                break;
            }
            const PacketClass packet_class = ClassifyPacket(header, guest_markers_enabled);
            if (packet_class == PacketClass::Other) {
                break;
            }
            ApplyPacketState(state, header);
            if (packet_class == PacketClass::Draw) {
                // The entry of the last result taken stays as it is: its draw may be using it.
                // Entries the binding thread has not finished are its to use.
                while (produced - lookahead_consumed.load(std::memory_order_acquire) >=
                           NumLookaheadResults - 1 ||
                       produced - lookahead_ready.load(std::memory_order_acquire) >=
                           NumLookaheadResults - 1) {
                    if (stoken.stop_requested()) {
                        return;
                    }
                    ThreadPause();
                }
                Vulkan::DrawAhead& ahead = lookahead_draws[produced % NumLookaheadResults];
                switch (header->type3.opcode) {
                case PM4ItOpcode::DrawIndex2:
                    rasterizer->LookAhead(ahead, state.regs, true, 0, run, *lookahead_cp_perf);
                    break;
                case PM4ItOpcode::DrawIndexOffset2:
                    rasterizer->LookAhead(
                        ahead, state.regs, true,
                        reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header)->index_offset, run,
                        *lookahead_cp_perf);
                    break;
                case PM4ItOpcode::DrawIndexAuto:
                    rasterizer->LookAhead(ahead, state.regs, false, 0, run, *lookahead_cp_perf);
                    break;
                default:
                    ahead.lookup.pipeline = nullptr;
                    break;
                }
                lookahead_results[produced % NumLookaheadResults] = {packet};
                lookahead_produced.store(++produced, std::memory_order_release);
            }
            packet += header->type3.NumWords() + 1;
        }
        lookahead_stop = packet;
        lookahead_scanned_run.store(run, std::memory_order_release);
        lookahead_perf.scan_ns.fetch_add(PerfNs() - scan_t0, std::memory_order_relaxed);
    }
}

void Liverpool::SleepUntilLookaheadRun(std::stop_token stoken, u64 run) {
    std::unique_lock lk{lookahead_mutex};
    lookahead_sleepers.fetch_add(1);
    lookahead_cv.wait(lk, stoken, [&] { return lookahead_started_run.load() != run; });
    lookahead_sleepers.fetch_sub(1);
}

void Liverpool::LookaheadBindingThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuLookaheadBind");
    Common::SamplingProfiler::RegisterCurrentThread("GpuLookaheadBind");

    // The last run this thread declared stopped, and the results it has finished.
    u64 run = 0;
    u64 ready = 0;
    u32 idle_spins = 0;
    while (!stoken.stop_requested()) {
        const u64 started = lookahead_started_run.load(std::memory_order_acquire);
        if (started == run) {
            if (SpinsForLookaheadRun(++idle_spins)) {
                ThreadPause();
                continue;
            }
            idle_spins = 0;
            SleepUntilLookaheadRun(stoken, run);
            continue;
        }
        idle_spins = 0;
        if (ready != lookahead_produced.load(std::memory_order_acquire)) {
            // A result below lookahead_consumed that is not finished was dropped by the GPU
            // thread: nobody takes it.
            if (ready >= lookahead_consumed.load(std::memory_order_acquire)) {
                const u64 bind_t0 = PerfNs();
                Vulkan::DrawAhead& ahead = lookahead_draws[ready % NumLookaheadResults];
                if (ahead.lookup.pipeline) {
                    rasterizer->BindAhead(ahead, *lookahead_binding_perf);
                }
                lookahead_perf.stream_ns.fetch_add(PerfNs() - bind_t0, std::memory_order_relaxed);
            } else {
                lookahead_perf.stream_skips.fetch_add(1, std::memory_order_relaxed);
            }
            lookahead_ready.store(++ready, std::memory_order_release);
            continue;
        }
        // The lookahead thread stores its last result of a run before it names the run scanned,
        // and makes no result of the next run before this thread has named this one stopped.
        if (lookahead_scanned_run.load(std::memory_order_acquire) == started &&
            ready == lookahead_produced.load(std::memory_order_acquire)) {
            run = started;
            lookahead_stopped_run.store(run, std::memory_order_release);
            continue;
        }
        ThreadPause();
    }
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb) {
    FIBER_ENTER(dcb_task_name);

    const GfxState gfx_state = LiveGfxState();
    u32 run_length = 0;
    const auto end_run = [&run_length](u32 ender) {
        if (run_length == 0) {
            return;
        }
        ++cp_perf.run_n;
        cp_perf.run_draws[std::min(std::bit_width(run_length) - 1, 5)] += run_length;
        ++perf_run_enders[ender];
        run_length = 0;
    };

    auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    while (!dcb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            const PacketClass packet_class = ClassifyPacket(header, guest_markers_enabled);
            if (packet_class == PacketClass::Other) {
                end_run(static_cast<u32>(opcode) & 0xff);
                EndLookaheadRun(dcb.data());
                perf_state_changed = true;
            } else {
                if (rasterizer) {
                    if (!is_lookahead_running && !is_lookahead_solo) {
                        if (HasLookaheadDraw(dcb)) {
                            StartLookaheadRun(dcb);
                        } else {
                            is_lookahead_solo = true;
                            ++cp_perf.la_solo_runs;
                        }
                    }
                    if (is_lookahead_solo) {
                        ApplySoloPacketState(header, dcb.size());
                        if (packet_class == PacketClass::Draw) {
                            rasterizer->SetDrawAhead(nullptr);
                        }
                    } else if (packet_class == PacketClass::Draw) {
                        TakeLookaheadResult(dcb);
                    }
                }
                if (packet_class == PacketClass::Draw) {
                    ++run_length;
                    if (perf_bind_check) {
                        ++cp_perf.bc_draws;
                        cp_perf.bc_draws_state_same += !perf_state_changed;
                        perf_state_changed = false;
                    }
                } else {
                    ++cp_perf.state_packets;
                    cp_perf.state_words += count;
                    if (perf_bind_check && !perf_state_changed) {
                        perf_state_changed = ChangesStateRegs(header);
                    }
                }
            }
            ApplyPacketState(gfx_state, header);
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl:
            case PM4ItOpcode::ClearState:
            case PM4ItOpcode::SetConfigReg:
            case PM4ItOpcode::SetContextReg:
            case PM4ItOpcode::SetShReg:
            case PM4ItOpcode::SetUconfigReg:
            case PM4ItOpcode::IndexType:
            case PM4ItOpcode::NumInstances:
            case PM4ItOpcode::IndexBase:
            case PM4ItOpcode::IndexBufferSize:
            case PM4ItOpcode::SetBase: {
                // ApplyPacketState has done all of it.
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(true); });
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexOffset2", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->Draw(true, draw_index_off->index_offset); });
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->Draw(false); });
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectCountMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectCountMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            false, indirect_args_addr, offset, draw_indirect->stride,
                            draw_indirect->count, draw_indirect->count_addr,
                            draw_indirect->base_vtx_loc, draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0,
                                                 draw_indirect->base_vtx_loc,
                                                 draw_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirect", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0,
                                                 draw_index_indirect->base_vtx_loc,
                                                 draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count, 0, draw_index_indirect->base_vtx_loc,
                            draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (!rasterizer) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DrawIndexIndirectCountMulti", fmt::make_format_args(cmd_address), [&] {
                        rasterizer->DrawIndirect(
                            true, indirect_args_addr, offset, draw_index_indirect->stride,
                            draw_index_indirect->count,
                            draw_index_indirect->count_indirect_enable.Value()
                                ? draw_index_indirect->count_addr
                                : 0,
                            draw_index_indirect->base_vtx_loc, draw_index_indirect->start_inst_loc);
                    });
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                auto& cs_program = GetCsRegs();
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(cmd_address),
                                        [&] { rasterizer->DispatchDirect(); });
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                    break;
                }
                const auto cmd_address = reinterpret_cast<const void*>(header);
                rasterizer->ScopeMarker(
                    "gfx:{}:DispatchIndirect", fmt::make_format_args(cmd_address),
                    [&] { rasterizer->DispatchIndirect(indirect_args_addr, offset, size); });
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() != EventType::SoVgtStreamoutFlush &&
                    event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        ++cp_perf.zpass_dumps;
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* results = event->Address<u64*>();
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const CpPerfScope perf_eos{cp_perf.eos_ns};
                ++cp_perf.eos;
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                event_eos->SignalFence([](void* address, u64 data, u32 num_bytes) {
                    auto* memory = Core::Memory::Instance();
                    ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                });
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        *event_eos->Address() = value;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const CpPerfScope perf_eop{cp_perf.eop_ns};
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                event_eop->SignalFence(
                    [](void* address, u64 data, u32 num_bytes) {
                        auto* memory = Core::Memory::Instance();
                        ASSERT(memory->TryWriteBacking(address, &data, num_bytes));
                    },
                    [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                ASSERT(dma_data->command.das == 0);
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                    u32(dma_data->dst_sel));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                u64* address = write_data->Address<u64*>();
                if (!write_data->wr_one_addr.Value()) {
                    if (rasterizer) {
                        rasterizer->OnFence();
                    }
                    // The driver's patched flip: the label write directly followed by the flip
                    // NOP. The flip is finished on the GPU right here.
                    const size_t this_size = header->type3.NumWords() + 1;
                    const auto* next_nop =
                        dcb.size() > this_size + 1
                            ? reinterpret_cast<const PM4CmdNop*>(dcb.data() + this_size)
                            : nullptr;
                    const bool flip_label =
                        next_nop && vo_port && vo_port->IsVoLabel(address) &&
                        next_nop->header.type == 3 && next_nop->header.opcode == PM4ItOpcode::Nop &&
                        next_nop->data_block[0] == PM4CmdNop::PayloadType::PatchedFlip;
                    if (flip_label) {
                        vo_port->WriteFlipLabel(address, write_data->data, data_size);
                    } else {
                        std::memcpy(address, write_data->data, data_size);
                    }
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                } else {
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                    mem_semaphore->Decrement();
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                const bool vo_label = vo_port->IsVoLabel(wait_addr);
                if (vo_label && num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    const CpPerfScope perf_vo{cp_perf.vo_block_ns};
                    ++cp_perf.vo_blocks;
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    break;
                }
                if (!wait_reg_mem->Test(regs.reg_array)) {
                    const u64 t0 = PerfNs();
                    while (!wait_reg_mem->Test(regs.reg_array)) {
                        YIELD_GFX();
                    }
                    if (vo_label) {
                        cp_perf.vo_spin_ns += PerfNs() - t0;
                    } else {
                        cp_perf.mem_spin_ns += PerfNs() - t0;
                        ++cp_perf.mem_spins;
                    }
                }
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                if (indirect_buffer->chain) {
                    // A chained buffer replaces the current one: the CP never comes back to the
                    // rest of this buffer, which in reused segments still holds old commands.
                    dcb = {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size};
                    base_addr = reinterpret_cast<uintptr_t>(dcb.data());
                    continue;
                }
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size});
                SCOPE_EXIT {
                    task.handle.destroy();
                };
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                const u32 buf_idx = strmout->buffer_select.Value();
                LOG_DEBUG(Render,
                          "IT_STRMOUT_BUFFER_UPDATE buffer_select = {}, source_select = {}, "
                          "update_memory = {}",
                          buf_idx, magic_enum::enum_name(strmout->source_select.Value()),
                          strmout->update_memory.Value());
                u32 offset = vgt_strmout_buffer_filled_size[buf_idx];
                switch (strmout->source_select.Value()) {
                case SourceSelect::BufferOffset:
                    offset = strmout->buffer_offset;
                    break;
                case SourceSelect::VgtStrmoutBufferFilledSize:
                    offset = vgt_strmout_buffer_filled_size[buf_idx];
                    break;
                case SourceSelect::SrcAddress: {
                    const VAddr src_addr = strmout->SrcAddress<VAddr>();
                    auto* memory = Core::Memory::Instance();
                    if (src_addr && memory && memory->IsValidMapping(src_addr, sizeof(offset))) {
                        std::memcpy(&offset, reinterpret_cast<const void*>(src_addr),
                                    sizeof(offset));
                    }
                    break;
                }
                case SourceSelect::None:
                    break;
                }
                if (strmout->source_select.Value() != SourceSelect::None) {
                    vgt_strmout_buffer_filled_size[buf_idx] = offset;
                }
                if (strmout->update_memory.Value()) {
                    const VAddr dst_addr = strmout->DstAddress<VAddr>();
                    auto* memory = Core::Memory::Instance();
                    if (dst_addr && memory && memory->IsValidMapping(dst_addr, sizeof(offset))) {
                        std::memcpy(reinterpret_cast<void*>(dst_addr),
                                    &vgt_strmout_buffer_filled_size[buf_idx], sizeof(offset));
                    }
                }
                regs.cp_strmout_cntl.offset_update_done = 1;
                lookahead_needs_copy = true;
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                const auto* get_lod = reinterpret_cast<const PM4CmdGetLodStats*>(header);
                HandleGetLodStats(get_lod);
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            // The size read before running the packet: a flip or an EOP interrupt lets the guest
            // reuse this buffer while we are still on it, so the header may be gone by now.
            dcb = NextPacket(dcb, count + 1);
            break;
        }
    }
    end_run(0);
    EndLookaheadRun(dcb.data());

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    auto& queue = asc_queues[{vqid}];

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            if constexpr (is_indirect) {
                if (indirect_buffer->chain) {
                    // As for graphics, a chained buffer replaces the current one.
                    acb = {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size};
                    base_addr = reinterpret_cast<VAddr>(acb.data());
                    acb_size = acb.size_bytes();
                    continue;
                }
            }
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            SCOPE_EXIT {
                task.handle.destroy();
            };
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            ASSERT(dma_data->command.das == 0);
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel),
                                u32(dma_data->dst_sel));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                lookahead_needs_copy = true;
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchDirect(); });
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (!rasterizer || (cs_program.dispatch_initiator & 1) == 0) {
                break;
            }
            const auto cmd_address = reinterpret_cast<const void*>(header);
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, cmd_address),
                                    [&] { rasterizer->DispatchIndirect(ib_address, 0, size); });
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                if (rasterizer) {
                    rasterizer->OnFence();
                }
                std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
            } else {
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                mem_semaphore->Decrement();
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            if (!wait_reg_mem->Test(regs.reg_array)) {
                const u64 t0 = PerfNs();
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    YIELD_ASC(vqid);
                }
                cp_perf.asc_spin_ns += PerfNs() - t0;
            }
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                rasterizer->OnFence();
            }
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        case PM4ItOpcode::GetLodStats: {
            const auto* get_lod = reinterpret_cast<const PM4CmdGetLodStats*>(header);
            HandleGetLodStats(get_lod);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

void Liverpool::HandleGetLodStats(const PM4CmdGetLodStats* get_lod) {
    LOG_DEBUG(Render, "IT_GET_LOD_STATS: buffer = {:#x}, offset = {:#x}", get_lod->Address(),
              get_lod->DataOffset());
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    auto task = ProcessGraphics(dcb);
    {
        std::scoped_lock lock{submit_mutex, queue.m_access};
        if (!ccb.empty()) {
            ce_submits.emplace(ProcessCeUpdate(ccb).handle);
            ++num_submits;
        }
        queue.submits.emplace(task.handle);
        ++num_submits;
        submit_cv.notify_one();
    }
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    submit_cv.notify_one();
}

} // namespace AmdGpu
