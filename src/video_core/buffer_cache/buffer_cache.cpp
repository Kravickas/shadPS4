// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <Zydis/Zydis.h>
#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "common/alignment.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t NULL_INDEX_BUFFER_SIZE = 16;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;
static constexpr u64 STREAM_SHADOW_SIZE = 32_MB;
static constexpr u64 MinReadbackStagingSize = 1_MB;
static constexpr u64 MaxReadbackStagingSize = 32_MB;
constexpr std::optional<u32> CmaskColorExpandedValue(u32 num_samples) {
    switch (num_samples) {
    case 1:
        return 0xFFFFFFFF;
    case 2:
        return 0xDDDDDDDD;
    case 4:
        return 0xEEEEEEEE;
    case 8:
        return 0xFFFFFFFF;
    default:
        return 0;
    }
}
static_assert(CmaskColorExpandedValue(1) == 0xFFFFFFFF);
static_assert(CmaskColorExpandedValue(2) == 0xDDDDDDDD);
static_assert(CmaskColorExpandedValue(4) == 0xEEEEEEEE);
static_assert(CmaskColorExpandedValue(8) == 0xFFFFFFFF);

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

struct BufferCache::ReadbackContext {
    vk::UniqueCommandPool command_pool;
    vk::CommandBuffer cmdbuf;
    vk::UniqueFence fence;
    std::unique_ptr<Buffer> staging;
};

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      stream_shadow(STREAM_SHADOW_SIZE),
      emulated_chunks{std::make_unique<std::atomic<u64>[]>(NumEmulatedChunkWords)},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = SupportedUsage(instance, ARENA_USAGE),
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    if (!instance.IsMaintenance6Supported()) {
        null_index_buffer = std::make_unique<Buffer>(instance, 0, NULL_INDEX_BUFFER_SIZE,
                                                     MemoryType::DeviceLocal, "Null Index Buffer");
        runtime.FillBuffer(null_index_buffer.get(), 0u, NULL_INDEX_BUFFER_SIZE, 0u);
    }
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, false, assume_locks);
    });
}

void BufferCache::UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write) {
    std::scoped_lock lk{modified_mutex};
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, is_write);
    gpu_modified_ranges.Subtract(device_addr, size);
}

bool BufferCache::AccessBesideGpuData(VAddr device_addr, u32 size, bool is_write, u8* data) {
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = page_start + BYTES_PER_PAGE;
    ASSERT(device_addr + size <= page_end);
    std::scoped_lock lk{modified_mutex};
    if (!memory_tracker->IsRegionGpuModified(device_addr, size)) {
        // Not a fault on GPU-modified memory.
        return false;
    }
    if (gpu_modified_ranges.Intersects(device_addr, size)) {
        // The access needs the GPU data.
        return false;
    }
    if (OverlapsDownloadInFlight(page_start, page_end) ||
        OverlapsDeferredDownload(page_start, page_end)) {
        // GPU data of the page is being written back, its ranges are not tracked meanwhile.
        return false;
    }
    if (IsGpuWriteSubmitted(page_start, BYTES_PER_PAGE)) {
        // The readback of the page runs ahead of the open command buffer and unprotects the
        // page, which serves further accesses better than faulting on each one.
        return false;
    }
    auto& accesses = emulated_accesses[page_start / BYTES_PER_PAGE];
    if (const u64 tick = scheduler.CurrentTick(); accesses.tick != tick) {
        accesses = {tick, 0};
    }
    if (++accesses.count > MaxEmulatedAccesses) {
        // The page is accessed often, a readback unprotects it.
        accesses.count = 0;
        return false;
    }
    AmdGpu::readback_wait.emulated.fetch_add(1, std::memory_order_relaxed);
    if (is_write) {
        if (!memory->TryWriteBacking(std::bit_cast<u8*>(device_addr), data, size)) {
            return false;
        }
        emulated_writes.Add(device_addr, size);
        const u64 chunk = device_addr >> EmulatedChunkBits;
        emulated_chunks[chunk >> 6].fetch_or(1ULL << (chunk & 63), std::memory_order_relaxed);
    } else {
        memory->CopySparseMemory(device_addr, data, size);
    }
    return true;
}

void BufferCache::WriteAndUnmarkGpuModified(VAddr device_addr, const u8* data, u64 size) {
    std::unique_lock lk{modified_mutex};
    // A download in flight would write the older buffer cache data over this data.
    ReadbackStats stats{};
    WaitDownloadsInFlight(lk, device_addr, device_addr + size, stats);
    memory->TryWriteBacking(std::bit_cast<u8*>(device_addr), data, size);
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, true);
    gpu_modified_ranges.Subtract(device_addr, size);
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool async,
                             bool assume_locks) {
    ReadbackStats stats{};
    const u64 rb_t0 = AmdGpu::PerfNs();
    // Whether the faulting access itself touches GPU-modified bytes, or only shares the page.
    const FaultAccess access = current_fault_access;
    int touches_gpu = -1;
    if (access.valid && access.bytes != 0) {
        std::scoped_lock lk{modified_mutex};
        touches_gpu = gpu_modified_ranges.Intersects(device_addr, access.bytes) ? 1 : 0;
    }
    // 0: on the calling thread, 1: sent to the GPU thread, 2: already on the GPU thread.
    u32 path = 0;
    if (assume_locks) {
        path = 2;
        ReadMemoryOnGpuThread(device_addr, size, is_write, async, stats);
    } else if (async || !ReadMemoryOnCallingThread(device_addr, size, is_write, stats)) {
        path = 1;
        liverpool->SendCommand<true>([this, device_addr, size, is_write, async, &stats] {
            stats.exec_t0 = AmdGpu::PerfNs();
            ReadMemoryOnGpuThread(device_addr, size, is_write, async, stats);
        });
    }
    const u64 rb_t1 = AmdGpu::PerfNs();
    if (!assume_locks) {
        const u64 wait_ns = rb_t1 - rb_t0;
        auto& perf = AmdGpu::readback_wait;
        perf.count.fetch_add(1, std::memory_order_relaxed);
        perf.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
        u64 prev = perf.max_ns.load(std::memory_order_relaxed);
        while (prev < wait_ns &&
               !perf.max_ns.compare_exchange_weak(prev, wait_ns, std::memory_order_relaxed)) {
        }
    }
    LOG_INFO(Render,
             "PERF rb addr={:#x} size={:#x} write={} gpu_thread={} path={} t0={} t1={} "
             "exec_t0={} in_flight_us={} ahead={} batch_windows={} dl_bytes={} wait_tick={} "
             "gpu_tick={} gpu_us={} insn={} acc_bytes={} lock={} rep={} touches_gpu={} rip={:#x} "
             "thread={}",
             device_addr, size, is_write, assume_locks, path, rb_t0, rb_t1, stats.exec_t0,
             stats.in_flight_ns / 1000, stats.ahead, stats.windows, stats.bytes, stats.wait_tick,
             stats.gpu_tick, stats.gpu_ns / 1000,
             access.mnemonic != 0
                 ? ZydisMnemonicGetString(static_cast<ZydisMnemonic>(access.mnemonic))
                 : "none",
             access.bytes, access.lock, access.rep, touches_gpu, access.rip,
             Common::GetCurrentThreadName());
}

bool BufferCache::ReadMemoryOnCallingThread(VAddr device_addr, u64 size, bool is_write,
                                            ReadbackStats& stats) {
    std::unique_lock lk{modified_mutex};
    DownloadWindow window;
    do {
        // Arenas are created and merged by the GPU thread, look again after waiting.
        const Buffer* arena = address_space[device_addr >> ARENA_PAGE_BITS];
        if (arena != address_space[(device_addr + size - 1) >> ARENA_PAGE_BITS]) {
            // The request spans two arenas, only the GPU thread can merge them.
            return false;
        }
        if (!arena) {
            // The GPU never wrote this memory through the buffer cache.
            FinishReadback(lk, device_addr, size, is_write, stats);
            return true;
        }
        window = GetDownloadWindow(arena, device_addr, size);
    } while (WaitDownloadsInFlight(lk, window.start, window.start + window.size, stats));
    if (IsGpuWriteSubmitted(window.start, window.size)) {
        hot_windows[window.start] = {window.size, scheduler.CurrentTick()};
    } else {
        // A GPU write to the window is still in the open command buffer. The faulting pages
        // alone can still be downloaded ahead when their own GPU writes were all submitted.
        const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
        const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
        if (page_end - page_start > MaxPageTickRange ||
            !IsGpuWriteSubmitted(page_start, page_end - page_start)) {
            return false;
        }
        window = {window.arena, page_start, page_end - page_start};
    }
    boost::container::small_vector<DownloadWindow, 16> windows{window};
    AddHotWindows(windows);
    stats.ahead = true;
    DownloadAhead(lk, windows, stats);
    FinishReadback(lk, device_addr, size, is_write, stats);
    return true;
}

void BufferCache::ReadMemoryOnGpuThread(VAddr device_addr, u64 size, bool is_write, bool async,
                                        ReadbackStats& stats) {
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    // May create or merge arenas, which takes modified_mutex.
    const Buffer* arena = GetArena(first_block, last_block);
    std::unique_lock lk{modified_mutex};
    const DownloadWindow window = GetDownloadWindow(arena, device_addr, size);
    // Only the GPU thread changes arenas, so the window stays the same while waiting.
    WaitDownloadsInFlight(lk, window.start, window.start + window.size, stats);
    // When the GPU writes were all submitted, the download can run ahead of the open command
    // buffer instead of flushing and draining it.
    const bool ahead = !async && IsGpuWriteSubmitted(window.start, window.size);
    if (!async) {
        hot_windows[window.start] = {window.size, scheduler.CurrentTick()};
    }
    if (ahead) {
        boost::container::small_vector<DownloadWindow, 16> windows{window};
        AddHotWindows(windows);
        stats.ahead = true;
        DownloadAhead(lk, windows, stats);
    } else {
        DownloadMemory(lk, arena, window.start, window.size, async, stats);
    }
    if (stats.bytes != 0) {
        ++AmdGpu::cp_perf.rb_downloads;
        AmdGpu::cp_perf.rb_bytes += stats.bytes;
    }
    FinishReadback(lk, device_addr, size, is_write, stats);
}

BufferCache::DownloadWindow BufferCache::GetDownloadWindow(const Buffer* arena, VAddr device_addr,
                                                           u64 size) const {
    // GPU-modified ranges come as many small scattered islands,
    // so the download is widened to a window around the request
    constexpr u64 WindowSize = 512_KB;
    const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
    const VAddr window_end =
        std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
    return {arena, window_start, window_end - window_start};
}

void BufferCache::AddHotWindows(boost::container::small_vector<DownloadWindow, 16>& windows) {
    // Download the other recently read-back windows in the same submission, but only those whose
    // GPU writes are done no later than the ones of the requested window, so the download does
    // not wait longer than the request needs.
    const u64 current_tick = scheduler.CurrentTick();
    const VAddr main_start = windows.front().start;
    auto* const work_semaphore = scheduler.GetWorkSemaphore();
    work_semaphore->Refresh();
    const u64 ready_tick =
        std::max(LastWriteTick(main_start, windows.front().size), work_semaphore->KnownGpuTick());
    for (auto it = hot_windows.begin(); it != hot_windows.end();) {
        auto& [window_size, tick] = it->second;
        if (current_tick - tick > HotWindowTicks) {
            it = hot_windows.erase(it);
            continue;
        }
        const VAddr start = it->first;
        const Buffer* arena = address_space[start >> ARENA_PAGE_BITS];
        if (start != main_start && arena &&
            arena == address_space[(start + window_size - 1) >> ARENA_PAGE_BITS] &&
            !OverlapsDownloadInFlight(start, start + window_size) &&
            memory_tracker->IsRegionGpuModified(start, window_size) &&
            LastWriteTick(start, window_size) <= ready_tick &&
            IsGpuWriteSubmitted(start, window_size)) {
            windows.push_back({arena, start, window_size});
            tick = current_tick;
        }
        ++it;
    }
}

bool BufferCache::IsGpuWriteSubmitted(VAddr device_addr, u64 size) const {
    return LastWriteTick(device_addr, size) < scheduler.CurrentTick();
}

u64 BufferCache::LastWriteTick(VAddr device_addr, u64 size) const {
    u64 last_tick = 0;
    if (size <= MaxPageTickRange) {
        for (u64 page = device_addr / BYTES_PER_PAGE;
             page <= (device_addr + size - 1) / BYTES_PER_PAGE; ++page) {
            if (const auto it = page_write_ticks.find(page); it != page_write_ticks.end()) {
                last_tick = std::max(last_tick, it->second);
            }
        }
        return last_tick;
    }
    for (u64 chunk = device_addr >> WriteTickChunkBits;
         chunk <= (device_addr + size - 1) >> WriteTickChunkBits; ++chunk) {
        if (const auto it = write_ticks.find(chunk); it != write_ticks.end()) {
            last_tick = std::max(last_tick, it->second);
        }
    }
    return last_tick;
}

bool BufferCache::OverlapsDownloadInFlight(VAddr start, VAddr end) const {
    return std::ranges::any_of(downloads_in_flight, [start, end](const DownloadInFlight& d) {
        return d.start < end && start < d.end;
    });
}

bool BufferCache::OverlapsDeferredDownload(VAddr start, VAddr end) const {
    return std::ranges::any_of(deferred_downloads, [start, end](const DownloadInFlight& d) {
        return d.start < end && start < d.end;
    });
}

void BufferCache::EraseDeferredDownload(u64 download_id) {
    std::erase_if(deferred_downloads,
                  [download_id](const DownloadInFlight& d) { return d.id == download_id; });
}

bool BufferCache::WaitDownloadsInFlight(std::unique_lock<std::mutex>& lk, VAddr start, VAddr end,
                                        ReadbackStats& stats) {
    const auto overlaps = [this, start, end, self = std::this_thread::get_id()] {
        return std::ranges::any_of(downloads_in_flight, [&](const DownloadInFlight& d) {
            return d.owner != self && d.start < end && start < d.end;
        });
    };
    if (!overlaps()) {
        return false;
    }
    const u64 wait_t0 = AmdGpu::PerfNs();
    downloads_cv.wait(lk, [&] { return !overlaps(); });
    stats.in_flight_ns += AmdGpu::PerfNs() - wait_t0;
    return true;
}

void BufferCache::UnmarkDownloadedWindow(VAddr start, u64 size, u64 download_id) {
    const VAddr end = start + size;
    boost::container::small_vector<std::pair<VAddr, VAddr>, 16> keep;
    gpu_modified_ranges.ForEachInRange(start, size, [&](VAddr range_start, VAddr range_end) {
        keep.emplace_back(Common::AlignDown(range_start, BYTES_PER_PAGE),
                          Common::AlignUp(range_end, BYTES_PER_PAGE));
    });
    for (const auto* downloads : {&downloads_in_flight, &deferred_downloads}) {
        for (const auto& download : *downloads) {
            if (download.id != download_id && download.start < end && start < download.end) {
                keep.emplace_back(
                    Common::AlignDown(std::max(download.start, start), BYTES_PER_PAGE),
                    Common::AlignUp(std::min(download.end, end), BYTES_PER_PAGE));
            }
        }
    }
    std::ranges::sort(keep);
    VAddr cursor = start;
    for (const auto& [keep_start, keep_end] : keep) {
        if (keep_start > cursor) {
            memory_tracker->UnmarkRegionAsGpuModified(cursor, keep_start - cursor, false);
        }
        cursor = std::max(cursor, keep_end);
    }
    if (cursor < end) {
        memory_tracker->UnmarkRegionAsGpuModified(cursor, end - cursor, false);
    }
}

void BufferCache::FinishReadback(std::unique_lock<std::mutex>& lk, VAddr device_addr, u64 size,
                                 bool is_write, ReadbackStats& stats) {
    const VAddr page_start = Common::AlignDown(device_addr, BYTES_PER_PAGE);
    const VAddr page_end = Common::AlignUp(device_addr + size, BYTES_PER_PAGE);
    // Another download in flight may still write data back to the pages.
    WaitDownloadsInFlight(lk, page_start, page_end, stats);
    if (memory_tracker->IsRegionGpuModified(device_addr, size)) {
        if (gpu_modified_ranges.Intersects(page_start, page_end - page_start)) {
            // A GPU write recorded while the download was in flight has to be read back first,
            // the access faults again for it.
            return;
        }
        if (OverlapsDeferredDownload(page_start, page_end)) {
            // The data is written back by a deferred operation, the access faults again until
            // it ran.
            return;
        }
        // The pages are marked GPU-modified but hold no GPU data, so nothing was downloaded for
        // them and no download unmarked them. Unmark them like a downloaded window: a page
        // marked CPU- and GPU-modified at once would be protected write-only.
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size, false);
    }
    if (is_write) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    }
}

void BufferCache::DownloadMemory(std::unique_lock<std::mutex>& lk, const Buffer* arena,
                                 VAddr device_addr, u64 size, bool async, ReadbackStats& stats) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    stats.bytes += total_size_bytes;
    stats.windows = 1;
    // An async download is written back by a deferred operation of the GPU thread, which a
    // waiter must not block on, so it is tracked as deferred instead of in flight.
    const u64 download_id = ++next_download_id;
    auto& downloads = async ? deferred_downloads : downloads_in_flight;
    downloads.push_back({download_id, device_addr, device_addr + size, std::this_thread::get_id()});
    lk.unlock();

    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 16, async);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);

    const auto write_data = [this, arena_base, device_addr, size, download, async, download_id,
                             copies = std::move(copies)] {
        download.Invalidate();
        for (const auto& copy : copies) {
            auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
            memory->TryWriteBacking(dst_addr, download.mapped + (copy.dstOffset - download.offset),
                                    copy.size);
        }
        {
            std::scoped_lock modified_lk{modified_mutex};
            UnmarkDownloadedWindow(device_addr, size, download_id);
            if (async) {
                EraseDeferredDownload(download_id);
            } else {
                std::erase_if(downloads_in_flight, [download_id](const DownloadInFlight& d) {
                    return d.id == download_id;
                });
            }
        }
        if (async) {
            staging_pool.FreeDeferred(download);
        } else {
            downloads_cv.notify_all();
        }
    };

    if (async) {
        scheduler.DeferOperation(std::move(write_data));
    } else {
        {
            const AmdGpu::CpPerfScope perf_finish{AmdGpu::cp_perf.rb_finish_ns};
            scheduler.Finish();
        }
        write_data();
    }
    lk.lock();
}

void BufferCache::DownloadAhead(std::unique_lock<std::mutex>& lk,
                                std::span<const DownloadWindow> windows, ReadbackStats& stats) {
    struct WindowCopies {
        boost::container::small_vector<vk::BufferCopy, 8> copies;
    };
    boost::container::small_vector<WindowCopies, 16> window_copies(windows.size());
    u64 total_size_bytes = 0;
    for (size_t i = 0; i < windows.size(); ++i) {
        const auto& window = windows[i];
        auto& copies = window_copies[i].copies;
        const VAddr arena_base = window.arena->cpu_addr;
        memory_tracker->ForEachDownloadRange<false>(
            window.start, window.size, [&](u64 address, u64 size) {
                const auto add_download = [&](VAddr start, VAddr end) {
                    copies.push_back(vk::BufferCopy{
                        .srcOffset = start - arena_base,
                        .dstOffset = total_size_bytes,
                        .size = end - start,
                    });
                    // Align up to avoid cache conflicts
                    constexpr u64 align = 64ULL;
                    total_size_bytes += Common::AlignUp(end - start, align);
                };
                gpu_modified_ranges.ForEachInRange(address, size, add_download);
                gpu_modified_ranges.Subtract(address, size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    const u64 download_id = ++next_download_id;
    u64 wait_tick = 0;
    for (size_t i = 0; i < windows.size(); ++i) {
        if (!window_copies[i].copies.empty()) {
            wait_tick = std::max(wait_tick, LastWriteTick(windows[i].start, windows[i].size));
            downloads_in_flight.push_back({download_id, windows[i].start,
                                           windows[i].start + windows[i].size,
                                           std::this_thread::get_id()});
            ++stats.windows;
        }
    }
    stats.bytes += total_size_bytes;
    stats.wait_tick = wait_tick;
    lk.unlock();

    ReadbackContext* context = AcquireReadbackContext(total_size_bytes);
    const vk::CommandBuffer cmdbuf = context->cmdbuf;
    const vk::Device device = instance.GetDevice();
    Vulkan::Check<"reset readback command pool">(device.resetCommandPool(*context->command_pool));
    Vulkan::Check<"begin readback command buffer">(cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    // Without a readback queue, the barrier's first scope holds every command submitted to the
    // graphics queue before this one, which includes all GPU writes to the windows. With one,
    // the wait for wait_tick orders the copies after them.
    const vk::MemoryBarrier2 gpu_writes = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &gpu_writes,
    });
    for (size_t i = 0; i < windows.size(); ++i) {
        const auto& copies = window_copies[i].copies;
        if (!copies.empty()) {
            cmdbuf.copyBuffer(windows[i].arena->Handle(), context->staging->Handle(),
                              std::span<const vk::BufferCopy>{copies.data(), copies.size()});
        }
    }
    const vk::MemoryBarrier2 host_read = {
        .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &host_read,
    });
    Vulkan::Check<"end readback command buffer">(cmdbuf.end());
    auto* const work_semaphore = scheduler.GetWorkSemaphore();
    work_semaphore->Refresh();
    stats.gpu_tick = work_semaphore->KnownGpuTick();
    const u64 gpu_t0 = AmdGpu::PerfNs();
    scheduler.SubmitOutOfBand(cmdbuf, *context->fence, wait_tick);
    Vulkan::Check<"wait for readback">(
        device.waitForFences(*context->fence, true, std::numeric_limits<u64>::max()));
    stats.gpu_ns = AmdGpu::PerfNs() - gpu_t0;
    Vulkan::Check<"reset readback fence">(device.resetFences(*context->fence));

    context->staging->Invalidate(0, total_size_bytes);
    const u8* staging_data = context->staging->mapped_data.data();
    for (size_t i = 0; i < windows.size(); ++i) {
        const VAddr arena_base = windows[i].arena->cpu_addr;
        for (const auto& copy : window_copies[i].copies) {
            memory->TryWriteBacking(std::bit_cast<u8*>(arena_base + copy.srcOffset),
                                    staging_data + copy.dstOffset, copy.size);
        }
    }
    if (context->staging->SizeBytes() > MaxReadbackStagingSize) {
        // Only staging buffers of the usual download size are kept.
        context->staging.reset();
    }
    ReleaseReadbackContext(context);

    lk.lock();
    for (size_t i = 0; i < windows.size(); ++i) {
        if (!window_copies[i].copies.empty()) {
            UnmarkDownloadedWindow(windows[i].start, windows[i].size, download_id);
        }
    }
    std::erase_if(downloads_in_flight,
                  [download_id](const DownloadInFlight& d) { return d.id == download_id; });
    downloads_cv.notify_all();
}

BufferCache::ReadbackContext* BufferCache::AcquireReadbackContext(u64 staging_size) {
    ReadbackContext* context = nullptr;
    {
        std::scoped_lock lk{readback_contexts_mutex};
        if (!free_readback_contexts.empty()) {
            context = free_readback_contexts.back();
            free_readback_contexts.pop_back();
        } else {
            context = readback_contexts.emplace_back(std::make_unique<ReadbackContext>()).get();
        }
    }
    const vk::Device device = instance.GetDevice();
    if (!context->command_pool) {
        context->command_pool =
            Vulkan::Check<"create readback command pool">(device.createCommandPoolUnique({
                .flags = vk::CommandPoolCreateFlagBits::eTransient,
                .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
            }));
        context->cmdbuf = Vulkan::Check<"allocate readback command buffer">(
                              device.allocateCommandBuffers({
                                  .commandPool = *context->command_pool,
                                  .level = vk::CommandBufferLevel::ePrimary,
                                  .commandBufferCount = 1,
                              }))
                              .front();
        context->fence = Vulkan::Check<"create readback fence">(device.createFenceUnique({}));
    }
    if (!context->staging || context->staging->SizeBytes() < staging_size) {
        // The previous staging buffer is idle, its last download was waited for.
        context->staging = std::make_unique<Buffer>(
            instance, 0, std::bit_ceil(std::max(staging_size, MinReadbackStagingSize)),
            MemoryType::HostCached, "Readback Staging Buffer");
    }
    return context;
}

void BufferCache::ReleaseReadbackContext(ReadbackContext* context) {
    std::scoped_lock lk{readback_contexts_mutex};
    free_readback_contexts.push_back(context);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        ObtainBufferFlags flags) {
    const bool is_written = True(flags & ObtainBufferFlags::IsWritten);
    const bool is_texel_buffer = True(flags & ObtainBufferFlags::IsTexelBuffer);
    const bool skip_stream_buffer = True(flags & ObtainBufferFlags::IgnoreStreamBuffer);
    ++AmdGpu::cp_perf.obtains;
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && !skip_stream_buffer && size <= STREAM_THRESHOLD &&
        !IsRegionGpuModified(device_addr, size)) {
        ++AmdGpu::cp_perf.obtain_stream;
        AmdGpu::cp_perf.obtain_stream_bytes += size;
        const AmdGpu::VkCallPerf stream_perf{AmdGpu::cp_perf.sc_total_ns};
        auto& copy = stream_copies[StreamCopySlot(device_addr, size)];
        if (copy.tick == scheduler.CurrentTick() && copy.device_addr == device_addr &&
            copy.size == size) {
            const AmdGpu::VkCallPerf compare_perf{AmdGpu::cp_perf.sc_compare_ns};
            if (memory->CompareSparseMemory(device_addr, stream_shadow.data() + copy.shadow_offset,
                                            size)) {
                ++AmdGpu::cp_perf.obtain_stream_reused;
                AmdGpu::cp_perf.obtain_stream_reused_bytes += size;
                return {&stream_buffer, copy.stream_offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        // Mapping can wait for the GPU and submit, which starts a new tick.
        const u64 tick = scheduler.CurrentTick();
        if (stream_shadow_tick != tick) {
            stream_shadow_tick = tick;
            stream_shadow_used = 0;
        }
        if (stream_shadow_used + size <= stream_shadow.size()) {
            // The stream buffer gets the very bytes that are kept to compare with.
            u8* const shadow = stream_shadow.data() + stream_shadow_used;
            {
                const AmdGpu::VkCallPerf guest_perf{AmdGpu::cp_perf.sc_guest_ns};
                memory->CopySparseMemory(device_addr, shadow, size);
            }
            const AmdGpu::VkCallPerf write_perf{AmdGpu::cp_perf.sc_write_ns};
            if (stream_buffer.is_coherent) {
                // The shadow keeps these bytes until the tick ends, and the commands that read
                // the stream buffer are recorded after this one.
                scheduler.Record([data = data, shadow, size,
                                  is_timed = AmdGpu::perf_vk_sample](vk::CommandBuffer) {
                    const u64 write_t0 = is_timed ? AmdGpu::PerfNs() : 0;
                    std::memcpy(data, shadow, size);
                    if (is_timed) {
                        AmdGpu::recorder_perf.stream_write_ns.fetch_add(
                            (AmdGpu::PerfNs() - write_t0) * AmdGpu::PerfVkSampleRate,
                            std::memory_order_relaxed);
                    }
                });
            } else {
                std::memcpy(data, shadow, size);
            }
            copy = {device_addr, size, tick, offset, stream_shadow_used};
            stream_shadow_used += size;
        } else {
            const AmdGpu::VkCallPerf guest_perf{AmdGpu::cp_perf.sc_guest_ns};
            memory->CopySparseMemory(device_addr, data, size);
        }
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    if (is_written) {
        ++AmdGpu::cp_perf.obtain_written;
        texture_cache.MarkAsMaybeGpuDirty(device_addr, size);
    }
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::MarkRegionAsGpuModified(VAddr addr, size_t size) {
    texture_cache.MarkAsMaybeGpuDirty(addr, size);
    std::scoped_lock lk{modified_mutex};
    memory_tracker->MarkRegionAsGpuModified(addr, size);
    gpu_modified_ranges.Add(addr, size);
}

void BufferCache::ReadEdgeImagePages(const Image& image) {
    // May happen that after downloading the image and invalidating region,
    // that there were GPU modified ranges that are lost due to CPU reuploading.
    // This doesn't change tracker state and it is spected to call DownloadImageMemory after this.
    const VAddr image_addr = image.info.guest_address;
    const u64 image_size = image.info.guest_size;
    const VAddr image_end = image_addr + image_size;
    const VAddr page_start = PageManager::GetPageAddr(image_addr);
    const VAddr page_end = PageManager::GetNextPageAddr(image_end - 1);
    boost::container::small_vector<vk::BufferCopy, 2> copies;
    u64 total_size_bytes = 0;
    const auto [buffer, offset] = ObtainBuffer(page_start, page_end - page_start);
    const auto add_download = [&](VAddr start, VAddr end) {
        const u64 new_offset = start - buffer->CpuAddr();
        const u64 new_size = end - start;
        copies.push_back(vk::BufferCopy{
            .srcOffset = new_offset,
            .dstOffset = total_size_bytes,
            .size = new_size,
        });
        // Align up to avoid cache conflicts
        constexpr u64 align = 64ULL;
        constexpr u64 mask = ~(align - 1ULL);
        total_size_bytes += (new_size + align - 1) & mask;
    };
    u64 download_id = 0;
    {
        std::scoped_lock lk{modified_mutex};
        gpu_modified_ranges.ForEachInRange(page_start, image_addr - page_start, add_download);
        gpu_modified_ranges.ForEachInRange(image_end, page_end - image_end, add_download);
        gpu_modified_ranges.Subtract(page_start, page_end - page_start);
        if (total_size_bytes == 0) {
            return;
        }
        // The pages keep their GPU-modified state until the edge data is written back.
        download_id = ++next_download_id;
        deferred_downloads.push_back(
            {download_id, page_start, page_end, std::this_thread::get_id()});
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 16, true);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += download.offset;
    }
    scheduler.EndRendering();
    runtime.CopyBuffer(buffer, download.buffer, copies);
    scheduler.DeferOperation(
        [this, base = buffer->CpuAddr(), download, download_id, copies = std::move(copies)]() {
            download.Invalidate();
            auto* memory = Core::Memory::Instance();
            for (const auto& copy : copies) {
                const VAddr copy_device_addr = base + copy.srcOffset;
                const u64 dst_offset = copy.dstOffset - download.offset;
                memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr),
                                        download.mapped + dst_offset, copy.size);
            }
            {
                std::scoped_lock lk{modified_mutex};
                EraseDeferredDownload(download_id);
            }
            staging_pool.FreeDeferred(download);
        });
}

void BufferCache::SynchronizeDmaBuffers() {
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            // Readbacks on guest threads look the arenas up.
            std::scoped_lock lk{modified_mutex};
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + start - backing.start) << block_shift,
        });
    });

    // Readbacks on guest threads look the arenas up.
    std::scoped_lock lk{modified_mutex};
    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }

    const bool has_bda = instance.IsBufferDeviceAddressSupported();
    const vk::MemoryAllocateFlagsInfo alloc_flags = {
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::MemoryAllocateInfo alloc_info = {
        .pNext = has_bda ? &alloc_flags : nullptr,
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset >> block_shift;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        if (!has_bda) {
            continue;
        }
        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    if (has_bda) {
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    }
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    {
        // A written range becomes GPU-modified, which readbacks on guest threads look at.
        const bool emulated_pending = HasEmulatedWrites(device_addr, size);
        std::unique_lock lk{modified_mutex, std::defer_lock};
        if (is_written || emulated_pending) {
            lk.lock();
        }
        memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
            copies.emplace_back(total_size_bytes, addr, size);
            total_size_bytes += size;
        });
        if (emulated_pending) {
            // Upload the CPU writes done beside GPU data, unless their page is uploaded anyway.
            const size_t num_page_copies = copies.size();
            emulated_writes.ForEachInRange(device_addr, size, [&](VAddr start, VAddr end) {
                const auto in_page_copy = [&](const vk::BufferCopy& copy) {
                    return copy.dstOffset <= start && end <= copy.dstOffset + copy.size;
                };
                if (std::none_of(copies.begin(), copies.begin() + num_page_copies, in_page_copy)) {
                    copies.emplace_back(total_size_bytes, start, end - start);
                    total_size_bytes += end - start;
                }
            });
            emulated_writes.Subtract(device_addr, size);
            for (u64 chunk = device_addr >> EmulatedChunkBits;
                 chunk <= (device_addr + size - 1) >> EmulatedChunkBits; ++chunk) {
                if (!emulated_writes.Intersects(chunk << EmulatedChunkBits,
                                                1ULL << EmulatedChunkBits)) {
                    emulated_chunks[chunk >> 6].fetch_and(~(1ULL << (chunk & 63)),
                                                          std::memory_order_relaxed);
                }
            }
        }
        if (is_written) {
            gpu_modified_ranges.Add(device_addr, size);
            const u64 tick = scheduler.CurrentTick();
            for (u64 chunk = device_addr >> WriteTickChunkBits;
                 chunk <= (device_addr + size - 1) >> WriteTickChunkBits; ++chunk) {
                write_ticks[chunk] = tick;
            }
            for (u64 page = device_addr / BYTES_PER_PAGE;
                 page <= (device_addr + size - 1) / BYTES_PER_PAGE; ++page) {
                page_write_ticks[page] = tick;
            }
        }
    }
    if (!copies.empty()) {
        ++AmdGpu::cp_perf.obtain_uploads;
        AmdGpu::cp_perf.obtain_upload_bytes += total_size_bytes;
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        ++AmdGpu::cp_perf.obtain_texel;
        const bool from_image = SynchronizeMemoryFromImage(arena, device_addr, size);
        AmdGpu::cp_perf.obtain_texel_image += from_image ? 1 : 0;
        return from_image;
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (const auto meta = texture_cache.GetMetaInfo(device_addr)) {
        if (meta->type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else if (meta->type == TextureCache::MetaType::CMask) {
            const auto expanded_value = CmaskColorExpandedValue(meta->num_samples);
            if (!expanded_value) {
                LOG_WARNING(Render_Vulkan,
                            "Unsupported CMask sample count: address={:#x}, size={:#x}, "
                            "samples={}, owner={}:{}",
                            device_addr, size, meta->num_samples, meta->owner_id.index,
                            meta->owner_uid);
                return false;
            }
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, *expanded_value);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}",
                        magic_enum::enum_name(meta->type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
