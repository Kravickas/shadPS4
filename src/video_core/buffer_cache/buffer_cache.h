// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <boost/container/small_vector.hpp>

#include "common/enum.h"
#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"
#include "video_core/texture_cache/image.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
struct StagingBufferRef;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class PageManager;

enum class ObtainBufferFlags {
    None = 0,
    IsWritten = 1 << 0,
    IsTexelBuffer = 1 << 1,
    IgnoreStreamBuffer = 1 << 2,
    InvalidateTextureCache = 1 << 3,
};
DECLARE_ENUM_FLAG_OPERATORS(ObtainBufferFlags)

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

public:
    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    [[nodiscard]] const Buffer* GetNullIndexBuffer() const noexcept {
        return null_index_buffer.get();
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    [[nodiscard]] StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    void TickFrame();

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Unmark a region as modified from the GPU.
    void UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write = false);

    /// Writes data that replaces the buffer cache contents of a region to guest memory and
    /// unmarks the region as modified from the GPU, after downloads in flight to it are done.
    void WriteAndUnmarkGpuModified(VAddr device_addr, const u8* data, u64 size);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool async = false,
                    bool assume_locks = false);

    /// Flushes GPU modified ranges of the uncovered part of the edge pages of an image.
    void ReadEdgeImagePages(const Image& image);

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(
        VAddr device_addr, u32 size, ObtainBufferFlags flags = ObtainBufferFlags::None);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Mark region as modified from the GPU
    void MarkRegionAsGpuModified(VAddr addr, size_t size);

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    struct DownloadWindow {
        const Buffer* arena;
        VAddr start;
        u64 size;
    };
    struct ReadbackStats {
        u64 exec_t0;      ///< When the GPU thread started the readback, 0 if it did not run it
        u64 in_flight_ns; ///< Time spent waiting for overlapping downloads in flight
        u64 bytes;
        u32 windows;
        bool ahead;
    };
    struct ReadbackContext;

    /// Runs a readback on the calling guest thread. Returns false when it needs the GPU thread,
    /// because a GPU write to the window is still in the open command buffer.
    bool ReadMemoryOnCallingThread(VAddr device_addr, u64 size, bool is_write,
                                   ReadbackStats& stats);

    /// Runs a readback on the GPU thread.
    void ReadMemoryOnGpuThread(VAddr device_addr, u64 size, bool is_write, bool async,
                               ReadbackStats& stats);

    /// The download window around a readback request in its arena.
    [[nodiscard]] DownloadWindow GetDownloadWindow(const Buffer* arena, VAddr device_addr,
                                                   u64 size) const;

    /// Adds the recently read-back windows that are ready to the download. Lock held.
    void AddHotWindows(boost::container::small_vector<DownloadWindow, 16>& windows);

    /// Downloads the GPU-modified ranges of the windows with a submission of its own, queued
    /// after all submitted work. Every GPU write to them must have been submitted already.
    /// Called with modified_mutex held through lk, released while the GPU copies.
    void DownloadAhead(std::unique_lock<std::mutex>& lk, std::span<const DownloadWindow> windows,
                       ReadbackStats& stats);

    /// Downloads the GPU-modified ranges of the window in the open command buffer. GPU thread
    /// only. Called with modified_mutex held through lk, released while the GPU copies.
    void DownloadMemory(std::unique_lock<std::mutex>& lk, const Buffer* arena, VAddr device_addr,
                        u64 size, bool async, ReadbackStats& stats);

    /// Whether every GPU write to the range was submitted before the open command buffer.
    /// Lock held.
    [[nodiscard]] bool IsGpuWriteSubmitted(VAddr device_addr, u64 size) const;

    /// Whether a download in flight overlaps the range. Lock held.
    [[nodiscard]] bool OverlapsDownloadInFlight(VAddr start, VAddr end) const;

    /// Waits until no download in flight of another thread overlaps the range. A download of
    /// the calling thread is not waited for: the GPU thread runs deferred operations inside its
    /// own download, which keeps the order of the GPU thread. Returns true if it had to wait,
    /// which released the lock meanwhile. Lock held.
    bool WaitDownloadsInFlight(std::unique_lock<std::mutex>& lk, VAddr start, VAddr end,
                               ReadbackStats& stats);

    /// Clears the GPU-modified state of the pages of a downloaded window, except pages that
    /// received GPU writes again while the download was in flight and pages another download in
    /// flight covers. Lock held.
    void UnmarkDownloadedWindow(VAddr start, u64 size, u64 download_id);

    /// Ends a readback: a write fault may proceed unless the range got GPU writes again. Lock
    /// held.
    void FinishReadback(VAddr device_addr, u64 size, bool is_write);

    ReadbackContext* AcquireReadbackContext(u64 staging_size);
    void ReleaseReadbackContext(ReadbackContext* context);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    /// Guards the GPU-modified state against readbacks running on guest threads: the tracker's
    /// GPU bits, gpu_modified_ranges, write_ticks, hot_windows, downloads_in_flight and the
    /// address_space entries. The GPU thread changes them with it held and other threads read
    /// them with it held. It is never held across a scheduler call.
    std::mutex modified_mutex;
    RangeSet gpu_modified_ranges;
    /// Scheduler tick of the last writable binding per WriteTickChunk bytes of guest memory.
    std::unordered_map<u64, u64> write_ticks;
    static constexpr u64 WriteTickChunkBits = 16;
    /// Readback windows that held GPU-modified data recently: start -> {size, scheduler tick}.
    /// A readback downloads all of them that are ready, so the CPU does not fault on each one.
    std::unordered_map<VAddr, std::pair<u64, u64>> hot_windows;
    static constexpr u64 HotWindowTicks = 4;
    /// Windows whose GPU-modified ranges were taken out of gpu_modified_ranges and whose data
    /// is not written back yet.
    struct DownloadInFlight {
        u64 id;
        VAddr start;
        VAddr end;
        std::thread::id owner;
    };
    std::vector<DownloadInFlight> downloads_in_flight;
    u64 next_download_id{};
    std::condition_variable downloads_cv;
    /// Command buffers, fences and staging memory for downloads ahead of the open command buffer.
    std::mutex readback_contexts_mutex;
    std::vector<std::unique_ptr<ReadbackContext>> readback_contexts;
    std::vector<ReadbackContext*> free_readback_contexts;

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    bool fault_process_pending{};

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
