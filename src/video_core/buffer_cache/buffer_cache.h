// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
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

inline constexpr std::size_t CacheLineSize = 64;

/// Tells the processor that the cache line of address is read soon.
inline void PrefetchRead(const void* address) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address);
#endif
}

/// The same for every cache line with a byte of the range.
inline void PrefetchRead(const void* address, std::size_t size) {
    if (size == 0) {
        return;
    }
    const auto* const bytes = static_cast<const u8*>(address);
    for (std::size_t offset = 0; offset < size; offset += CacheLineSize) {
        PrefetchRead(bytes + offset);
    }
    PrefetchRead(bytes + size - 1);
}

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

    /// Called at every submit of the GPU thread's command buffer, with the tick advanced.
    /// TickFrame is only called once per frame.
    void OnSubmit();

    /// Whether the open command buffer holds a GPU write to memory the readback thread
    /// downloads for the game's threads.
    [[nodiscard]] bool IsEagerWritePending() const {
        return eager_write_pending.load(std::memory_order_relaxed);
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// Unmark a region as modified from the GPU.
    void UnmarkRegionAsGpuModified(VAddr device_addr, u64 size, bool is_write = false);

    /// Performs a CPU access that faulted on a GPU-modified page directly on guest memory, when
    /// the accessed bytes hold no GPU data: the page only shares them with GPU-modified bytes,
    /// so the access needs nothing from the GPU. A write is uploaded when the GPU next uses the
    /// bytes. Returns false if the access has to go through a readback. The access must lie in
    /// one page.
    bool AccessBesideGpuData(VAddr device_addr, u32 size, bool is_write, u8* data);

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

    /// Bytes of a read-only binding, read from guest memory ahead of the draw that binds it.
    struct StreamHint {
        VAddr device_addr;
        u32 size;
        u64 id; ///< Names the bytes: hints with the same id hold the same bytes
        const u8* data;
    };

    enum class StreamAheadResult { None, Known, Same, Copied, Full };

    /// For the lookahead thread: reads the bytes of a binding the stream buffer could serve.
    /// The bytes stay where the hint points until the hints of a later draw are released.
    /// run names the run of draws the binding belongs to, gpu_write_epoch is the count of GPU
    /// writes to guest memory from before the draw's reads. A range already read in the same
    /// run with no GPU write to guest memory since is not read again.
    StreamAheadResult ReadStreamAhead(VAddr device_addr, u64 size, u64 run, u64 gpu_write_epoch,
                                      StreamHint& hint);

    /// For the lookahead thread: where its next bytes go. Hints made after this was read point
    /// to bytes not more than StreamAheadWindow behind it.
    u64 StreamAheadPosition() const noexcept {
        return stream_ahead_position;
    }

    /// Lets the lookahead thread reuse the space of bytes no hint made at position or later
    /// can point to. Nothing recorded for the GPU may still read hints made before position.
    void ReleaseStreamAhead(u64 position);

    /// Hints for the ObtainBuffer calls of the draw in progress, made when the count of GPU
    /// writes to guest memory was gpu_write_epoch.
    void SetStreamHints(std::span<const StreamHint> hints, u64 gpu_write_epoch) {
        stream_hints = hints;
        stream_hint_cursor = 0;
        stream_hints_epoch = gpu_write_epoch;
    }

    /// Asks for the entries the bindings of the hints look their last copy up in.
    void PrefetchStreamCopies(std::span<const StreamHint> hints) const {
        for (const StreamHint& hint : hints) {
            PrefetchRead(&stream_copies[StreamCopySlot(hint.device_addr, hint.size)],
                         sizeof(StreamCopy));
        }
    }

    void ClearStreamHints() {
        stream_hints = {};
    }

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
        u64 wait_tick; ///< Tick of the last GPU write to the downloaded windows
        u64 gpu_tick;  ///< Tick the GPU had reached when the download was submitted
        u64 gpu_ns;    ///< Download submit to fence signalled
        u64 submit_ns; ///< Of it, the submit call with its wait for the submit mutex
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

    /// Adds the recently read-back windows whose GPU writes are done no later than those of the
    /// requested window to the download. Lock held.
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

    /// Scheduler tick of the last writable binding of the range, 0 if there was none. Exact to
    /// the page for ranges up to MaxPageTickRange, to WriteTickChunk bytes above. Lock held.
    [[nodiscard]] u64 LastWriteTick(VAddr device_addr, u64 size) const;

    /// Whether a download in flight overlaps the range. Lock held.
    [[nodiscard]] bool OverlapsDownloadInFlight(VAddr start, VAddr end) const;

    /// Whether a deferred download overlaps the range. Lock held.
    [[nodiscard]] bool OverlapsDeferredDownload(VAddr start, VAddr end) const;

    /// Removes a deferred download once its data is written back. Lock held.
    void EraseDeferredDownload(u64 download_id);

    /// Waits until no download in flight of another thread overlaps the range. A download of
    /// the calling thread is not waited for: the GPU thread runs deferred operations inside its
    /// own download, which keeps the order of the GPU thread. Returns true if it had to wait,
    /// which released the lock meanwhile. Lock held.
    bool WaitDownloadsInFlight(std::unique_lock<std::mutex>& lk, VAddr start, VAddr end,
                               ReadbackStats& stats);

    /// Clears the GPU-modified state of the pages of a downloaded window, except pages that
    /// received GPU writes again while the download was in flight and pages another download in
    /// flight or a deferred download covers. Lock held.
    void UnmarkDownloadedWindow(VAddr start, u64 size, u64 download_id);

    /// Ends a readback: the access may proceed unless the range got GPU writes again. Lock held.
    void FinishReadback(std::unique_lock<std::mutex>& lk, VAddr device_addr, u64 size,
                        bool is_write, ReadbackStats& stats);

    ReadbackContext* AcquireReadbackContext(u64 staging_size);
    void ReleaseReadbackContext(ReadbackContext* context);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    static constexpr u64 StreamCopySlot(VAddr device_addr, u64 size) {
        return ((device_addr ^ (size << 40)) * 0x9E3779B97F4A7C15ULL) >> (64 - StreamCopySlotBits);
    }

    static constexpr u64 StreamAheadSlot(VAddr device_addr, u64 size) {
        return ((device_addr ^ (size << 40)) * 0x9E3779B97F4A7C15ULL) >> (64 - StreamAheadSlotBits);
    }

    /// The hint of the draw in progress for a binding, if it has one that still holds.
    const StreamHint* FindStreamHint(VAddr device_addr, u32 size);

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
    /// Read-only bindings copied into the stream buffer by the open command buffer. A binding
    /// whose guest memory holds the same bytes as an earlier copy of the same range reuses that
    /// copy. Only copies of the current scheduler tick are reused: the stream buffer keeps their
    /// space until the command buffer of that tick is done.
    struct StreamCopy {
        VAddr device_addr;
        u64 size;
        u64 tick;
        u64 stream_offset;
        u64 shadow_offset;
        /// Id of the hint the copy was made from, 0 when its bytes are in the shadow.
        u64 ahead_id;
    };
    static constexpr u64 StreamCopySlotBits = 12;
    std::array<StreamCopy, 1ULL << StreamCopySlotBits> stream_copies{};
    /// Host copy of the bytes written to the stream buffer in the current tick, to compare with.
    std::vector<u8> stream_shadow;
    u64 stream_shadow_used{};
    u64 stream_shadow_tick{};
    std::span<const StreamHint> stream_hints;
    u32 stream_hint_cursor{};
    u64 stream_hints_epoch{};
    u32 stream_hint_uses{};
    /// One hint in this many is compared with guest memory when it is used.
    static constexpr u32 StreamHintCheckInterval = 64;
    static constexpr u32 MaxStreamHintDiffsLogged = 64;
    u32 stream_hint_diffs_logged{};

    /// Bytes the lookahead thread read, in a ring it fills in order. A position is a count of
    /// bytes since the start; the bytes at a position are at position % StreamAheadSize.
    static constexpr u64 StreamAheadSize = 128_MB;
    static constexpr u64 StreamAheadWindow = 32_MB;
    /// The last read of each range, to give equal bytes the same id. Lookahead thread only.
    struct StreamAheadCopy {
        VAddr device_addr;
        u64 size;
        u64 position;
        u64 id;
        /// The run the bytes were last read or compared in, and the epoch of that read.
        u64 run;
        u64 gpu_write_epoch;
    };
    static constexpr u64 StreamAheadSlotBits = 15;
    alignas(64) std::unique_ptr<u8[]> stream_ahead_data;
    std::unique_ptr<StreamAheadCopy[]> stream_ahead_copies;
    u64 stream_ahead_position{};
    u64 stream_ahead_last_id{};
    /// Positions below this are free. Set by the GPU thread, read by the lookahead thread.
    alignas(64) std::atomic<u64> stream_ahead_released{};
    Buffer gds_buffer;
    std::unique_ptr<Buffer> null_index_buffer;
    /// Guards the GPU-modified state against readbacks running on guest threads: the tracker's
    /// GPU bits, gpu_modified_ranges, write_ticks, hot_windows, downloads_in_flight and the
    /// address_space entries. The GPU thread changes them with it held and other threads read
    /// them with it held. It is never held across a scheduler call.
    std::mutex modified_mutex;
    RangeSet gpu_modified_ranges;
    /// CPU writes performed by AccessBesideGpuData that the arenas do not hold yet. Their pages
    /// are not marked CPU-modified, so they are uploaded by range.
    RangeSet emulated_writes;
    /// One bit per EmulatedChunk bytes of the address space, set while emulated_writes holds a
    /// range in the chunk. Changed with the lock held; the GPU thread reads it without the lock
    /// to skip bindings that have none.
    static constexpr u64 EmulatedChunkBits = 16;
    static constexpr u64 NumEmulatedChunkWords = u64{1}
                                                 << (ADDRESS_SPACE_BITS - EmulatedChunkBits - 6);
    std::unique_ptr<std::atomic<u64>[]> emulated_chunks;

    /// Whether a CPU write done beside GPU data may be pending in the range. Lock-free.
    [[nodiscard]] bool HasEmulatedWrites(VAddr device_addr, u64 size) const {
        for (u64 chunk = device_addr >> EmulatedChunkBits;
             chunk <= (device_addr + size - 1) >> EmulatedChunkBits; ++chunk) {
            if ((emulated_chunks[chunk >> 6].load(std::memory_order_relaxed) >> (chunk & 63)) & 1) {
                return true;
            }
        }
        return false;
    }
    /// Accesses performed by AccessBesideGpuData per page while one command buffer is open. A
    /// page that is accessed more often than that gets a readback, which unprotects it.
    struct EmulatedAccesses {
        u64 tick;
        u32 count;
    };
    std::unordered_map<u64, EmulatedAccesses> emulated_accesses;
    static constexpr u32 MaxEmulatedAccesses = 512;
    /// Scheduler tick of the last writable binding of each page of guest memory, as ranges of
    /// pages with the same tick.
    RangeMap<u64> write_ticks;
    static constexpr u64 WriteTickChunkBits = 16;
    static constexpr u64 MaxPageTickRange = 1ULL << WriteTickChunkBits;
    /// Readback windows that held GPU-modified data recently: start -> {size, scheduler tick}.
    /// A readback downloads all of them that are ready, so the CPU does not fault on each one.
    std::unordered_map<VAddr, std::pair<u64, u64>> hot_windows;
    static constexpr u64 HotWindowTicks = 4;
    /// Pages a readback was needed for: page address -> scheduler tick of the last one. A game
    /// that reuses memory each frame faults on the same pages again, so once the GPU has run
    /// the last write to such a page the readback thread downloads it, which is what a fault
    /// at that moment would do. A page without a fault for EagerPageTicks is dropped.
    std::unordered_map<VAddr, u64> eager_pages;
    static constexpr u64 EagerPageTicks = 1024;
    std::atomic<bool> has_eager_pages{};
    /// The 64 KB chunks that hold eager pages, to find them from a written range.
    static constexpr u64 EagerChunkBits = 16;
    std::unordered_set<u64> eager_chunks;
    /// A GPU write to an eager page was recorded since the last submit.
    std::atomic<bool> eager_write_pending{};
    [[nodiscard]] bool WritesEagerPage(VAddr device_addr, u64 size) const;
    std::mutex eager_mutex;
    std::condition_variable_any eager_cv;
    u64 eager_kicks{};

    /// Remembers the pages of an access that faulted on GPU-modified memory. is_readback: it
    /// took a readback, else it was emulated.
    void NoteFaultedPages(VAddr device_addr, u64 size, bool is_readback = true);
    /// The eager pages with GPU-modified data whose last write the GPU ran by gpu_tick.
    void CollectEagerPages(u64 gpu_tick,
                           boost::container::small_vector<DownloadWindow, 16>& windows);
    void EagerReadbackThread(std::stop_token stoken);
    /// Windows whose GPU-modified ranges were taken out of gpu_modified_ranges and whose data
    /// is not written back yet.
    struct DownloadInFlight {
        u64 id;
        VAddr start;
        VAddr end;
        std::thread::id owner;
    };
    std::vector<DownloadInFlight> downloads_in_flight;
    /// Downloads written back by a deferred operation of the GPU thread. They are not waited
    /// for, a fault on their pages repeats until the operation ran.
    std::vector<DownloadInFlight> deferred_downloads;
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
    /// Block ranges found resident, per slot of their first block. Resident ranges are only
    /// added, so such a range stays resident.
    struct ResidentHit {
        u64 first_block{1};
        u64 last_block{0};
    };
    std::array<ResidentHit, 4096> resident_hits{};

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};

    /// Last member: it stops before anything it uses is destroyed.
    std::jthread eager_readback_thread;
};

} // namespace VideoCore
