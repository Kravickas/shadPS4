// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

struct UffdImpl;
struct SignalImpl;

/// The CPU access that faulted on a tracked page, decoded from the faulting instruction. Set by
/// the signal handler for the duration of its readback on the faulting thread.
struct FaultAccess {
    VAddr rip;
    u32 mnemonic; ///< ZydisMnemonic, 0 if the instruction was not decoded
    u32 bytes;    ///< Size of the memory operand, 0 if unknown
    bool lock;
    bool rep;
    bool valid;
};
inline thread_local FaultAccess current_fault_access{};

enum class PageOp : s8 {
    None = 0,
    Track = 1,
    Untrack = -1,
};

class PageManager {
    static constexpr size_t PM_PAGE_BITS = 12;
    static constexpr size_t PM_PAGE_SIZE = 1ULL << PM_PAGE_BITS;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    /// Updates watches in the pages touching the specified region.
    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const;

    /// Updates watches in the pages touching the inclusive bounds using a mask.
    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) const;

    /// Probe: pages the calling thread's watcher updates went over, and their protection calls.
    /// The image counts are the part of UpdatePageWatchers, which only images use.
    struct WatchCounts {
        u64 pages;
        u64 protects;
        u64 image_pages;
        u64 image_protects;
    };
    static WatchCounts WatchCountsOfThread() noexcept;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

    static constexpr size_t GetPageSize() {
        return PM_PAGE_SIZE;
    }

private:
    friend struct UffdImpl;
    friend struct SignalImpl;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCore
