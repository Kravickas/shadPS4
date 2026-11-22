// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstring>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/decoder.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

static thread_local PageManager::WatchCounts watch_counts{};

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers;
        u8 num_read_watchers;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <bool is_read>
        u8 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            if (update_read) {
                if (read_op == PageOp::Track) {
                    ASSERT_MSG(num_read_watchers == 0, "Too many watchers");
                } else if (read_op == PageOp::Untrack) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                }
                num_read_watchers += std::to_underlying(read_op);
            }
            if (update_write) {
                if (write_op == PageOp::Track) {
                    ASSERT_MSG(num_write_watchers < 255, "Too many watchers");
                } else if (write_op == PageOp::Untrack) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                }
                num_write_watchers += std::to_underlying(write_op);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
        EnsurePages(address, address + size);
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        locks.reserve(start_page, end_page);
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        watch_counts.pages += page_end - page_start;
        watch_counts.image_pages += page_end - page_start;

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                ++watch_counts.protects;
                ++watch_counts.image_protects;
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            EnsurePages(aligned_addr, aligned_end);
        }

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            locks[page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;
        watch_counts.pages += page_end - page_start;

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                ++watch_counts.protects;
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            locks[base_page + page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(base_page + page)) {
                lock->unlock();
            }
        }
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    struct MutexTraits {
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        using Entry = Common::AdaptiveMutex;
#else
        using Entry = std::mutex;
#endif
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadPS4:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            rasterizer->InvalidateMemory(addr, 1,
                                         ptid == rasterizer->GetGpuCommandProcessorThreadId());

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        impl.Protect(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        const auto is_gpu_thread =
            std::this_thread::get_id() == rasterizer->GetGpuCommandProcessorThread();
        if (!is_gpu_thread && TryEmulateAccess(context, addr)) {
            return true;
        }
        DecodeFaultAccess(context);
        const u64 fault_t0 = AmdGpu::PerfNs();
        const bool is_write = Common::IsWriteError(context);
        bool handled;
        if (is_write) {
            handled = rasterizer->InvalidateMemory(addr, size, is_gpu_thread);
        } else {
            handled = rasterizer->ReadMemory(addr, size, is_gpu_thread);
        }
        current_fault_access.valid = false;
        if (handled && !is_gpu_thread) {
            auto& perf = AmdGpu::guest_faults;
            const u64 fault_ns = AmdGpu::PerfNs() - fault_t0;
            if (is_write) {
                static thread_local VAddr last_write_page = 0;
                const VAddr page = addr >> PageManager::PM_PAGE_BITS;
                if (page == last_write_page + 1) {
                    perf.next_page_writes.fetch_add(1, std::memory_order_relaxed);
                }
                last_write_page = page;
                perf.writes.fetch_add(1, std::memory_order_relaxed);
                perf.write_ns.fetch_add(fault_ns, std::memory_order_relaxed);
            } else {
                perf.reads.fetch_add(1, std::memory_order_relaxed);
                perf.read_ns.fetch_add(fault_ns, std::memory_order_relaxed);
            }
        }
        return handled;
    }

    /// Value of a general purpose register operand, false if it is not a plain one.
    static bool ReadGpr(void* context, ZydisRegister reg, u64& value) {
        const auto reg_class = ZydisRegisterGetClass(reg);
        if (reg_class != ZYDIS_REGCLASS_GPR8 && reg_class != ZYDIS_REGCLASS_GPR16 &&
            reg_class != ZYDIS_REGCLASS_GPR32 && reg_class != ZYDIS_REGCLASS_GPR64) {
            return false;
        }
        if (reg >= ZYDIS_REGISTER_AH && reg <= ZYDIS_REGISTER_BH) {
            return false;
        }
        const u64* const pointer = GprPointer(context, reg);
        if (!pointer) {
            return false;
        }
        value = *pointer;
        return true;
    }

    /// The 64-bit register of the context that holds a general purpose register operand.
    static u64* GprPointer(void* context, ZydisRegister reg) {
        // The register number is the one of the enclosing 64-bit register: the 8-bit class
        // numbers its registers differently.
        const auto full = ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, reg);
        if (ZydisRegisterGetClass(full) != ZYDIS_REGCLASS_GPR64) {
            return nullptr;
        }
        return Common::GetGprPointer(context, u8(ZydisRegisterGetId(full)));
    }

    /// Performs the faulting instruction's memory access on guest memory and skips the
    /// instruction, when the access touches no GPU data. Handles the plain forms of mov, cmp
    /// and 128-bit vector stores; anything else is left to the readback.
    static bool TryEmulateAccess(void* context, VAddr fault_addr) {
        ZydisDecodedInstruction inst;
        ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
        const VAddr rip = reinterpret_cast<VAddr>(Common::GetRip(context));
        if (!ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                inst, operands, reinterpret_cast<void*>(rip)))) {
            return false;
        }
        constexpr ZydisInstructionAttributes Unsupported =
            ZYDIS_ATTRIB_HAS_LOCK | ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE |
            ZYDIS_ATTRIB_HAS_REPNE | ZYDIS_ATTRIB_HAS_ADDRESSSIZE;
        if ((inst.attributes & Unsupported) != 0 || inst.operand_count_visible != 2) {
            return false;
        }
        const bool mem_first = operands[0].type == ZYDIS_OPERAND_TYPE_MEMORY;
        const bool mem_second = operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY;
        if (mem_first == mem_second) {
            return false;
        }
        const auto& mem_op = operands[mem_first ? 0 : 1];
        const auto& other_op = operands[mem_first ? 1 : 0];
        const auto& mem = mem_op.mem;
        if (mem.type != ZYDIS_MEMOP_TYPE_MEM || mem.segment == ZYDIS_REGISTER_FS ||
            mem.segment == ZYDIS_REGISTER_GS) {
            return false;
        }
        const u32 size = mem_op.size / 8;
        if (size == 0 || size > 16) {
            return false;
        }

        // The effective address has to be the faulting one, within a single page.
        u64 address = static_cast<u64>(mem.disp.value);
        if (mem.base == ZYDIS_REGISTER_RIP) {
            address += rip + inst.length;
        } else if (mem.base != ZYDIS_REGISTER_NONE) {
            u64 base;
            if (ZydisRegisterGetClass(mem.base) != ZYDIS_REGCLASS_GPR64 ||
                !ReadGpr(context, mem.base, base)) {
                return false;
            }
            address += base;
        }
        if (mem.index != ZYDIS_REGISTER_NONE) {
            u64 index;
            if (ZydisRegisterGetClass(mem.index) != ZYDIS_REGCLASS_GPR64 ||
                !ReadGpr(context, mem.index, index)) {
                return false;
            }
            address += index * mem.scale;
        }
        if (address != fault_addr || PageManager::GetNextPageAddr(address) < address + size) {
            return false;
        }

        // The other operand: a register or an immediate.
        u64 other = 0;
        const bool other_is_reg = other_op.type == ZYDIS_OPERAND_TYPE_REGISTER;
        if (other_op.type == ZYDIS_OPERAND_TYPE_IMMEDIATE) {
            other = other_op.imm.is_signed ? static_cast<u64>(other_op.imm.value.s)
                                           : other_op.imm.value.u;
        } else if (!other_is_reg) {
            return false;
        }
        const u64 mask = size >= 8 ? ~0ULL : (1ULL << (size * 8)) - 1;
        const bool is_write_fault = Common::IsWriteError(context);
        u8 data[16];

        switch (inst.mnemonic) {
        case ZYDIS_MNEMONIC_MOV: {
            if (size > 8) {
                return false;
            }
            if (mem_first) {
                // Store of a register or an immediate.
                if (!is_write_fault ||
                    (other_is_reg && !ReadGpr(context, other_op.reg.value, other))) {
                    return false;
                }
                std::memcpy(data, &other, size);
                if (!rasterizer->AccessBesideGpuData(address, size, true, data)) {
                    return false;
                }
            } else {
                // Load into a 32 or 64-bit register; a 32-bit load clears the upper half.
                const auto reg_class = ZydisRegisterGetClass(other_op.reg.value);
                if (is_write_fault || !other_is_reg ||
                    (reg_class != ZYDIS_REGCLASS_GPR32 && reg_class != ZYDIS_REGCLASS_GPR64) ||
                    size != (reg_class == ZYDIS_REGCLASS_GPR32 ? 4U : 8U)) {
                    return false;
                }
                u64* const dest = GprPointer(context, other_op.reg.value);
                if (!dest || !rasterizer->AccessBesideGpuData(address, size, false, data)) {
                    return false;
                }
                u64 value = 0;
                std::memcpy(&value, data, size);
                *dest = value;
            }
            break;
        }
        case ZYDIS_MNEMONIC_CMP: {
            if (size > 8 || is_write_fault ||
                (other_is_reg && !ReadGpr(context, other_op.reg.value, other)) ||
                !rasterizer->AccessBesideGpuData(address, size, false, data)) {
                return false;
            }
            u64 value = 0;
            std::memcpy(&value, data, size);
            // cmp subtracts the second operand from the first and only sets the flags.
            const u64 lhs = (mem_first ? value : other) & mask;
            const u64 rhs = (mem_first ? other : value) & mask;
            const u64 result = (lhs - rhs) & mask;
            const u64 sign = 1ULL << (size * 8 - 1);
            constexpr u64 CF = 1ULL << 0, PF = 1ULL << 2, AF = 1ULL << 4, ZF = 1ULL << 6,
                          SF = 1ULL << 7, OF = 1ULL << 11;
            u64 flags = Common::GetFlags(context) & ~(CF | PF | AF | ZF | SF | OF);
            flags |= lhs < rhs ? CF : 0;
            flags |= (std::popcount(static_cast<u8>(result)) & 1) == 0 ? PF : 0;
            flags |= ((lhs ^ rhs ^ result) & 0x10) != 0 ? AF : 0;
            flags |= result == 0 ? ZF : 0;
            flags |= (result & sign) != 0 ? SF : 0;
            flags |= ((lhs ^ rhs) & (lhs ^ result) & sign) != 0 ? OF : 0;
            Common::SetFlags(context, flags);
            break;
        }
        case ZYDIS_MNEMONIC_MOVAPS:
        case ZYDIS_MNEMONIC_MOVAPD:
        case ZYDIS_MNEMONIC_MOVDQA:
        case ZYDIS_MNEMONIC_VMOVAPS:
        case ZYDIS_MNEMONIC_VMOVAPD:
        case ZYDIS_MNEMONIC_VMOVDQA:
            if ((address & 0xf) != 0) {
                // An unaligned access of these raises a general protection fault.
                return false;
            }
            [[fallthrough]];
        case ZYDIS_MNEMONIC_MOVUPS:
        case ZYDIS_MNEMONIC_MOVUPD:
        case ZYDIS_MNEMONIC_MOVDQU:
        case ZYDIS_MNEMONIC_VMOVUPS:
        case ZYDIS_MNEMONIC_VMOVUPD:
        case ZYDIS_MNEMONIC_VMOVDQU: {
            // Store of an XMM register. Loads are left alone: the VEX forms also clear the
            // upper bits of the YMM register, which the context here does not hold.
            if (!mem_first || !is_write_fault || size != 16 || !other_is_reg ||
                ZydisRegisterGetClass(other_op.reg.value) != ZYDIS_REGCLASS_XMM) {
                return false;
            }
            const void* const xmm =
                Common::GetXmmRegisterPointer(context, u8(ZydisRegisterGetId(other_op.reg.value)));
            if (!xmm) {
                return false;
            }
            std::memcpy(data, xmm, 16);
            if (!rasterizer->AccessBesideGpuData(address, size, true, data)) {
                return false;
            }
            break;
        }
        default:
            return false;
        }
        Common::SkipInstruction(context, inst.length);
        return true;
    }

    /// Records which instruction faulted and the size of its memory access, for the readback
    /// trace.
    static void DecodeFaultAccess(void* context) {
        auto& access = current_fault_access;
        access = {};
        access.rip = reinterpret_cast<VAddr>(Common::GetRip(context));
        access.valid = true;
        ZydisDecodedInstruction inst;
        ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
        if (!ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                inst, operands, reinterpret_cast<void*>(access.rip)))) {
            return;
        }
        access.mnemonic = inst.mnemonic;
        access.lock = (inst.attributes & ZYDIS_ATTRIB_HAS_LOCK) != 0;
        access.rep = (inst.attributes &
                      (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE)) != 0;
        for (u32 i = 0; i < inst.operand_count; ++i) {
            const auto& operand = operands[i];
            if (operand.type == ZYDIS_OPERAND_TYPE_MEMORY &&
                operand.mem.type == ZYDIS_MEMOP_TYPE_MEM) {
                access.bytes = operand.size / 8;
                break;
            }
        }
    }
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

PageManager::WatchCounts PageManager::WatchCountsOfThread() noexcept {
    return watch_counts;
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
