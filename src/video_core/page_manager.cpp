// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <map>
#include <thread>
#include <tuple>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/signal_context.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/linker.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#if defined(__linux__) && !defined(_WIN32)
#include <sys/ucontext.h>
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

// PADDIAG: test build only. Records the last CPU writers of every page of the GPU pool.
namespace {
VAddr pad_eboot_base{};
VAddr pad_eboot_end{};

struct PadDiagWriter {
    u64 rip{};
    u64 caller{};
    u64 caller2{};
    u64 seq{};
};

struct PadDiagPool {
    VAddr base{};
    u64 num_pages{};
    std::unique_ptr<std::atomic<u8>[]> armed;
    std::unique_ptr<std::atomic<u8>[]> sources;
    std::unique_ptr<std::atomic<u16>[]> faults;
    std::unique_ptr<PadDiagWriter[]> last;
    std::unique_ptr<PadDiagWriter[]> prev;
};
PadDiagPool pad_pool;
std::atomic<bool> pad_pool_ready{false};
std::atomic<u64> pad_seq{0};

u64 PadDiagGetRsp(void* ctx) {
#if defined(_WIN32)
    return ((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rsp;
#elif defined(__linux__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RSP];
#else
    return 0;
#endif
}

void PadDiagFindCallers(void* ctx, u64& caller, u64& caller2) {
    caller = 0;
    caller2 = 0;
    const u64 rsp = PadDiagGetRsp(ctx);
    if (rsp == 0) {
        return;
    }
    const u64* stack = reinterpret_cast<const u64*>(rsp);
    for (u32 i = 0; i < 256; ++i) {
        const u64 value = stack[i];
        if (value < pad_eboot_base || value >= pad_eboot_end) {
            continue;
        }
        if (caller == 0) {
            caller = value;
        } else if (value != caller) {
            caller2 = value;
            return;
        }
    }
}

constexpr u64 PadPageBits = std::countr_zero(PageManager::GetPageSize());

// Guest breakpoints on the eboot texture loaders (offsets from the eboot base).
constexpr u64 PadBpStreamRead = 0x2905322;    // call [rax+0x68]: stream->Read(dst, size)
constexpr u64 PadBpStreamReadRet = 0x2905325; // mov rax, [r14]
constexpr u64 PadBpGnfMemcpy = 0x2905daa;     // call memcpy(dst, src, size)
constexpr u64 PadBpCreateTexture = 0x28efa72; // mov ecx, [r14+0x20] after the object ctor
constexpr u64 PadMemcpyTarget = 0x4a2d3d0;
constexpr u64 PadBpMinSize = 0x10000;

struct PadDiagPendingRead {
    u64 dst{};
    u64 size{};
    u64 stream{};
};
thread_local PadDiagPendingRead pad_pending_read;

// Reads through the backing view so protected pages do not fault inside the handler.
u64 PadDiagNonZero(u64 addr, u64 size) {
    const auto& backing_pages = Core::Memory::Instance()->GetAddressSpace().BackingPages();
    u64 count = 0;
    while (size) {
        const u64 offset_in_page = addr & (Core::GUEST_PAGE_SIZE - 1);
        const u64 chunk = std::min<u64>(Core::GUEST_PAGE_SIZE - offset_in_page, size);
        if (const auto* entry = backing_pages.find(addr >> Core::GUEST_PAGE_BITS);
            entry && *entry) {
            const u8* data = *entry + offset_in_page;
            for (u64 i = 0; i < chunk; ++i) {
                count += data[i] != 0;
            }
        } else {
            const u8* data = reinterpret_cast<const u8*>(addr);
            for (u64 i = 0; i < chunk; ++i) {
                count += data[i] != 0;
            }
        }
        addr += chunk;
        size -= chunk;
    }
    return count;
}

void PadDiagInstallBreakpoints() {
#ifdef _WIN32
    static constexpr std::array<std::pair<u64, u8>, 4> breakpoints{{
        {PadBpStreamRead, 0xff},
        {PadBpStreamReadRet, 0x49},
        {PadBpGnfMemcpy, 0xe8},
        {PadBpCreateTexture, 0x41},
    }};
    for (const auto [offset, expected] : breakpoints) {
        u8* code = reinterpret_cast<u8*>(pad_eboot_base + offset);
        if (*code != expected) {
            LOG_ERROR(Render, "PADDIAG breakpoint {:#x} unexpected byte {:#04x}", offset, *code);
            continue;
        }
        DWORD old_protect{};
        if (!VirtualProtect(code, 1, PAGE_EXECUTE_READWRITE, &old_protect)) {
            LOG_ERROR(Render, "PADDIAG breakpoint {:#x} protect failed", offset);
            continue;
        }
        *code = 0xCC;
        DWORD unused{};
        VirtualProtect(code, 1, old_protect, &unused);
        FlushInstructionCache(GetCurrentProcess(), code, 1);
        LOG_WARNING(Render, "PADDIAG breakpoint {:#x} installed", offset);
    }
#endif
}

bool PadDiagInPool(VAddr addr) {
    return addr >= pad_pool.base && addr < pad_pool.base + (pad_pool.num_pages << PadPageBits);
}
} // namespace

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

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
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

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
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
        instance = this;

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
        if (True(perms & Core::MemoryPermission::Write) &&
            pad_pool_ready.load(std::memory_order_acquire)) {
            // Armed pool pages stay read-only so the next guest write is recorded.
            const auto read_perms = perms & ~Core::MemoryPermission::Write;
            VAddr run_begin = address;
            Core::MemoryPermission run_perms = perms;
            for (VAddr page = address; page < address + size; page += PageManager::PM_PAGE_SIZE) {
                const bool armed =
                    PadDiagInPool(page) &&
                    pad_pool.armed[(page - pad_pool.base) >> PageManager::PM_PAGE_BITS].load() != 0;
                const auto page_perms = armed ? read_perms : perms;
                if (page != address && page_perms != run_perms) {
                    impl.Protect(run_begin, page - run_begin, run_perms);
                    run_begin = page;
                }
                run_perms = page_perms;
            }
            impl.Protect(run_begin, address + size - run_begin, run_perms);
            return;
        }
        impl.Protect(address, size, perms);
    }

    static bool PadDiagFault(void* context, VAddr addr) {
        if (!pad_pool_ready.load(std::memory_order_acquire) || !Common::IsWriteError(context) ||
            !PadDiagInPool(addr)) {
            return false;
        }
        const u64 index = (addr - pad_pool.base) >> PageManager::PM_PAGE_BITS;
        if (pad_pool.armed[index].exchange(0) == 0) {
            return false;
        }
        PadDiagWriter writer{
            .rip = reinterpret_cast<u64>(Common::GetRip(context)),
            .seq = pad_seq.fetch_add(1) + 1,
        };
        PadDiagFindCallers(context, writer.caller, writer.caller2);
        const u64 page = addr >> PageManager::PM_PAGE_BITS;
        auto& lock = instance->locks[page];
        std::scoped_lock lk{lock};
        pad_pool.prev[index] = pad_pool.last[index];
        pad_pool.last[index] = writer;
        pad_pool.sources[index].fetch_or(1);
        if (const u16 faults = pad_pool.faults[index].load(); faults != 0xffff) {
            pad_pool.faults[index].store(faults + 1);
        }
        const PageState* state = instance->cached_pages.find(page);
        if (state && (state->num_write_watchers != 0 || state->num_read_watchers != 0)) {
            // Page is watched by the page manager, let it handle the fault as usual.
            return false;
        }
        pad_pool.armed[index].store(0);
        Core::Memory::Instance()->GetAddressSpace().Protect(page << PageManager::PM_PAGE_BITS,
                                                            PageManager::PM_PAGE_SIZE,
                                                            Core::MemoryPermission::ReadWrite);
        return true;
    }

    // Re-arms written pool pages. Pages written in many windows are re-armed less often.
    static void PadDiagRearm(u64 pass) {
        if (!instance) {
            return;
        }
        auto& address_space = Core::Memory::Instance()->GetAddressSpace();
        const u64 base_page = pad_pool.base >> PageManager::PM_PAGE_BITS;
        for (u64 index = 0; index < pad_pool.num_pages; ++index) {
            if (pad_pool.armed[index].load(std::memory_order_relaxed) != 0) {
                continue;
            }
            if (pad_pool.faults[index].load(std::memory_order_relaxed) >= 8 && pass % 25 != 0) {
                continue;
            }
            const u64 page = base_page + index;
            auto& lock = instance->locks[page];
            std::scoped_lock lk{lock};
            pad_pool.armed[index].store(1);
            const PageState* state = instance->cached_pages.find(page);
            if (state && (state->num_write_watchers != 0 || state->num_read_watchers != 0)) {
                // Already protected by the page manager.
                continue;
            }
            address_space.Protect(page << PageManager::PM_PAGE_BITS, PageManager::PM_PAGE_SIZE,
                                  Core::MemoryPermission::Read);
        }
    }

    inline static SignalImpl* instance;

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        if (PadDiagFault(context, addr)) {
            return true;
        }
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        const auto is_gpu_thread =
            std::this_thread::get_id() == rasterizer->GetGpuCommandProcessorThread();
        if (Common::IsWriteError(context)) {
            return rasterizer->InvalidateMemory(addr, size, is_gpu_thread);
        } else {
            return rasterizer->ReadMemory(addr, size, is_gpu_thread);
        }
        return false;
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

void PageManager::PadDiagArmPool(VAddr address, size_t size) {
    if (pad_pool_ready.load()) {
        return;
    }
    if (const auto* eboot = Common::Singleton<Core::Linker>::Instance()->GetModule(0)) {
        pad_eboot_base = eboot->GetBaseAddress();
        pad_eboot_end = pad_eboot_base + eboot->aligned_base_size;
        LOG_WARNING(Render, "PADDIAG module 0 {} {:#x}-{:#x}", eboot->name, pad_eboot_base,
                    pad_eboot_end);
    }
    if (pad_eboot_base != 0) {
        PadDiagInstallBreakpoints();
    }
    const VAddr base = Common::AlignDown(address, PM_PAGE_SIZE);
    const u64 num_pages = (Common::AlignUp(address + size, PM_PAGE_SIZE) - base) >> PM_PAGE_BITS;
    pad_pool.base = base;
    pad_pool.num_pages = num_pages;
    pad_pool.armed = std::make_unique<std::atomic<u8>[]>(num_pages);
    pad_pool.sources = std::make_unique<std::atomic<u8>[]>(num_pages);
    pad_pool.faults = std::make_unique<std::atomic<u16>[]>(num_pages);
    pad_pool.last = std::make_unique<PadDiagWriter[]>(num_pages);
    pad_pool.prev = std::make_unique<PadDiagWriter[]>(num_pages);
    for (u64 i = 0; i < num_pages; ++i) {
        pad_pool.armed[i].store(1, std::memory_order_relaxed);
    }
    pad_pool_ready.store(true, std::memory_order_release);
    Core::Memory::Instance()->GetAddressSpace().Protect(base, num_pages << PM_PAGE_BITS,
                                                        Core::MemoryPermission::Read);
    std::thread([] {
        Common::SetCurrentThreadName("shadPS4:PadDiag");
        for (u64 pass = 1;; ++pass) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            SignalImpl::PadDiagRearm(pass);
        }
    }).detach();
    LOG_WARNING(Render, "PADDIAG armed pool {:#x} size {:#x}", base, num_pages << PM_PAGE_BITS);
}

// Sources: 1 CPU write fault, 2 GPU DMA/copy, 4 shader storage write, 8 readback to guest,
// 16 marked GPU modified.
void PageManager::PadDiagRecord(VAddr address, size_t size, u8 source) {
    if (!pad_pool_ready.load(std::memory_order_acquire) || size == 0) {
        return;
    }
    const VAddr pool_end = pad_pool.base + (pad_pool.num_pages << PM_PAGE_BITS);
    const VAddr end = std::min<VAddr>(address + size, pool_end);
    for (VAddr page = std::max(GetPageAddr(address), pad_pool.base); page < end;
         page += PM_PAGE_SIZE) {
        pad_pool.sources[(page - pad_pool.base) >> PM_PAGE_BITS].fetch_or(source);
    }
}

void PageManager::PadDiagReport(VAddr image, VAddr address, size_t size, const char* label) {
    if (!pad_pool_ready.load(std::memory_order_acquire) || !PadDiagInPool(address) ||
        !PadDiagInPool(address + size - 1)) {
        LOG_WARNING(Render, "PADDIAG writers image={:#x} {}: outside armed pool", image, label);
        return;
    }
    const auto offset = [](u64 addr) { return addr ? addr - pad_eboot_base : 0; };
    const auto report = [&](const char* which, auto get) {
        struct Group {
            u32 pages{};
            u64 min_seq{~0ULL};
            u64 max_seq{};
            u32 max_faults{};
        };
        std::map<std::tuple<u8, u64, u64, u64>, Group> groups;
        for (VAddr page = GetPageAddr(address); page < address + size; page += PM_PAGE_SIZE) {
            const u64 index = (page - pad_pool.base) >> PM_PAGE_BITS;
            const PadDiagWriter& writer = get(index);
            auto& group =
                groups[{pad_pool.sources[index].load(), writer.rip, writer.caller, writer.caller2}];
            ++group.pages;
            group.min_seq = std::min(group.min_seq, writer.seq);
            group.max_seq = std::max(group.max_seq, writer.seq);
            group.max_faults = std::max<u32>(group.max_faults, pad_pool.faults[index].load());
        }
        for (const auto& [key, group] : groups) {
            const auto [sources, rip, caller, caller2] = key;
            LOG_WARNING(Render,
                        "PADDIAG writers image={:#x} {} {}: sources={:#x} rip={:#x} "
                        "caller_off={:#x} caller2_off={:#x} seq={}-{} faults<={} pages={} now={}",
                        image, label, which, sources, rip, offset(caller), offset(caller2),
                        group.min_seq, group.max_seq, group.max_faults, group.pages,
                        pad_seq.load());
        }
    };
    report("last", [](u64 index) -> const PadDiagWriter& { return pad_pool.last[index]; });
    report("prev", [](u64 index) -> const PadDiagWriter& { return pad_pool.prev[index]; });
}

bool PageManager::PadDiagBreakpoint(void* context) {
#ifdef _WIN32
    if (pad_eboot_base == 0) {
        return false;
    }
    CONTEXT* ctx = reinterpret_cast<EXCEPTION_POINTERS*>(context)->ContextRecord;
    const auto is_bp = [](u64 rip) {
        const u64 offset = rip - pad_eboot_base;
        return offset == PadBpStreamRead || offset == PadBpStreamReadRet ||
               offset == PadBpGnfMemcpy || offset == PadBpCreateTexture;
    };
    u64 bp = ctx->Rip;
    if (!is_bp(bp)) {
        if (!is_bp(bp - 1)) {
            return false;
        }
        bp -= 1;
    }
    const auto push = [ctx](u64 value) {
        ctx->Rsp -= 8;
        *reinterpret_cast<u64*>(ctx->Rsp) = value;
    };
    const auto offset = [](u64 addr) {
        return addr >= pad_eboot_base && addr < pad_eboot_end ? addr - pad_eboot_base : addr;
    };
    const u64 caller = *reinterpret_cast<const u64*>(ctx->Rbp + 8);
    // Return addresses of the two frames above the loader, following the rbp chain.
    std::array<u64, 2> callers{};
    u64 frame = *reinterpret_cast<const u64*>(ctx->Rbp);
    for (u64& ret : callers) {
        if (frame <= ctx->Rsp || frame - ctx->Rsp > 0x100000) {
            break;
        }
        ret = *reinterpret_cast<const u64*>(frame + 8);
        frame = *reinterpret_cast<const u64*>(frame);
    }
    switch (bp - pad_eboot_base) {
    case PadBpStreamRead: {
        const u64 vtable = *reinterpret_cast<const u64*>(ctx->Rdi);
        const u64 target = *reinterpret_cast<const u64*>(ctx->Rax + 0x68);
        pad_pending_read = {.dst = ctx->Rsi, .size = ctx->Rdx, .stream = ctx->Rdi};
        if (ctx->Rdx >= PadBpMinSize) {
            const u64* stream = reinterpret_cast<const u64*>(ctx->Rdi);
            LOG_WARNING(Render,
                        "PADDIAG bp stream read stream={:#x} vtable_off={:#x} read_off={:#x} "
                        "dst={:#x} size={:#x} caller_off={:#x} seq={} stream_words={:016x} "
                        "{:016x} {:016x} {:016x} {:016x} {:016x}",
                        ctx->Rdi, offset(vtable), offset(target), ctx->Rsi, ctx->Rdx,
                        offset(caller), pad_seq.load(), stream[1], stream[2], stream[3], stream[4],
                        stream[5], stream[6]);
        }
        push(bp + 3);
        ctx->Rip = target;
        return true;
    }
    case PadBpStreamReadRet: {
        const PadDiagPendingRead read = pad_pending_read;
        if (read.size >= PadBpMinSize) {
            LOG_WARNING(Render,
                        "PADDIAG bp stream read done stream={:#x} dst={:#x} size={:#x} got={:#x} "
                        "nonzero={} seq={}",
                        read.stream, read.dst, read.size, ctx->Rax,
                        PadDiagNonZero(read.dst, read.size), pad_seq.load());
        }
        pad_pending_read = {};
        ctx->Rax = *reinterpret_cast<const u64*>(ctx->R14);
        ctx->Rip = bp + 3;
        return true;
    }
    case PadBpGnfMemcpy: {
        if (ctx->Rdx >= PadBpMinSize) {
            LOG_WARNING(Render,
                        "PADDIAG bp gnf memcpy dst={:#x} src={:#x} size={:#x} src_nonzero={} "
                        "caller_off={:#x} {:#x} {:#x} seq={}",
                        ctx->Rdi, ctx->Rsi, ctx->Rdx, PadDiagNonZero(ctx->Rsi, ctx->Rdx),
                        offset(caller), offset(callers[0]), offset(callers[1]), pad_seq.load());
        }
        push(bp + 5);
        ctx->Rip = pad_eboot_base + PadMemcpyTarget;
        return true;
    }
    case PadBpCreateTexture: {
        const u32* desc = reinterpret_cast<const u32*>(ctx->R14);
        if (desc[0] == 2048 && desc[1] == 1024) {
            LOG_WARNING(Render,
                        "PADDIAG bp create texture object={:#x} desc={:08x} {:08x} {:08x} "
                        "{:08x} {:08x} {:08x} {:08x} {:08x} {:08x} caller_off={:#x} "
                        "{:#x} {:#x} seq={}",
                        ctx->Rbx, desc[0], desc[1], desc[2], desc[3], desc[4], desc[5], desc[6],
                        desc[7], desc[8], offset(caller), offset(callers[0]), offset(callers[1]),
                        pad_seq.load());
        }
        ctx->Rcx = desc[8];
        ctx->Rip = bp + 4;
        return true;
    }
    default:
        return false;
    }
#else
    return false;
#endif
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
