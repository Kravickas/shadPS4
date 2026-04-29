// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/types.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"

#ifdef _WIN64
#include <intrin.h>
#include <windows.h>
#endif

namespace Libraries::Kernel {

namespace {

constexpr uintptr_t kPlausibleHeapMin = 0x10000ULL;

// Validates whether the bytes at `addr` look like a fully-initialized
// PthreadMutex — i.e. they are internally consistent with the layout produced
// by MutexInit. Returns true if all sanity checks pass.
bool LooksLikePthreadMutex(const PthreadMutex* candidate) {
#ifdef _WIN64
    if (candidate == nullptr) {
        return false;
    }
    // Must be readable for sizeof(PthreadMutex) = ~72 bytes.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(candidate, &mbi, sizeof(mbi)) == 0) {
        return false;
    }
    if (!(mbi.State & MEM_COMMIT) || (mbi.Protect & PAGE_NOACCESS) ||
        (mbi.Protect & PAGE_GUARD)) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(mbi.BaseAddress);
    const uintptr_t end = base + mbi.RegionSize;
    if (reinterpret_cast<uintptr_t>(candidate) + sizeof(PthreadMutex) > end) {
        return false;
    }

    // Field-by-field consistency checks. We DO NOT touch the std::string
    // because that could contain heap pointers we'd fault on.
    const u32 flags_bits = static_cast<u32>(candidate->m_flags);
    const u32 type_bits = flags_bits & 0xff; // PthreadMutexFlags::TypeMask
    const u32 nontype_bits = flags_bits & ~0xff;
    if (type_bits == 0 || type_bits >= static_cast<u32>(PthreadMutexType::Max)) {
        return false;
    }
    // Only Deferred(0x200) is currently valid in the high bits.
    if (nontype_bits != 0 && nontype_bits != 0x200) {
        return false;
    }
    const u32 protocol = static_cast<u32>(candidate->m_protocol);
    if (protocol > static_cast<u32>(PthreadMutexProt::Protect)) {
        return false;
    }
    if (candidate->m_count < 0 || candidate->m_spinloops < 0 ||
        candidate->m_yieldloops < 0) {
        return false;
    }
    return true;
#else
    (void)candidate;
    return false;
#endif
}

void DumpMemSafe(const char* label, const void* addr, size_t bytes) {
#ifdef _WIN64
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(addr, &mbi, sizeof(mbi)) == 0 || !(mbi.State & MEM_COMMIT) ||
        (mbi.Protect & PAGE_NOACCESS) || (mbi.Protect & PAGE_GUARD)) {
        LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   {}: NOT readable", label);
        return;
    }
    const u8* base = reinterpret_cast<const u8*>(mbi.BaseAddress);
    const u8* p = reinterpret_cast<const u8*>(addr);
    const size_t avail = mbi.RegionSize - static_cast<size_t>(p - base);
    const size_t to_dump = avail < bytes ? avail : bytes;
    std::string line;
    line.reserve(80);
    for (size_t i = 0; i < to_dump; ++i) {
        line += fmt::format("{:02x} ", p[i]);
        if ((i & 0xf) == 0xf || i == to_dump - 1) {
            LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   {}+{:#04x}: {}", label,
                         i & ~size_t{0xf}, line);
            line.clear();
        }
    }
#else
    (void)label;
    (void)addr;
    (void)bytes;
#endif
}

void DiagSuspiciousMutex(const char* fn, PthreadMutexT* mutex, const void* return_addr) {
    if (mutex == nullptr) {
        LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2] {}: mutex** is NULL caller={}", fn,
                     fmt::ptr(return_addr));
        return;
    }
    const uintptr_t m_val = reinterpret_cast<uintptr_t>(*mutex);
    if (m_val <= 2 || m_val >= kPlausibleHeapMin) {
        return; // valid magic or plausible heap pointer; not suspicious
    }

    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2] ============================");
    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2] {} *mutex={:#x} (suspicious)", fn, m_val);
    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   mutex** parameter = {}", fmt::ptr(mutex));
    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   caller PC         = {}", fmt::ptr(return_addr));
    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   g_curthread       = {} name='{}'",
                 fmt::ptr(g_curthread),
                 g_curthread ? g_curthread->name.c_str() : "<null>");

    // CRITICAL TEST: does the buffer at `mutex` itself look like a valid
    // PthreadMutex? If yes, the game is passing the heap PthreadMutex* directly
    // (single-pointer ABI), and the FIX is to interpret `mutex` as PthreadMutex*
    // rather than dereferencing it once more.
    const PthreadMutex* as_pmutex = reinterpret_cast<const PthreadMutex*>(mutex);
    const bool looks_valid_at_mutex = LooksLikePthreadMutex(as_pmutex);
    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   `mutex` itself looks like PthreadMutex? {}",
                 looks_valid_at_mutex ? "YES (game passes single-ptr)" : "no");
    if (looks_valid_at_mutex) {
        // Print the raw 8 bytes at +0x00 (the m_lock HANDLE, opaque to us)
        u64 raw_lock = 0;
        std::memcpy(&raw_lock, mutex, sizeof(raw_lock));
        LOG_CRITICAL(Kernel_Pthread,
                     "[MTX-DIAG-V2]   -> as PthreadMutex*: lock={:#x} flags={:#x}"
                     " owner={} count={} prot={}",
                     raw_lock, static_cast<u32>(as_pmutex->m_flags),
                     fmt::ptr(as_pmutex->m_owner), as_pmutex->m_count,
                     static_cast<u32>(as_pmutex->m_protocol));
    }

    // ALSO check if *mutex looks like a valid PthreadMutex (i.e. just because
    // it's small doesn't necessarily mean it's invalid — but for 0x6eec, no).
    const PthreadMutex* via_deref = reinterpret_cast<const PthreadMutex*>(*mutex);
    const bool looks_valid_via_deref = LooksLikePthreadMutex(via_deref);
    LOG_CRITICAL(Kernel_Pthread, "[MTX-DIAG-V2]   *mutex looks like PthreadMutex? {}",
                 looks_valid_via_deref ? "YES (FreeBSD ABI)" : "no");

    DumpMemSafe("buf_at_mutex", mutex, 80);
}

} // namespace

#ifdef _WIN64
#define MTX_DIAG(fn_name, mutex_ptr) DiagSuspiciousMutex(fn_name, mutex_ptr, _ReturnAddress())
#else
#define MTX_DIAG(fn_name, mutex_ptr)                                                               \
    DiagSuspiciousMutex(fn_name, mutex_ptr, __builtin_return_address(0))
#endif

static constexpr u32 MUTEX_ADAPTIVE_SPINS = 2000;
static std::mutex MutxStaticLock;

#define THR_MUTEX_INITIALIZER ((PthreadMutex*)NULL)
#define THR_ADAPTIVE_MUTEX_INITIALIZER ((PthreadMutex*)1)
#define THR_MUTEX_DESTROYED ((PthreadMutex*)2)

#define CPU_SPINWAIT __asm__ volatile("pause")

#define CHECK_AND_INIT_MUTEX                                                                       \
    if (PthreadMutex* m = *mutex; m <= THR_MUTEX_DESTROYED) [[unlikely]] {                         \
        if (m == THR_MUTEX_DESTROYED) {                                                            \
            return POSIX_EINVAL;                                                                   \
        }                                                                                          \
        if (s32 ret = InitStatic(g_curthread, mutex); ret) {                                       \
            return ret;                                                                            \
        }                                                                                          \
        m = *mutex;                                                                                \
    }

static constexpr PthreadMutexAttr PthreadMutexattrDefault = {
    .m_type = PthreadMutexType::ErrorCheck, .m_protocol = PthreadMutexProt::None, .m_ceiling = 0};

static constexpr PthreadMutexAttr PthreadMutexattrAdaptiveDefault = {
    .m_type = PthreadMutexType::AdaptiveNp, .m_protocol = PthreadMutexProt::None, .m_ceiling = 0};

using CallocFun = void* (*)(size_t, size_t);

static s32 MutexInit(PthreadMutexT* mutex, const PthreadMutexAttr* mutex_attr, const char* name) {
    const PthreadMutexAttr* attr;
    if (mutex_attr == nullptr) {
        attr = &PthreadMutexattrDefault;
    } else {
        attr = mutex_attr;
        if (attr->m_type < PthreadMutexType::ErrorCheck || attr->m_type >= PthreadMutexType::Max) {
            return POSIX_EINVAL;
        }
        if (attr->m_protocol > PthreadMutexProt::Protect) {
            return POSIX_EINVAL;
        }
    }
    auto* pmutex = new (std::nothrow) PthreadMutex{};
    if (pmutex == nullptr) {
        return POSIX_ENOMEM;
    }

    if (name) {
        pmutex->name = name;
    } else {
        static std::atomic<s32> MutexId{0};
        pmutex->name = fmt::format("Mutex{}", MutexId.fetch_add(1));
    }

    pmutex->m_flags = PthreadMutexFlags(attr->m_type);
    pmutex->m_owner = nullptr;
    pmutex->m_count = 0;
    pmutex->m_spinloops = 0;
    pmutex->m_yieldloops = 0;
    pmutex->m_protocol = attr->m_protocol;
    if (attr->m_type == PthreadMutexType::AdaptiveNp) {
        pmutex->m_spinloops = MUTEX_ADAPTIVE_SPINS;
        // pmutex->m_yieldloops = _thr_yieldloops;
    }

    *mutex = pmutex;
    return 0;
}

static s32 InitStatic(Pthread* thread, PthreadMutexT* mutex) {
    std::scoped_lock lk{MutxStaticLock};

    if (*mutex == THR_MUTEX_INITIALIZER) {
        return MutexInit(mutex, &PthreadMutexattrDefault, nullptr);
    } else if (*mutex == THR_ADAPTIVE_MUTEX_INITIALIZER) {
        return MutexInit(mutex, &PthreadMutexattrAdaptiveDefault, nullptr);
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_init(PthreadMutexT* mutex,
                                          const PthreadMutexAttrT* mutex_attr) {
    return MutexInit(mutex, mutex_attr ? *mutex_attr : nullptr, nullptr);
}

s32 PS4_SYSV_ABI scePthreadMutexInit(PthreadMutexT* mutex, const PthreadMutexAttrT* mutex_attr,
                                     const char* name) {
    return MutexInit(mutex, mutex_attr ? *mutex_attr : nullptr, name);
}

s32 PS4_SYSV_ABI posix_pthread_mutex_destroy(PthreadMutexT* mutex) {
    PthreadMutexT m = *mutex;
    if (m < THR_MUTEX_DESTROYED) {
        return 0;
    }
    if (m == THR_MUTEX_DESTROYED) {
        return POSIX_EINVAL;
    }
    if (m->m_owner != nullptr) {
        return POSIX_EBUSY;
    }
    *mutex = THR_MUTEX_DESTROYED;
    delete m;
    return 0;
}

s32 PthreadMutex::SelfTryLock() {
    switch (Type()) {
    case PthreadMutexType::ErrorCheck:
    case PthreadMutexType::Normal:
    case PthreadMutexType::AdaptiveNp:
        return POSIX_EBUSY;
    case PthreadMutexType::Recursive: {
        /* Increment the lock count: */
        if (m_count + 1 > 0) {
            m_count++;
            return 0;
        }
        return POSIX_EAGAIN;
    }
    default:
        return POSIX_EINVAL;
    }
}

s32 PthreadMutex::SelfLock(const OrbisKernelTimespec* abstime, u64 usec) {
    const auto DoSleep = [&] {
        if (abstime == THR_RELTIME) {
            std::this_thread::sleep_for(std::chrono::microseconds(usec));
            return POSIX_ETIMEDOUT;
        } else {
            if (abstime->tv_sec < 0 || abstime->tv_nsec < 0 || abstime->tv_nsec >= 1000000000) {
                return POSIX_EINVAL;
            } else {
                std::this_thread::sleep_until(abstime->TimePoint());
                return POSIX_ETIMEDOUT;
            }
        }
    };
    switch (Type()) {
    case PthreadMutexType::ErrorCheck:
    case PthreadMutexType::AdaptiveNp: {
        if (abstime) {
            return DoSleep();
        }
        /*
         * POSIX specifies that mutexes should return
         * EDEADLK if a recursive lock is detected.
         */
        return POSIX_EDEADLK;
    }
    case PthreadMutexType::Normal: {
        /*
         * What SS2 define as a 'normal' mutex.  Intentionally
         * deadlock on attempts to get a lock you already own.
         */
        if (abstime) {
            return DoSleep();
        }
        UNREACHABLE_MSG("Mutex deadlock occured");
        return 0;
    }
    case PthreadMutexType::Recursive: {
        /* Increment the lock count: */
        if (m_count + 1 > 0) {
            m_count++;
            return 0;
        }
        return POSIX_EAGAIN;
    }
    default:
        return POSIX_EINVAL;
    }
}

s32 PthreadMutex::Lock(const OrbisKernelTimespec* abstime, u64 usec) {
    Pthread* curthread = g_curthread;
    if (m_owner == curthread) {
        return SelfLock(abstime, usec);
    }

    /*
     * For adaptive mutexes, spin for a bit in the expectation
     * that if the application requests this mutex type then
     * the lock is likely to be released quickly and it is
     * faster than entering the kernel
     */
    if (m_protocol == PthreadMutexProt::None) [[likely]] {
        s32 count = m_spinloops;
        while (count--) {
            if (m_lock.try_lock()) {
                m_owner = curthread;
                return 0;
            }
            CPU_SPINWAIT;
        }

        count = m_yieldloops;
        while (count--) {
            std::this_thread::yield();
            if (m_lock.try_lock()) {
                m_owner = curthread;
                return 0;
            }
        }
    }

    s32 ret = 0;
    if (abstime == nullptr) {
        m_lock.lock();
    } else if (abstime != THR_RELTIME && (abstime->tv_nsec < 0 || abstime->tv_nsec >= 1000000000))
        [[unlikely]] {
        ret = POSIX_EINVAL;
    } else {
        if (abstime == THR_RELTIME) {
            ret = m_lock.try_lock_for(std::chrono::microseconds(usec)) ? 0 : POSIX_ETIMEDOUT;
        } else {
            ret = m_lock.try_lock_until(abstime->TimePoint()) ? 0 : POSIX_ETIMEDOUT;
        }
    }
    if (ret == 0) {
        m_owner = curthread;
    }
    return ret;
}

s32 PthreadMutex::TryLock() {
    Pthread* curthread = g_curthread;
    if (m_owner == curthread) {
        return SelfTryLock();
    }
    const s32 ret = m_lock.try_lock() ? 0 : POSIX_EBUSY;
    if (ret == 0) {
        m_owner = curthread;
    }
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_trylock(PthreadMutexT* mutex) {
    MTX_DIAG("posix_pthread_mutex_trylock", mutex);
    CHECK_AND_INIT_MUTEX
    return (*mutex)->TryLock();
}

s32 PS4_SYSV_ABI posix_pthread_mutex_lock(PthreadMutexT* mutex) {
    MTX_DIAG("posix_pthread_mutex_lock", mutex);
    CHECK_AND_INIT_MUTEX
    return (*mutex)->Lock(nullptr);
}

s32 PS4_SYSV_ABI posix_pthread_mutex_timedlock(PthreadMutexT* mutex,
                                               const OrbisKernelTimespec* abstime) {
    MTX_DIAG("posix_pthread_mutex_timedlock", mutex);
    CHECK_AND_INIT_MUTEX
    return (*mutex)->Lock(abstime);
}

s32 PS4_SYSV_ABI posix_pthread_mutex_reltimedlock_np(PthreadMutexT* mutex, u64 usec) {
    MTX_DIAG("posix_pthread_mutex_reltimedlock_np", mutex);
    CHECK_AND_INIT_MUTEX
    return (*mutex)->Lock(THR_RELTIME, usec);
}

s32 PthreadMutex::Unlock() {
    Pthread* curthread = g_curthread;
    /*
     * Check if the running thread is not the owner of the mutex.
     */
    if (m_owner != curthread) [[unlikely]] {
        return POSIX_EPERM;
    }

    if (Type() == PthreadMutexType::Recursive && m_count > 0) [[unlikely]] {
        m_count--;
    } else {
        const bool deferred = True(m_flags & PthreadMutexFlags::Deferred);
        m_flags &= ~PthreadMutexFlags::Deferred;

        m_owner = nullptr;
        m_lock.unlock();

        if (curthread->will_sleep == 0 && deferred) {
            curthread->WakeAll();
        }
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_unlock(PthreadMutexT* mutex) {
    PthreadMutex* mp = *mutex;
    if (mp <= THR_MUTEX_DESTROYED) [[unlikely]] {
        if (mp == THR_MUTEX_DESTROYED) {
            return POSIX_EINVAL;
        }
        return POSIX_EPERM;
    }
    return mp->Unlock();
}

s32 PS4_SYSV_ABI posix_pthread_mutex_getspinloops_np(PthreadMutexT* mutex, int* count) {
    CHECK_AND_INIT_MUTEX
    *count = (*mutex)->m_spinloops;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_setspinloops_np(PthreadMutexT* mutex, s32 count) {
    CHECK_AND_INIT_MUTEX(*mutex)->m_spinloops = count;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_getyieldloops_np(PthreadMutexT* mutex, int* count) {
    CHECK_AND_INIT_MUTEX
    *count = (*mutex)->m_yieldloops;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_setyieldloops_np(PthreadMutexT* mutex, s32 count) {
    CHECK_AND_INIT_MUTEX(*mutex)->m_yieldloops = count;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_isowned_np(PthreadMutexT* mutex) {
    PthreadMutex* m = *mutex;
    if (m <= THR_MUTEX_DESTROYED) {
        return 0;
    }
    return m->m_owner == g_curthread;
}

s32 PthreadMutex::IsOwned(Pthread* curthread) const {
    if (this <= THR_MUTEX_DESTROYED) [[unlikely]] {
        if (this == THR_MUTEX_DESTROYED) {
            return POSIX_EINVAL;
        }
        return POSIX_EPERM;
    }
    if (m_owner != curthread) {
        return POSIX_EPERM;
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_init(PthreadMutexAttrT* attr) {
    auto pattr = new (std::nothrow) PthreadMutexAttr{};
    if (pattr == nullptr) {
        return POSIX_ENOMEM;
    }
    memcpy(pattr, &PthreadMutexattrDefault, sizeof(PthreadMutexAttr));
    *attr = pattr;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setkind_np(PthreadMutexAttrT* attr,
                                                    PthreadMutexType kind) {
    if (attr == nullptr || *attr == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    (*attr)->m_type = kind;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_getkind_np(PthreadMutexAttrT attr) {
    if (attr == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    return static_cast<int>(attr->m_type);
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_settype(PthreadMutexAttrT* attr, PthreadMutexType type) {
    if (attr == nullptr || *attr == nullptr || type < PthreadMutexType::ErrorCheck ||
        type >= PthreadMutexType::Max) {
        return POSIX_EINVAL;
    }
    (*attr)->m_type = type;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_gettype(PthreadMutexAttrT* attr, PthreadMutexType* type) {
    if (attr == nullptr || *attr == nullptr || (*attr)->m_type >= PthreadMutexType::Max) {
        return POSIX_EINVAL;
    }
    *type = (*attr)->m_type;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_destroy(PthreadMutexAttrT* attr) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    delete *attr;
    *attr = nullptr;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_getprotocol(PthreadMutexAttrT* mattr,
                                                     PthreadMutexProt* protocol) {
    if (mattr == nullptr || *mattr == nullptr) {
        return POSIX_EINVAL;
    }
    *protocol = (*mattr)->m_protocol;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setprotocol(PthreadMutexAttrT* mattr,
                                                     PthreadMutexProt protocol) {
    if (mattr == nullptr || *mattr == nullptr || (protocol < PthreadMutexProt::None) ||
        (protocol > PthreadMutexProt::Protect)) {
        return POSIX_EINVAL;
    }
    (*mattr)->m_protocol = protocol;
    //(*mattr)->m_ceiling = THR_MAX_RR_PRIORITY;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setpshared(PthreadMutexAttrT* attr, s32 pshared) {
    constexpr s32 POSIX_PTHREAD_PROCESS_PRIVATE = 0;
    constexpr s32 POSIX_PTHREAD_PROCESS_SHARED = 1;
    if (!attr || !*attr || pshared != POSIX_PTHREAD_PROCESS_PRIVATE) {
        return POSIX_EINVAL;
    }
    return ORBIS_OK;
}

void RegisterMutex(Core::Loader::SymbolsResolver* sym) {
    // Posix
    LIB_FUNCTION("ttHNfU+qDBU", "libScePosix", 1, "libkernel", posix_pthread_mutex_init);
    LIB_FUNCTION("7H0iTOciTLo", "libScePosix", 1, "libkernel", posix_pthread_mutex_lock);
    LIB_FUNCTION("Io9+nTKXZtA", "libScePosix", 1, "libkernel", posix_pthread_mutex_timedlock);
    LIB_FUNCTION("2Z+PpY6CaJg", "libScePosix", 1, "libkernel", posix_pthread_mutex_unlock);
    LIB_FUNCTION("ltCfaGr2JGE", "libScePosix", 1, "libkernel", posix_pthread_mutex_destroy);
    LIB_FUNCTION("dQHWEsJtoE4", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_init);
    LIB_FUNCTION("mDmgMOGVUqg", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_settype);
    LIB_FUNCTION("5txKfcMUAok", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_setprotocol);
    LIB_FUNCTION("HF7lK46xzjY", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_destroy);
    LIB_FUNCTION("K-jXhbt2gn4", "libScePosix", 1, "libkernel", posix_pthread_mutex_trylock);
    LIB_FUNCTION("EXv3ztGqtDM", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_setpshared);

    // Posix-Kernel
    LIB_FUNCTION("ttHNfU+qDBU", "libkernel", 1, "libkernel", posix_pthread_mutex_init);
    LIB_FUNCTION("7H0iTOciTLo", "libkernel", 1, "libkernel", posix_pthread_mutex_lock);
    LIB_FUNCTION("2Z+PpY6CaJg", "libkernel", 1, "libkernel", posix_pthread_mutex_unlock);
    LIB_FUNCTION("ltCfaGr2JGE", "libkernel", 1, "libkernel", posix_pthread_mutex_destroy);
    LIB_FUNCTION("dQHWEsJtoE4", "libkernel", 1, "libkernel", posix_pthread_mutexattr_init);
    LIB_FUNCTION("mDmgMOGVUqg", "libkernel", 1, "libkernel", posix_pthread_mutexattr_settype);
    LIB_FUNCTION("HF7lK46xzjY", "libkernel", 1, "libkernel", posix_pthread_mutexattr_destroy);
    LIB_FUNCTION("K-jXhbt2gn4", "libkernel", 1, "libkernel", posix_pthread_mutex_trylock);
    LIB_FUNCTION("EXv3ztGqtDM", "libkernel", 1, "libkernel", posix_pthread_mutexattr_setpshared);

    // Orbis
    LIB_FUNCTION("cmo1RIYva9o", "libkernel", 1, "libkernel", ORBIS(scePthreadMutexInit));
    LIB_FUNCTION("2Of0f+3mhhE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_destroy));
    LIB_FUNCTION("F8bUHwAG284", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutexattr_init));
    LIB_FUNCTION("smWEktiyyG0", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_destroy));
    LIB_FUNCTION("iMp8QpE+XO4", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_settype));
    LIB_FUNCTION("1FGvU0i9saQ", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_setprotocol));
    LIB_FUNCTION("9UK1vLZQft4", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_lock));
    LIB_FUNCTION("tn3VlD0hG60", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_unlock));
    LIB_FUNCTION("upoVrzMHFeE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_trylock));
    LIB_FUNCTION("IafI2PxcPnQ", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_reltimedlock_np));
    LIB_FUNCTION("qH1gXoq71RY", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_init));
    LIB_FUNCTION("n2MMpvU8igI", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutexattr_init));
}

} // namespace Libraries::Kernel
