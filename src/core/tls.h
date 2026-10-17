// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include "common/types.h"
#ifdef _WIN32
#include <malloc.h>
#endif

namespace Xbyak {
class CodeGenerator;
}

namespace Libraries::Fiber {
struct OrbisFiberContext;
}

namespace Core {

union DtvEntry {
    std::size_t counter;
    u8* pointer;
};

struct Tcb {
    Tcb* tcb_self;
    DtvEntry* tcb_dtv;
    void* tcb_thread;
    void* tcb_spare[2];
    u64 tcb_canary;
    ::Libraries::Fiber::OrbisFiberContext* tcb_fiber;
};

#ifdef _WIN32
/// Gets the thread local storage key for the TCB block.
u32 GetTcbKey();
#endif

/// Sets the data pointer to the TCB block.
void SetTcbBase(void* image_address);

/// Retrieves Tcb structure for the calling thread.
Tcb* GetTcbBase();

/// Makes sure TLS is initialized for the thread before entering guest.
void InitializeTLS();

template <auto f>
struct HostCallWrapperImpl;

template <class ReturnType, class... Args, PS4_SYSV_ABI ReturnType (*func)(Args...)>
struct HostCallWrapperImpl<func> {
    static ReturnType PS4_SYSV_ABI wrap(Args... args) {
        return func(args...);
    }
};

#define HOST_CALL(func) (Core::HostCallWrapperImpl<func>::wrap)

template <auto f>
struct HostCallCallerWrapperImpl;

/// Like HostCallWrapperImpl, and passes the guest return address of the call as the first
/// argument. The guest calls wrap directly, so its return address is the guest's call site.
template <class ReturnType, class... Args, PS4_SYSV_ABI ReturnType (*func)(VAddr, Args...)>
struct HostCallCallerWrapperImpl<func> {
    static ReturnType PS4_SYSV_ABI wrap(Args... args) {
        return func(reinterpret_cast<VAddr>(__builtin_return_address(0)), args...);
    }
};

#define HOST_CALL_WITH_CALLER(func) (Core::HostCallCallerWrapperImpl<func>::wrap)

} // namespace Core
