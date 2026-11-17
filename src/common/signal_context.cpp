// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/signal_context.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__FreeBSD__)
#include <machine/npx.h>
#include <sys/ucontext.h>
#else
#include <sys/ucontext.h>
#endif

namespace Common {

void* GetRip(void* ctx) {
#if defined(_WIN32)
    return (void*)((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rip;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext->__ss.__rip;
#elif defined(__APPLE__) && defined(ARCH_ARM64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext->__ss.__pc;
#elif defined(__FreeBSD__)
    return (void*)((ucontext_t*)ctx)->uc_mcontext.mc_rip;
#elif defined(ARCH_X86_64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP];
#else
#error "Unsupported architecture"
#endif
}

u64* GetGprPointer(void* ctx, u8 index) {
    if (index > 15) {
        return nullptr;
    }
#if defined(_WIN32)
    auto* const context = ((EXCEPTION_POINTERS*)ctx)->ContextRecord;
    DWORD64* const registers[] = {
        &context->Rax, &context->Rcx, &context->Rdx, &context->Rbx, &context->Rsp, &context->Rbp,
        &context->Rsi, &context->Rdi, &context->R8,  &context->R9,  &context->R10, &context->R11,
        &context->R12, &context->R13, &context->R14, &context->R15,
    };
    return reinterpret_cast<u64*>(registers[index]);
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    auto& ss = ((ucontext_t*)ctx)->uc_mcontext->__ss;
    __uint64_t* const registers[] = {
        &ss.__rax, &ss.__rcx, &ss.__rdx, &ss.__rbx, &ss.__rsp, &ss.__rbp, &ss.__rsi, &ss.__rdi,
        &ss.__r8,  &ss.__r9,  &ss.__r10, &ss.__r11, &ss.__r12, &ss.__r13, &ss.__r14, &ss.__r15,
    };
    return reinterpret_cast<u64*>(registers[index]);
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    auto& mc = ((ucontext_t*)ctx)->uc_mcontext;
    __register_t* const registers[] = {
        &mc.mc_rax, &mc.mc_rcx, &mc.mc_rdx, &mc.mc_rbx, &mc.mc_rsp, &mc.mc_rbp,
        &mc.mc_rsi, &mc.mc_rdi, &mc.mc_r8,  &mc.mc_r9,  &mc.mc_r10, &mc.mc_r11,
        &mc.mc_r12, &mc.mc_r13, &mc.mc_r14, &mc.mc_r15,
    };
    return reinterpret_cast<u64*>(registers[index]);
#elif defined(ARCH_X86_64)
    static constexpr int registers[] = {
        REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
        REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15,
    };
    return reinterpret_cast<u64*>(&((ucontext_t*)ctx)->uc_mcontext.gregs[registers[index]]);
#else
    return nullptr;
#endif
}

void* GetXmmRegisterPointer(void* ctx, u8 index) {
    if (index > 15) {
        return nullptr;
    }
#if defined(_WIN32)
    return &(&((EXCEPTION_POINTERS*)ctx)->ContextRecord->Xmm0)[index];
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return &(&((ucontext_t*)ctx)->uc_mcontext->__fs.__fpu_xmm0)[index];
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    auto& mc = ((ucontext_t*)ctx)->uc_mcontext;
    if (mc.mc_fpformat != _MC_FPFMT_XMM) {
        return nullptr;
    }
    return &((struct savefpu*)(&mc.mc_fpstate[0]))->sv_xmm[index];
#elif defined(ARCH_X86_64)
    return &((ucontext_t*)ctx)->uc_mcontext.fpregs->_xmm[index];
#else
    return nullptr;
#endif
}

u64 GetFlags(void* ctx) {
#if defined(_WIN32)
    return ((EXCEPTION_POINTERS*)ctx)->ContextRecord->EFlags;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext->__ss.__rflags;
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.mc_rflags;
#elif defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_EFL];
#else
    return 0;
#endif
}

void SetFlags(void* ctx, u64 flags) {
#if defined(_WIN32)
    ((EXCEPTION_POINTERS*)ctx)->ContextRecord->EFlags = static_cast<DWORD>(flags);
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext->__ss.__rflags = flags;
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext.mc_rflags = flags;
#elif defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_EFL] = flags;
#endif
}

void SkipInstruction(void* ctx, u64 length) {
#if defined(_WIN32)
    ((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rip += length;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext->__ss.__rip += length;
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext.mc_rip += length;
#elif defined(ARCH_X86_64)
    ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP] += length;
#endif
}

bool IsWriteError(void* ctx) {
#if defined(_WIN32)
    return ((EXCEPTION_POINTERS*)ctx)->ExceptionRecord->ExceptionInformation[0] == 1;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__err & 0x2;
#elif defined(__APPLE__) && defined(ARCH_ARM64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__esr & 0x40;
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.mc_err & 0x2;
#elif defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_ERR] & 0x2;
#else
#error "Unsupported architecture"
#endif
}

bool IsExecuteError(void* ctx) {
#if defined(_WIN32)
    return ((EXCEPTION_POINTERS*)ctx)->ExceptionRecord->ExceptionInformation[0] == 0xf;
#elif defined(__APPLE__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__err & 0x10;
#elif defined(__FreeBSD__) && defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.mc_err & 0x10;
#elif defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_ERR] & 0x10;
#else
#error "Unsupported architecture"
#endif
}

} // namespace Common
