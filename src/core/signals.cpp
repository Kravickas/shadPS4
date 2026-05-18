// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdio>
#include <filesystem>

#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/logging/log.h"
#include "common/signal_context.h"
#include "common/singleton.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/linker.h"
#include "core/module.h"
#include "core/signals.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

#ifndef _WIN32
namespace Libraries::Kernel {
void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context);
extern std::array<OrbisKernelExceptionHandler, 32> Handlers;
} // namespace Libraries::Kernel
#endif

namespace Core {

#if defined(_WIN32)

// Identifies which module a code address belongs to. PS4 modules (eboot.bin and
// loaded PRX libraries) are mapped via VirtualAlloc rather than LoadLibrary, so they
// are invisible to Win32 module enumeration APIs. We must consult shadPS4's own
// Linker module list for those, then fall back to host Win32 modules for DLLs.
static std::string IdentifyModule(void* address) {
    const VAddr addr = reinterpret_cast<VAddr>(address);

    // 1. Try shadPS4 PS4 module list first.
    if (auto* linker = Common::Singleton<Core::Linker>::Instance()) {
        if (auto* module = linker->FindByAddress(addr)) {
            const VAddr base = module->GetBaseAddress();
            const u64 offset = addr - base;
            const std::string& name =
                !module->name.empty() ? module->name : module->file.filename().string();
            return fmt::format("{}+{:#x} (base {:#x})", name, offset, base);
        }
    }

    // 2. Fall back to host Win32 modules (for shadPS4 itself, system DLLs).
    HMODULE host_module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &host_module)) {
        char path[MAX_PATH] = {};
        if (GetModuleFileNameA(host_module, path, sizeof(path)) > 0) {
            const u64 offset = addr - reinterpret_cast<VAddr>(host_module);
            const std::string filename = std::filesystem::path(path).filename().string();
            return fmt::format("{}+{:#x} (base {})", filename, offset, fmt::ptr(host_module));
        }
    }

    return "<unmapped or unknown>";
}

// Reads sizeof(u64) from `addr` into `out`. Returns false on access fault.
// Used by post-mortem stack/code dumpers, which must not double-fault if the
// target address is corrupted or unmapped.
static bool SafeReadU64(const void* addr, u64& out) noexcept {
    __try {
        out = *static_cast<const volatile u64*>(addr);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

static bool SafeReadU8(const void* addr, u8& out) noexcept {
    __try {
        out = *static_cast<const volatile u8*>(addr);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// Dumps an `n`-qword window of stack as hex on a single log line. `offset_first`
// is the byte offset (negative or positive) from `rsp` of the FIRST qword printed.
static void DumpStackRange(VAddr rsp, s64 offset_first, int count, const char* label) {
    char hex[64 * 17 + 64] = {};
    char* p = hex;
    char* const end = hex + sizeof(hex);

    for (int i = 0; i < count; ++i) {
        const auto addr = rsp + offset_first + static_cast<s64>(i) * 8;
        u64 value = 0;
        const bool ok = SafeReadU64(reinterpret_cast<const void*>(addr), value);
        const int n = std::snprintf(p, static_cast<size_t>(end - p), "%016llx ",
                                    ok ? static_cast<unsigned long long>(value) : 0ULL);
        if (n <= 0 || n >= end - p) {
            break;
        }
        p += n;
    }
    LOG_CRITICAL(Common, "{} (rsp{:+#x}, {} qwords): {}", label, offset_first, count, hex);
}

// Dumps `n` bytes of code at `addr`, useful to verify that no in-memory patching
// occurred between AOT decoding and execution. Bloodborne's crash function has
// no instructions in shadPS4's JIT-patcher set (no SSE4a, no FS-segment), so
// these bytes should match the eboot file exactly.
static void DumpCodeBytes(VAddr addr, int count, const char* label) {
    char hex[256 * 4 + 64] = {};
    char* p = hex;
    char* const end = hex + sizeof(hex);

    for (int i = 0; i < count; ++i) {
        u8 byte = 0;
        const bool ok = SafeReadU8(reinterpret_cast<const void*>(addr + i), byte);
        const int n = std::snprintf(p, static_cast<size_t>(end - p), "%02x ", ok ? byte : 0);
        if (n <= 0 || n >= end - p) {
            break;
        }
        p += n;
    }
    LOG_CRITICAL(Common, "{} ({:#x}, {} bytes): {}", label, static_cast<unsigned long long>(addr),
                 count, hex);
}

static void DumpPostMortem(const CONTEXT* ctx) {
    const auto rsp = ctx->Rsp;
    const auto rip = ctx->Rip;

    DumpStackRange(rsp, -256, 16, "RedZoneFar ");
    DumpStackRange(rsp, -128, 16, "RedZoneNear");
    DumpStackRange(rsp, 0, 12, "SavedRegs  ");
    DumpCodeBytes(rip - 32, 48, "CodeBytes  ");

    LOG_CRITICAL(
        Common,
        "Regs: Rax={:#018x} Rbx={:#018x} Rcx={:#018x} Rdx={:#018x} "
        "Rsi={:#018x} Rdi={:#018x} Rbp={:#018x} Rsp={:#018x}",
        static_cast<unsigned long long>(ctx->Rax), static_cast<unsigned long long>(ctx->Rbx),
        static_cast<unsigned long long>(ctx->Rcx), static_cast<unsigned long long>(ctx->Rdx),
        static_cast<unsigned long long>(ctx->Rsi), static_cast<unsigned long long>(ctx->Rdi),
        static_cast<unsigned long long>(ctx->Rbp), static_cast<unsigned long long>(ctx->Rsp));
    LOG_CRITICAL(
        Common,
        "Regs: R8 ={:#018x} R9 ={:#018x} R10={:#018x} R11={:#018x} "
        "R12={:#018x} R13={:#018x} R14={:#018x} R15={:#018x}",
        static_cast<unsigned long long>(ctx->R8), static_cast<unsigned long long>(ctx->R9),
        static_cast<unsigned long long>(ctx->R10), static_cast<unsigned long long>(ctx->R11),
        static_cast<unsigned long long>(ctx->R12), static_cast<unsigned long long>(ctx->R13),
        static_cast<unsigned long long>(ctx->R14), static_cast<unsigned long long>(ctx->R15));
}

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    bool handled = false;
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions. Always continuable.
        return EXCEPTION_CONTINUE_EXECUTION;
    default:
        break;
    }

    if (!handled) {
        const auto code = pExp->ExceptionRecord->ExceptionCode;
        const auto* rec = pExp->ExceptionRecord;
        void* fault_addr = nullptr;
        const char* kind = "unknown";
        if (code == EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters >= 2) {
            fault_addr = reinterpret_cast<void*>(rec->ExceptionInformation[1]);
            kind = rec->ExceptionInformation[0] == 0 ? "read" : "write";
        }
        LOG_CRITICAL(Common,
                     "Unhandled Windows exception {:#010x} at RIP {} ({} fault at {}). "
                     "Process will terminate.",
                     static_cast<u32>(code), fmt::ptr(rec->ExceptionAddress), kind,
                     fmt::ptr(fault_addr));
        LOG_CRITICAL(Common, "Module: {}", IdentifyModule(rec->ExceptionAddress));
        DumpPostMortem(pExp->ContextRecord);
    }

    return handled ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

#else

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            // If the guest has installed a custom signal handler, and the access violation didn't
            // come from HLE memory tracking, pass the signal on
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (!signals->DispatchIllegalInstruction(raw_context)) {
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address));
        }
        break;
    default:
        if (sig == SIGSLEEP) {
            // Sleep thread until signal is received again
            sigset_t sigset;
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGSLEEP);
            sigwait(&sigset, &sig);
        }
        break;
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action {};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to register access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
    ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action {};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
