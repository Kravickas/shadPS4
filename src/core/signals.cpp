// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <fmt/format.h>
#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/logging/log.h"
#include "common/signal_context.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection
#include "common/string_util.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/linker.h"
#include "core/signals.h"
#include "emulator.h"

#ifdef _WIN32
#include <windows.h>
static constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;
#else
#include <csignal>
#include <pthread.h>
#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif
#endif

namespace Core {

#if defined(_WIN32)

namespace {

constexpr size_t MaxInstructionBytes = 16;
constexpr size_t MaxBacktraceFrames = 32;
constexpr size_t StackDumpBytes = 256;
constexpr size_t GprDumpBytes = 128;
constexpr size_t ChaseDumpBytes = 64;
constexpr size_t MaxChaseTargets = 24;
constexpr size_t ChasePerRegister = 3;
constexpr size_t MaxScanSlots = 64;
constexpr DWORD PageExecutable =
    PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
constexpr DWORD PageReadable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;

thread_local bool in_crash_report = false;

size_t ReadableSpan(const void* address, size_t wanted) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(address, &mbi, sizeof(mbi)) != sizeof(mbi)) {
        return 0;
    }
    if (mbi.State != MEM_COMMIT || (mbi.Protect & PageReadable) == 0 ||
        (mbi.Protect & PAGE_GUARD) != 0) {
        return 0;
    }
    const auto region_end = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
    return std::min<size_t>(wanted, region_end - reinterpret_cast<uintptr_t>(address));
}

std::string DescribeRegion(const void* address) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(address, &mbi, sizeof(mbi)) != sizeof(mbi)) {
        return "<query failed>";
    }
    // Protect and Type are undefined for MEM_FREE, Protect is undefined for MEM_RESERVE.
    if (mbi.State == MEM_FREE) {
        return fmt::format("free size={:#x}", mbi.RegionSize);
    }
    if (mbi.State == MEM_RESERVE) {
        return fmt::format("reserved alloc_base={} size={:#x} type={:#x}", mbi.AllocationBase,
                           mbi.RegionSize, mbi.Type);
    }
    return fmt::format("committed alloc_base={} size={:#x} prot={:#x} type={:#x}",
                       mbi.AllocationBase, mbi.RegionSize, mbi.Protect, mbi.Type);
}

std::string DescribeModule(const void* address) {
    HMODULE module{};
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCWSTR>(address), &module) ||
        module == nullptr) {
        return {};
    }
    wchar_t path[MAX_PATH]{};
    const DWORD length = GetModuleFileNameW(module, path, MAX_PATH);
    const auto offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(module);
    if (length == 0) {
        return fmt::format("{}+{:#x}", static_cast<const void*>(module), offset);
    }
    std::wstring_view name{path, length};
    if (const auto slash = name.find_last_of(L"\\/"); slash != std::wstring_view::npos) {
        name.remove_prefix(slash + 1);
    }
    return fmt::format("{}+{:#x}", Common::UTF16ToUTF8(name), offset);
}

std::string DescribeGuest(const void* address) {
    auto* linker = Common::Singleton<Linker>::Instance();
    if (linker == nullptr) {
        return {};
    }
    const auto guest_addr = reinterpret_cast<VAddr>(address);
    auto* module = linker->FindByAddress(guest_addr);
    if (module == nullptr) {
        return {};
    }
    return fmt::format("{}+{:#x}", module->name, guest_addr - module->GetBaseAddress());
}

std::string DescribeCode(const void* address) {
    if (auto host = DescribeModule(address); !host.empty()) {
        return host;
    }
    if (auto guest = DescribeGuest(address); !guest.empty()) {
        return guest;
    }
    return "<unknown>";
}

std::string DumpBytes(const void* address, size_t count) {
    const auto* bytes = static_cast<const u8*>(address);
    std::string out;
    out.reserve(count * 3);
    for (size_t i = 0; i < count; ++i) {
        out += fmt::format("{:02x} ", bytes[i]);
    }
    return out;
}

#ifdef ARCH_X86_64
bool IsExecutable(uintptr_t address) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &mbi, sizeof(mbi)) != sizeof(mbi)) {
        return false;
    }
    return mbi.State == MEM_COMMIT && (mbi.Protect & PageExecutable) != 0 &&
           (mbi.Protect & PAGE_GUARD) == 0;
}

void DumpMemory(const char* label, uintptr_t address, size_t wanted) {
    const auto readable = ReadableSpan(reinterpret_cast<const void*>(address), wanted);
    if (readable == 0) {
        return;
    }
    const auto* bytes = reinterpret_cast<const u8*>(address);
    for (size_t offset = 0; offset < readable; offset += 16) {
        const auto count = std::min<size_t>(16, readable - offset);
        std::string hex;
        std::string ascii;
        for (size_t i = 0; i < count; ++i) {
            const u8 value = bytes[offset + i];
            hex += fmt::format("{:02x} ", value);
            ascii += value >= 0x20 && value < 0x7f ? static_cast<char>(value) : '.';
        }
        LOG_CRITICAL(Debug, "  {:<3} {:#018x}  {:<48}{}", label, address + offset, hex, ascii);
    }
}

// Collects up to ChasePerRegister pointers from one register's dump. The per-register budget
// matters: a single stack-pointing register otherwise fills the whole table with spilled scalars
// and starves the later registers, which are the ones usually holding the object of interest.
size_t CollectPointers(uintptr_t address, size_t length, uintptr_t* targets, size_t max_targets,
                       size_t found) {
    const auto* words = reinterpret_cast<const u64*>(address);
    size_t taken = 0;
    for (size_t i = 0; i < length / sizeof(u64) && found < max_targets; ++i) {
        if (taken >= ChasePerRegister) {
            break;
        }
        const auto candidate = static_cast<uintptr_t>(words[i]);
        if (ReadableSpan(reinterpret_cast<const void*>(candidate), ChaseDumpBytes) <
            ChaseDumpBytes) {
            continue;
        }
        bool duplicate = false;
        for (size_t j = 0; j < found; ++j) {
            duplicate = duplicate || targets[j] == candidate;
        }
        if (!duplicate) {
            targets[found++] = candidate;
            ++taken;
        }
    }
    return found;
}

std::string Disassemble(const void* address, size_t size) {
    ZydisDecodedInstruction inst;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    auto* decoder = Common::Decoder::Instance();
    if (!ZYAN_SUCCESS(
            decoder->decodeInstruction(inst, operands, const_cast<void*>(address), size))) {
        return "<decode failed>";
    }
    return decoder->disassembleInst(inst, operands, reinterpret_cast<u64>(address));
}

enum class FrameSource { Exact, Unwind, FramePointer, StackScan };

struct Frame {
    u64 pc;
    FrameSource source;
};

char SourceMark(FrameSource source) {
    switch (source) {
    case FrameSource::Exact:
        return ' ';
    case FrameSource::Unwind:
        return ' ';
    case FrameSource::FramePointer:
        return '+';
    default:
        return '?';
    }
}

// Guest modules carry no Windows unwind data, so RtlLookupFunctionEntry always misses on them.
// Fall back to the frame pointer chain, and only if that fails to scanning the stack for a
// plausible return address. Frames are tagged with how they were obtained so a guess is never
// presented as a real unwind.
size_t CaptureBacktrace(const CONTEXT& initial, Frame* frames, size_t max_frames) {
    CONTEXT ctx = initial;
    size_t count = 0;
    frames[count++] = {ctx.Rip, FrameSource::Exact};

    while (count < max_frames && ctx.Rip != 0) {
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(ctx.Rip, &image_base, nullptr);
        if (entry != nullptr) {
            PVOID handler_data = nullptr;
            DWORD64 establisher_frame = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, ctx.Rip, entry, &ctx, &handler_data,
                             &establisher_frame, nullptr);
            if (ctx.Rip == 0) {
                break;
            }
            frames[count++] = {ctx.Rip, FrameSource::Unwind};
            continue;
        }

        const auto frame_ptr = static_cast<uintptr_t>(ctx.Rbp);
        const auto* saved = reinterpret_cast<const u64*>(frame_ptr);
        if (frame_ptr >= ctx.Rsp && ReadableSpan(saved, 2 * sizeof(u64)) == 2 * sizeof(u64) &&
            static_cast<uintptr_t>(saved[0]) > frame_ptr && IsExecutable(saved[1])) {
            ctx.Rip = saved[1];
            ctx.Rsp = frame_ptr + 2 * sizeof(u64);
            ctx.Rbp = saved[0];
            frames[count++] = {ctx.Rip, FrameSource::FramePointer};
            continue;
        }

        bool found = false;
        for (size_t slot = 0; slot < MaxScanSlots && !found; ++slot) {
            const auto address = ctx.Rsp + slot * sizeof(u64);
            const auto* word = reinterpret_cast<const u64*>(address);
            if (ReadableSpan(word, sizeof(u64)) < sizeof(u64)) {
                break;
            }
            if (!IsExecutable(*word)) {
                continue;
            }
            ctx.Rip = *word;
            ctx.Rsp = address + sizeof(u64);
            frames[count++] = {ctx.Rip, FrameSource::StackScan};
            found = true;
        }
        if (!found) {
            break;
        }
    }
    return count;
}
#endif

void LogCrashContext(EXCEPTION_POINTERS* pExp, DWORD code, const void* address) {
    if (in_crash_report || pExp == nullptr || pExp->ExceptionRecord == nullptr ||
        pExp->ContextRecord == nullptr) {
        return;
    }
    in_crash_report = true;

    const EXCEPTION_RECORD& record = *pExp->ExceptionRecord;
    LOG_CRITICAL(Debug, "  rip          {} {}", address, DescribeCode(address));
    LOG_CRITICAL(Debug, "  rip region   {}", DescribeRegion(address));

    if (code == EXCEPTION_ACCESS_VIOLATION && record.NumberParameters >= 2) {
        const auto operation = record.ExceptionInformation[0];
        const auto* fault = reinterpret_cast<const void*>(record.ExceptionInformation[1]);
        const char* kind = operation == 0   ? "read"
                           : operation == 1 ? "write"
                           : operation == 8 ? "execute"
                                            : "unknown";
        LOG_CRITICAL(Debug, "  fault        {} at {} {}", kind, fault, DescribeCode(fault));
        LOG_CRITICAL(Debug, "  fault region {}", DescribeRegion(fault));
        if (record.NumberParameters >= 3) {
            LOG_CRITICAL(Debug, "  ntstatus     {:#x}", record.ExceptionInformation[2]);
        }
    }

    if (const auto readable = ReadableSpan(address, MaxInstructionBytes); readable > 0) {
        LOG_CRITICAL(Debug, "  bytes        {}", DumpBytes(address, readable));
#ifdef ARCH_X86_64
        LOG_CRITICAL(Debug, "  inst         {}", Disassemble(address, readable));
#endif
    } else {
        LOG_CRITICAL(Debug, "  bytes        <not readable>");
    }

#ifdef ARCH_X86_64
    const CONTEXT& ctx = *pExp->ContextRecord;
    LOG_CRITICAL(Debug, "  rax {:016x} rbx {:016x} rcx {:016x} rdx {:016x}", ctx.Rax, ctx.Rbx,
                 ctx.Rcx, ctx.Rdx);
    LOG_CRITICAL(Debug, "  rsi {:016x} rdi {:016x} rbp {:016x} rsp {:016x}", ctx.Rsi, ctx.Rdi,
                 ctx.Rbp, ctx.Rsp);
    LOG_CRITICAL(Debug, "  r8  {:016x} r9  {:016x} r10 {:016x} r11 {:016x}", ctx.R8, ctx.R9,
                 ctx.R10, ctx.R11);
    LOG_CRITICAL(Debug, "  r12 {:016x} r13 {:016x} r14 {:016x} r15 {:016x}", ctx.R12, ctx.R13,
                 ctx.R14, ctx.R15);
    LOG_CRITICAL(Debug, "  rip {:016x} eflags {:08x}", ctx.Rip, ctx.EFlags);
#endif
    Common::Log::Flush();

#ifdef ARCH_X86_64
    DumpMemory("rsp", static_cast<uintptr_t>(ctx.Rsp), StackDumpBytes);

    const std::array<std::pair<const char*, DWORD64>, 15> gprs{{
        {"rax", ctx.Rax},
        {"rbx", ctx.Rbx},
        {"rcx", ctx.Rcx},
        {"rdx", ctx.Rdx},
        {"rsi", ctx.Rsi},
        {"rdi", ctx.Rdi},
        {"rbp", ctx.Rbp},
        {"r8", ctx.R8},
        {"r9", ctx.R9},
        {"r10", ctx.R10},
        {"r11", ctx.R11},
        {"r12", ctx.R12},
        {"r13", ctx.R13},
        {"r14", ctx.R14},
        {"r15", ctx.R15},
    }};
    std::array<uintptr_t, MaxChaseTargets> chase{};
    size_t chase_count = 0;
    for (const auto& [name, value] : gprs) {
        const auto base = static_cast<uintptr_t>(value);
        const auto readable = ReadableSpan(reinterpret_cast<const void*>(base), GprDumpBytes);
        if (readable == 0) {
            continue;
        }
        DumpMemory(name, base, GprDumpBytes);
        chase_count = CollectPointers(base, readable, chase.data(), chase.size(), chase_count);
    }
    for (size_t i = 0; i < chase_count; ++i) {
        DumpMemory("ptr", chase[i], ChaseDumpBytes);
    }
    Common::Log::Flush();

    std::array<Frame, MaxBacktraceFrames> frames{};
    const auto frame_count = CaptureBacktrace(ctx, frames.data(), frames.size());
    for (size_t i = 0; i < frame_count; ++i) {
        const auto* pc = reinterpret_cast<const void*>(frames[i].pc);
        LOG_CRITICAL(Debug, "  #{:02}{} {} {}", i, SourceMark(frames[i].source), pc,
                     DescribeCode(pc));
    }
    Common::Log::Flush();
#endif

    in_crash_report = false;
}

} // Anonymous namespace

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    using namespace Libraries::Kernel;
    const auto* signals = Signals::Instance();
    // Windows static guest red-zone protection
    const bool use_static_windows_guest_red_zone_protection =
        WindowsGuestRedZoneProtection::IsStaticPatchingEnabled();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    Ucontext guest_context{pExp->ContextRecord};
    Siginfo guest_info{
        ._si_signo = 0,
        ._si_errno = 0,
        ._si_code = POSIX_SI_NOINFO,
        ._si_addr = (void*)guest_context.uc_mcontext.mc_rip,
    };

    bool handled = false;
    bool static_protection_exception = false; // Windows static guest red-zone protection
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        guest_info._si_signo = POSIX_SIGSEGV;
        guest_info._si_code = POSIX_SEGV_MAPERR;
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        guest_info._si_signo = POSIX_SIGILL;
        guest_info._si_code = POSIX_ILL_ILLOPC;
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case EXCEPTION_PRIV_INSTRUCTION: // Windows static guest red-zone protection
        if (use_static_windows_guest_red_zone_protection) {
            static_protection_exception = true;
            handled = signals->DispatchIllegalInstruction(pExp);
        }
        break;
    case EXCEPTION_IN_PAGE_ERROR:
        guest_info._si_signo = POSIX_SIGBUS;
        guest_info._si_code = POSIX_BUS_ADRALN;
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_INTDIV;
        break;
    case EXCEPTION_INT_OVERFLOW:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_INTOVF;
        break;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTDIV;
        break;
    case EXCEPTION_FLT_INVALID_OPERATION:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTINV;
        break;
    case EXCEPTION_FLT_OVERFLOW:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTOVF;
        break;
    case EXCEPTION_FLT_UNDERFLOW:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTUND;
        break;
    case EXCEPTION_FLT_DENORMAL_OPERAND:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTSUB; // i am not sure about this one
        break;
    case EXCEPTION_FLT_INEXACT_RESULT:
        guest_info._si_signo = POSIX_SIGFPE;
        guest_info._si_code = POSIX_FPE_FLTRES;
        break;
    case EXCEPTION_FLT_STACK_CHECK:
        guest_info._si_signo = POSIX_SIGILL;
        guest_info._si_code = POSIX_ILL_BADSTK; // i am not sure about this one either
        break;
    case EXCEPTION_BREAKPOINT:
    case EXCEPTION_SINGLE_STEP:
        guest_info._si_signo = POSIX_SIGTRAP;
        guest_info._si_code = POSIX_TRAP_BRKPT;
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    case MS_VC_EXCEPTION:
        LOG_DEBUG(Debug, "Pass MS_VC_EXCEPTION at {} to handler", address);
        return EXCEPTION_EXECUTE_HANDLER;
    default:
        break;
    }

    if (handled) {
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    if (guest_info._si_signo != 0) {
        if (g_curthread &&
            g_curthread->DispatchSignal(guest_info._si_signo, &guest_info, &guest_context)) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }

    // Windows static guest red-zone protection
    const bool report_unhandled = use_static_windows_guest_red_zone_protection
                                      ? static_protection_exception
                                      : code != EXCEPTION_BREAKPOINT;
    if (report_unhandled) { // Windows static guest red-zone protection
        LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {}", code, address);
        LogCrashContext(pExp, code, address);
        Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    }

    return EXCEPTION_CONTINUE_SEARCH;
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

static s32 NativeSiCodeToGuest(s32 sig, s32 code) {
    using namespace Libraries::Kernel;
    switch (sig) {
    case SIGUSR1:
        return POSIX_SI_LWP;
    case SIGSEGV:
        switch (code) {
        case SEGV_MAPERR:
            return POSIX_SEGV_MAPERR;
        case SEGV_ACCERR:
            return POSIX_SEGV_ACCERR;
        }
    case SIGBUS:
        switch (code) {
        case BUS_ADRALN:
            return POSIX_BUS_ADRALN;
        case BUS_ADRERR:
            return POSIX_BUS_ADRERR;
        case BUS_OBJERR:
            return POSIX_BUS_OBJERR;
        }
    case SIGILL:
        switch (code) {
        case ILL_ILLOPC:
            return POSIX_ILL_ILLOPC;
        case ILL_ILLOPN:
            return POSIX_ILL_ILLOPN;
        case ILL_ILLADR:
            return POSIX_ILL_ILLADR;
        case ILL_ILLTRP:
            return POSIX_ILL_ILLTRP;
        case ILL_PRVOPC:
            return POSIX_ILL_PRVOPC;
        case ILL_PRVREG:
            return POSIX_ILL_PRVREG;
        case ILL_COPROC:
            return POSIX_ILL_COPROC;
        case ILL_BADSTK:
            return POSIX_ILL_BADSTK;
        }
    case SIGFPE:
        switch (code) {
        case FPE_INTOVF:
            return POSIX_FPE_INTOVF;
        case FPE_INTDIV:
            return POSIX_FPE_INTDIV;
        case FPE_FLTDIV:
            return POSIX_FPE_FLTDIV;
        case FPE_FLTOVF:
            return POSIX_FPE_FLTOVF;
        case FPE_FLTUND:
            return POSIX_FPE_FLTUND;
        case FPE_FLTRES:
            return POSIX_FPE_FLTRES;
        case FPE_FLTINV:
            return POSIX_FPE_FLTINV;
        case FPE_FLTSUB:
            return POSIX_FPE_FLTSUB;
        }
    case SIGTRAP:
        switch (code) {
        case TRAP_BRKPT:
            return POSIX_TRAP_BRKPT;
        case TRAP_TRACE:
            return POSIX_TRAP_TRACE;
#ifdef __FreeBSD__
        case TRAP_DTRACE:
            return POSIX_TRAP_DTRACE;
#endif
        }

    default:
        return POSIX_SI_NOINFO;
    }
}

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    using namespace Libraries::Kernel;
    auto* thread = g_curthread;
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    Ucontext context{info, reinterpret_cast<ucontext_t*>(raw_context)};
    Siginfo guest_info{};
    if (info) {
        guest_info = *reinterpret_cast<Siginfo*>(info);
        guest_info._si_signo = sig == SIGUSR1 ? 0 : NativeToOrbisSignal(info->si_signo);
        guest_info._si_errno = NativeToPosixErrno(info->si_errno);
        guest_info._si_code = NativeSiCodeToGuest(sig, info->si_code);
        guest_info._si_addr = (void*)context.uc_mcontext.mc_rip;
    }
    Siginfo* info_p = info ? &guest_info : nullptr;
    Ucontext* context_p = raw_context ? &context : nullptr;

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            if (thread && thread->DispatchSignal(NativeToOrbisSignal(sig), info_p, context_p)) {
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (signals->DispatchIllegalInstruction(raw_context)) {
            return;
        }
    case SIGFPE:
    case SIGTRAP:
    case SIGSYS: {
        if (thread && thread->DispatchSignal(NativeToOrbisSignal(sig), info_p, context_p)) {
            return;
        }

        UNREACHABLE_MSG("Unhandled signal {} at code address {}", sig, fmt::ptr(code_address));
    }
    case SIGSLEEP: {
        // Sleep thread until signal is received again
        sigset_t sigset;
        sigemptyset(&sigset);
        sigaddset(&sigset, SIGSLEEP);
        sigwait(&sigset, &sig);
        break;
    }
    case SIGUSR1:
        if (thread) {
            thread->DispatchPendingSignals(info_p, context_p);
        }
        break;
    default:
        UNREACHABLE_MSG("Unhandled signal {} at code address {}", sig, fmt::ptr(code_address));
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(
        sigaction(SIGSEGV, &action, nullptr) == 0 && sigaction(SIGBUS, &action, nullptr) == 0 &&
            sigaction(SIGILL, &action, nullptr) == 0 && sigaction(SIGFPE, &action, nullptr) == 0 &&
            sigaction(SIGTRAP, &action, nullptr) == 0 && sigaction(SIGSYS, &action, nullptr) == 0 &&
            sigaction(SIGUSR1, &action, nullptr) == 0 && sigaction(SIGSLEEP, &action, nullptr) == 0,
        "Failed to register signal handlers.");
#endif
}

void SignalDispatch::RemoveHandlers() {
    // asserting here would get into an infinite loop until too
    // many nested exceptions makes the OS kill the process
#if defined(_WIN32)
    if (!(RemoveVectoredExceptionHandler(handle))) {
        LOG_CRITICAL(Core, "Failed to remove exception handler.");
        std::quick_exit(1);
    }
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    if (!(sigaction(SIGSEGV, &action, nullptr) == 0 && sigaction(SIGBUS, &action, nullptr) == 0 &&
          sigaction(SIGILL, &action, nullptr) == 0 && sigaction(SIGFPE, &action, nullptr) == 0 &&
          sigaction(SIGTRAP, &action, nullptr) == 0 && sigaction(SIGSYS, &action, nullptr) == 0 &&
          sigaction(SIGUSR1, &action, nullptr) == 0 &&
          sigaction(SIGSLEEP, &action, nullptr) == 0)) {
        LOG_CRITICAL(Core, "Failed to remove signal handlers.");
        std::quick_exit(1);
    }
#endif
}

SignalDispatch::~SignalDispatch() {
    RemoveHandlers();
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
