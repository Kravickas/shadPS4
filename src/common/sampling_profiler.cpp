// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/sampling_profiler.h"

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"

#include <windows.h>

namespace Common::SamplingProfiler {

namespace {

constexpr u32 MaxThreads = 4;
constexpr u32 MaxModules = 64;
constexpr u32 TableBits = 17;
constexpr u32 TableSize = 1u << TableBits;
constexpr u32 MaxProbes = 64;
constexpr u32 StackWords = 192;
constexpr u64 FirstReportSamples = 15'000;
constexpr u64 ReportSamples = 30'000;
constexpr u32 EntriesPerLine = 40;

/// The executable is module 0 and addresses without a module are module 1.
constexpr u32 ExeModule = 0;
constexpr u32 NoModule = 1;

/// A key holds the module in its top byte. Below it is the address in the executable, or for
/// the other modules two addresses of callers in the executable.
constexpr u32 ModuleShift = 56;
constexpr u32 CallerBits = 28;
constexpr u64 CallerMask = (1ULL << CallerBits) - 1;
constexpr u32 NumCallers = 2;

struct Slot {
    u64 key;
    u32 count;
};

struct Thread {
    HANDLE handle;
    const char* name;
    u64 stack_low;
    u64 stack_high;
    std::unique_ptr<Slot[]> slots;
    u64 samples;
    u64 dropped;
};

struct Module {
    u64 base;
    u64 size;
    std::string name;
};

std::array<Thread, MaxThreads> threads;
std::atomic<u32> num_threads{};
std::atomic<bool> enabled{};
std::mutex register_mutex;
std::once_flag sampler_started;

std::vector<Module> modules;
u64 exe_text_begin{};
u64 exe_text_end{};

Module DescribeModule(HMODULE handle) {
    const auto* base = reinterpret_cast<const u8*>(handle);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    char path[MAX_PATH];
    const DWORD length = GetModuleFileNameA(handle, path, MAX_PATH);
    std::string name{path, length};
    if (const auto slash = name.find_last_of("\\/"); slash != std::string::npos) {
        name.erase(0, slash + 1);
    }
    return {reinterpret_cast<u64>(base), nt->OptionalHeader.SizeOfImage, std::move(name)};
}

void SamplerThread();

void DescribeExe() {
    const HMODULE handle = GetModuleHandleW(nullptr);
    modules.reserve(MaxModules);
    modules.push_back(DescribeModule(handle));
    modules.push_back({0, 0, "(none)"});

    // The code of the executable is the section this file's code is in. Data lies between it
    // and the other executable sections.
    const auto* base = reinterpret_cast<const u8*>(handle);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    const IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
    const u64 own_code = reinterpret_cast<u64>(&SamplerThread);
    for (u32 i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        const u64 begin = modules[ExeModule].base + section->VirtualAddress;
        if (own_code - begin < section->Misc.VirtualSize) {
            exe_text_begin = begin;
            exe_text_end = begin + section->Misc.VirtualSize;
        }
    }
}

/// Whether the address in the code of the executable follows a call instruction.
bool IsReturnAddress(u64 address) {
    if (address < exe_text_begin + 8 || address >= exe_text_end) {
        return false;
    }
    const auto* code = reinterpret_cast<const u8*>(address);
    const auto is_indirect_call = [code](u32 length) {
        return code[-static_cast<s32>(length)] == 0xFF &&
               (code[1 - static_cast<s32>(length)] & 0x38) == 0x10;
    };
    // call rel32, and call r/m64 with its two to seven bytes.
    return code[-5] == 0xE8 || is_indirect_call(2) || is_indirect_call(3) || is_indirect_call(4) ||
           is_indirect_call(6) || is_indirect_call(7);
}

u32 FindModule(u64 address) {
    for (u32 i = 0; i < modules.size(); ++i) {
        if (address - modules[i].base < modules[i].size) {
            return i;
        }
    }
    HMODULE handle;
    if (modules.size() == MaxModules ||
        !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(address), &handle)) {
        return NoModule;
    }
    modules.push_back(DescribeModule(handle));
    return static_cast<u32>(modules.size() - 1);
}

void Count(Thread& thread, u64 key) {
    ++thread.samples;
    u32 index = static_cast<u32>((key * 0x9E3779B97F4A7C15ULL) >> (64 - TableBits));
    for (u32 probe = 0; probe < MaxProbes; ++probe, index = (index + 1) & (TableSize - 1)) {
        Slot& slot = thread.slots[index];
        if (slot.key == key || slot.count == 0) {
            slot.key = key;
            ++slot.count;
            return;
        }
    }
    ++thread.dropped;
}

void Sample(Thread& thread) {
    CONTEXT context;
    context.ContextFlags = CONTEXT_CONTROL;
    std::array<u64, StackWords> stack;
    u32 num_words = 0;

    // Nothing that can take a lock runs while the thread is suspended: it may hold that lock.
    if (SuspendThread(thread.handle) == static_cast<DWORD>(-1)) {
        return;
    }
    const bool has_context = GetThreadContext(thread.handle, &context) != 0;
    if (has_context && (context.Rip < exe_text_begin || context.Rip >= exe_text_end)) {
        const u64 rsp = (context.Rsp + 7) & ~7ULL;
        if (rsp >= thread.stack_low && rsp < thread.stack_high) {
            num_words = static_cast<u32>(std::min<u64>(StackWords, (thread.stack_high - rsp) / 8));
            std::memcpy(stack.data(), reinterpret_cast<const void*>(rsp), num_words * sizeof(u64));
        }
    }
    ResumeThread(thread.handle);
    if (!has_context) {
        return;
    }

    const u64 exe_base = modules[ExeModule].base;
    if (context.Rip - exe_base < modules[ExeModule].size) {
        Count(thread, (u64{ExeModule + 1} << ModuleShift) | (context.Rip - exe_base));
        return;
    }
    // Outside of the executable the sample is counted for the first words on the stack that
    // are return addresses into the code of the executable, which is where it was probably
    // called from.
    u64 callers = 0;
    u32 num_callers = 0;
    for (u32 i = 0; i < num_words && num_callers < NumCallers; ++i) {
        if (IsReturnAddress(stack[i])) {
            callers |= (stack[i] - exe_base) << (num_callers * CallerBits);
            ++num_callers;
        }
    }
    Count(thread, (u64{FindModule(context.Rip) + 1} << ModuleShift) | callers);
}

void Report() {
    for (u32 i = 0; i < modules.size(); ++i) {
        LOG_INFO(Render, "PERF prof_mod index={} base={:#x} size={:#x} name={}", i, modules[i].base,
                 modules[i].size, modules[i].name);
    }
    const u32 count = num_threads.load(std::memory_order_acquire);
    std::vector<Slot> entries;
    for (u32 t = 0; t < count; ++t) {
        const Thread& thread = threads[t];
        entries.clear();
        for (u32 i = 0; i < TableSize; ++i) {
            if (thread.slots[i].count != 0) {
                entries.push_back(thread.slots[i]);
            }
        }
        std::ranges::sort(entries, std::ranges::greater{}, &Slot::count);
        LOG_INFO(Render, "PERF prof thread={} name={} samples={} dropped={} keys={}", t,
                 thread.name, thread.samples, thread.dropped, entries.size());
        std::string line;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const u64 key = entries[i].key;
            const u64 module = (key >> ModuleShift) - 1;
            if (module == ExeModule) {
                fmt::format_to(std::back_inserter(line), " 0:{:x}={}",
                               key & ((1ULL << ModuleShift) - 1), entries[i].count);
            } else {
                fmt::format_to(std::back_inserter(line), " {:x}:{:x}:{:x}={}", module,
                               key & CallerMask, (key >> CallerBits) & CallerMask,
                               entries[i].count);
            }
            if ((i + 1) % EntriesPerLine == 0 || i + 1 == entries.size()) {
                LOG_INFO(Render, "PERF prof_data thread={}{}", t, line);
                line.clear();
            }
        }
    }
}

void SamplerThread() {
    SetCurrentThreadName("shadPS4:SamplingProfiler");
    SetCurrentThreadPriority(ThreadPriority::VeryHigh);
    DescribeExe();
    if (modules[ExeModule].size > CallerMask || exe_text_begin >= exe_text_end) {
        LOG_ERROR(Render, "PERF prof cannot describe the executable");
        return;
    }

    u64 random = 0x9E3779B97F4A7C15ULL;
    u64 next_report = FirstReportSamples;
    while (true) {
        // A random interval keeps the samples from falling in step with the frames.
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        AccurateSleep(std::chrono::microseconds(600 + random % 800), nullptr, false);
        if (!enabled.load(std::memory_order_relaxed)) {
            continue;
        }
        const u32 count = num_threads.load(std::memory_order_acquire);
        for (u32 t = 0; t < count; ++t) {
            Sample(threads[t]);
        }
        if (count != 0 && threads[0].samples >= next_report) {
            next_report = threads[0].samples + ReportSamples;
            Report();
        }
    }
}

} // Anonymous namespace

void RegisterCurrentThread(const char* name) {
    {
        std::scoped_lock lock{register_mutex};
        const u32 index = num_threads.load(std::memory_order_relaxed);
        if (index == MaxThreads) {
            return;
        }
        Thread& thread = threads[index];
        if (!DuplicateHandle(
                GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &thread.handle,
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0)) {
            return;
        }
        ULONG_PTR stack_low;
        ULONG_PTR stack_high;
        GetCurrentThreadStackLimits(&stack_low, &stack_high);
        thread.name = name;
        thread.stack_low = stack_low;
        thread.stack_high = stack_high;
        thread.slots = std::make_unique<Slot[]>(TableSize);
        num_threads.store(index + 1, std::memory_order_release);
    }
    std::call_once(sampler_started, [] { std::thread{SamplerThread}.detach(); });
}

void SetEnabled(bool is_enabled) {
    enabled.store(is_enabled, std::memory_order_relaxed);
}

} // namespace Common::SamplingProfiler

#else

namespace Common::SamplingProfiler {

void RegisterCurrentThread(const char*) {}

void SetEnabled(bool) {}

} // namespace Common::SamplingProfiler

#endif
