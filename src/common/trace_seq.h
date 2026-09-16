// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <thread>
#include "common/types.h"

namespace Common {

// Shared by the debug tracers so lines from different files can be ordered against each other.
inline std::atomic<u64> trace_sequence{0};

inline u64 NextTraceSeq() {
    return trace_sequence.fetch_add(1, std::memory_order_relaxed);
}

inline u32 TraceThreadId() {
    static std::atomic<u32> next_id{0};
    static thread_local u32 id = next_id.fetch_add(1, std::memory_order_relaxed);
    return id;
}

} // namespace Common
