// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include "common/types.h"

namespace Common {

/// Microseconds on one steady clock, shared by every frame trace line.
inline u64 FrameTraceNow() {
    static const auto origin = std::chrono::steady_clock::now();
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - origin)
                                .count());
}

} // namespace Common
