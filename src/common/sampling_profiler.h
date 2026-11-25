// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Common::SamplingProfiler {

/// Adds the calling thread to the threads whose instruction pointer is sampled about once a
/// millisecond. The counts go to the log as "PERF prof" lines. Windows only.
void RegisterCurrentThread(const char* name);

/// Samples are only taken while enabled.
void SetEnabled(bool is_enabled);

/// Probe: the logical processor the calling thread runs on. 0 where it is not known.
unsigned CurrentProcessor();

} // namespace Common::SamplingProfiler
