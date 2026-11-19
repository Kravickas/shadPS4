// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Common {

void* GetXmmPointer(void* ctx, u8 index);

void* GetRip(void* ctx);

void IncrementRip(void* ctx, u64 length);

/// Pointer to a general purpose register of the context by its x86 register number
/// (0 = RAX, 1 = RCX, 2 = RDX, 3 = RBX, 4 = RSP, 5 = RBP, 6 = RSI, 7 = RDI, 8-15 = R8-R15),
/// nullptr if the context has no x86-64 registers.
u64* GetGprPointer(void* ctx, u8 index);

/// Pointer to the 128 bits of an XMM register of the context, nullptr if unavailable.
void* GetXmmRegisterPointer(void* ctx, u8 index);

u64 GetFlags(void* ctx);
void SetFlags(void* ctx, u64 flags);

/// Moves the instruction pointer of the context past an instruction of the given length.
void SkipInstruction(void* ctx, u64 length);

bool IsWriteError(void* ctx);
bool IsExecuteError(void* ctx);

} // namespace Common