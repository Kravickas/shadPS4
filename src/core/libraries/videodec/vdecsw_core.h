// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "core/libraries/videodec/vdecsw_memory.h"

namespace Libraries::Vdecsw {

// libSceVdecCore error codes used by libSceVdecsw.
constexpr s32 ORBIS_VDECCORE_ERROR_FAIL = static_cast<s32>(0x80c00001);
constexpr s32 ORBIS_VDECCORE_ERROR_RESOURCE_TYPE = static_cast<s32>(0x80c00003);
constexpr s32 ORBIS_VDECCORE_ERROR_MEMORY_POINTER = static_cast<s32>(0x80c00007);
constexpr s32 ORBIS_VDECCORE_ERROR_CPU_MEMORY_SIZE = static_cast<s32>(0x80c00008);
constexpr s32 ORBIS_VDECCORE_ERROR_CPU_GPU_MEMORY_SIZE = static_cast<s32>(0x80c00009);
constexpr s32 ORBIS_VDECCORE_ERROR_FRAME_BUFFER_SIZE = static_cast<s32>(0x80c0000a);
constexpr s32 ORBIS_VDECCORE_ERROR_FRAME_BUFFER_ALIGNMENT = static_cast<s32>(0x80c0000b);
constexpr s32 ORBIS_VDECCORE_ERROR_INPUT_QUEUE_FULL = static_cast<s32>(0x80c0000e);
constexpr s32 ORBIS_VDECCORE_ERROR_NEW_SEQUENCE = static_cast<s32>(0x80c0000f);
constexpr s32 ORBIS_VDECCORE_ERROR_INVALID_SEQUENCE = static_cast<s32>(0x80c00010);
constexpr s32 ORBIS_VDECCORE_ERROR_OVERSIZE_DECODE = static_cast<s32>(0x80c00011);
constexpr s32 ORBIS_VDECCORE_ERROR_ACCESS_UNIT = static_cast<s32>(0x80c00012);
constexpr s32 ORBIS_VDECCORE_ERROR_FATAL_STREAM = static_cast<s32>(0x80c00013);
constexpr s32 ORBIS_VDECCORE_ERROR_GPU_MEMORY_SIZE = static_cast<s32>(0x80c00017);
constexpr s32 ORBIS_VDECCORE_ERROR_NOT_READY = static_cast<s32>(0x80c00018);
constexpr s32 ORBIS_VDECCORE_ERROR_BUSY = static_cast<s32>(0x80c00019);
constexpr s32 ORBIS_VDECCORE_ERROR_NO_OUTPUT_BUFFER = static_cast<s32>(0x80c0001a);
constexpr s32 ORBIS_VDECCORE_ERROR_NO_OUTPUT_FRAME = static_cast<s32>(0x80c0001b);
constexpr s32 ORBIS_VDECCORE_ERROR_OUTPUT_PENDING = static_cast<s32>(0x80c0001c);
constexpr s32 ORBIS_VDECCORE_ERROR_STOPPED = static_cast<s32>(0x80c0001d);

struct VdecCoreComputeParams {
    u64 cpu_gpu_memory_size;
    void* cpu_gpu_memory;
    u32 compute_pipe_id;
    u32 compute_queue_id;
};

struct VdecCoreMemoryParams {
    u64 cpu_memory_size;
    void* cpu_memory;
    u64 cpu_gpu_memory_size;
    void* cpu_gpu_memory;
    u64 gpu_memory_size;
    void* gpu_memory;
};

struct VdecCoreInput {
    void* au_data;
    u64 au_size;
    u64 pts_data;
    u64 dts_data;
    u64 attached_data;
};

struct VdecCoreOutput {
    u32 frame_width;
    u32 frame_pitch;
    u32 frame_height;
    u8 picture_count;
    u32 frame_format;
    u32 codec;
    u32 is_last_frame;
    u32 error_mb_count;
    u32 error_status;
    u32 frame_pitch_in_bytes;
    void* frame_buffer;
    u64 frame_buffer_size;
    u32 is_discarded_frame;
};

s32 VdecCoreQueryComputeResourceInfo(u64* size, u64* base);
s32 VdecCoreInitializeComputeResource(const VdecCoreComputeParams& params, void** handle, u32 sdk);
s32 VdecCoreFinalizeComputeResource(void* handle);

s32 VdecCoreCreateDecoder(const VdecCoreConfig& cfg, const VdecCoreMemoryParams& mem, void* compute,
                          void** core, u32 sdk);
s32 VdecCoreDeleteDecoder(void* core);
s32 VdecCoreResetDecoder(void* core);
s32 VdecCoreSetDecodeInput(void* core, const VdecCoreInput& input);
s32 VdecCoreSyncDecodeWptr(void* core, u32* frame_count, void** decoded_au);
s32 VdecCoreTrySyncDecodeWptr(void* core, u32* frame_count, void** decoded_au);
s32 VdecCoreSetDecodeOutputSw(void* core, void* frame_buffer, u64 frame_buffer_size);
s32 VdecCoreSyncDecodeOutputSw(void* core, VdecCoreOutput* output);
s32 VdecCoreTrySyncDecodeOutputSw(void* core, VdecCoreOutput* output);
s32 VdecCoreFlushDecodeOutput(void* core, s32* frame_count);

} // namespace Libraries::Vdecsw
