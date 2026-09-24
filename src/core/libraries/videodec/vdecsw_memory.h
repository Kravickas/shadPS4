// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "core/libraries/videodec/vdecsw.h"

namespace Libraries::Vdecsw {

// libSceVdecCore decoder configuration built by libSceVdecsw (FUN_00003440).
struct VdecCoreConfig {
    u32 resource_type;
    u32 codec;
    u32 profile;
    u32 level;
    s32 max_dpb_frame_count;
    s32 max_frame_width;
    s32 max_frame_height;
    u32 pitch_align;
    u32 output_format;
    u32 unk24;
    u8 hevc_ext_mode;
    u8 hevc_align;
    u8 bit_depth_luma;
    u8 bit_depth_chroma;
    s32 hevc_ext_2c;
    u8 hevc_ext_30;
    u8 pad31[7];
    u32 decode_pipeline_depth;
    s32 cpu_thread_priority;
    u32 not_optimize_progressive;
    s32 extra_dpb_frame_count;
    u64 cpu_affinity_mask;
    s32 max_pending_sync_count;
    u32 pad54;
};
static_assert(sizeof(VdecCoreConfig) == 0x58);

// Decoder parameters resolved by libSceVdecCore FUN_00001070 for a libSceVdecsw configuration.
struct VdecCoreDecoderParams {
    u32 codec;
    u32 profile;
    u32 level;
    s32 width_units;
    s32 height_units;
    u32 unit_size;
    s32 max_dpb_frame_count;
    u32 decode_pipeline_depth;
    s32 extra_dpb_frame_count;
    s32 max_pending_sync_count;
    u32 optimize_progressive;
    u64 frame_buffer_size;
    u32 frame_buffer_alignment;
    u32 pitch_align;
};

s32 VdecswBuildCoreConfig(const OrbisVdecswDecoderConfigInfo& cfg,
                          OrbisVdecswDecoderMemoryInfo& mem, VdecCoreConfig& core, u32 sdk,
                          u32 cpumode);

s32 VdecCoreQueryInstanceSize(const VdecCoreConfig& cfg, u64* cpu_size, u64* gpu_size,
                              u64* cpu_gpu_size, u32 sdk);

s32 VdecCoreQueryFrameBufferInfo(const VdecCoreConfig& cfg, s32 info[6], u32 sdk);

s32 VdecCoreGetDecoderParams(const VdecCoreConfig& cfg, VdecCoreDecoderParams& params, u32 sdk);

} // namespace Libraries::Vdecsw
