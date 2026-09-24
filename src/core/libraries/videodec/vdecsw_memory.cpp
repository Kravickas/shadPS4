// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Port of the libSceVdecsw -> libSceVdecCore -> libSceVdecSavc2 memory queries (FW 12.02).
// Function comments name the SPRX function each block was ported from.

#include <algorithm>
#include <bit>
#include <cstring>

#include "core/libraries/kernel/memory.h"
#include "core/libraries/videodec/vdecsw_memory.h"
#include "core/libraries/videodec/videodec_error.h"
#include "core/memory.h"

namespace Libraries::Vdecsw {

namespace {

struct BufInfo {
    u64 base;
    u64 size;
    u32 align;
};

// ---------------------------------------------------------------------------------------------
// libSceVdecSavc2
// ---------------------------------------------------------------------------------------------
namespace Savc2 {

// FUN_00001fe0 / DAT_0005ad90
u32 BytesPerElement(u32 fmt) {
    switch (fmt & 0xff) {
    case 1:
        return 1;
    case 2:
    case 3:
        return 2;
    case 4:
    case 5:
    case 0xa:
        return 4;
    case 0xc:
        return 8;
    default:
        return 0;
    }
}

struct SurfaceSize {
    u32 size;
    u32 align;
};

// FUN_00002440 + FUN_000029a0, kTileModeDisplay_LinearAligned, 2D, 1 mip.
SurfaceSize LinearAlignedSurface(u32 width, u32 height, u32 slices, u32 fmt) {
    const u32 bpe = BytesPerElement(fmt);
    const u32 pitch_align = std::max(8u, 64u / bpe);
    const u32 slice_align = std::max(256u, 64u * bpe);
    u64 pitch = (width + pitch_align - 1) / pitch_align * pitch_align;
    while ((pitch * height * bpe) % slice_align) {
        pitch += pitch_align;
    }
    return {static_cast<u32>(pitch * height * bpe * slices), 0x100};
}

u32 AlignU32(u32 value, u32 align) {
    return -align & (align - 1 + value);
}

// FUN_000244d0 (via FUN_000243f0).
s32 ComputeResources(s32 width, s32 height, s32 dpb, s32 slice_pipes, u8 field, BufInfo* small,
                     BufInfo* big, u8 sdk_ge_400) {
    *small = {};
    *big = {};
    const s32 limit = (field ^ 1) * 0x2000 + 0x2000;
    const s32 h15 = (height * 3) / 2;
    if (width < 0x10 || limit < width || height < 0x10 || limit < h15 || dpb < 1 || 0x2000 < dpb) {
        return 0x200000;
    }
    s32 count_2230 = slice_pipes;
    s32 count_2234;
    s32 count_2238 = slice_pipes;
    if (!sdk_ge_400) {
        count_2234 = std::max(slice_pipes, 2) - 1;
    } else {
        const s32 m = slice_pipes < 2 ? slice_pipes : 2;
        if (slice_pipes != 1) {
            count_2238 = m;
        }
        count_2234 = 1;
    }

    const u32 w = static_cast<u32>(width);
    const u32 h = static_cast<u32>(height);
    const u32 w2 = w >> 1;
    const auto t5 = LinearAlignedSurface(w2, h, 1, 0x2c505);
    u32 max_align = t5.align;
    SurfaceSize t58{0, 0};
    if (!sdk_ge_400) {
        t58 = LinearAlignedSurface(w2, h >> 1, 1, 0x2c505);
        max_align = std::max(t58.align, t5.align);
    }
    const auto t6 = LinearAlignedSurface(w >> 2, h, 1, 0x4404);
    max_align = std::max(max_align, t6.align);
    const auto t7 = LinearAlignedSurface(w >> 3, h, 1, 0x4404);
    max_align = std::max(max_align, t7.align);
    const auto t8 = LinearAlignedSurface(w >> 3, h, 1, 0x4404);
    max_align = std::max(max_align, t8.align);
    const u32 w16 = w >> 4;
    const auto t9 = LinearAlignedSurface(w16, h >> 2, 1, 0xfac40a);
    max_align = std::max(max_align, t9.align);
    const auto t10 = LinearAlignedSurface(w16, h >> 2, 1, 0xfac40a);
    max_align = std::max(max_align, t10.align);

    u32 view_a = AlignU32(t5.size, t5.align);
    if (!sdk_ge_400) {
        view_a = (view_a - 0x200) + AlignU32(t58.size, t58.align);
    }
    u32 view_b = AlignU32(t6.size, t6.align);
    view_b += AlignU32(t7.size, t7.align);
    view_b += AlignU32(t8.size, t8.align);
    view_b += AlignU32(t9.size, t9.align);
    view_b += AlignU32(t10.size, t10.align);
    const u32 scratch_bytes = std::max(view_b, view_a);

    const u32 dpb_width = field ? ((w + 0xff) & 0xffffff00) : w;
    const u32 dpb_slices = static_cast<u32>(dpb);
    const auto t12 = LinearAlignedSurface(dpb_width, static_cast<u32>(h15), dpb_slices, 0x4401);
    u32 a = std::max(t12.align, max_align);
    const auto t13 = LinearAlignedSurface(dpb_width, static_cast<u32>(h15), dpb_slices, 0x4401);
    a = std::max(t13.align, a);
    const auto t14 = LinearAlignedSurface(static_cast<u32>(static_cast<s32>(dpb_width) >> 1),
                                          static_cast<u32>(h15), dpb_slices, 0x2c403);
    a = std::max(t14.align, a);
    const auto t36 =
        LinearAlignedSurface(static_cast<u32>(static_cast<s32>(dpb_width) >> 2), h, 1, 0x4404);
    a = std::max(t36.align, a);
    const auto t41 =
        LinearAlignedSurface(static_cast<u32>(static_cast<s32>(dpb_width) >> 2), h >> 1, 1, 0x4404);
    u32 max_tex_align = std::max(t41.align, a);

    const u32 slice_h = field ? ((h + 0x1f) >> 1 & 0xfffffff0) : h;
    const u32 slice_h16 = slice_h >> 4;
    SurfaceSize per_pipe[5]{};
    if (0 < count_2230) {
        per_pipe[0] = LinearAlignedSurface(w16, slice_h16, 1, 0x4404);
        per_pipe[1] = LinearAlignedSurface(w16, slice_h16, 1, 0x4404);
        per_pipe[2] = LinearAlignedSurface(w16, slice_h16, 1, 0xfac50a);
        per_pipe[3] = LinearAlignedSurface(w, slice_h16, 1, 0xfac50c);
        per_pipe[4] = LinearAlignedSurface(w, slice_h16, 1, 0xfac50c);
        for (const auto& s : per_pipe) {
            max_tex_align = std::max(max_tex_align, s.align);
        }
    }

    const s32 mb_count = static_cast<s32>(slice_h * w16);
    s32 row_blocks = mb_count;
    if (mb_count < 0) {
        row_blocks = mb_count + 0xf;
    }
    SurfaceSize per_slice[6]{};
    s32 cb_size = 0;
    if (0 < count_2238) {
        row_blocks >>= 4;
        const u32 h8 = static_cast<u32>(static_cast<s32>(slice_h) >> 3);
        const u32 w8 = static_cast<u32>(width >> 3);
        cb_size = row_blocks * 4 + 0x14;
        per_slice[0] = LinearAlignedSurface(w8, h8, 1, 0x2c505);
        per_slice[1] = LinearAlignedSurface(w8, h8, 1, 0x2c505);
        per_slice[2] = LinearAlignedSurface(w8, h8, 1, 0x2c505);
        per_slice[3] = LinearAlignedSurface(w8, h8, 1, 0xfac50a);
        per_slice[4] = LinearAlignedSurface(w8, h8, 1, 0xfac50a);
        per_slice[5] = LinearAlignedSurface(w8, h8, 1, 0xfac50a);
        for (const auto& s : per_slice) {
            max_tex_align = std::max(max_tex_align, s.align);
        }
    }

    const u32 align = std::max(4u, max_tex_align);
    const auto buffer = [](u32 stride, u32 records) { return (stride * records + 3) & 0xfffffffc; };
    const auto const_buffer = [&](u32 bytes) { return buffer(16, (bytes + 0xf) >> 4); };

    u32 big_cursor = 0;
    big_cursor += buffer(1, scratch_bytes);
    for (const u32 bytes : {0x180u, 0x600u, 0x300u, 0xd0u, 0x340u, 0x20u, 0x10u, 0x10u, 0x28u}) {
        big_cursor += const_buffer(bytes);
    }
    big_cursor = AlignU32(big_cursor, align);
    big_cursor += AlignU32(t13.size, t13.align);
    big->size = big_cursor;
    big->align = align;

    u32 small_cursor = const_buffer(0x20);
    for (s32 i = 0; i < count_2230; i++) {
        small_cursor += buffer(4, static_cast<u32>(static_cast<s32>(slice_h * w) / 2));
        small_cursor += buffer(2, (slice_h >> 1) * w);
    }
    for (s32 i = 0; i < count_2234; i++) {
        const u32 mbs = slice_h16 * w16;
        small_cursor += buffer(1, ((mbs * 0x18 + 0x33) & 0x7ffffff8) * 2 + mbs * 0x44);
    }
    for (s32 i = 0; i < count_2238; i++) {
        small_cursor += buffer(4, static_cast<u32>(row_blocks * 5));
        small_cursor += buffer(4, static_cast<u32>(row_blocks * 2));
        small_cursor += buffer(4, static_cast<u32>(row_blocks + 5));
        small_cursor += buffer(4, static_cast<u32>(row_blocks + 5));
        small_cursor += const_buffer(static_cast<u32>(cb_size));
        small_cursor += const_buffer(static_cast<u32>(cb_size));
    }
    small_cursor = AlignU32(small_cursor, align);
    for (s32 i = 0; i < count_2230; i++) {
        for (const auto& s : per_pipe) {
            small_cursor += AlignU32(s.size, s.align);
        }
    }
    for (s32 i = 0; i < count_2238; i++) {
        for (const auto& s : per_slice) {
            small_cursor += AlignU32(s.size, s.align);
        }
    }
    small->size = small_cursor;
    small->align = align;
    return 0;
}

// FUN_0001d370
s32 LevelMaxFs(u32* out, u32 level, bool strict, u32 sdk) {
    u32 v = 99;
    switch (level) {
    case 10:
    case 0x6f:
        break;
    case 0xb:
    case 0xc:
    case 0xd:
    case 0x14:
        v = 0x18c;
        break;
    case 0x15:
        v = 0x318;
        break;
    case 0x16:
    case 0x1e:
        v = 0x654;
        break;
    case 0x1f:
        v = 0xe10;
        break;
    case 0x20:
        v = 0x1400;
        break;
    case 0x28:
    case 0x29:
        v = 0x2000;
        break;
    case 0x2a:
        v = 0x2200;
        break;
    case 0x32:
        v = 0x5640;
        break;
    case 0x33:
        v = 0x9000;
        break;
    case 0x34:
        v = strict ? 0x10000 : 0x9000;
        if (sdk < 0x5000000) {
            v = 0x10000;
        }
        if (0x64fffff < sdk) {
            v = 0x9000;
        }
        break;
    case 0x3c:
    case 0x3d:
    case 0x3e:
        v = 0x22000;
        break;
    default:
        *out = 0;
        return 1;
    }
    *out = v;
    return 0;
}

// FUN_0001d910
s32 LevelMaxDims(u32 level, u32* max_w, u32* max_h, u32 sdk) {
    u32 v = 0x1c;
    switch (level) {
    case 10:
    case 0x6f:
        break;
    case 0xb:
    case 0xc:
    case 0xd:
    case 0x14:
        v = 0x38;
        break;
    case 0x15:
        v = 0x4f;
        break;
    case 0x16:
    case 0x1e:
        v = 0x71;
        break;
    case 0x1f:
        v = 0xa9;
        break;
    case 0x20:
        v = 0xca;
        break;
    case 0x28:
    case 0x29:
        v = 0x100;
        break;
    case 0x2a:
        v = 0x107;
        break;
    case 0x32:
        v = 0x1a4;
        break;
    case 0x33:
    case 0x34:
        v = 0x21f;
        break;
    case 0x3c:
    case 0x3d:
    case 0x3e:
        if (sdk < 0x6500000) {
            return 3;
        }
        v = 0x41f;
        break;
    default:
        return 3;
    }
    const bool big = level != 0x6f && 0x33 < static_cast<s32>(level);
    const u32 w = big ? 0x200 : 0x100;
    const u32 h = big ? 0x110 : 0x88;
    *max_w = std::min(w, v);
    *max_h = std::min(h, v);
    return 0;
}

// FUN_0001d6a0
u32 DpbFrames(s32 dpb, s32 w_mbs, s32 h_mbs, u32 level, u32 sdk) {
    u32 max_dpb_mbs = 0x18c;
    bool legacy = false;
    switch (level) {
    case 10:
        break;
    case 0xb:
        max_dpb_mbs = 900;
        break;
    case 0xc:
    case 0xd:
    case 0x14:
        max_dpb_mbs = 0x948;
        break;
    case 0x15:
        max_dpb_mbs = 0x1290;
        break;
    case 0x16:
    case 0x1e:
        max_dpb_mbs = 0x1fa4;
        break;
    case 0x1f:
        max_dpb_mbs = 18000;
        break;
    case 0x20:
        max_dpb_mbs = 0x5000;
        break;
    case 0x28:
    case 0x29:
        max_dpb_mbs = 0x8000;
        break;
    case 0x2a:
        max_dpb_mbs = 0x8800;
        break;
    case 0x32:
        max_dpb_mbs = 0x1af40;
        break;
    case 0x33:
        max_dpb_mbs = 0x2d000;
        break;
    case 0x34:
        legacy = true;
        break;
    default:
        max_dpb_mbs = 0xaa000;
        if (sdk < 0x6500000) {
            legacy = true;
        }
        break;
    }
    if (legacy) {
        max_dpb_mbs = 0x4ffffff < sdk ? 0x2d000 : 0x5a000;
    }
    if (dpb >= 0) {
        return dpb < 0x10 ? static_cast<u32>(dpb) : 0x10;
    }
    if (w_mbs < 0 || h_mbs < 0) {
        u32 frames = 0x10;
        u32 max_fs = 0;
        if (0x64fffff < sdk && LevelMaxFs(&max_fs, level, false, sdk) == 0 && max_fs != 0) {
            if (max_dpb_mbs / max_fs < 0x10) {
                frames = max_dpb_mbs / max_fs;
            }
        }
        return frames;
    }
    const u32 q = max_dpb_mbs / static_cast<u32>(h_mbs * w_mbs);
    return q < 0x10 ? q : 0x10;
}

// FUN_0001d560
s32 SliceDpbFrames(u32 w_mbs, u32 h_mbs, s32 dpb, u32 level, u32 extra, u32 sdk) {
    const u32 base = DpbFrames(dpb, static_cast<s32>(w_mbs), static_cast<s32>(h_mbs), level, sdk);
    u32 add = extra;
    if (dpb < 0) {
        u32 max_fs = 0;
        if (LevelMaxFs(&max_fs, level, false, sdk) != 0) {
            return static_cast<s32>(base + 1 + 0x10);
        }
        u32 mbs = h_mbs * w_mbs;
        if (mbs == 0) {
            mbs = 1;
        }
        u32 q = (max_fs * extra) / mbs;
        q += (mbs * q < max_fs * extra) ? 1 : 0;
        add = q;
        if (0x10 < q) {
            if (extra < 0x11) {
                return static_cast<s32>(base + 1 + 0x10);
            }
            add = q <= extra ? q : extra;
        }
    }
    const u32 clamped = static_cast<s32>(add) < 1 ? 0 : add;
    return static_cast<s32>(base + 1 + clamped);
}

// FUN_0001d140
u64 CpuWorkSize(s32 w_mbs, s32 h_mbs, s32 frames, s32 extra, s32 pipes, u8 field, u8 sdk_ge_400,
                u32 sdk) {
    const s32 total_frames = extra + frames;
    s32 frame_slots = total_frames;
    if (field) {
        h_mbs = (h_mbs + 1) >> 1;
        frame_slots = total_frames * 2;
    }
    const s64 lp = pipes;
    const s64 mbs = h_mbs * lp * static_cast<s64>(w_mbs);
    const u64 ctx =
        static_cast<u64>(static_cast<u32>(w_mbs + 1 + (static_cast<s32>(h_mbs + 1u) >> 1))) * 0x98 +
            0x1f &
        0xffffffffffffffe0ULL;
    const s32 pipes_min2 = std::max(pipes, 2);
    const u64 sum = (static_cast<u64>(static_cast<s64>(total_frames)) + 0x1f & ~0x1fULL) +
                    static_cast<u64>(static_cast<s64>(total_frames * 2) * 0xc0) +
                    (static_cast<u64>(static_cast<s64>(total_frames) * 0x30) + 0x1f & ~0x1fULL) +
                    (static_cast<u64>(lp * 0x138) + 0x1f & ~0x1fULL) +
                    (static_cast<u64>(lp * 0x10) + 0x1f & ~0x1fULL) +
                    (static_cast<u64>(lp * 0x3fec8) + 0x1f & ~0x1fULL) +
                    (static_cast<u64>(static_cast<s64>(pipes << 4)) + 0x1f & ~0x1fULL) +
                    static_cast<u64>(mbs * 0x80) +
                    (static_cast<u64>(static_cast<s64>(pipes * 8) * 2) + 0x3e & ~0x3fULL) +
                    (static_cast<u64>(static_cast<s64>(frame_slots) * 0x18) + 0x1f & ~0x1fULL) +
                    static_cast<u64>(static_cast<s64>(frames << (field & 0x1f))) *
                        static_cast<u64>((static_cast<u32>(w_mbs) * 0x80 +
                                          (static_cast<u32>(w_mbs) + 0x7f & 0xffffff80)) *
                                         (static_cast<u32>(h_mbs) + 1 & 0xfffffffe)) +
                    (static_cast<u64>(pipes_min2 - 1) * 0x70 + 0x1f & ~0x1fULL);
    u64 total = sum + 0x6b8de0 + ctx * 6;
    if (0x64fffff < sdk) {
        total = sum + 0x6b51c0 + ctx * 6;
    }
    if (!sdk_ge_400) {
        total += static_cast<u64>(mbs * 0xe0);
    }
    return total;
}

// Savc2 query spec (built by libSceVdecCore FUN_00019b10).
struct Spec {
    s32 codec;
    s32 profile;
    u32 level;
    s32 unk0c;
    s32 format;
    s32 unk14;
    s32 pitch_align;
    s32 dpb;
    s32 w_mbs;
    s32 h_mbs;
    s32 w_rem;
    s32 h_rem;
    u8 bit_depth_luma;
    u8 bit_depth_chroma;
    u8 pad32[2];
    s32 optimize_progressive;
};
static_assert(sizeof(Spec) == 0x38);

// Savc2 query decoder parameters (built by libSceVdecCore FUN_000180c0).
struct QueryDec {
    s32 mode;
    s32 pipe_mode;
    u32 pipeline_depth;
    s32 extra_dpb;
    u64 unk10;
    s32 cpu_thread_priority;
    u32 pad1c;
    u64 cpu_affinity_mask;
};
static_assert(sizeof(QueryDec) == 0x28);

// FUN_0001daa0
s32 QueryCpu(const Spec& spec, u32 frames, u32 extra, s32 pipes, BufInfo* out, u32 field,
             u8 sdk_ge_400, u8* oversize, u32 sdk) {
    if (spec.codec != 0 || spec.w_mbs == 0 || spec.h_mbs == 0) {
        return 3;
    }
    u32 max_fs = 0;
    s32 add = 0;
    if (0x64fffff < sdk) {
        add = 0x10 - static_cast<s32>(extra);
        if (0x10 < extra) {
            add = 0;
        }
        if (frames < 0x10) {
            add = (add - static_cast<s32>(frames)) + 0x10;
        }
    }
    u64 best = 0;
    if (spec.w_mbs < 0 || spec.h_mbs < 0) {
        if (LevelMaxFs(&max_fs, spec.level, false, sdk) != 0) {
            return 3;
        }
        u32 max_w = 0;
        u32 max_h = 0;
        if (LevelMaxDims(spec.level, &max_w, &max_h, sdk) != 0) {
            return 3;
        }
        const auto sweep = [&](bool by_width, s32 work_pipes, u8 work_field) -> bool {
            const u32 count = by_width ? max_w : max_h;
            const u32 other = by_width ? max_h : max_w;
            if (static_cast<s32>(count) < 4) {
                return true;
            }
            for (u32 i = 4;; i++) {
                const u32 q = max_fs / i;
                u32 dim = other;
                if (static_cast<s32>(q) <= static_cast<s32>(other)) {
                    dim = q;
                }
                if (0x64fffff < sdk) {
                    if (dim == 0) {
                        dim = 1;
                    }
                    frames = static_cast<u32>(
                        by_width ? SliceDpbFrames(i, dim, spec.dpb, spec.level, extra, sdk)
                                 : SliceDpbFrames(dim, i, spec.dpb, spec.level, extra, sdk));
                }
                const u64 size = by_width ? CpuWorkSize(static_cast<s32>(i), static_cast<s32>(dim),
                                                        static_cast<s32>(frames), add, work_pipes,
                                                        work_field, sdk_ge_400, sdk)
                                          : CpuWorkSize(static_cast<s32>(dim), static_cast<s32>(i),
                                                        static_cast<s32>(frames), add, work_pipes,
                                                        work_field, sdk_ge_400, sdk);
                if (size == 0) {
                    return false;
                }
                best = std::max(best, size);
                if ((1 - count) + i == 1) {
                    break;
                }
            }
            return true;
        };
        if (!sweep(true, pipes, 0) || !sweep(false, pipes, 0)) {
            return 3;
        }
        if (field != 0) {
            if (!sweep(true, pipes * 2, 1) || !sweep(false, pipes * 2, 1)) {
                return 3;
            }
        }
    } else {
        if (LevelMaxFs(&max_fs, spec.level, true, sdk) != 0) {
            return 3;
        }
        if (static_cast<s32>(max_fs) < spec.h_mbs * spec.w_mbs) {
            if (oversize != nullptr) {
                *oversize = 1;
            }
            return 3;
        }
        best = CpuWorkSize(spec.w_mbs, spec.h_mbs, static_cast<s32>(frames), add, pipes, 0,
                           sdk_ge_400, sdk);
        if (best == 0) {
            return 3;
        }
        if (field != 0) {
            const u64 size = CpuWorkSize(spec.w_mbs, spec.h_mbs, static_cast<s32>(frames), add,
                                         pipes * 2, 1, sdk_ge_400, sdk);
            if (size == 0) {
                return 3;
            }
            best = std::max(best, size);
        }
    }
    out->align = 0;
    out->size = best;
    out->base = 0;
    return 0;
}

// FUN_000237c0: shader constant pool, fixed for FW 12.02.
void ShaderPool(BufInfo* onion, BufInfo* garlic) {
    *onion = {0, 0x80018, 0x100};
    *garlic = {0, 0, 0};
}

// FUN_0001e0a0
s32 QueryGpu(const Spec& spec, u32 frames, u32 extra, s32 pipes, BufInfo* onion, BufInfo* garlic,
             u32 field, u8 sdk_ge_400, u32 sdk) {
    if (spec.codec != 0 || spec.w_mbs == 0 || spec.h_mbs == 0) {
        return 3;
    }
    BufInfo small{};
    BufInfo big{};
    u64 small_size = 0;
    u64 small_align = 0;
    u64 big_size = 0;
    u32 big_align = 0;
    bool found = false;
    bool skip_check = false;
    if (spec.w_mbs < 0 || spec.h_mbs < 0) {
        u32 max_fs = 0;
        if (LevelMaxFs(&max_fs, spec.level, false, sdk) != 0) {
            return 3;
        }
        u32 max_w = 0;
        u32 max_h = 0;
        if (LevelMaxDims(spec.level, &max_w, &max_h, sdk) != 0) {
            return 3;
        }
        u32 cur_frames = frames;
        const auto sweep = [&](bool by_width, s32 work_pipes, u8 work_field) -> bool {
            const u32 count = by_width ? max_w : max_h;
            const u32 other = by_width ? max_h : max_w;
            if (static_cast<s32>(count) < 4) {
                return true;
            }
            u32 px = 0x40;
            for (u32 i = 4;; i++) {
                u32 dim = other;
                if (static_cast<s32>(max_fs / i) <= static_cast<s32>(other)) {
                    dim = max_fs / i;
                }
                if (0x64fffff < sdk) {
                    const u32 d = dim != 0 ? dim : 1;
                    cur_frames = static_cast<u32>(
                        by_width ? SliceDpbFrames(i, d, spec.dpb, spec.level, extra, sdk)
                                 : SliceDpbFrames(d, i, spec.dpb, spec.level, extra, sdk));
                }
                const s32 r =
                    by_width ? ComputeResources(static_cast<s32>(px), static_cast<s32>(dim << 4),
                                                static_cast<s32>(cur_frames), work_pipes,
                                                work_field, &small, &big, sdk_ge_400)
                             : ComputeResources(static_cast<s32>(dim << 4), static_cast<s32>(px),
                                                static_cast<s32>(cur_frames), work_pipes,
                                                work_field, &small, &big, sdk_ge_400);
                if (r < 0) {
                    return false;
                }
                if (r == 0) {
                    if (small_size < small.size) {
                        found = true;
                        small_align = small.align;
                        small_size = small.size;
                    }
                    small_align &= 0xffffffff;
                    if (big_size < big.size) {
                        found = true;
                        big_align = big.align;
                        big_size = big.size;
                    }
                }
                px += 0x10;
                if ((1 - count) + i == 1) {
                    break;
                }
            }
            return true;
        };
        if (!sweep(true, pipes, 0) || !sweep(false, pipes, 0)) {
            return 3;
        }
        if (field != 0) {
            if (!sweep(true, pipes * 2, 1) || !sweep(false, pipes * 2, 1)) {
                return 3;
            }
        }
    } else {
        u32 max_fs = 0;
        if (LevelMaxFs(&max_fs, spec.level, true, sdk) != 0) {
            return 3;
        }
        if (static_cast<s32>(max_fs) < spec.h_mbs * spec.w_mbs) {
            return 3;
        }
        const s32 w = static_cast<s32>(static_cast<u32>(spec.w_mbs) << 4);
        const s32 h = static_cast<s32>(static_cast<u32>(spec.h_mbs) << 4);
        s32 r =
            ComputeResources(w, h, static_cast<s32>(frames), pipes, 0, &small, &big, sdk_ge_400);
        if (r < 0) {
            return 3;
        }
        if (r == 0 && small.size != 0) {
            small_size = small.size;
            small_align = small.align;
            big_align = big.align;
            big_size = big.size;
            found = true;
        }
        if (field != 0) {
            r = ComputeResources(w, h, static_cast<s32>(frames), pipes * 2, 1, &small, &big,
                                 sdk_ge_400);
            if (r < 0) {
                return 3;
            }
            if (r == 0) {
                if (small_size < small.size) {
                    small_align = small.align;
                    small_size = small.size;
                    found = true;
                }
                if (big_size < big.size) {
                    big_align = big.align;
                    big_size = big.size;
                    skip_check = true;
                }
            }
        }
    }
    if (!found && !skip_check) {
        return 3;
    }
    BufInfo pool_onion{};
    BufInfo pool_garlic{};
    ShaderPool(&pool_onion, &pool_garlic);
    u32 a = static_cast<u32>(small_align);
    if (a < pool_onion.align) {
        a = pool_onion.align;
    }
    onion->base = 0;
    onion->size = ((a - 1) + static_cast<u32>(pool_onion.size) & -a) + small_size;
    onion->align = a;
    u32 b = big_align;
    if (b < pool_garlic.align) {
        b = pool_garlic.align;
    }
    garlic->base = 0;
    garlic->size = ((b - 1) + static_cast<u32>(pool_garlic.size) & -b) + big_size;
    garlic->align = b;
    return 0;
}

// sceSdecQueryMemorySw2 -> FUN_0001eb50
s32 QueryMemorySw2(const Spec& spec, const QueryDec& dec, s32 one, BufInfo mem[3], s32* dpb_out,
                   u8 sdk_ge_400, u8* oversize, u32 sdk) {
    s32 r = 3;
    if (one != 0 && (spec.bit_depth_luma == spec.bit_depth_chroma ||
                     (spec.bit_depth_luma != 8 && spec.bit_depth_chroma != 8))) {
        const u32 dpb = DpbFrames(spec.dpb, spec.w_mbs, spec.h_mbs, spec.level, sdk);
        *dpb_out = static_cast<s32>(dpb);
        u32 pipes = dec.pipeline_depth;
        if (pipes < 0x11) {
            if (dec.pipe_mode != 2) {
                pipes = dec.pipe_mode == 1 ? 1 : 2;
            }
            const s32 extra = dec.extra_dpb;
            const u32 frames = dpb + 1 + static_cast<u32>(extra < 0 ? 0 : extra);
            const u32 field = spec.optimize_progressive == 0 && spec.codec != 1;
            r = QueryCpu(spec, frames, static_cast<u32>(extra), static_cast<s32>(pipes), &mem[0],
                         field, sdk_ge_400, oversize, sdk);
            if (r == 0) {
                r = QueryGpu(spec, frames, static_cast<u32>(dec.extra_dpb), static_cast<s32>(pipes),
                             &mem[1], &mem[2], field, sdk_ge_400, sdk);
            }
        }
    }
    mem[0].size += 0x38;
    return r;
}

} // namespace Savc2

// ---------------------------------------------------------------------------------------------
// libSceVdecCore
// ---------------------------------------------------------------------------------------------
namespace Core {

constexpr s32 kErrorArg = static_cast<s32>(0x80c00001);
constexpr s32 kErrorCodec = static_cast<s32>(0x80c00003);
constexpr s32 kErrorPitch = static_cast<s32>(0x80c0000c);
constexpr s32 kErrorFormat = static_cast<s32>(0x80c0000d);

constexpr u32 kFormatMap[7] = {0, 1, 2, 0, 0, 5, 6};                          // DAT_0007bea0
constexpr u32 kSeqFormatMap[9] = {0, 1, 3, 2, 4, 5, 6, 7, 8};                 // DAT_0007c794
constexpr u32 kSpecFormatMap[8] = {1, 2, 3, 4, 5, 0, 0, 6};                   // DAT_0007d168
constexpr u32 kCodecConst[4] = {0x608cee17, 0xd9114df8, 0x8d68c, 0x669a0c20}; // DAT_0007bce0

using Config = VdecCoreConfig;

struct CodecInfo {
    u32 codec;
    u32 profile;
    u32 level;
    s32 hevc_ext_mode;
    u32 optimize_progressive;
    u32 output_format;
    s32 max_dpb_frame_count;
    u32 pitch_align;
    s32 width_units;
    s32 height_units;
    u32 work_unit;
    u32 codec_const[4];
    u32 is_codec1;
    u8 bit_depth_luma;
    u8 bit_depth_chroma;
    u8 unk42;
    u8 pad43;
    u32 unit_size;
    s32 hevc_ext_2c;
    u8 unk4c;
    u8 pad4d[3];
    u32 unk50;
    u32 unk54;
};
static_assert(sizeof(CodecInfo) == 0x58);

struct DecoderInfo {
    u32 backend;
    u32 pipeline_depth_minus1;
    u32 async;
    u32 unk0c;
    u64 unk10;
    u32 hevc_ext_fmt;
    s32 extra_dpb_frame_count;
    u32 unk20;
    s32 cpu_thread_priority;
    u64 cpu_affinity_mask;
    u8 is_sw;
    u8 pad31[3];
    s32 max_pending_sync_count;
    u64 cpu_affinity_mask2;
    s32 cpu_thread_priority2;
    u32 unk44;
};
static_assert(sizeof(DecoderInfo) == 0x48);

// FUN_000033e0 output.
struct SeqInfo {
    s32 codec;
    s32 profile;
    u32 level;
    s32 hevc_ext_mode;
    u32 optimize_progressive;
    u32 format;
    u32 pitch_align;
    s32 max_dpb_frame_count;
    s32 width_units;
    s32 height_units;
    u32 unit;
    u32 work_unit;
    u32 is_codec1;
    s32 hevc_ext_2c;
    u32 codec_const[4];
    u8 bit_depth_luma;
    u8 bit_depth_chroma;
    u8 unk4a;
    u8 sei_enable;
    u8 sei_4c;
    u8 sei_4d;
    u8 sei_4e;
    s8 sei_4f;
    u8 sei_50;
    s8 sei_51;
    s8 sei_52;
    u8 pad53;
    s16 sei_54;
    s16 sei_56;
    u32 pad58[2];
};
static_assert(sizeof(SeqInfo) == 0x60);

// FUN_000045d0 output.
struct DecInfo {
    u32 backend;
    u32 pipeline_depth_minus1;
    u32 async;
    u32 unk0c;
    u64 unk10;
    u32 hevc_ext_fmt;
    u32 work_frames;
    s32 extra_dpb_frame_count;
    u32 unk24;
    s32 cpu_thread_priority;
    u32 pad2c;
    u64 cpu_affinity_mask;
    s32 max_pending_sync_count;
    u32 pad3c;
};
static_assert(sizeof(DecInfo) == 0x40);

// FUN_000033e0
bool BuildSeqInfo(u32 sw_avc, const CodecInfo& ci, SeqInfo& seq, u32 sdk) {
    bool bad = false;
    bool invalid;
    seq.codec = static_cast<s32>(ci.codec);
    switch (ci.codec) {
    case 0: {
        u32 profile = ci.profile;
        if (0x3e < profile - 0x42 ||
            ((0x4000000400000801ULL >> ((profile - 0x42) & 0x3f)) & 1) == 0) {
            bad = true;
            profile = 100;
        }
        seq.profile = static_cast<s32>(profile);
        switch (ci.level) {
        case 10:
        case 0xb:
        case 0xc:
        case 0xd:
        case 0x14:
        case 0x15:
        case 0x16:
        case 0x1e:
        case 0x1f:
        case 0x20:
        case 0x28:
        case 0x29:
        case 0x2a:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x6f:
            seq.level = ci.level;
            break;
        case 0x3c:
        case 0x3d:
        case 0x3e: {
            bool b = true;
            u32 level = 0x3e;
            if (0x64fffff < sdk) {
                level = 0x34;
                if (sw_avc != 0) {
                    level = ci.level;
                    b = bad;
                    if (level != 0x3c) {
                        level = level == 0x3d ? 0x3d : 0x3e;
                    }
                }
            } else {
                level = 0x34;
            }
            bad = b;
            seq.level = level;
            break;
        }
        default:
            seq.level = 0x64fffff < sdk ? 0x3e : 0x34;
            bad = true;
            break;
        }
        break;
    }
    case 1: {
        if (ci.profile == 2) {
            seq.profile = 2;
            bad = false;
        } else {
            if (ci.profile == 1) {
                seq.profile = 1;
                bad = !(ci.bit_depth_luma == 8 && ci.bit_depth_chroma == 8);
            } else {
                seq.profile = 2;
                bad = true;
            }
        }
        s32 ext = ci.hevc_ext_mode;
        if (ext != 0) {
            if (ext != 1) {
                bad = true;
            }
            ext = 1;
        }
        seq.hevc_ext_mode = ext;
        switch (ci.level) {
        case 0x1e:
        case 0x3c:
        case 0x3f:
        case 0x5a:
        case 0x5d:
        case 0x78:
        case 0x7b:
        case 0x96:
        case 0x99:
        case 0x9c:
            seq.level = ci.level;
            break;
        default:
            seq.level = 0x9c;
            bad = true;
            break;
        }
        break;
    }
    default:
        // Codec types 2..4 are not produced by libSceVdecsw.
        return true;
    }
    invalid = true;
    if (static_cast<u64>(static_cast<s64>(static_cast<s32>(ci.output_format))) < 9) {
        seq.format = kSeqFormatMap[ci.output_format];
        invalid = bad;
    }
    u32 unit = 0x10;
    seq.optimize_progressive = ci.optimize_progressive;
    seq.pitch_align = ci.pitch_align;
    seq.max_dpb_frame_count = ci.max_dpb_frame_count;
    s32 w = ci.width_units;
    seq.width_units = w;
    s32 h = ci.height_units;
    seq.height_units = h;
    bool ok_unit = true;
    if (ci.codec == 1) {
        unit = ci.unit_size;
        switch (unit) {
        case 8:
        case 0x10:
        case 0x20:
        case 0x40:
            break;
        case 0xffffffff:
            seq.width_units = -1;
            seq.height_units = -1;
            w = -1;
            h = -1;
            break;
        default:
            ok_unit = false;
            break;
        }
    }
    seq.unit = unit;
    if (!ok_unit) {
        invalid = true;
    }
    if (w == 0 || w == 0x7fffffff || h == 0x7fffffff || h == 0) {
        invalid = true;
    } else {
        if (w != -1 && 0x2000 < static_cast<u32>(w) * unit) {
            invalid = true;
        }
        if (h != -1 && 0x1100 < static_cast<u32>(h) * unit) {
            invalid = true;
        }
    }
    seq.work_unit = ci.work_unit;
    std::memcpy(seq.codec_const, ci.codec_const, sizeof(seq.codec_const));
    seq.is_codec1 = ci.is_codec1;
    if (2 < static_cast<u8>(ci.bit_depth_luma - 8) ||
        2 < static_cast<u8>(ci.bit_depth_chroma - 8)) {
        invalid = true;
    }
    if (ci.codec == 1) {
        seq.bit_depth_luma = ci.bit_depth_luma;
        seq.bit_depth_chroma = ci.bit_depth_chroma;
    } else {
        seq.bit_depth_luma = 8;
        seq.bit_depth_chroma = 8;
    }
    seq.unk4a = ci.unk42 != 0;
    seq.hevc_ext_2c = ci.hevc_ext_2c;
    seq.pad58[1] = 0;
    seq.sei_enable = 0;
    seq.sei_4c = 0;
    seq.sei_4d = 0;
    seq.sei_4e = 0;
    seq.sei_4f = 0;
    seq.sei_50 = 0;
    seq.sei_51 = 0;
    seq.sei_52 = 0;
    seq.pad53 = 0;
    return invalid;
}

// FUN_000045d0
s32 BuildDecInfo(const DecoderInfo& di, DecInfo& dec) {
    if (7 < di.pipeline_depth_minus1) {
        return 1;
    }
    dec.backend = di.backend;
    dec.pipeline_depth_minus1 = di.pipeline_depth_minus1;
    dec.async = di.async;
    u32 fmt = di.hevc_ext_fmt;
    switch (fmt) {
    case 0:
        break;
    case 1:
        break;
    case 2:
        if (di.async == 0 || (di.backend & 0xfffffffe) != 2) {
            fmt = 2;
        } else {
            fmt = 1;
        }
        break;
    case 3:
    case 4:
        if (di.async != 0 && (di.backend & 0xfffffffe) == 2) {
            return 1;
        }
        break;
    default:
        return 1;
    }
    dec.hevc_ext_fmt = fmt;
    dec.extra_dpb_frame_count = di.extra_dpb_frame_count;
    dec.unk24 = di.unk20;
    dec.cpu_thread_priority = di.cpu_thread_priority;
    if (di.async == 0) {
        dec.cpu_affinity_mask = 0;
    } else if (di.cpu_affinity_mask < 0x80) {
        dec.cpu_affinity_mask = di.cpu_affinity_mask;
    } else {
        dec.cpu_affinity_mask = 0x7f;
    }
    dec.unk10 = di.backend - 1 < 2 ? di.unk10 : 0;
    dec.max_pending_sync_count = di.max_pending_sync_count;
    return 0;
}

// FUN_0001b860
u32 MapAvcLevel(u32 level, u32 sdk) {
    switch (level) {
    case 10:
    case 0xb:
    case 0xc:
    case 0xd:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x1e:
    case 0x1f:
    case 0x20:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x6f:
        return level;
    default:
        return 0x64fffff < sdk ? 0x3e : 0x34;
    }
}

// FUN_00019b10 (AVC)
s32 BuildSpec(const SeqInfo& seq, const DecInfo& dec, Savc2::Spec& spec, u32 sdk) {
    if (seq.codec == 0) {
        const s32 p = seq.profile;
        spec.profile = (p == 0x80 || p == 100 || p == 0x4d) ? p : 0x42;
        spec.level = MapAvcLevel(seq.level, sdk);
    }
    // FUN_0001b810
    if (static_cast<u32>(seq.codec) - 1 < 4) {
        spec.codec = seq.codec;
    } else if (seq.codec == 0) {
        spec.codec = (dec.backend - 4 < 3) ? 5 : 0;
    } else {
        spec.codec = 0;
    }
    // FUN_0001b9f0
    u32 fmt = 0;
    if (seq.format - 1 < 8) {
        fmt = kSpecFormatMap[seq.format - 1];
    }
    spec.format = static_cast<s32>(fmt);
    spec.dpb = seq.max_dpb_frame_count;
    if (dec.async != 0 && fmt == 6) {
        spec.format = 0;
    }
    if (seq.width_units == -1 || seq.height_units == -1) {
        spec.w_mbs = -1;
        spec.h_mbs = -1;
        spec.w_rem = 0;
        spec.h_rem = 0;
    } else {
        const u32 w = static_cast<u32>(seq.width_units) * seq.unit;
        const u32 h = seq.unit * static_cast<u32>(seq.height_units);
        spec.w_mbs = static_cast<s32>(w >> 4);
        spec.h_mbs = static_cast<s32>(h >> 4);
        spec.w_rem = static_cast<s32>(w & 0xf);
        spec.h_rem = static_cast<s32>(h & 0xf);
        if (((h | w) & 1) != 0) {
            return 1;
        }
    }
    spec.unk14 = static_cast<s32>(seq.is_codec1);
    spec.pitch_align = seq.pitch_align != 0 ? static_cast<s32>(seq.pitch_align) : 0x10;
    spec.bit_depth_luma = seq.bit_depth_luma;
    spec.bit_depth_chroma = seq.bit_depth_chroma;
    spec.optimize_progressive = seq.codec == 1 ? 1 : static_cast<s32>(seq.optimize_progressive);
    return 0;
}

// FUN_00019d50
u64 AuxSize(const SeqInfo& seq) {
    const u32 fields = (seq.optimize_progressive == 0) + 1;
    u32 header = fields * 0x180;
    if (seq.codec != 3) {
        header = fields * 0x100;
    }
    const u64 base = static_cast<u64>((seq.work_unit + 0x7f & 0xffffff80) * fields + header);
    u32 sei = 0;
    if (seq.sei_enable != 0) {
        u32 a = 0;
        u32 b = 0;
        u32 c = 0;
        if (seq.sei_4c != 0) {
            s32 v = 0x60;
            if (seq.sei_4f == 2 || seq.sei_4f == -1) {
                v = (2 << (seq.sei_50 & 0x1f)) + 0x60;
            }
            if (seq.sei_4f == 3 || seq.sei_4f == -1) {
                v = v + seq.sei_54 * 4;
            }
            a = static_cast<u32>(v) + 7 & 0xfffffff8;
        }
        if (seq.sei_4e != 0) {
            b = static_cast<u32>((seq.sei_52 + seq.sei_51) * 0xc) + 0xa7 & 0xfffffff8;
        }
        if (seq.sei_4d != 0) {
            c = static_cast<u32>(seq.sei_56 * 4) + 0x47 & 0xfffffff8;
        }
        sei = c * seq.sei_4d + 0xcf + b * seq.sei_4e + a * seq.sei_4c & 0xffffff80;
    }
    return static_cast<u64>(fields * sei) + base;
}

// FUN_00018b20, first output only.
s32 SeiWorkSize(const SeqInfo& seq) {
    if (seq.sei_enable == 0) {
        return 0;
    }
    u32 a = 0;
    if (seq.sei_4c != 0) {
        s32 v;
        const s8 mode = seq.sei_4f;
        if (mode == 2 || mode == -1) {
            v = (2 << (seq.sei_50 & 0x1f)) + 0x60;
        } else {
            v = 0x60;
        }
        if (mode == 3 || mode == -1) {
            v += seq.sei_54 * 4;
        }
        a = static_cast<u32>(v) + 7 & 0xfffffff8;
    }
    u32 b = 0;
    if (seq.sei_4e != 0) {
        b = static_cast<u32>((seq.sei_52 + seq.sei_51) * 0xc) + 0xa7 & 0xfffffff8;
    }
    u32 c = 0;
    if (seq.sei_4d != 0) {
        c = static_cast<u32>(seq.sei_56 * 4) + 0x47 & 0xfffffff8;
    }
    return static_cast<s32>(seq.sei_4d * c + 0x50 + seq.sei_4e * b + seq.sei_4c * a);
}

// DAT_00086d00: {level, MaxDpbMbs, MaxFS, ...} entries used by FUN_00073fc0.
struct AvcLevelLimits {
    u32 level;
    u32 max_dpb_mbs;
    u32 max_fs;
};
constexpr AvcLevelLimits kAvcLevelLimits[] = {
    {0x0a, 0x18c, 0x63},     {0x6f, 0x18c, 0x63},     {0x0b, 0x384, 0x18c},
    {0x0c, 0x948, 0x18c},    {0x0d, 0x948, 0x18c},    {0x14, 0x948, 0x18c},
    {0x15, 0x1290, 0x318},   {0x16, 0x1fa4, 0x654},   {0x1e, 0x1fa4, 0x654},
    {0x1f, 0x4650, 0xe10},   {0x20, 0x5000, 0x1400},  {0x28, 0x8000, 0x2000},
    {0x29, 0x8000, 0x2000},  {0x2a, 0x8800, 0x2200},  {0x32, 0x1af40, 0x5640},
    {0x33, 0x2d000, 0x9000}, {0x34, 0x2d000, 0x9000}, {0x3c, 0x2d000, 0x9000},
    {0x3d, 0x2d000, 0x9000}, {0x3e, 0x2d000, 0x9000},
};

// FUN_000732b0 called with param_1 = 1 from FUN_000180c0 (AVC, all dimensions unknown).
s32 DpbFromLevel(const Savc2::Spec& spec, s32* dpb_out, u32 sdk) {
    const u32 w = static_cast<u32>(spec.w_mbs);
    const u32 h = static_cast<u32>(spec.h_mbs);
    if (static_cast<s32>(w) < -1 || w == 0 || static_cast<s32>(h) < -1 || h == 0) {
        return 3;
    }
    if (0 < static_cast<s32>(w) && 0x74 < w - 4) {
        return 3;
    }
    if (0 < static_cast<s32>(h) && 0x40 < h - 4) {
        return 3;
    }
    if (0x10 < spec.dpb) {
        return 3;
    }
    if (spec.codec != 0) {
        return 3;
    }
    u32 views;
    const s32 profile = spec.profile;
    if (profile == 0x42 || profile == 0x4d || profile == 100) {
        views = 1;
    } else if (profile == 0x80) {
        views = 2;
    } else {
        return 3;
    }
    if (6 < static_cast<u32>(spec.format)) {
        return 3;
    }
    const AvcLevelLimits* limits = nullptr;
    for (const auto& l : kAvcLevelLimits) {
        if (l.level == spec.level) {
            limits = &l;
        }
    }
    if (limits == nullptr) {
        return 3;
    }
    *dpb_out = -1;
    u32 frame_mbs;
    if (w != 0xffffffff && h != 0xffffffff) {
        frame_mbs = h * w;
    } else {
        frame_mbs = 0x64fffff < sdk ? limits->max_fs : 0;
    }
    if (frame_mbs != 0) {
        if (spec.dpb < 0) {
            const u32 q = (limits->max_dpb_mbs * views) / frame_mbs;
            const s32 frames = static_cast<s32>(q) < 0x10 ? static_cast<s32>(q) : 0x10;
            *dpb_out = frames;
            if (frames < 1) {
                return 3;
            }
        } else {
            *dpb_out = spec.dpb;
        }
    }
    return 0;
}

// FUN_000180c0, software (Savc2) path.
s32 QueryMemorySw(const SeqInfo& seq, u32 is_sw, const DecInfo& dec, BufInfo mem[3], BufInfo* fb,
                  BufInfo* aux, s32* dpb_out, u8* oversize, u32 sdk) {
    Savc2::Spec spec{};
    if (BuildSpec(seq, dec, spec, sdk) != 0) {
        return 1;
    }
    if (is_sw == 0 || dec.backend == 1 || dec.backend == 3) {
        // Hardware, HEVC (Shevc) and Savc backends are not reached by the AVC libSceVdecsw path.
        return 1;
    }
    Savc2::QueryDec qd{};
    qd.mode = dec.backend == 5 ? 1 : (dec.backend == 6) * 2;
    qd.extra_dpb = dec.extra_dpb_frame_count;
    qd.pipeline_depth = dec.work_frames;
    qd.cpu_thread_priority = dec.cpu_thread_priority;
    qd.unk10 = dec.unk10;
    qd.cpu_affinity_mask = dec.cpu_affinity_mask;
    if (dec.async == 0) {
        qd.pipe_mode = 0;
    } else {
        qd.pipe_mode = static_cast<s32>(dec.hevc_ext_fmt);
        if (dec.backend - 1 < 2) {
            if (dec.pipeline_depth_minus1 < 8) {
                qd.pipeline_depth = dec.pipeline_depth_minus1 + 1;
                qd.pipe_mode = 2;
            }
            if (dec.backend == 1) {
                qd.pipe_mode = 2 < dec.hevc_ext_fmt ? static_cast<s32>(dec.hevc_ext_fmt) : 2;
            }
        }
    }
    if (Savc2::QueryMemorySw2(spec, qd, 1, mem, dpb_out, 0x3ffffff < sdk, oversize, sdk) != 0) {
        return 1;
    }
    if (0x64fffff < sdk && dec.backend == 2 && seq.max_dpb_frame_count < 0 && seq.width_units < 0 &&
        seq.height_units < 0) {
        if (DpbFromLevel(spec, dpb_out, sdk) != 0) {
            return 1;
        }
    }
    s32 width;
    s32 height;
    if (seq.width_units == -1 || seq.height_units == -1) {
        if (seq.codec == 0) {
            const bool l6 = seq.level - 0x3c < 3;
            height = l6 ? 0x1100 : (seq.level == 0x34 ? 0x1000 : 0x880);
            width = l6 ? 0x2000 : 0x1000;
        } else {
            height = 0x880;
            width = 0x1000;
        }
    } else {
        width = seq.width_units * static_cast<s32>(seq.unit);
        height = seq.height_units * static_cast<s32>(seq.unit);
    }
    const u32 pitch_align = static_cast<u32>(spec.pitch_align);
    u32 pitch = pitch_align - 1 + static_cast<u32>(width);
    pitch = pitch - pitch % pitch_align;
    const u32 rows = static_cast<u32>(height) + 0xf & 0xfffffff0;
    const u32 bytes = rows * 3 * pitch;
    fb->align = 0x100;
    fb->size = (bytes >> 1) + 0xff & 0xffffff00;
    aux->size = 0;
    aux->align = 0;
    const u64 aux_size = AuxSize(seq);
    fb->size = aux_size + 0xc00 + static_cast<u64>(static_cast<u32>(fb->size) + 0x7f & 0xffffff80);
    if (sdk < 0x4000000) {
        mem[1].size += 0x800000;
        mem[0].size += 0x7bfe00;
    } else {
        mem[1].size += 0x100000;
        mem[0].size += sdk < 0x6500000 ? 0xc0000 : 0xb5f80;
    }
    if (0x4ffffff < sdk) {
        mem[0].size -= 0x1000;
    }
    return 0;
}

// FUN_000188b0
s32 FrameCounts(u32 sdk, const DecInfo& dec, const SeqInfo& seq, s32* dpb_out, s32* cpu_frames,
                s32* ref_frames, s32* sei_frames, s32* views) {
    *sei_frames = 0;
    *views = 0;
    *ref_frames = 0;
    *cpu_frames = 0;
    s32 pipes;
    if (dec.async == 0) {
        pipes = 5;
    } else if (dec.backend == 3) {
        pipes = dec.hevc_ext_fmt != 1 ? 2 - (dec.extra_dpb_frame_count == 0) : 1;
    } else if (dec.pipeline_depth_minus1 < 8) {
        pipes = static_cast<s32>(dec.pipeline_depth_minus1 + 1);
    } else {
        return 1;
    }
    u32 extra = static_cast<u32>(dec.extra_dpb_frame_count);
    if (0x64fffff < sdk) {
        const u32 e = 0xf < extra ? extra : 0x10;
        if (dec.backend == 2) {
            extra = e;
        }
    }
    s32 max_refs;
    s32 view_count;
    if (seq.codec == 1) {
        max_refs = 0x11;
        view_count = 1;
    } else if (seq.codec == 0) {
        view_count = (seq.profile == 0x80) + 1;
        max_refs = (seq.profile == 0x80) + 0x11;
    } else {
        view_count = 1;
        max_refs = 3;
    }
    const s32 dpb = seq.max_dpb_frame_count;
    if (dpb < 0) {
        if (*dpb_out < 0) {
            *dpb_out = max_refs - view_count;
        }
    } else {
        if (-1 < *dpb_out && dpb != *dpb_out) {
            return 1;
        }
        if (*dpb_out < 0) {
            *dpb_out = dpb;
        }
    }
    s32 cpu;
    if (dec.async == 0) {
        cpu = (sdk < 0x5000000 ? pipes + 1 : pipes + 0x13) + static_cast<s32>(extra);
    } else {
        cpu = (dec.backend == 3) + static_cast<s32>(extra);
        const s32 add = 0x4ffffff < sdk ? 0x13 : 1;
        if (dec.backend == 3) {
            cpu += add;
            pipes = 0;
        } else {
            cpu += add + pipes;
        }
    }
    const s32 refs = static_cast<s32>(extra) + max_refs + pipes;
    if (refs == 0) {
        return 2;
    }
    *cpu_frames = cpu;
    *ref_frames = refs;
    *views = view_count;
    *sei_frames = max_refs;
    return 0;
}

// FUN_00019240
s32 InstanceMemory(const SeqInfo& seq, const DecInfo& dec, BufInfo out[3], BufInfo* frames,
                   u8* oversize, u32 sdk) {
    *frames = {};
    out[0] = {};
    out[1] = {};
    out[2] = {};
    if (dec.async != 0 && 1 < static_cast<u32>(seq.codec)) {
        return 1;
    }
    const u32 pitch = seq.pitch_align;
    if ((pitch & 0xf) != 0 || 1 < __builtin_popcount(pitch)) {
        return 1;
    }
    if (0xffff < seq.work_unit) {
        return 1;
    }
    if (0x10 < seq.max_dpb_frame_count) {
        return 1;
    }
    u32 max_w;
    u32 max_h;
    const u32 async = dec.async;
    const u32 backend = dec.backend;
    if (async != 0) {
        if (backend == 1) {
            max_w = 0x100;
            max_h = 0x88;
        } else if (backend == 2) {
            max_w = 0x200;
            max_h = 0x110;
        } else if (backend == 3) {
            if (seq.format != 0) {
                return 1;
            }
            max_w = 0x100;
            max_h = 0x88;
        } else {
            return 1;
        }
        if (seq.format - 7 < 2 && pitch < 0x40) {
            return 1;
        }
    } else if ((backend | 2) == 6) {
        if (1 < static_cast<u32>(seq.codec)) {
            return 1;
        }
        max_w = 0x100;
        max_h = 0x88;
    } else {
        max_w = 0x78;
        max_h = 0x44;
    }
    const u32 w = static_cast<u32>(seq.width_units);
    const u32 h = static_cast<u32>(seq.height_units);
    if (w != 0xffffffff) {
        if (h != 0xffffffff) {
            if (h == 0 || static_cast<s32>(w) < -1 || w == 0 || static_cast<s32>(h) < -1) {
                return 1;
            }
            if (async == 0 || backend != 2) {
                if (max_w << 4 < seq.unit * w || max_h * 0x10 < seq.unit * h) {
                    return 1;
                }
            } else if (static_cast<s32>(0x22000 / static_cast<u64>(h)) < static_cast<s32>(w)) {
                return 1;
            }
        }
        if (static_cast<s32>(w) < static_cast<s32>(0x40 / static_cast<u64>(seq.unit))) {
            return 1;
        }
    }
    if (h != 0xffffffff &&
        static_cast<s32>(h) < static_cast<s32>(0x40 / static_cast<u64>(seq.unit))) {
        return 1;
    }
    if (async != 0 && seq.codec == 1) {
        if (seq.bit_depth_chroma != seq.bit_depth_luma &&
            (seq.bit_depth_chroma == 8 || seq.bit_depth_luma == 8)) {
            return 1;
        }
    }
    BufInfo mem[3]{};
    BufInfo fb{};
    BufInfo aux{};
    s32 dpb_out = 0;
    if (QueryMemorySw(seq, async, dec, mem, &fb, &aux, &dpb_out, oversize, sdk) != 0) {
        return 1;
    }
    s32 cpu_frames;
    s32 ref_frames;
    s32 sei_frames;
    s32 views;
    if (FrameCounts(sdk, dec, seq, &dpb_out, &cpu_frames, &ref_frames, &sei_frames, &views) != 0) {
        return 1;
    }
    frames->size = static_cast<u64>(static_cast<s64>(
                       static_cast<s32>(views + dpb_out + dec.extra_dpb_frame_count))) *
                   fb.size;
    frames->align = fb.align;
    u64 base = 0x1ce100;
    u64 cg_extra = 0;
    u32 cg_align = 0;
    switch (seq.codec) {
    case 0:
        break;
    case 1:
        base = 0x47e400;
        break;
    case 2:
        base = 0xfe850;
        cg_extra = fb.size;
        cg_align = fb.align;
        break;
    case 3:
        base = 0x1026c0;
        break;
    case 4:
        base = 0x102418;
        break;
    default:
        base = 0;
        break;
    }
    u32 aux_bytes = static_cast<u32>(aux.size);
    if (aux.align != 0) {
        aux_bytes = static_cast<u32>(aux.size) + (aux.align - 1) & -aux.align;
    }
    const u32 work_unit = seq.work_unit;
    s32 sei = SeiWorkSize(seq);
    if (sdk < 0x7fffffff || dec.async != 0) {
        sei = (ref_frames * 2 + 2) * sei;
    } else {
        sei = sei * ref_frames + sei;
    }
    const u32 cpu_align = mem[0].align;
    if (cpu_align != 0) {
        base = static_cast<u32>(static_cast<s32>(base) - 1 + static_cast<s32>(cpu_align) &
                                -static_cast<s32>(cpu_align));
    }
    const u32 cpu_mem = static_cast<u32>(mem[0].size);
    out[0].size =
        static_cast<u64>(static_cast<u32>(sei_frames * 0x110) + 0x7f & 0xffffff80) +
        static_cast<u64>(static_cast<u32>(ref_frames * 0x2b0) + 0x7f & 0xffffff80) +
        static_cast<u64>(aux_bytes * 2 + 0x7f & 0xffffff80) +
        static_cast<u64>(cpu_mem + 0x7f & 0xffffff80) +
        static_cast<u64>(static_cast<u32>(cpu_frames * 0xe0) + 0x7f & 0xffffff80) +
        static_cast<u64>((static_cast<u32>(ref_frames) * work_unit + work_unit) * 2 + 0x7f &
                         0xffffff80) +
        static_cast<u64>(static_cast<u32>(sei) + 0x7f & 0xffffff80) + base;
    out[0].align = 0x80;
    u32 onion_align = mem[1].align;
    if (onion_align < cg_align) {
        onion_align = cg_align;
    }
    out[1].align = onion_align;
    out[1].size = cg_extra + mem[1].size;
    out[2].size = mem[2].size;
    out[2].align = mem[2].align;
    return 0;
}

// FUN_00003e00 (libSceVdecsw path: codec 0/1, async software decoder).
s32 QueryInstance(const CodecInfo& ci, const DecoderInfo& di, BufInfo out[3], BufInfo* frames,
                  u32 sdk) {
    bool is_avc = false;
    bool is_hevc = false;
    u32 codec = ci.codec;
    if (codec == 1) {
        // FUN_00004520 (debug hook, no effect) when async.
        if (ci.bit_depth_chroma != ci.bit_depth_luma &&
            (ci.bit_depth_chroma == 8 || ci.bit_depth_luma == 8)) {
            return 1;
        }
        is_hevc = true;
    } else if (codec == 0) {
        is_avc = true;
        if (di.backend <= 5 && ((0x29u >> di.backend) & 1) != 0 &&
            !(static_cast<s32>(ci.level) < 0x34 || ci.level == 0x6f)) {
            return 1;
        }
    }
    const u32 fmt = di.hevc_ext_fmt;
    if (4 < static_cast<s32>(fmt)) {
        return 1;
    }
    if (1 < static_cast<s32>(fmt) && is_avc && di.async != 0) {
        return 1;
    }
    if (di.async == 0) {
        // Hardware decoders are not reached by libSceVdecsw.
        return 1;
    }
    if (ci.output_format == 5 || ci.unk42 != 0) {
        return 1;
    }
    const u32 backend = di.backend;
    if (backend - 2 < 2) {
        if (!is_avc) {
            return 1;
        }
        if (ci.profile == 0x80) {
            return 1;
        }
    } else if (backend == 1) {
        if (!is_hevc) {
            return 1;
        }
        if (is_avc && ci.profile == 0x80) {
            return 1;
        }
    } else {
        return 1;
    }
    if (ci.output_format == 6) {
        if (ci.codec != 1 || di.async == 0) {
            return 1;
        }
    }
    *frames = {};
    out[0] = {};
    out[1] = {};
    out[2] = {};
    const u32 sw_avc = di.async != 0 ? (backend == 2) : 0;
    SeqInfo seq{};
    if (BuildSeqInfo(sw_avc, ci, seq, sdk)) {
        return 1;
    }
    DecInfo dec{};
    if (BuildDecInfo(di, dec) != 0) {
        return 1;
    }
    dec.work_frames = 7;
    if (0x4ffffff < sdk) {
        const bool b8 = ci.bit_depth_luma == 8 && ci.bit_depth_chroma == 8;
        const bool b10 = ci.bit_depth_luma == 10 && ci.bit_depth_chroma == 10;
        if (!b8 && !b10) {
            return 4;
        }
    }
    BufInfo mem[3];
    BufInfo fr;
    if (InstanceMemory(seq, dec, mem, &fr, nullptr, sdk) != 0) {
        return 1;
    }
    out[0] = mem[0];
    out[1] = mem[1];
    out[2] = mem[2];
    *frames = fr;
    const u64 align = out[0].align;
    u64 sync = 0xf10;
    if (static_cast<u32>(dec.max_pending_sync_count) != 0xffffffff) {
        sync = static_cast<u64>(static_cast<u32>(dec.max_pending_sync_count)) * 0x18 + 0xf10;
    }
    if (1 < align) {
        sync = (align - 1) + sync;
        sync = sync - sync % align;
    }
    if (sync + out[0].size < sync) {
        return 4;
    }
    out[0].size += sync;
    return 0;
}

// FUN_0001a1b0
s32 QueryFrameBufferSw(const SeqInfo& seq, const DecInfo& dec, BufInfo* aux, BufInfo* fb,
                       s32* count, u32 sdk) {
    *aux = {};
    *fb = {};
    if (0xffff < seq.work_unit) {
        return 1;
    }
    s32 views = 1;
    if (seq.codec == 0) {
        views = (seq.profile == 0x80) + 1;
    }
    BufInfo mem[3]{};
    BufInfo f{};
    BufInfo a{};
    s32 dpb_out = 0;
    if (QueryMemorySw(seq, dec.async, dec, mem, &f, &a, &dpb_out, nullptr, sdk) != 0) {
        return 1;
    }
    fb->size = f.size;
    fb->align = f.align;
    *count = dpb_out < 0 ? -1 : views + dpb_out;
    aux->align = 0x80;
    aux->size = AuxSize(seq);
    return 0;
}

// FUN_000046f0
s32 QueryFrameBuffer(const CodecInfo& ci, const DecoderInfo& di, BufInfo* fb, BufInfo* aux,
                     s32* count, u32 sdk) {
    *fb = {};
    *aux = {};
    *count = -1;
    const u32 sw_avc = di.async != 0 ? (di.backend == 2) : 0;
    SeqInfo seq{};
    if (BuildSeqInfo(sw_avc, ci, seq, sdk)) {
        return 1;
    }
    DecInfo dec{};
    if (BuildDecInfo(di, dec) != 0) {
        return 1;
    }
    dec.work_frames = 7;
    BufInfo a{};
    BufInfo f{};
    if (QueryFrameBufferSw(seq, dec, &a, &f, count, sdk) != 0) {
        return 1;
    }
    if (sdk < 0x6500000) {
        if (di.async == 0 && (seq.width_units == -1 || seq.height_units == -1) && *count != -1) {
            return 4;
        }
    } else if (di.async != 0 && di.backend == 2 &&
               (seq.width_units == -1 || seq.height_units == -1) && *count == -1) {
        return 4;
    }
    BufInfo inst[3];
    BufInfo frames;
    const s32 r = QueryInstance(ci, di, inst, &frames, sdk);
    if (r != 0) {
        return r;
    }
    if (frames.align != f.align) {
        return 4;
    }
    *fb = f;
    *aux = a;
    return 0;
}

// FUN_00001070, resource type 3 (libSceVdecsw).
s32 BuildCodecInfo(const Config& cfg, CodecInfo& ci, DecoderInfo& di, BufInfo* fb, BufInfo* aux,
                   s32* count, u32 sdk) {
    ci.unk42 = 0;
    ci.unit_size = 0x10;
    ci.hevc_ext_2c = -1;
    ci.bit_depth_luma = 8;
    ci.bit_depth_chroma = 8;
    ci.unk4c = 0;
    ci.hevc_ext_mode = -1;
    ci.optimize_progressive = 0;
    di.hevc_ext_fmt = 0;
    u32 unit = 0x10;
    switch (cfg.codec) {
    case 0:
        ci.codec = 0;
        ci.profile = cfg.profile;
        ci.level = cfg.level;
        break;
    case 4:
        ci.codec = 1;
        ci.profile = cfg.profile;
        ci.level = cfg.level;
        if (cfg.hevc_ext_mode > 1) {
            return kErrorArg;
        }
        ci.hevc_ext_mode = cfg.hevc_ext_mode;
        ci.unit_size = 8;
        unit = 8;
        ci.bit_depth_luma = cfg.bit_depth_luma;
        ci.bit_depth_chroma = cfg.bit_depth_chroma;
        ci.hevc_ext_2c = cfg.hevc_ext_2c;
        if (cfg.hevc_ext_30 != 0 && (cfg.resource_type & 0xfffffffe) == 2) {
            if (cfg.hevc_ext_30 == 2) {
                di.hevc_ext_fmt = 4;
            } else if (cfg.hevc_ext_30 == 1) {
                di.hevc_ext_fmt = 3;
            } else {
                return kErrorArg;
            }
        }
        break;
    case 1:
    case 2:
    case 3:
        // Not produced by libSceVdecsw.
        return kErrorCodec;
    default:
        return kErrorCodec;
    }
    const u32 fmt = cfg.output_format;
    if (6 < fmt || ((0x67u >> (fmt & 0x1f)) & 1) == 0) {
        return kErrorFormat;
    }
    ci.output_format = kFormatMap[fmt];
    const u32 dpb = static_cast<u32>(cfg.max_dpb_frame_count);
    if (0x11 < dpb + 1) {
        return kErrorArg;
    }
    ci.max_dpb_frame_count = static_cast<s32>(dpb);
    const s32 w = cfg.max_frame_width;
    const s32 h = cfg.max_frame_height;
    if (w < -1 || h < -1) {
        return kErrorArg;
    }
    if (static_cast<u32>(w) < 0x31 || static_cast<u32>(h) < 0x31) {
        return kErrorArg;
    }
    const auto units = [&](u32 v) -> s32 {
        u32 q;
        u32 r;
        if (cfg.codec == 4) {
            q = v / unit;
            r = (unit - 1) & v;
        } else {
            r = v & 0xf;
            q = v >> 4;
        }
        return static_cast<s32>((q + 1) - (r == 0));
    };
    s32 wu = -1;
    s32 hu = -1;
    if (w != -1) {
        wu = units(static_cast<u32>(w));
    }
    ci.width_units = wu;
    if (h == -1) {
        ci.width_units = -1;
        ci.height_units = -1;
    } else {
        hu = units(static_cast<u32>(h));
        ci.height_units = hu;
        if (wu == -1 || hu == -1) {
            ci.width_units = -1;
            ci.height_units = -1;
        }
    }
    const s32 pitch = cfg.pitch_align != 0xffffffff ? static_cast<s32>(cfg.pitch_align) : 0;
    if (pitch < 0) {
        return kErrorPitch;
    }
    ci.pitch_align = static_cast<u32>(pitch);
    ci.is_codec1 = cfg.codec == 1;
    std::memcpy(ci.codec_const, kCodecConst, sizeof(kCodecConst));
    di.max_pending_sync_count = 0;
    di.is_sw = 0;
    di.extra_dpb_frame_count = 0xff;
    di.unk20 = 0;
    if (cfg.resource_type != 3) {
        // Only the libSceVdecsw resource type is ported.
        return kErrorCodec;
    }
    const u32 depth = cfg.decode_pipeline_depth;
    if (7 < depth - 1) {
        return kErrorArg;
    }
    di.pipeline_depth_minus1 = depth - 1;
    di.cpu_affinity_mask = cfg.cpu_affinity_mask;
    di.cpu_thread_priority = cfg.cpu_thread_priority;
    ci.optimize_progressive = cfg.not_optimize_progressive == 0;
    di.is_sw = 1;
    di.cpu_affinity_mask2 = cfg.cpu_affinity_mask;
    di.cpu_thread_priority2 = cfg.cpu_thread_priority;
    di.async = 1;
    di.backend = 2 - (cfg.codec == 4);
    ci.work_unit = 0x300;
    di.extra_dpb_frame_count = cfg.extra_dpb_frame_count;
    di.max_pending_sync_count = cfg.max_pending_sync_count;
    if (di.extra_dpb_frame_count == 0xff) {
        return kErrorArg;
    }
    if (QueryFrameBuffer(ci, di, fb, aux, count, sdk) != 0) {
        return kErrorArg;
    }
    if (*count == 0) {
        return kErrorArg;
    }
    if ((fb->align - 1 & fb->align) != 0 || (aux->align - 1 & aux->align) != 0) {
        return kErrorArg;
    }
    u32 extra = static_cast<u32>(di.extra_dpb_frame_count);
    if (extra == 0) {
        if (sdk < 0x5000000) {
            if ((cfg.resource_type & 0xfffffffe) == 2 && ci.max_dpb_frame_count != -1) {
                di.extra_dpb_frame_count = ci.max_dpb_frame_count < static_cast<s32>(depth)
                                               ? static_cast<s32>(depth)
                                               : ci.max_dpb_frame_count;
            } else {
                di.extra_dpb_frame_count = 0x10;
            }
        } else if (static_cast<s32>(depth) < *count) {
            di.extra_dpb_frame_count = *count - 1;
        } else {
            di.extra_dpb_frame_count = static_cast<s32>(depth);
        }
        return 0;
    }
    if (extra - 1 < 0x10) {
        return 0;
    }
    return kErrorArg;
}

// sceVdecCoreQueryInstanceSize
s32 QueryInstanceSize(const Config& cfg, u64* cpu, u64* gpu, u64* cpu_gpu, u32 sdk) {
    CodecInfo ci{};
    DecoderInfo di{};
    BufInfo fb{};
    BufInfo aux{};
    s32 count = 0;
    const s32 r = BuildCodecInfo(cfg, ci, di, &fb, &aux, &count, sdk);
    if (r != 0) {
        return r;
    }
    BufInfo mem[3];
    BufInfo frames;
    if (QueryInstance(ci, di, mem, &frames, sdk) != 0) {
        return kErrorArg;
    }
    if ((mem[0].size != 0 && (mem[0].align - 1 & mem[0].align) != 0) ||
        (mem[2].size != 0 && (mem[2].align - 1 & mem[2].align) != 0) ||
        (mem[1].size != 0 && (mem[1].align - 1 & mem[1].align) != 0)) {
        return kErrorArg;
    }
    *cpu = mem[0].size + 0x40000 + mem[0].align;
    *gpu = mem[2].size + mem[2].align;
    *cpu_gpu = mem[1].size + mem[1].align;
    return 0;
}

// sceVdecCoreQueryFrameBufferInfo
s32 QueryFrameBufferInfo(const Config& cfg, s32 info[6], u32 sdk) {
    CodecInfo ci{};
    DecoderInfo di{};
    BufInfo fb{};
    BufInfo aux{};
    s32 count = 0;
    const s32 r = BuildCodecInfo(cfg, ci, di, &fb, &aux, &count, sdk);
    if (r != 0) {
        return r;
    }
    info[0] = static_cast<s32>(fb.size);
    info[1] = static_cast<s32>(fb.align);
    info[2] = cfg.max_dpb_frame_count + 1;
    info[3] = count;
    info[4] = static_cast<s32>(aux.size);
    info[5] = static_cast<s32>(aux.align);
    return 0;
}

} // namespace Core

namespace Sw {

bool IsAvcLevel(u32 level) {
    switch (level) {
    case 10:
    case 0xb:
    case 0xc:
    case 0xd:
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x1e:
    case 0x1f:
    case 0x20:
    case 0x28:
    case 0x29:
    case 0x2a:
    case 0x32:
    case 0x33:
    case 0x34:
    case 0x3c:
    case 0x3d:
    case 0x3e:
    case 0x6f:
        return true;
    default:
        return false;
    }
}

bool IsHevcLevel(u32 level) {
    return (level - 0x1e < 0x40 && ((0x9000000240000001ULL >> ((level - 0x1e) & 0x3f)) & 1)) ||
           (level - 0x78 < 0x25 && ((0x1240000009ULL >> ((level - 0x78) & 0x3f)) & 1));
}

bool QueryMemory(const void* addr) {
    ::Libraries::Kernel::OrbisVirtualQueryInfo info{};
    return ::Core::Memory::Instance()->VirtualQuery(std::bit_cast<VAddr>(addr), 0, &info) == 0;
}

} // namespace Sw

} // namespace

// libSceVdecsw FUN_00003440
s32 VdecswBuildCoreConfig(const OrbisVdecswDecoderConfigInfo& cfg,
                          OrbisVdecswDecoderMemoryInfo& mem, VdecCoreConfig& core, u32 sdk,
                          u32 cpumode) {
    if (cfg.resource_type != 1) {
        return ORBIS_VDECSW_ERROR_RESOURCE_TYPE;
    }
    core.resource_type = 3;
    core.output_format = 0;
    const u32 codec_type = static_cast<u32>(cfg.codec_type);
    if (codec_type == static_cast<u32>(OrbisVdecswCodecType::Avc)) {
        core.codec = 0;
        if (cfg.extra_config_info != nullptr) {
            return ORBIS_VDECSW_ERROR_EXTRA_CONFIG_INFO;
        }
        const u32 p = cfg.profile - 0x42;
        if (!(p < 0x23 && ((0x400000801ULL >> (p & 0x3f)) & 1) != 0)) {
            return ORBIS_VDECSW_ERROR_PROFILE_LEVEL;
        }
        if (!Sw::IsAvcLevel(cfg.max_level)) {
            return ORBIS_VDECSW_ERROR_PROFILE_LEVEL;
        }
    } else if (codec_type == static_cast<u32>(OrbisVdecswCodecType::Hevc)) {
        core.codec = 4;
        core.hevc_align = (static_cast<u32>(cfg.max_frame_height) & 0xf) == 0 ? 0x10 : 8;
        const u32 profile = cfg.profile;
        if (1 < profile - 1) {
            return ORBIS_VDECSW_ERROR_PROFILE_LEVEL;
        }
        if (!Sw::IsHevcLevel(cfg.max_level)) {
            return ORBIS_VDECSW_ERROR_PROFILE_LEVEL;
        }
        if (cfg.extra_config_info == nullptr) {
            core.hevc_ext_mode = 1;
            core.hevc_ext_2c = -1;
            const u8 depth = static_cast<u8>((profile == 2) * 2 | 8);
            core.bit_depth_luma = depth;
            core.bit_depth_chroma = depth;
            core.hevc_ext_30 = 0;
        } else {
            if (!Sw::QueryMemory(cfg.extra_config_info)) {
                return ORBIS_VDECSW_ERROR_EXTRA_CONFIG_INFO;
            }
            const auto* extra =
                static_cast<const OrbisVdecswHevcExtraConfigInfo*>(cfg.extra_config_info);
            const u64 size = extra->this_size;
            if (size != 0x18) {
                if (size != 0x20) {
                    return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
                }
                if (extra->hdr_frame_format != 0) {
                    if (extra->hdr_frame_format != 0xc24a) {
                        return ORBIS_VDECSW_ERROR_EXTRA_CONFIG_INFO;
                    }
                    core.output_format = 6;
                }
                const u32 load = extra->gpu_load_level - 1;
                if (2 < load) {
                    return ORBIS_VDECSW_ERROR_EXTRA_CONFIG_INFO;
                }
                core.hevc_ext_30 = static_cast<u8>(load);
            }
            if (1 < static_cast<u32>(extra->tier_type)) {
                return ORBIS_VDECSW_ERROR_EXTRA_CONFIG_INFO;
            }
            core.hevc_ext_mode = static_cast<u8>(extra->tier_type);
            core.bit_depth_luma = static_cast<u8>(extra->max_bit_depth_luma);
            core.bit_depth_chroma = static_cast<u8>(extra->max_bit_depth_chroma);
            core.hevc_ext_2c = extra->max_temporal_id_to_decode;
        }
    } else {
        return ORBIS_VDECSW_ERROR_CODEC_TYPE;
    }
    core.profile = cfg.profile;
    core.level = cfg.max_level;
    if (cfg.reserved0 != 0) {
        return ORBIS_VDECSW_ERROR_CONFIG_INFO;
    }
    if (7 < cfg.decode_pipeline_depth - 1) {
        return ORBIS_VDECSW_ERROR_INPUT_QUEUE_DEPTH;
    }
    core.decode_pipeline_depth = cfg.decode_pipeline_depth;
    core.max_frame_width = cfg.max_frame_width;
    core.max_frame_height = cfg.max_frame_height;
    const u32 dpb = static_cast<u32>(cfg.max_dpb_frame_count);
    if (!(dpb + 1 < 0x12)) {
        return ORBIS_VDECSW_ERROR_DPB_FRAME_COUNT;
    }
    core.max_dpb_frame_count = static_cast<s32>(dpb);
    core.pitch_align = 0x100;
    core.unk24 = 0;
    core.not_optimize_progressive = static_cast<u32>(cfg.optimize_progressive_video) ^ 1;
    const s32 prio = cfg.cpu_thread_priority;
    if (prio != -1 && 0x1ff < static_cast<u32>(prio) - 0x100) {
        return ORBIS_VDECSW_ERROR_THREAD_PRIORITY;
    }
    core.cpu_thread_priority = prio;
    const u64 mask = cfg.cpu_affinity_mask;
    if (((static_cast<u64>((cpumode & 1) == 0) << 6 | 0xffffffffffffff80ULL) & mask) != 0) {
        return ORBIS_VDECSW_ERROR_AFFINITY_MASK;
    }
    core.cpu_affinity_mask = mask;
    const s8 extra_dpb = cfg.extra_dpb_frame_count;
    if (extra_dpb == -1) {
        core.extra_dpb_frame_count = 0;
    } else if (static_cast<u8>(extra_dpb - 1) < 0x10) {
        core.extra_dpb_frame_count = extra_dpb;
    } else {
        return ORBIS_VDECSW_ERROR_DPB_FRAME_COUNT;
    }
    s32 pending = 0;
    if (cfg.this_size == sizeof(OrbisVdecswDecoderConfigInfo)) {
        pending = -1;
        if (!cfg.disable_sync_decode_output) {
            pending = cfg.max_pending_sync_count;
            if (pending == -1) {
                return ORBIS_VDECSW_ERROR_CONFIG_INFO;
            }
        }
    }
    core.max_pending_sync_count = pending;
    if (Core::QueryInstanceSize(core, &mem.cpu_memory_size, &mem.gpu_memory_size,
                                &mem.cpu_gpu_memory_size, sdk) != 0) {
        return ORBIS_VDECSW_ERROR_CONFIG_INFO;
    }
    mem.cpu_memory_size += 0x40000;
    s32 info[6]{};
    if (Core::QueryFrameBufferInfo(core, info, sdk) != 0) {
        return ORBIS_VDECSW_ERROR_CONFIG_INFO;
    }
    mem.max_frame_buffer_size = static_cast<u64>(static_cast<s64>(info[0]));
    mem.frame_buffer_alignment = static_cast<u32>(info[1]);
    return ORBIS_OK;
}

s32 VdecCoreQueryInstanceSize(const VdecCoreConfig& cfg, u64* cpu_size, u64* gpu_size,
                              u64* cpu_gpu_size, u32 sdk) {
    return Core::QueryInstanceSize(cfg, cpu_size, gpu_size, cpu_gpu_size, sdk);
}

s32 VdecCoreQueryFrameBufferInfo(const VdecCoreConfig& cfg, s32 info[6], u32 sdk) {
    return Core::QueryFrameBufferInfo(cfg, info, sdk);
}

s32 VdecCoreGetDecoderParams(const VdecCoreConfig& cfg, VdecCoreDecoderParams& params, u32 sdk) {
    Core::CodecInfo ci{};
    Core::DecoderInfo di{};
    BufInfo fb{};
    BufInfo aux{};
    s32 count = 0;
    const s32 r = Core::BuildCodecInfo(cfg, ci, di, &fb, &aux, &count, sdk);
    if (r != 0) {
        return r;
    }
    params.codec = ci.codec;
    params.profile = ci.profile;
    params.level = ci.level;
    params.width_units = ci.width_units;
    params.height_units = ci.height_units;
    params.unit_size = ci.unit_size;
    params.max_dpb_frame_count = ci.max_dpb_frame_count;
    params.decode_pipeline_depth = di.pipeline_depth_minus1 + 1;
    params.extra_dpb_frame_count = di.extra_dpb_frame_count;
    params.max_pending_sync_count = di.max_pending_sync_count;
    params.optimize_progressive = ci.optimize_progressive;
    params.frame_buffer_size = fb.size;
    params.frame_buffer_alignment = fb.align;
    params.pitch_align = ci.pitch_align;
    return 0;
}

} // namespace Libraries::Vdecsw
