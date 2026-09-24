// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// H.264 header parsing for the libSceVdecCore software decoder path (FW 12.02).
// Syntax follows ITU-T H.264 7.3/E.1/D.1; value ranges, defaults and the post-parse
// adjustments follow the libSceVdecCore syntax tables (DAT_000ba110 SPS, DAT_000beed0 VUI)
// and FUN_000542a0 / FUN_00052610 / FUN_00052680 / FUN_000526f0 / FUN_00011660.

#include <algorithm>
#include <vector>

#include "core/libraries/videodec/vdecsw_avc.h"

namespace Libraries::Vdecsw {

namespace {

// DAT_0007caf0: {level_idc, MaxDpbMbs, MaxFS}
struct LevelLimits {
    u32 level;
    u32 max_dpb_mbs;
    u32 max_fs;
};
constexpr LevelLimits kLevelLimits[] = {
    {0x0a, 0x18c, 0x63},      {0x6f, 0x18c, 0x63},      {0x0b, 0x384, 0x18c},
    {0x0c, 0x948, 0x18c},     {0x0d, 0x948, 0x18c},     {0x14, 0x948, 0x18c},
    {0x15, 0x1290, 0x318},    {0x16, 0x1fa4, 0x654},    {0x1e, 0x1fa4, 0x654},
    {0x1f, 0x4650, 0xe10},    {0x20, 0x5000, 0x1400},   {0x28, 0x8000, 0x2000},
    {0x29, 0x8000, 0x2000},   {0x2a, 0x8800, 0x2200},   {0x32, 0x1af40, 0x5640},
    {0x33, 0x2d000, 0x9000},  {0x34, 0x2d000, 0x9000},  {0x3c, 0xaa000, 0x22000},
    {0x3d, 0xaa000, 0x22000}, {0x3e, 0xaa000, 0x22000},
};
constexpr u32 kLevelCount = sizeof(kLevelLimits) / sizeof(kLevelLimits[0]);

s32 LevelIndex(u32 level) {
    for (u32 i = 0; i < kLevelCount; i++) {
        if (kLevelLimits[i].level == level) {
            return static_cast<s32>(i);
        }
    }
    return -1;
}

class BitReader {
public:
    BitReader(const u8* data, u64 size) : data{data}, size{size} {}

    bool Read(u32 bits, u32* value) {
        u32 v = 0;
        for (u32 i = 0; i < bits; i++) {
            if (pos >= size * 8) {
                return false;
            }
            v = (v << 1) | ((data[pos >> 3] >> (7 - (pos & 7))) & 1);
            pos++;
        }
        *value = v;
        return true;
    }

    bool ReadUe(u32* value) {
        u32 zeros = 0;
        u32 bit = 0;
        while (true) {
            if (!Read(1, &bit)) {
                return false;
            }
            if (bit != 0) {
                break;
            }
            if (++zeros > 31) {
                return false;
            }
        }
        u32 suffix = 0;
        if (!Read(zeros, &suffix)) {
            return false;
        }
        *value = static_cast<u32>((1ULL << zeros) - 1 + suffix);
        return true;
    }

    bool ReadSe(s32* value) {
        u32 k = 0;
        if (!ReadUe(&k)) {
            return false;
        }
        *value = (k & 1) ? static_cast<s32>((k + 1) / 2) : -static_cast<s32>(k / 2);
        return true;
    }

    bool Skip(u32 bits) {
        u32 v = 0;
        while (bits > 0) {
            const u32 n = bits > 32 ? 32 : bits;
            if (!Read(n, &v)) {
                return false;
            }
            bits -= n;
        }
        return true;
    }

    bool ByteAligned() const {
        return (pos & 7) == 0;
    }

    u64 BitsLeft() const {
        return size * 8 - pos;
    }

private:
    const u8* data;
    u64 size;
    u64 pos = 0;
};

std::vector<u8> Unescape(const u8* data, u64 size) {
    std::vector<u8> out;
    out.reserve(size);
    u32 zeros = 0;
    for (u64 i = 0; i < size; i++) {
        const u8 b = data[i];
        if (zeros >= 2 && b == 3) {
            zeros = 0;
            continue;
        }
        out.push_back(b);
        zeros = b == 0 ? zeros + 1 : 0;
    }
    return out;
}

bool InRange(u32 v, u32 lo, u32 hi) {
    return lo <= v && v <= hi;
}

bool SkipScalingList(BitReader& br, u32 count) {
    s32 last = 8;
    s32 next = 8;
    for (u32 j = 0; j < count; j++) {
        if (next != 0) {
            s32 delta = 0;
            if (!br.ReadSe(&delta)) {
                return false;
            }
            next = (last + delta + 256) % 256;
        }
        last = next == 0 ? last : next;
    }
    return true;
}

bool ParseHrd(BitReader& br, AvcHrd& hrd) {
    u32 v = 0;
    if (!br.ReadUe(&hrd.cpb_cnt_minus1) || hrd.cpb_cnt_minus1 > 31 || !br.Read(4, &v) ||
        !br.Read(4, &v)) {
        return false;
    }
    for (u32 i = 0; i <= hrd.cpb_cnt_minus1; i++) {
        if (!br.ReadUe(&v) || !br.ReadUe(&v) || !br.Read(1, &v)) {
            return false;
        }
    }
    return br.Read(5, &hrd.initial_cpb_removal_delay_length_minus1) &&
           br.Read(5, &hrd.cpb_removal_delay_length_minus1) &&
           br.Read(5, &hrd.dpb_output_delay_length_minus1) && br.Read(5, &hrd.time_offset_length);
}

void SetVuiDefaults(AvcSps& s) {
    s.aspect_ratio_info_present_flag = 0;
    s.aspect_ratio_idc = 0;
    s.sar_width = 0;
    s.sar_height = 0;
    s.video_signal_type_present_flag = 0;
    s.video_format = 5;
    s.video_full_range_flag = 0;
    s.colour_description_present_flag = 0;
    s.colour_primaries = 2;
    s.transfer_characteristics = 2;
    s.matrix_coefficients = 2;
    s.timing_info_present_flag = 0;
    s.num_units_in_tick = 0;
    s.time_scale = 0;
    s.fixed_frame_rate_flag = 0;
    s.nal_hrd_parameters_present_flag = 0;
    s.vcl_hrd_parameters_present_flag = 0;
    s.low_delay_hrd_flag = 1;
    s.pic_struct_present_flag = 0;
    s.bitstream_restriction_flag = 0;
    s.max_dec_frame_buffering = s.max_dpb_frames;
}

bool ParseVui(BitReader& br, AvcSps& s) {
    SetVuiDefaults(s);
    u32 v = 0;
    if (!br.Read(1, &s.aspect_ratio_info_present_flag)) {
        return false;
    }
    if (s.aspect_ratio_info_present_flag) {
        if (!br.Read(8, &s.aspect_ratio_idc)) {
            return false;
        }
        if (s.aspect_ratio_idc == 0xff &&
            (!br.Read(16, &s.sar_width) || !br.Read(16, &s.sar_height))) {
            return false;
        }
    }
    if (!br.Read(1, &v)) {
        return false;
    }
    if (v && !br.Read(1, &v)) {
        return false;
    }
    if (!br.Read(1, &s.video_signal_type_present_flag)) {
        return false;
    }
    if (s.video_signal_type_present_flag) {
        if (!br.Read(3, &s.video_format) || !br.Read(1, &s.video_full_range_flag) ||
            !br.Read(1, &s.colour_description_present_flag)) {
            return false;
        }
        if (s.colour_description_present_flag &&
            (!br.Read(8, &s.colour_primaries) || !br.Read(8, &s.transfer_characteristics) ||
             !br.Read(8, &s.matrix_coefficients))) {
            return false;
        }
    }
    if (!br.Read(1, &v)) {
        return false;
    }
    if (v && (!br.ReadUe(&v) || !br.ReadUe(&v))) {
        return false;
    }
    if (!br.Read(1, &s.timing_info_present_flag)) {
        return false;
    }
    if (s.timing_info_present_flag &&
        (!br.Read(32, &s.num_units_in_tick) || !br.Read(32, &s.time_scale) ||
         !br.Read(1, &s.fixed_frame_rate_flag))) {
        return false;
    }
    if (!br.Read(1, &s.nal_hrd_parameters_present_flag)) {
        return false;
    }
    if (s.nal_hrd_parameters_present_flag && !ParseHrd(br, s.nal_hrd)) {
        return false;
    }
    if (!br.Read(1, &s.vcl_hrd_parameters_present_flag)) {
        return false;
    }
    if (s.vcl_hrd_parameters_present_flag && !ParseHrd(br, s.vcl_hrd)) {
        return false;
    }
    if (s.nal_hrd_parameters_present_flag || s.vcl_hrd_parameters_present_flag) {
        if (!br.Read(1, &s.low_delay_hrd_flag)) {
            return false;
        }
    } else {
        s.low_delay_hrd_flag = 1 - s.fixed_frame_rate_flag;
    }
    if (!br.Read(1, &s.pic_struct_present_flag) || !br.Read(1, &s.bitstream_restriction_flag)) {
        return false;
    }
    if (s.bitstream_restriction_flag) {
        u32 max_bytes = 0;
        u32 max_bits = 0;
        u32 mv_h = 0;
        u32 mv_v = 0;
        u32 reorder = 0;
        if (!br.Read(1, &v) || !br.ReadUe(&max_bytes) || !br.ReadUe(&max_bits) ||
            !br.ReadUe(&mv_h) || !br.ReadUe(&mv_v) || !br.ReadUe(&reorder) ||
            !br.ReadUe(&s.max_dec_frame_buffering)) {
            return false;
        }
        if (!InRange(s.max_dec_frame_buffering, 0, 0x10)) {
            return false;
        }
    }
    return true;
}

bool ParseSps(const std::vector<u8>& rbsp, AvcSps& s) {
    BitReader br(rbsp.data() + 1, rbsp.size() - 1);
    s = {};
    u32 v = 0;
    if (!br.Read(8, &s.profile_idc) || !InRange(s.profile_idc, 0x2c, 0xf4)) {
        return false;
    }
    for (u32 i = 0; i < 6; i++) {
        if (!br.Read(1, &s.constraint_set[i])) {
            return false;
        }
    }
    if (!br.Read(2, &v) || !br.Read(8, &s.level_idc) || !InRange(s.level_idc, 10, 0x6f) ||
        !br.ReadUe(&s.seq_parameter_set_id) || s.seq_parameter_set_id > 0x1f) {
        return false;
    }
    s.chroma_format_idc = 1;
    switch (s.profile_idc) {
    case 100:
    case 110:
    case 122:
    case 244:
    case 44:
    case 83:
    case 86:
    case 118:
    case 128:
    case 138:
    case 139:
    case 134:
    case 135: {
        if (!br.ReadUe(&s.chroma_format_idc) || s.chroma_format_idc > 3) {
            return false;
        }
        if (s.chroma_format_idc == 3 && !br.Read(1, &s.separate_colour_plane_flag)) {
            return false;
        }
        if (!br.ReadUe(&s.bit_depth_luma_minus8) || s.bit_depth_luma_minus8 != 0 ||
            !br.ReadUe(&s.bit_depth_chroma_minus8) || s.bit_depth_chroma_minus8 != 0) {
            return false;
        }
        u32 matrix_present = 0;
        if (!br.Read(1, &v) || !br.Read(1, &matrix_present)) {
            return false;
        }
        if (matrix_present) {
            const u32 lists = s.chroma_format_idc != 3 ? 8 : 12;
            for (u32 i = 0; i < lists; i++) {
                u32 present = 0;
                if (!br.Read(1, &present)) {
                    return false;
                }
                if (present && !SkipScalingList(br, i < 6 ? 16 : 64)) {
                    return false;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    if (!br.ReadUe(&s.log2_max_frame_num_minus4) || s.log2_max_frame_num_minus4 > 0xc ||
        !br.ReadUe(&s.pic_order_cnt_type) || s.pic_order_cnt_type > 2) {
        return false;
    }
    if (s.pic_order_cnt_type == 0) {
        if (!br.ReadUe(&v) || v > 0xc) {
            return false;
        }
    } else if (s.pic_order_cnt_type == 1) {
        s32 sv = 0;
        u32 cycle = 0;
        if (!br.Read(1, &v) || !br.ReadSe(&sv) || !br.ReadSe(&sv) || !br.ReadUe(&cycle) ||
            cycle > 0xff) {
            return false;
        }
        for (u32 i = 0; i < cycle; i++) {
            if (!br.ReadSe(&sv)) {
                return false;
            }
        }
    }
    if (!br.ReadUe(&s.max_num_ref_frames) || s.max_num_ref_frames > 0x10 || !br.Read(1, &v) ||
        !br.ReadUe(&s.pic_width_in_mbs_minus1) || !br.ReadUe(&s.pic_height_in_map_units_minus1) ||
        !br.Read(1, &s.frame_mbs_only_flag)) {
        return false;
    }
    if (!s.frame_mbs_only_flag && !br.Read(1, &v)) {
        return false;
    }
    if (!br.Read(1, &v) || !br.Read(1, &s.frame_cropping_flag)) {
        return false;
    }
    if (s.frame_cropping_flag) {
        constexpr u32 kCropMax[4] = {0x780, 0x780, 0x440, 0x440};
        for (u32 i = 0; i < 4; i++) {
            if (!br.ReadUe(&s.frame_crop_offset[i]) || s.frame_crop_offset[i] > kCropMax[i]) {
                return false;
            }
        }
    }
    const s32 level = LevelIndex(s.level_idc);
    if (level < 0) {
        return false;
    }
    const u32 frame_mbs = (s.pic_height_in_map_units_minus1 + 1) * (2 - s.frame_mbs_only_flag) *
                          (s.pic_width_in_mbs_minus1 + 1);
    const u32 dpb_frames = kLevelLimits[level].max_dpb_mbs / frame_mbs;
    s.max_dpb_frames = dpb_frames < 0x10 ? dpb_frames : 0x10;
    if (!br.Read(1, &s.vui_parameters_present_flag)) {
        return false;
    }
    if (s.vui_parameters_present_flag) {
        return ParseVui(br, s);
    }
    SetVuiDefaults(s);
    return true;
}

// FUN_00011660
void AdjustSps(AvcSps& s) {
    if (s.max_dec_frame_buffering < s.max_num_ref_frames) {
        s.max_dec_frame_buffering = s.max_num_ref_frames;
    }
    u32 needed = s.max_dpb_frames;
    if (needed < s.max_dec_frame_buffering) {
        needed = s.max_dec_frame_buffering;
    }
    s32 index = LevelIndex(s.level_idc);
    if (index < 0) {
        index = 0;
    }
    const u32 frame_mbs = (2 - s.frame_mbs_only_flag) * (s.pic_width_in_mbs_minus1 + 1) *
                          (s.pic_height_in_map_units_minus1 + 1);
    for (u32 i = static_cast<u32>(index); i < kLevelCount; i++) {
        if (kLevelLimits[i].max_dpb_mbs / frame_mbs >= needed &&
            kLevelLimits[i].max_fs >= frame_mbs) {
            s.level_idc = kLevelLimits[i].level;
            return;
        }
    }
}

bool ParsePicTiming(BitReader& br, const AvcSps& s, AvcPicTiming& pt) {
    const AvcHrd* hrd = s.nal_hrd_parameters_present_flag   ? &s.nal_hrd
                        : s.vcl_hrd_parameters_present_flag ? &s.vcl_hrd
                                                            : nullptr;
    u32 v = 0;
    if (hrd != nullptr) {
        if (!br.Read(hrd->cpb_removal_delay_length_minus1 + 1, &v) ||
            !br.Read(hrd->dpb_output_delay_length_minus1 + 1, &v)) {
            return false;
        }
    }
    if (!s.pic_struct_present_flag) {
        return true;
    }
    if (!br.Read(4, &pt.pic_struct) || pt.pic_struct > 8) {
        return false;
    }
    constexpr u32 kNumClockTs[9] = {1, 1, 1, 2, 2, 3, 3, 2, 3};
    for (u32 i = 0; i < kNumClockTs[pt.pic_struct]; i++) {
        if (!br.Read(1, &pt.clock_timestamp_flag[i])) {
            return false;
        }
        if (!pt.clock_timestamp_flag[i]) {
            continue;
        }
        u32 full = 0;
        u32 discontinuity = 0;
        u32 cnt_dropped = 0;
        u32 n_frames = 0;
        if (!br.Read(2, &pt.ct_type[i]) || !br.Read(1, &v) || !br.Read(5, &v) ||
            !br.Read(1, &full) || !br.Read(1, &discontinuity) || !br.Read(1, &cnt_dropped) ||
            !br.Read(8, &n_frames)) {
            return false;
        }
        if (full) {
            if (!br.Read(6, &v) || !br.Read(6, &v) || !br.Read(5, &v)) {
                return false;
            }
        } else {
            u32 flag = 0;
            if (!br.Read(1, &flag)) {
                return false;
            }
            if (flag) {
                if (!br.Read(6, &v) || !br.Read(1, &flag)) {
                    return false;
                }
                if (flag) {
                    if (!br.Read(6, &v) || !br.Read(1, &flag)) {
                        return false;
                    }
                    if (flag && !br.Read(5, &v)) {
                        return false;
                    }
                }
            }
        }
        const u32 offset_len = hrd != nullptr ? hrd->time_offset_length : 24;
        if (offset_len > 0 && !br.Read(offset_len, &v)) {
            return false;
        }
    }
    return true;
}

bool ParseFramePacking(BitReader& br, AvcFramePacking& fp) {
    u32 v = 0;
    if (!br.ReadUe(&v) || !br.Read(1, &fp.cancel_flag)) {
        return false;
    }
    if (fp.cancel_flag) {
        return true;
    }
    u32 spatial_flipping = 0;
    u32 frame0_flipped = 0;
    u32 field_views = 0;
    u32 self_contained0 = 0;
    u32 self_contained1 = 0;
    if (!br.Read(7, &fp.arrangement_type) || !br.Read(1, &fp.quincunx_sampling_flag) ||
        !br.Read(6, &fp.content_interpretation_type) || !br.Read(1, &spatial_flipping) ||
        !br.Read(1, &frame0_flipped) || !br.Read(1, &field_views) ||
        !br.Read(1, &fp.current_frame_is_frame0_flag) || !br.Read(1, &self_contained0) ||
        !br.Read(1, &self_contained1)) {
        return false;
    }
    return true;
}

// FUN_0004c070, user_data_unregistered.
void ParseUserData(const u8* payload, u32 size, AvcAccessUnit& au) {
    // DAT_0007f954 / DAT_0007f974
    constexpr std::array<u8, 16> kUuidCc = {0x17, 0xee, 0x8c, 0x60, 0xf8, 0x4d, 0x11, 0xd9,
                                            0x8c, 0xd6, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x66};
    constexpr std::array<u8, 16> kUuidClid = {0xa7, 0x46, 0x02, 0xbb, 0xf8, 0xa1, 0x4c, 0xc0,
                                              0xa9, 0x36, 0x48, 0xe3, 0x91, 0xdc, 0xe7, 0x61};
    constexpr std::array<u8, 16> kUuidApp{};
    const auto identifier = [&]() -> u32 {
        return size < 0x14 ? 0
                           : (u32{payload[16]} << 24) | (u32{payload[17]} << 16) |
                                 (u32{payload[18]} << 8) | u32{payload[19]};
    };
    const bool is_cc =
        std::equal(kUuidCc.begin(), kUuidCc.end(), payload) && identifier() == 0x47413934;
    const bool is_clid = !is_cc && std::equal(kUuidClid.begin(), kUuidClid.end(), payload) &&
                         identifier() == 0x434c4944;
    if (is_cc || is_clid) {
        const u32 length = size - 0x14;
        u32& out_size = is_cc ? au.closed_caption_size : au.clid_size;
        auto& out = is_cc ? au.closed_caption : au.clid;
        out_size = 0;
        if (length <= out.size()) {
            std::copy(payload + 0x14, payload + 0x14 + length, out.begin());
            out_size = length;
        }
        return;
    }
    const bool is_app = std::equal(kUuidApp.begin(), kUuidApp.end(), payload);
    if (is_app) {
        au.user_data_status = 0;
        au.user_data_app = 1;
    } else {
        au.user_data_status = 2;
    }
}

} // namespace

void AvcHeaderParser::Reset() {
    sps.fill(std::nullopt);
    pps.fill(std::nullopt);
    pic_timing_state = {};
}

AvcAccessUnit AvcHeaderParser::Parse(const u8* data, u64 size) {
    AvcAccessUnit au{};
    au.status = AvcParseStatus::NoSlice;
    au.slice_type = 0xff;
    au.user_data_status = 1;
    std::vector<u8> sei_pic_timing;
    std::vector<u8> sei_buffering;
    u64 pos = 0;
    const auto next_nal = [&](u64& start, u64& end) -> bool {
        while (pos + 3 <= size) {
            if (data[pos] == 0 && data[pos + 1] == 0 && data[pos + 2] == 1) {
                start = pos + 3;
                u64 p = start;
                while (p + 3 <= size && !(data[p] == 0 && data[p + 1] == 0 &&
                                          (data[p + 2] == 1 || data[p + 2] == 0))) {
                    p++;
                }
                end = p + 3 <= size ? p : size;
                pos = end;
                return true;
            }
            pos++;
        }
        return false;
    };
    u64 start = 0;
    u64 end = 0;
    while (next_nal(start, end)) {
        if (end <= start) {
            continue;
        }
        const u32 nal_type = data[start] & 0x1f;
        if (7 < nal_type - 5 && (0xf < nal_type || ((0xc002u >> nal_type) & 1) == 0)) {
            au.status = AvcParseStatus::Invalid;
            return au;
        }
        switch (nal_type) {
        case 1:
        case 5: {
            const auto rbsp = Unescape(data + start, end - start);
            BitReader br(rbsp.data() + 1, rbsp.size() - 1);
            u32 first_mb = 0;
            u32 slice_type = 0;
            u32 pps_id = 0;
            if (!br.ReadUe(&first_mb) || !br.ReadUe(&slice_type) || slice_type > 9 ||
                !br.ReadUe(&pps_id) || pps_id > 0xff) {
                au.status = AvcParseStatus::Invalid;
                return au;
            }
            if (!pps[pps_id] || !sps[pps[pps_id]->seq_parameter_set_id]) {
                au.status = AvcParseStatus::Invalid;
                return au;
            }
            const AvcSps& s = *sps[pps[pps_id]->seq_parameter_set_id];
            if (first_mb != 0) {
                au.status = AvcParseStatus::Invalid;
                return au;
            }
            if (nal_type == 5 && slice_type != 7 && slice_type != 2) {
                au.status = AvcParseStatus::Invalid;
                return au;
            }
            u32 v = 0;
            if (s.separate_colour_plane_flag && !br.Read(2, &v)) {
                au.status = AvcParseStatus::Invalid;
                return au;
            }
            if (!br.Read(s.log2_max_frame_num_minus4 + 4, &v)) {
                au.status = AvcParseStatus::Invalid;
                return au;
            }
            if (!s.frame_mbs_only_flag) {
                if (!br.Read(1, &au.field_pic_flag)) {
                    au.status = AvcParseStatus::Invalid;
                    return au;
                }
                if (au.field_pic_flag && !br.Read(1, &au.bottom_field_flag)) {
                    au.status = AvcParseStatus::Invalid;
                    return au;
                }
            }
            au.nal_unit_type = nal_type;
            au.slice_type = slice_type;
            au.sps = s;
            AdjustSps(au.sps);
            au.idr_picture =
                (slice_type == 7 || slice_type == 2) && au.sps_present && au.pps_present;
            au.idr_flag = nal_type == 5;
            if (!sei_buffering.empty()) {
                BitReader bp(sei_buffering.data(), sei_buffering.size());
                u32 bp_sps = 0;
                if (bp.ReadUe(&bp_sps) && bp_sps == s.seq_parameter_set_id) {
                    au.buffering_period_present = 1;
                }
            }
            if (!sei_pic_timing.empty() && (sei_buffering.empty() || au.buffering_period_present)) {
                BitReader pt(sei_pic_timing.data(), sei_pic_timing.size());
                AvcPicTiming timing{};
                timing.ct_type = {2, 2, 2};
                if (ParsePicTiming(pt, s, timing)) {
                    pic_timing_state = timing;
                    au.pic_timing_present = 1;
                }
            }
            au.pic_timing = pic_timing_state;
            au.status = AvcParseStatus::Ok;
            return au;
        }
        case 6: {
            const auto rbsp = Unescape(data + start, end - start);
            u64 p = 1;
            while (p < rbsp.size() && !(rbsp.size() - p == 1 && rbsp[p] == 0x80)) {
                u32 type = 0;
                while (p < rbsp.size() && rbsp[p] == 0xff) {
                    type += 0xff;
                    p++;
                }
                if (p >= rbsp.size()) {
                    break;
                }
                type += rbsp[p++];
                u32 psize = 0;
                while (p < rbsp.size() && rbsp[p] == 0xff) {
                    psize += 0xff;
                    p++;
                }
                if (p >= rbsp.size()) {
                    break;
                }
                psize += rbsp[p++];
                if (p + psize > rbsp.size()) {
                    break;
                }
                const u8* payload = rbsp.data() + p;
                if (type == 0) {
                    sei_buffering.assign(payload, payload + (psize < 0x400 ? psize : 0x400));
                } else if (type == 1) {
                    sei_pic_timing.assign(payload, payload + (psize < 0x40 ? psize : 0x40));
                } else if (type == 5) {
                    if (psize < 0x10) {
                        au.status = AvcParseStatus::Invalid;
                        return au;
                    }
                    ParseUserData(payload, psize, au);
                } else if (type == 45) {
                    BitReader br(payload, psize);
                    AvcFramePacking fp{};
                    if (ParseFramePacking(br, fp)) {
                        au.frame_packing = fp;
                        au.frame_packing_present = 1;
                    }
                }
                p += psize;
            }
            break;
        }
        case 7: {
            const auto rbsp = Unescape(data + start, end - start);
            AvcSps s{};
            if (ParseSps(rbsp, s)) {
                sps[s.seq_parameter_set_id] = s;
                au.sps_present = 1;
            } else {
                sps.fill(std::nullopt);
            }
            break;
        }
        case 8: {
            const auto rbsp = Unescape(data + start, end - start);
            BitReader br(rbsp.data() + 1, rbsp.size() - 1);
            AvcPps p{};
            if (br.ReadUe(&p.pic_parameter_set_id) && p.pic_parameter_set_id <= 0xff &&
                br.ReadUe(&p.seq_parameter_set_id) && p.seq_parameter_set_id <= 0x1f &&
                sps[p.seq_parameter_set_id]) {
                pps[p.pic_parameter_set_id] = p;
                au.pps_present = 1;
            }
            break;
        }
        case 9:
            au.aud_present = 1;
            break;
        case 12:
            au.filler_present = 1;
            break;
        default:
            break;
        }
    }
    return au;
}

} // namespace Libraries::Vdecsw
