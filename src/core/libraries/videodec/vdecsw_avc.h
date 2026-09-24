// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <optional>
#include "common/types.h"

namespace Libraries::Vdecsw {

struct AvcHrd {
    u32 cpb_cnt_minus1;
    u32 initial_cpb_removal_delay_length_minus1;
    u32 cpb_removal_delay_length_minus1;
    u32 dpb_output_delay_length_minus1;
    u32 time_offset_length;
};

struct AvcSps {
    u32 profile_idc;
    std::array<u32, 6> constraint_set;
    u32 level_idc;
    u32 seq_parameter_set_id;
    u32 chroma_format_idc;
    u32 separate_colour_plane_flag;
    u32 bit_depth_luma_minus8;
    u32 bit_depth_chroma_minus8;
    u32 log2_max_frame_num_minus4;
    u32 pic_order_cnt_type;
    u32 max_num_ref_frames;
    u32 pic_width_in_mbs_minus1;
    u32 pic_height_in_map_units_minus1;
    u32 frame_mbs_only_flag;
    u32 frame_cropping_flag;
    std::array<u32, 4> frame_crop_offset;
    u32 vui_parameters_present_flag;
    u32 aspect_ratio_info_present_flag;
    u32 aspect_ratio_idc;
    u32 sar_width;
    u32 sar_height;
    u32 video_signal_type_present_flag;
    u32 video_format;
    u32 video_full_range_flag;
    u32 colour_description_present_flag;
    u32 colour_primaries;
    u32 transfer_characteristics;
    u32 matrix_coefficients;
    u32 timing_info_present_flag;
    u32 num_units_in_tick;
    u32 time_scale;
    u32 fixed_frame_rate_flag;
    u32 nal_hrd_parameters_present_flag;
    u32 vcl_hrd_parameters_present_flag;
    AvcHrd nal_hrd;
    AvcHrd vcl_hrd;
    u32 low_delay_hrd_flag;
    u32 pic_struct_present_flag;
    u32 bitstream_restriction_flag;
    u32 max_dec_frame_buffering;
    u32 max_dpb_frames;
};

struct AvcPps {
    u32 pic_parameter_set_id;
    u32 seq_parameter_set_id;
};

struct AvcPicTiming {
    u32 pic_struct;
    std::array<u32, 3> clock_timestamp_flag;
    std::array<u32, 3> ct_type;
};

struct AvcFramePacking {
    u32 cancel_flag;
    u32 arrangement_type;
    u32 quincunx_sampling_flag;
    u32 content_interpretation_type;
    u32 current_frame_is_frame0_flag;
};

enum class AvcParseStatus {
    Ok,
    NoSlice,
    Invalid,
};

struct AvcAccessUnit {
    AvcParseStatus status;
    AvcSps sps;
    u32 nal_unit_type;
    u32 slice_type;
    u32 field_pic_flag;
    u32 bottom_field_flag;
    u32 idr_flag;
    u32 idr_picture;
    u32 sps_present;
    u32 pps_present;
    u32 aud_present;
    u32 filler_present;
    u32 pic_timing_present;
    u32 buffering_period_present;
    u32 frame_packing_present;
    u32 user_data_status;
    u32 user_data_app;
    u32 closed_caption_size;
    std::array<u8, 12> closed_caption;
    u32 clid_size;
    std::array<u8, 12> clid;
    AvcPicTiming pic_timing;
    AvcFramePacking frame_packing;
};

class AvcHeaderParser {
public:
    void Reset();
    AvcAccessUnit Parse(const u8* data, u64 size);

private:
    std::array<std::optional<AvcSps>, 32> sps{};
    std::array<std::optional<AvcPps>, 256> pps{};
    AvcPicTiming pic_timing_state{};
};

} // namespace Libraries::Vdecsw
