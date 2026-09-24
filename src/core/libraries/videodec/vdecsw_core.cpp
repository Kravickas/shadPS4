// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// libSceVdecCore software decoder path (resource type 3) used by libSceVdecsw (FW 12.02).
// Control flow, queue limits, output ordering and the picture information trailer follow
// libSceVdecCore; the H.264 reconstruction itself is done by libavcodec.

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

#include "common/alignment.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/videodec/vdecsw_avc.h"
#include "core/libraries/videodec/vdecsw_core.h"

namespace Libraries::Vdecsw {

namespace {

constexpr u64 kComputeMemorySize = 0x495200;
constexpr u32 kComputeMemoryAlign = 0x100;
constexpr u64 kComputeHandleOffset = 0x290500;
constexpr u64 kComputeMinimumSize = 0x294650;
// DAT_0007ef40
constexpr std::array<u8, 16> kComputeMagic = {0xb1, 0x9e, 0x90, 0xb7, 0x85, 0x6e, 0x47, 0xa7,
                                              0x95, 0x54, 0x4b, 0x4a, 0x71, 0x07, 0xe2, 0x94};
constexpr u64 kCoreMagic = 0x344c4338424b51;
constexpr u32 kTrailerSize = 0x400;
constexpr u32 kTrailerInfoSize = 0x100;

struct ComputeHandle {
    u64 vtable;
    std::array<u8, 16> magic;
    u8 pad18[0x100];
    u32 attached_decoders;
};
static_assert(offsetof(ComputeHandle, attached_decoders) == 0x118);

struct CoreHeader {
    u64 magic;
    u32 level;
    u32 profile;
    u32 new_sequence;
    u32 mode;
    u32 codec;
};

struct DecodedFrame {
    std::vector<u8> luma;
    std::vector<u8> chroma;
    u32 width;
    u32 height;
    u32 error;
    std::array<u8, kTrailerInfoSize> info;
};

struct PendingInput {
    void* au_data;
    u32 frame_count;
};

class CoreDecoder {
public:
    CoreDecoder(const VdecCoreDecoderParams& params, ComputeHandle* compute, u32 sdk)
        : params{params}, compute{compute}, sdk{sdk} {
        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
        context = avcodec_alloc_context3(codec);
        context->opaque = this;
        context->thread_count = 1;
        context->get_buffer2 = &CoreDecoder::GetBuffer;
        avcodec_open2(context, codec, nullptr);
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        Reset();
    }

    ~CoreDecoder() {
        ReleaseCapture();
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&context);
    }

    CoreDecoder(const CoreDecoder&) = delete;
    CoreDecoder& operator=(const CoreDecoder&) = delete;

    ComputeHandle* Compute() const {
        return compute;
    }

    void Reset() {
        std::scoped_lock lk{mutex};
        parser.Reset();
        inputs.clear();
        frames.clear();
        produced = 0;
        consumed = 0;
        output_state = 0;
        output_buffer = nullptr;
        output_buffer_size = 0;
        flushed = false;
        new_sequence = false;
        has_sequence = false;
        sequence_width = 0;
        sequence_height = 0;
        pending_field = false;
        avcodec_flush_buffers(context);
        ReleaseCapture();
    }

    s32 SetDecodeInput(const VdecCoreInput& input) {
        std::scoped_lock lk{mutex};
        if (inputs.size() >= params.decode_pipeline_depth) {
            return ORBIS_VDECCORE_ERROR_INPUT_QUEUE_FULL;
        }
        const u64 held = inputs.size() + frames.size() + (output_state == 2 ? 1 : 0);
        if (held >= static_cast<u64>(params.extra_dpb_frame_count) + params.decode_pipeline_depth) {
            return ORBIS_VDECCORE_ERROR_BUSY;
        }
        const auto* data = static_cast<const u8*>(input.au_data);
        const AvcAccessUnit au = parser.Parse(data, input.au_size);
        if (au.status != AvcParseStatus::Ok) {
            return ORBIS_VDECCORE_ERROR_INVALID_SEQUENCE;
        }
        const AvcSps& sps = au.sps;
        const u32 width_mbs = sps.pic_width_in_mbs_minus1 + 1;
        const u32 height_mbs =
            (sps.pic_height_in_map_units_minus1 + 1) * (2 - sps.frame_mbs_only_flag);
        if (has_sequence && (width_mbs != sequence_width || height_mbs != sequence_height)) {
            new_sequence = true;
            return ORBIS_VDECCORE_ERROR_NEW_SEQUENCE;
        }
        if (IsOversize(sps, width_mbs, height_mbs)) {
            return ORBIS_VDECCORE_ERROR_OVERSIZE_DECODE;
        }
        has_sequence = true;
        sequence_width = width_mbs;
        sequence_height = height_mbs;

        const u32 count = Decode(input, au, width_mbs, height_mbs);
        inputs.push_back({input.au_data, count});
        return ORBIS_OK;
    }

    s32 SyncDecodeWptr(u32* frame_count, void** decoded_au) {
        std::scoped_lock lk{mutex};
        if (inputs.empty()) {
            return ORBIS_VDECCORE_ERROR_NOT_READY;
        }
        const PendingInput input = inputs.front();
        inputs.pop_front();
        *frame_count = input.frame_count;
        *decoded_au = input.au_data;
        produced += input.frame_count;
        WriteOutputIfReady();
        return ORBIS_OK;
    }

    s32 SetDecodeOutput(void* buffer, u64 size) {
        std::scoped_lock lk{mutex};
        if (output_state != 0) {
            return ORBIS_VDECCORE_ERROR_BUSY;
        }
        if (buffer == nullptr) {
            return ORBIS_VDECCORE_ERROR_FAIL;
        }
        if (size == 0 || size < params.frame_buffer_size) {
            return ORBIS_VDECCORE_ERROR_FRAME_BUFFER_SIZE;
        }
        output_buffer = buffer;
        output_buffer_size = size;
        output_state = 1;
        WriteOutputIfReady();
        return ORBIS_OK;
    }

    s32 SyncDecodeOutput(VdecCoreOutput* out, bool try_only) {
        std::scoped_lock lk{mutex};
        std::memset(out, 0, sizeof(*out));
        if (output_state != 1 && output_state != 2) {
            return ORBIS_VDECCORE_ERROR_NO_OUTPUT_BUFFER;
        }
        if (output_state == 1) {
            if (produced == consumed) {
                return try_only ? ORBIS_VDECCORE_ERROR_NO_OUTPUT_FRAME
                                : ORBIS_VDECCORE_ERROR_NOT_READY;
            }
            return try_only ? ORBIS_VDECCORE_ERROR_OUTPUT_PENDING : ORBIS_VDECCORE_ERROR_NOT_READY;
        }
        *out = written;
        consumed++;
        output_state = 0;
        return ORBIS_OK;
    }

    s32 Flush(s32* frame_count) {
        std::scoped_lock lk{mutex};
        *frame_count = 0;
        flushed = true;
        return ORBIS_OK;
    }

private:
    static int GetBuffer(AVCodecContext* ctx, AVFrame* pic, int flags) {
        auto* self = static_cast<CoreDecoder*>(ctx->opaque);
        const int ret = avcodec_default_get_buffer2(ctx, pic, flags);
        if (ret == 0) {
            self->ReleaseCapture();
            self->captured = av_frame_alloc();
            if (self->captured != nullptr && av_frame_ref(self->captured, pic) < 0) {
                av_frame_free(&self->captured);
            }
            self->captured_new = true;
        }
        return ret;
    }

    void ReleaseCapture() {
        if (captured != nullptr) {
            av_frame_free(&captured);
        }
        captured_new = false;
    }

    bool IsOversize(const AvcSps& sps, u32 width_mbs, u32 height_mbs) const {
        if (params.width_units != -1 && params.height_units != -1) {
            const u32 max_w = static_cast<u32>(params.width_units) * params.unit_size / 16;
            const u32 max_h = static_cast<u32>(params.height_units) * params.unit_size / 16;
            if (width_mbs > max_w || height_mbs > max_h) {
                return true;
            }
        }
        if (params.max_dpb_frame_count >= 0 &&
            sps.max_num_ref_frames > static_cast<u32>(params.max_dpb_frame_count)) {
            return true;
        }
        return false;
    }

    u32 Decode(const VdecCoreInput& input, const AvcAccessUnit& au, u32 width_mbs, u32 height_mbs) {
        captured_new = false;
        bitstream.assign(input.au_size + AV_INPUT_BUFFER_PADDING_SIZE, 0);
        std::memcpy(bitstream.data(), input.au_data, input.au_size);
        packet->data = bitstream.data();
        packet->size = static_cast<int>(input.au_size);
        packet->pts = static_cast<s64>(input.pts_data);
        packet->dts = static_cast<s64>(input.dts_data);
        const int send = avcodec_send_packet(context, packet);
        while (avcodec_receive_frame(context, frame) == 0) {
            av_frame_unref(frame);
        }
        if (captured == nullptr) {
            return 0;
        }
        if (au.field_pic_flag && !captured_new && pending_field) {
            pending_field = false;
            StoreFrame(input, au, width_mbs, height_mbs, send < 0);
            return 1;
        }
        if (au.field_pic_flag) {
            pending_field = true;
            return 0;
        }
        pending_field = false;
        StoreFrame(input, au, width_mbs, height_mbs, send < 0);
        return 1;
    }

    void StoreFrame(const VdecCoreInput& input, const AvcAccessUnit& au, u32 width_mbs,
                    u32 height_mbs, bool error) {
        DecodedFrame out{};
        out.width = width_mbs * 16;
        out.height = height_mbs * 16;
        out.error = error ? 1 : 0;
        out.luma.resize(static_cast<size_t>(out.width) * out.height);
        out.chroma.resize(static_cast<size_t>(out.width) * out.height / 2);
        const AVFrame* src = captured;
        if (src != nullptr && src->data[0] != nullptr) {
            const u32 copy_w = std::min<u32>(out.width, static_cast<u32>(src->width));
            const u32 copy_h = std::min<u32>(out.height, static_cast<u32>(src->height));
            for (u32 y = 0; y < copy_h; y++) {
                std::memcpy(out.luma.data() + static_cast<size_t>(y) * out.width,
                            src->data[0] + static_cast<size_t>(y) * src->linesize[0], copy_w);
            }
            const bool has_chroma = src->data[1] != nullptr && src->data[2] != nullptr;
            for (u32 y = 0; y < copy_h / 2; y++) {
                u8* dst = out.chroma.data() + static_cast<size_t>(y) * out.width;
                if (has_chroma) {
                    const u8* cb = src->data[1] + static_cast<size_t>(y) * src->linesize[1];
                    const u8* cr = src->data[2] + static_cast<size_t>(y) * src->linesize[2];
                    for (u32 x = 0; x < copy_w / 2; x++) {
                        dst[x * 2] = cb[x];
                        dst[x * 2 + 1] = cr[x];
                    }
                } else {
                    std::memset(dst, 0x80, copy_w);
                }
            }
        }
        BuildPictureInfo(out.info, input, au);
        frames.push_back(std::move(out));
    }

    // FUN_00012120 (field index 0) and FUN_0001d080.
    static void BuildPictureInfo(std::array<u8, kTrailerInfoSize>& info, const VdecCoreInput& input,
                                 const AvcAccessUnit& au) {
        info.fill(0);
        const AvcSps& s = au.sps;
        const auto put32 = [&](u32 off, u32 v) { std::memcpy(info.data() + off, &v, 4); };
        const auto put64 = [&](u32 off, u64 v) { std::memcpy(info.data() + off, &v, 8); };
        const auto put16 = [&](u32 off, u16 v) { std::memcpy(info.data() + off, &v, 2); };
        put32(0x00, 0);
        put64(0x08, input.attached_data);
        put64(0x10, input.pts_data);
        put64(0x18, input.dts_data);
        put32(0x20, 1);
        put32(0x28, s.pic_width_in_mbs_minus1);
        put32(0x2c, s.pic_height_in_map_units_minus1);
        put32(0x30, s.frame_crop_offset[0]);
        put32(0x34, s.frame_crop_offset[1]);
        put32(0x38, s.frame_crop_offset[2]);
        put32(0x3c, s.frame_crop_offset[3]);
        put32(0x40, s.num_units_in_tick);
        put32(0x44, s.time_scale);
        put32(0x48, 0);
        u32 flags = 0;
        flags |= (s.frame_mbs_only_flag & 1) << 0;
        flags |= (s.frame_cropping_flag & 1) << 1;
        flags |= (s.timing_info_present_flag & 1) << 2;
        flags |= (s.pic_struct_present_flag & 1) << 3;
        flags |= (s.aspect_ratio_info_present_flag & 1) << 4;
        flags |= (s.video_full_range_flag & 1) << 5;
        flags |= (s.colour_description_present_flag & 1) << 6;
        flags |= (s.bitstream_restriction_flag & 1) << 7;
        flags |= (au.pic_timing.pic_struct & 0xf) << 8;
        flags |= (s.video_format & 7) << 12;
        flags |= (s.video_signal_type_present_flag & 1) << 15;
        flags |= (s.low_delay_hrd_flag & 1) << 16;
        flags |= (au.idr_flag & 1) << 17;
        flags |= (au.field_pic_flag & 1) << 18;
        flags |= (au.bottom_field_flag & 1) << 19;
        if (au.frame_packing_present) {
            flags |= (au.frame_packing.cancel_flag & 1) << 20;
            flags |= (au.frame_packing.current_frame_is_frame0_flag & 1) << 21;
            flags |= (au.frame_packing.quincunx_sampling_flag & 1) << 22;
        }
        flags |= (s.fixed_frame_rate_flag & 1) << 23;
        for (u32 i = 0; i < 6; i++) {
            flags |= (s.constraint_set[i] & 1) << (24 + i);
        }
        put32(0x4c, flags);
        put16(0x50, 0);
        put16(0x52, static_cast<u16>(s.sar_width));
        put16(0x54, static_cast<u16>(s.sar_height));
        info[0x56] = static_cast<u8>(s.profile_idc);
        info[0x57] = static_cast<u8>(s.level_idc);
        info[0x58] = static_cast<u8>(s.max_dec_frame_buffering);
        info[0x59] = static_cast<u8>(s.aspect_ratio_idc);
        info[0x5a] = static_cast<u8>(s.colour_primaries);
        info[0x5b] = static_cast<u8>(s.transfer_characteristics);
        info[0x5c] = static_cast<u8>(s.matrix_coefficients);
        info[0x5d] = 0;
        u16 flags2 = 0;
        flags2 |= static_cast<u16>((au.sps_present & 1) << 0);
        flags2 |= static_cast<u16>((au.pps_present & 1) << 1);
        flags2 |= static_cast<u16>((au.aud_present & 1) << 2);
        flags2 |= static_cast<u16>((au.filler_present & 1) << 5);
        flags2 |= static_cast<u16>((au.pic_timing_present & 1) << 6);
        flags2 |= static_cast<u16>((au.buffering_period_present & 1) << 7);
        flags2 |= static_cast<u16>((au.frame_packing_present & 1) << 8);
        flags2 |= static_cast<u16>((au.user_data_app & 1) << 9);
        put16(0x60, flags2);
        info[0x64] = static_cast<u8>(au.slice_type);
        if (au.frame_packing_present) {
            info[0x65] = static_cast<u8>(au.frame_packing.arrangement_type);
            info[0x66] = static_cast<u8>(au.frame_packing.content_interpretation_type);
        }
        info[0x68] = static_cast<u8>(au.idr_picture);
        if (au.closed_caption_size != 0) {
            std::memcpy(info.data() + 0x6a, au.closed_caption.data(), au.closed_caption_size);
            info[0x69] = static_cast<u8>(au.closed_caption_size);
        }
        if (au.clid_size != 0) {
            std::memcpy(info.data() + 0x77, au.clid.data(), au.clid_size);
            info[0x76] = static_cast<u8>(au.clid_size);
        }
        info[0x83] = static_cast<u8>(au.pic_timing.ct_type[0]);
        info[0x84] = static_cast<u8>(au.pic_timing.ct_type[1]);
        info[0x85] = static_cast<u8>(au.pic_timing.ct_type[2]);
        put32(0xd8, au.user_data_status);
    }

    // Worker side of FUN_0000df20 / FUN_0000e510: the next decoded frame is written into the
    // registered output buffer.
    void WriteOutputIfReady() {
        if (output_state != 1 || frames.empty() || produced == consumed) {
            return;
        }
        DecodedFrame f = std::move(frames.front());
        frames.pop_front();
        const u32 pitch = Common::AlignUp(f.width, 64u);
        auto* dst = static_cast<u8*>(output_buffer);
        for (u32 y = 0; y < f.height; y++) {
            std::memcpy(dst + static_cast<size_t>(y) * pitch,
                        f.luma.data() + static_cast<size_t>(y) * f.width, f.width);
        }
        u8* chroma = dst + static_cast<size_t>(pitch) * f.height;
        for (u32 y = 0; y < f.height / 2; y++) {
            std::memcpy(chroma + static_cast<size_t>(y) * pitch,
                        f.chroma.data() + static_cast<size_t>(y) * f.width, f.width);
        }
        const u64 fb_size = params.frame_buffer_size;
        std::memcpy(dst + fb_size - kTrailerSize, f.info.data(), f.info.size());

        written = {};
        written.frame_width = f.width;
        written.frame_pitch = pitch;
        written.frame_height = f.height;
        written.picture_count = 1;
        written.frame_format = 0;
        written.codec = 0;
        written.error_status = f.error;
        written.frame_pitch_in_bytes = pitch;
        written.frame_buffer = output_buffer;
        written.frame_buffer_size = fb_size;
        const bool empty = frames.empty() && inputs.empty();
        written.is_last_frame = empty ? 1 : 0;
        if (0x84fffff < sdk && empty) {
            written.is_last_frame = flushed ? 1 : 0;
        }
        output_state = 2;
    }

    VdecCoreDecoderParams params;
    ComputeHandle* compute;
    u32 sdk;
    std::mutex mutex;
    AvcHeaderParser parser;
    AVCodecContext* context = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* captured = nullptr;
    bool captured_new = false;
    std::vector<u8> bitstream;
    std::deque<PendingInput> inputs;
    std::deque<DecodedFrame> frames;
    bool pending_field = false;
    u32 produced = 0;
    u32 consumed = 0;
    u32 output_state = 0;
    void* output_buffer = nullptr;
    u64 output_buffer_size = 0;
    VdecCoreOutput written{};
    bool flushed = false;
    bool new_sequence = false;
    bool has_sequence = false;
    u32 sequence_width = 0;
    u32 sequence_height = 0;
};

std::mutex g_decoders_mutex;
std::unordered_map<void*, std::unique_ptr<CoreDecoder>> g_decoders;

CoreDecoder* FindDecoder(void* core) {
    std::scoped_lock lk{g_decoders_mutex};
    const auto it = g_decoders.find(core);
    return it != g_decoders.end() ? it->second.get() : nullptr;
}

bool IsValidCompute(const ComputeHandle* handle) {
    return handle != nullptr && handle->magic == kComputeMagic;
}

} // namespace

s32 VdecCoreQueryComputeResourceInfo(u64* size, u64* base) {
    *size = kComputeMemorySize;
    *base = 0;
    return ORBIS_OK;
}

s32 VdecCoreInitializeComputeResource(const VdecCoreComputeParams& params, void** handle, u32 sdk) {
    if (6 < params.compute_pipe_id || 7 < params.compute_queue_id) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    const u64 base = reinterpret_cast<u64>(params.cpu_gpu_memory);
    if (params.cpu_gpu_memory_size < kComputeMemorySize || base % kComputeMemoryAlign != 0) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    auto* compute = reinterpret_cast<ComputeHandle*>(base + kComputeHandleOffset);
    if (compute->magic == kComputeMagic && 0x5ffffff < sdk) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    if (base + params.cpu_gpu_memory_size < base + kComputeMinimumSize) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    std::memset(params.cpu_gpu_memory, 0, params.cpu_gpu_memory_size);
    compute->attached_decoders = 0;
    compute->magic = kComputeMagic;
    *handle = compute;
    return ORBIS_OK;
}

s32 VdecCoreFinalizeComputeResource(void* handle) {
    auto* compute = static_cast<ComputeHandle*>(handle);
    if (!IsValidCompute(compute) || compute->attached_decoders != 0) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    compute->magic = {};
    return ORBIS_OK;
}

s32 VdecCoreCreateDecoder(const VdecCoreConfig& cfg, const VdecCoreMemoryParams& mem, void* compute,
                          void** core, u32 sdk) {
    if (8 < cfg.resource_type) {
        return static_cast<s32>(0x80c00002);
    }
    if (cfg.resource_type != 3) {
        return ORBIS_VDECCORE_ERROR_RESOURCE_TYPE;
    }
    if (mem.cpu_memory == nullptr || mem.gpu_memory == nullptr || mem.cpu_gpu_memory == nullptr) {
        return ORBIS_VDECCORE_ERROR_MEMORY_POINTER;
    }
    VdecCoreDecoderParams params{};
    s32 ret = VdecCoreGetDecoderParams(cfg, params, sdk);
    if (ret != 0) {
        return ret;
    }
    u64 cpu = 0;
    u64 gpu = 0;
    u64 cpu_gpu = 0;
    if (VdecCoreQueryInstanceSize(cfg, &cpu, &gpu, &cpu_gpu, sdk) != 0) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    if (mem.cpu_memory_size < cpu) {
        return ORBIS_VDECCORE_ERROR_CPU_MEMORY_SIZE;
    }
    if (mem.cpu_gpu_memory_size < cpu_gpu) {
        return ORBIS_VDECCORE_ERROR_CPU_GPU_MEMORY_SIZE;
    }
    if (mem.gpu_memory_size < gpu) {
        return ORBIS_VDECCORE_ERROR_GPU_MEMORY_SIZE;
    }
    auto* compute_handle = static_cast<ComputeHandle*>(compute);
    if (!IsValidCompute(compute_handle)) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    auto* header = static_cast<CoreHeader*>(mem.cpu_memory);
    header->magic = kCoreMagic;
    header->mode = 2;
    header->level = cfg.level;
    header->profile = cfg.profile;
    header->codec = cfg.codec;
    header->new_sequence = 0;
    compute_handle->attached_decoders++;
    {
        std::scoped_lock lk{g_decoders_mutex};
        g_decoders[mem.cpu_memory] = std::make_unique<CoreDecoder>(params, compute_handle, sdk);
    }
    *core = mem.cpu_memory;
    return ORBIS_OK;
}

s32 VdecCoreDeleteDecoder(void* core) {
    std::unique_ptr<CoreDecoder> decoder;
    {
        std::scoped_lock lk{g_decoders_mutex};
        const auto it = g_decoders.find(core);
        if (it == g_decoders.end()) {
            return ORBIS_VDECCORE_ERROR_FAIL;
        }
        decoder = std::move(it->second);
        g_decoders.erase(it);
    }
    ComputeHandle* compute = decoder->Compute();
    if (IsValidCompute(compute) && compute->attached_decoders != 0) {
        compute->attached_decoders--;
    }
    return ORBIS_OK;
}

s32 VdecCoreResetDecoder(void* core) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    decoder->Reset();
    static_cast<CoreHeader*>(core)->new_sequence = 0;
    return ORBIS_OK;
}

s32 VdecCoreSetDecodeInput(void* core, const VdecCoreInput& input) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    const s32 ret = decoder->SetDecodeInput(input);
    if (ret == ORBIS_VDECCORE_ERROR_NEW_SEQUENCE) {
        static_cast<CoreHeader*>(core)->new_sequence = 1;
    }
    return ret;
}

s32 VdecCoreSyncDecodeWptr(void* core, u32* frame_count, void** decoded_au) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    const s32 ret = decoder->SyncDecodeWptr(frame_count, decoded_au);
    return ret == ORBIS_VDECCORE_ERROR_NOT_READY ? ORBIS_VDECCORE_ERROR_FAIL : ret;
}

s32 VdecCoreTrySyncDecodeWptr(void* core, u32* frame_count, void** decoded_au) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    return decoder->SyncDecodeWptr(frame_count, decoded_au);
}

s32 VdecCoreSetDecodeOutputSw(void* core, void* frame_buffer, u64 frame_buffer_size) {
    if (static_cast<CoreHeader*>(core)->mode != 2) {
        return ORBIS_VDECCORE_ERROR_RESOURCE_TYPE;
    }
    if ((reinterpret_cast<u64>(frame_buffer) & 0xff) != 0) {
        return ORBIS_VDECCORE_ERROR_FRAME_BUFFER_ALIGNMENT;
    }
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    return decoder->SetDecodeOutput(frame_buffer, frame_buffer_size);
}

s32 VdecCoreSyncDecodeOutputSw(void* core, VdecCoreOutput* output) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    return decoder->SyncDecodeOutput(output, false);
}

s32 VdecCoreTrySyncDecodeOutputSw(void* core, VdecCoreOutput* output) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    return decoder->SyncDecodeOutput(output, true);
}

s32 VdecCoreFlushDecodeOutput(void* core, s32* frame_count) {
    CoreDecoder* decoder = FindDecoder(core);
    if (decoder == nullptr) {
        return ORBIS_VDECCORE_ERROR_FAIL;
    }
    return decoder->Flush(frame_count);
}

} // namespace Libraries::Vdecsw
