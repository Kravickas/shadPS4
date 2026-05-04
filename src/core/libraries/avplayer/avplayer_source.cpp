// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/file_sys/fs.h"
#include "core/libraries/avplayer/avplayer_error.h"
#include "core/libraries/avplayer/avplayer_file_streamer.h"
#include "core/libraries/avplayer/avplayer_source.h"

#include <magic_enum/magic_enum.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include "common/support/avdec.h"

namespace Libraries::AvPlayer {

AvPlayerSource::AvPlayerSource(AvPlayerStateCallback& state, bool use_vdec2)
    : m_state(state), m_use_vdec2(use_vdec2) {}

AvPlayerSource::~AvPlayerSource() {
    Stop();
}

bool AvPlayerSource::Init(const AvPlayerInitData& init_data, std::string_view path) {
    m_memory_replacement = init_data.memory_replacement;
    m_max_num_video_framebuffers =
        std::min(std::max(2, init_data.num_output_video_framebuffers), 16);

    AVFormatContext* context = avformat_alloc_context();
    if (init_data.file_replacement.open != nullptr) {
        m_up_data_streamer = std::make_unique<AvPlayerFileStreamer>(init_data.file_replacement);
        if (!m_up_data_streamer->Init(path)) {
            return false;
        }
        context->pb = m_up_data_streamer->GetContext();
        if (AVPLAYER_IS_ERROR(avformat_open_input(&context, nullptr, nullptr, nullptr))) {
            return false;
        }
    } else {
        const auto mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
        const auto filepath = mnt->GetHostPath(path);
        if (AVPLAYER_IS_ERROR(
                avformat_open_input(&context, filepath.string().c_str(), nullptr, nullptr))) {
            return false;
        }
    }
    m_avformat_context = AVFormatContextPtr(context, &ReleaseAVFormatContext);
    return true;
}

bool AvPlayerSource::FindStreamInfo() {
    if (m_avformat_context == nullptr) {
        LOG_ERROR(Lib_AvPlayer, "Could not find stream info. NULL context.");
        return false;
    }
    if (m_avformat_context->nb_streams > 0) {
        return true;
    }
    return avformat_find_stream_info(m_avformat_context.get(), nullptr) == 0;
}

s32 AvPlayerSource::GetStreamCount() {
    if (m_avformat_context == nullptr) {
        LOG_ERROR(Lib_AvPlayer, "Could not get stream count. NULL context.");
        return -1;
    }
    LOG_INFO(Lib_AvPlayer, "Stream Count: {}", m_avformat_context->nb_streams);
    return m_avformat_context->nb_streams;
}

static AvPlayerStreamType CodecTypeToStreamType(AVMediaType codec_type) {
    switch (codec_type) {
    case AVMediaType::AVMEDIA_TYPE_VIDEO:
        return AvPlayerStreamType::Video;
    case AVMediaType::AVMEDIA_TYPE_AUDIO:
        return AvPlayerStreamType::Audio;
    case AVMediaType::AVMEDIA_TYPE_SUBTITLE:
        return AvPlayerStreamType::TimedText;
    default:
        LOG_ERROR(Lib_AvPlayer, "Unexpected AVMediaType {}", magic_enum::enum_name(codec_type));
        return AvPlayerStreamType::Unknown;
    }
}

bool AvPlayerSource::GetStreamInfo(u32 stream_index, AvPlayerStreamInfo& info) {
    info = {};
    if (m_avformat_context == nullptr || stream_index >= m_avformat_context->nb_streams) {
        LOG_ERROR(Lib_AvPlayer, "Could not get stream {} info.", stream_index);
        return false;
    }
    const auto p_stream = m_avformat_context->streams[stream_index];
    if (p_stream == nullptr || p_stream->codecpar == nullptr) {
        LOG_ERROR(Lib_AvPlayer, "Could not get stream {} info. NULL stream.", stream_index);
        return false;
    }
    info.type = CodecTypeToStreamType(p_stream->codecpar->codec_type);
    info.start_time = p_stream->start_time;
    info.duration = p_stream->duration;
    const auto p_lang_node = av_dict_get(p_stream->metadata, "language", nullptr, 0);
    if (p_lang_node != nullptr) {
        LOG_INFO(Lib_AvPlayer, "Stream {} language = {}", stream_index, p_lang_node->value);
    } else {
        LOG_WARNING(Lib_AvPlayer, "Stream {} language is unknown", stream_index);
    }
    switch (info.type) {
    case AvPlayerStreamType::Video: {
        LOG_INFO(Lib_AvPlayer, "Stream {} is a video stream.", stream_index);
        info.details.video.aspect_ratio =
            f32(p_stream->codecpar->width) / p_stream->codecpar->height;
        info.details.video.width = u32(p_stream->codecpar->width);
        info.details.video.height = u32(p_stream->codecpar->height);
        if (p_lang_node != nullptr) {
            std::memcpy(info.details.video.language_code, p_lang_node->value,
                        std::min(strlen(p_lang_node->value), size_t(3)));
        }
        break;
    }
    case AvPlayerStreamType::Audio: {
        LOG_INFO(Lib_AvPlayer, "Stream {} is an audio stream.", stream_index);
        info.details.audio.channel_count = p_stream->codecpar->ch_layout.nb_channels;
        info.details.audio.sample_rate = p_stream->codecpar->sample_rate;
        info.details.audio.size = 0; // sceAvPlayerGetStreamInfo() is expected to set this to 0
        if (p_lang_node != nullptr) {
            std::memcpy(info.details.audio.language_code, p_lang_node->value,
                        std::min(strlen(p_lang_node->value), size_t(3)));
        }
        break;
    }
    case AvPlayerStreamType::TimedText: {
        LOG_WARNING(Lib_AvPlayer, "Stream {} is a timedtext stream.", stream_index);
        info.details.subs.font_size = 12;
        info.details.subs.text_size = 12;
        if (p_lang_node != nullptr) {
            std::memcpy(info.details.subs.language_code, p_lang_node->value,
                        std::min(strlen(p_lang_node->value), size_t(3)));
        }
        break;
    }
    default: {
        LOG_ERROR(Lib_AvPlayer, "Stream {} type is unknown: {}.", stream_index,
                  magic_enum::enum_name(info.type));
        return false;
    }
    }
    return true;
}

bool AvPlayerSource::EnableStream(u32 stream_index) {
    if (m_avformat_context == nullptr || stream_index >= m_avformat_context->nb_streams) {
        return false;
    }
    const auto stream = m_avformat_context->streams[stream_index];
    switch (stream->codecpar->codec_type) {
    case AVMediaType::AVMEDIA_TYPE_VIDEO: {
        m_video_stream_index = stream_index;
        LOG_INFO(Lib_AvPlayer, "Video stream {} enabled", stream_index);
        break;
    }
    case AVMediaType::AVMEDIA_TYPE_AUDIO: {
        m_audio_stream_index = stream_index;
        LOG_INFO(Lib_AvPlayer, "Audio stream {} enabled", stream_index);
        break;
    }
    default:
        LOG_WARNING(Lib_AvPlayer, "Unknown stream type {} for stream {}",
                    magic_enum::enum_name(stream->codecpar->codec_type), stream_index);
        break;
    }
    return true;
}

void AvPlayerSource::SetLooping(bool is_looping) {
    m_is_looping = is_looping;
}

std::optional<bool> AvPlayerSource::HasFrames(u32 num_frames) {
    return m_video_packets.Size() > num_frames || m_is_eof;
}

bool AvPlayerSource::Start() {
    std::unique_lock lock(m_state_mutex);

    if (!m_video_stream_index && !m_audio_stream_index) {
        LOG_ERROR(Lib_AvPlayer, "Could not start playback. No streams.");
        return false;
    }
    if (m_video_stream_index) {
        const auto stream = m_avformat_context->streams[m_video_stream_index.value()];
        avformat_seek_file(m_avformat_context.get(), m_video_stream_index.value(), 0, 0,
                           stream->duration, 0);
        const auto decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (decoder == nullptr) {
            return false;
        }
        m_video_codec_context =
            AVCodecContextPtr(avcodec_alloc_context3(decoder), &ReleaseAVCodecContext);
        if (avcodec_parameters_to_context(m_video_codec_context.get(), stream->codecpar) < 0) {
            LOG_ERROR(Lib_AvPlayer, "Could not copy stream {} avcodec parameters to context.",
                      m_video_stream_index.value());
            return false;
        }
        if (avcodec_open2(m_video_codec_context.get(), decoder, nullptr) < 0) {
            LOG_ERROR(Lib_AvPlayer, "Could not open avcodec for video stream {}.",
                      m_video_stream_index.value());
            return false;
        }
        const auto width = u32(m_video_codec_context->width);
        const auto height = u32(m_video_codec_context->height);
        // Match ImageSizeLinearAligned pitch (tile.h): max(8, 64/Bpp) = 64 for R8.
        const auto pitch = Common::AlignUp(width, 64u);
        const auto size = (pitch * height * 3) / 2;
        for (u64 index = 0; index < m_max_num_video_framebuffers; ++index) {
            m_video_buffers.Push(GuestBuffer(m_memory_replacement, 0x100, size, true));
        }
    }
    if (m_audio_stream_index) {
        const auto stream = m_avformat_context->streams[m_audio_stream_index.value()];
        avformat_seek_file(m_avformat_context.get(), m_audio_stream_index.value(), 0, 0,
                           stream->duration, 0);
        const auto decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (decoder == nullptr) {
            return false;
        }
        m_audio_codec_context =
            AVCodecContextPtr(avcodec_alloc_context3(decoder), &ReleaseAVCodecContext);
        if (avcodec_parameters_to_context(m_audio_codec_context.get(), stream->codecpar) < 0) {
            LOG_ERROR(Lib_AvPlayer, "Could not copy stream {} avcodec parameters to context.",
                      m_audio_stream_index.value());
            return false;
        }
        if (avcodec_open2(m_audio_codec_context.get(), decoder, nullptr) < 0) {
            LOG_ERROR(Lib_AvPlayer, "Could not open avcodec for audio stream {}.",
                      m_audio_stream_index.value());
            return false;
        }
        const auto num_channels = m_audio_codec_context->ch_layout.nb_channels;
        const auto align = num_channels * sizeof(u16);
        const auto size = num_channels * sizeof(u16) * 1024;
        for (u64 index = 0; index < 8; ++index) {
            m_audio_buffers.Push(GuestBuffer(m_memory_replacement, 0x100, size, false));
        }
    }
    m_demuxer_thread.Run([this](std::stop_token stop) { this->DemuxerThread(stop); });
    if (m_video_stream_index) {
        m_video_decoder_thread.Run(
            [this](std::stop_token stop) { this->VideoDecoderThread(stop); });
    }
    if (m_audio_stream_index) {
        m_audio_decoder_thread.Run(
            [this](std::stop_token stop) { this->AudioDecoderThread(stop); });
    }
    m_start_time = std::chrono::high_resolution_clock::now();
    return true;
}

bool AvPlayerSource::Stop() {
    std::unique_lock lock(m_state_mutex);

    const bool post_eof_cleanup = !HasRunningThreads();
    if (post_eof_cleanup) {
        LOG_INFO(Lib_AvPlayer, "Stop: post-EOF cleanup (decoder + demuxer threads "
                               "already exited on their own; flushing queues)");
    } else {
        LOG_INFO(Lib_AvPlayer, "Stop: game-initiated, signalling threads");
    }

    if (m_up_data_streamer) {
        m_up_data_streamer->Reset();
    }

    // .Stop() is idempotent on a non-Joinable thread; both paths above end
    // up here, and threads that have already exited are a no-op.
    m_video_decoder_thread.Stop();
    m_audio_decoder_thread.Stop();
    m_demuxer_thread.Stop();

    m_current_audio_frame.reset();
    m_current_video_frame.reset();
    m_previous_video_frame.reset();

    m_video_buffers.Clear();
    m_audio_buffers.Clear();
    m_audio_packets.Clear();
    m_video_packets.Clear();
    m_audio_frames.Clear();
    m_video_frames.Clear();

    m_last_audio_ts.reset();
    m_start_time.reset();
    m_pause_time = {};
    m_pause_duration = {};

    m_is_paused = false;
    m_is_eof = false;

    return true;
}

void AvPlayerSource::Pause() {
    m_pause_time = std::chrono::high_resolution_clock::now();
    m_is_paused = true;
}

void AvPlayerSource::Resume() {
    m_pause_duration += std::chrono::high_resolution_clock::now() - m_pause_time;
    m_is_paused = false;
}

bool AvPlayerSource::GetVideoData(AvPlayerFrameInfo& video_info) {
    AvPlayerFrameInfoEx info{};
    if (!GetVideoData(info)) {
        return false;
    }
    video_info = {};
    video_info.timestamp = u64(info.timestamp);
    video_info.p_data = reinterpret_cast<u8*>(info.p_data);
    video_info.details.video.aspect_ratio = info.details.video.aspect_ratio;
    video_info.details.video.width = info.details.video.width;
    video_info.details.video.height = info.details.video.height;
    return true;
}

bool AvPlayerSource::GetVideoData(AvPlayerFrameInfoEx& video_info) {
    // Stage one frame back. This guards against the GPU still
    // reading the data of frame N when the game asks for frame N+1.
    if (m_previous_video_frame.has_value()) {
        m_video_buffers.Push(std::move(m_previous_video_frame->buffer));
        m_previous_video_frame.reset();
        m_video_buffers_cv.Notify();
    }
    if (m_current_video_frame.has_value()) {
        m_previous_video_frame = std::move(m_current_video_frame);
        m_current_video_frame.reset();
    }

    enum class Result { Released, NotActive, Paused, NoFrame, Early };
    Result outcome = Result::Released;
    s64 last_drift = 0;
    u64 last_pts = 0;
    u64 last_clock = 0;
    u64 qsize = m_video_frames.Size();

    if (!IsActive()) {
        outcome = Result::NotActive;
    } else if (m_is_paused) {
        outcome = Result::Paused;
    } else if (m_video_frames.Size() == 0) {
        outcome = Result::NoFrame;
    } else if (m_state.GetSyncMode() == AvPlayerAvSyncMode::Default) {
        const auto& new_frame = m_video_frames.Front();
        const auto current_time = CurrentTime();
        last_pts = new_frame.info.timestamp;
        last_clock = current_time;
        last_drift = s64(current_time) - s64(last_pts);
        if (0 < current_time && current_time < new_frame.info.timestamp) {
            outcome = Result::Early;
        }
    }

    // Throttled diagnostic: log aggregate stats once per second so we can see
    // what the gate is doing across the whole playback without per-call spam.
    {
        using namespace std::chrono;
        static std::atomic_uint64_t s_calls{0};
        static std::atomic_uint64_t s_released{0};
        static std::atomic_uint64_t s_early{0};
        static std::atomic_uint64_t s_no_frame{0};
        static std::atomic<int64_t> s_next_log_ms{0};

        s_calls.fetch_add(1, std::memory_order_relaxed);
        if (outcome == Result::Released)
            s_released.fetch_add(1, std::memory_order_relaxed);
        else if (outcome == Result::Early)
            s_early.fetch_add(1, std::memory_order_relaxed);
        else if (outcome == Result::NoFrame)
            s_no_frame.fetch_add(1, std::memory_order_relaxed);

        const auto now_ms =
            duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
        int64_t expected = s_next_log_ms.load(std::memory_order_relaxed);
        if (now_ms >= expected && s_next_log_ms.compare_exchange_strong(expected, now_ms + 1000)) {
            LOG_INFO(Lib_AvPlayer,
                     "GetVideoData/sec: calls={} released={} early={} noframe={} qsize={} "
                     "last_drift={}ms last_pts={}ms last_clock={}ms",
                     s_calls.exchange(0, std::memory_order_relaxed),
                     s_released.exchange(0, std::memory_order_relaxed),
                     s_early.exchange(0, std::memory_order_relaxed),
                     s_no_frame.exchange(0, std::memory_order_relaxed), qsize, last_drift, last_pts,
                     last_clock);
        }
    }

    if (outcome != Result::Released) {
        return false;
    }

    auto frame = m_video_frames.Pop();
    video_info = frame->info;
    m_current_video_frame = std::move(frame);
    return true;
}

bool AvPlayerSource::GetAudioData(AvPlayerFrameInfo& audio_info) {
    if (m_current_audio_frame.has_value()) {
        m_audio_buffers.Push(std::move(m_current_audio_frame->buffer));
        m_current_audio_frame.reset();
        m_audio_buffers_cv.Notify();
    }

    bool released = false;
    bool inactive = false;
    if (!IsActive() || m_is_paused) {
        inactive = true;
    } else if (m_audio_frames.Size() != 0) {
        auto frame = m_audio_frames.Pop();
        m_last_audio_ts = frame->info.timestamp;

        audio_info = {};
        audio_info.timestamp = frame->info.timestamp;
        audio_info.p_data = reinterpret_cast<u8*>(frame->info.p_data);
        audio_info.details.audio.sample_rate = frame->info.details.audio.sample_rate;
        audio_info.details.audio.size = frame->info.details.audio.size;
        audio_info.details.audio.channel_count = frame->info.details.audio.channel_count;
        m_current_audio_frame = std::move(frame);
        released = true;
    }

    {
        using namespace std::chrono;
        static std::atomic_uint64_t s_calls{0};
        static std::atomic_uint64_t s_released{0};
        static std::atomic_uint64_t s_inactive{0};
        static std::atomic<int64_t> s_next_log_ms{0};
        s_calls.fetch_add(1, std::memory_order_relaxed);
        if (released)
            s_released.fetch_add(1, std::memory_order_relaxed);
        if (inactive)
            s_inactive.fetch_add(1, std::memory_order_relaxed);

        const auto now_ms =
            duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
        int64_t expected = s_next_log_ms.load(std::memory_order_relaxed);
        if (now_ms >= expected && s_next_log_ms.compare_exchange_strong(expected, now_ms + 1000)) {
            LOG_INFO(Lib_AvPlayer,
                     "GetAudioData/sec: calls={} released={} inactive={} qsize={} last_ts={}ms",
                     s_calls.exchange(0, std::memory_order_relaxed),
                     s_released.exchange(0, std::memory_order_relaxed),
                     s_inactive.exchange(0, std::memory_order_relaxed), m_audio_frames.Size(),
                     m_last_audio_ts.value_or(0));
        }
    }

    return released;
}

u64 AvPlayerSource::CurrentTime() {
    if (!IsActive() || !m_start_time.has_value()) {
        return 0;
    }
    using namespace std::chrono;
    return duration_cast<milliseconds>(high_resolution_clock::now() - m_start_time.value() -
                                       m_pause_duration)
        .count();
}

bool AvPlayerSource::IsActive() {
    const bool active = !m_is_eof || m_audio_packets.Size() != 0 || m_video_packets.Size() != 0 ||
                        m_video_frames.Size() != 0 || m_audio_frames.Size() != 0;
    static std::atomic_bool s_last{true};
    bool prev = s_last.exchange(active, std::memory_order_relaxed);
    if (prev != active) {
        LOG_INFO(Lib_AvPlayer,
                 "IsActive transition: {} -> {} (eof={} apkt={} vpkt={} vfr={} afr={})", prev,
                 active, m_is_eof.load(), m_audio_packets.Size(), m_video_packets.Size(),
                 m_video_frames.Size(), m_audio_frames.Size());
    }
    return active;
}

void AvPlayerSource::ReleaseAVPacket(AVPacket* packet) {
    if (packet != nullptr) {
        av_packet_free(&packet);
    }
}

void AvPlayerSource::ReleaseAVFrame(AVFrame* frame) {
    if (frame != nullptr) {
        av_frame_free(&frame);
    }
}

void AvPlayerSource::ReleaseAVCodecContext(AVCodecContext* context) {
    if (context != nullptr) {
        avcodec_free_context(&context);
    }
}

void AvPlayerSource::ReleaseSWRContext(SwrContext* context) {
    if (context != nullptr) {
        swr_free(&context);
    }
}

void AvPlayerSource::ReleaseSWSContext(SwsContext* context) {
    if (context != nullptr) {
        sws_freeContext(context);
    }
}

void AvPlayerSource::ReleaseAVFormatContext(AVFormatContext* context) {
    if (context != nullptr) {
        avformat_close_input(&context);
    }
}

void AvPlayerSource::DemuxerThread(std::stop_token stop) {
    using namespace std::chrono;
    Common::SetCurrentThreadName("shadPS4:AvDemuxer");

    if (!m_audio_stream_index.has_value() && !m_video_stream_index.has_value()) {
        LOG_WARNING(Lib_AvPlayer, "Could not start DEMUXER thread. No streams enabled.");
        return;
    }
    LOG_INFO(Lib_AvPlayer, "Demuxer Thread started");

    bool natural_eof = false;
    while (!stop.stop_requested()) {
        if (m_video_packets.Size() > 30) {
            std::this_thread::sleep_for(milliseconds(5));
            continue;
        }
        if (m_audio_stream_index.has_value() && m_audio_packets.Size() > 8) {
            std::this_thread::sleep_for(milliseconds(5));
            continue;
        }
        AVPacketPtr up_packet(av_packet_alloc(), &ReleaseAVPacket);
        const auto res = av_read_frame(m_avformat_context.get(), up_packet.get());
        if (res < 0) {
            if (res == AVERROR_EOF) {
                if (m_is_looping) {
                    LOG_INFO(Lib_AvPlayer, "EOF reached in demuxer. Looping the source...");
                    m_state.OnWarning(ORBIS_AVPLAYER_ERROR_WAR_LOOPING_BACK);
                    avio_seek(m_avformat_context->pb, 0, SEEK_SET);
                    if (m_video_stream_index.has_value()) {
                        const auto index = m_video_stream_index.value();
                        const auto stream = m_avformat_context->streams[index];
                        avformat_seek_file(m_avformat_context.get(), index, 0, 0, stream->duration,
                                           0);
                        m_video_flush_pending.store(true);
                        m_video_packets_cv.Notify();
                    }
                    if (m_audio_stream_index.has_value()) {
                        const auto index = m_audio_stream_index.value();
                        const auto stream = m_avformat_context->streams[index];
                        avformat_seek_file(m_avformat_context.get(), index, 0, 0, stream->duration,
                                           0);
                        m_audio_flush_pending.store(true);
                        m_audio_packets_cv.Notify();
                    }
                    continue;
                } else {
                    LOG_INFO(Lib_AvPlayer, "EOF reached in demuxer. Exiting.");
                    natural_eof = true;
                    break;
                }
            } else {
                LOG_ERROR(Lib_AvPlayer, "Could not read AV frame: error = {}", res);
                m_state.OnError();
                return;
            }
            break;
        }
        if (up_packet->stream_index == m_video_stream_index) {
            m_video_packets.Push(std::move(up_packet));
            m_video_packets_cv.Notify();
        } else if (up_packet->stream_index == m_audio_stream_index) {
            m_audio_packets.Push(std::move(up_packet));
            m_audio_packets_cv.Notify();
        }
    }

    m_is_eof = true;

    m_video_packets_cv.Notify();
    m_audio_packets_cv.Notify();
    m_video_frames_cv.Notify();
    m_audio_frames_cv.Notify();
    // Wake decoder buffer-wait loops so they can observe m_is_eof and bail
    // out cleanly if the game has stopped recycling buffers.
    m_video_buffers_cv.Notify();
    m_audio_buffers_cv.Notify();

    m_video_decoder_thread.Join();
    m_audio_decoder_thread.Join();

    if (natural_eof) {
        m_state.OnEOF();
    }

    LOG_INFO(Lib_AvPlayer, "Demuxer Thread exited normally");
}

AvPlayerSource::AVFramePtr AvPlayerSource::ConvertVideoFrame(const AVFrame& frame) {
    auto nv12_frame = AVFramePtr{av_frame_alloc(), &ReleaseAVFrame};
    nv12_frame->pts = frame.pts;
    nv12_frame->pkt_dts = frame.pkt_dts < 0 ? 0 : frame.pkt_dts;
    nv12_frame->best_effort_timestamp = frame.best_effort_timestamp;
    nv12_frame->format = AV_PIX_FMT_NV12;
    nv12_frame->width = frame.width;
    nv12_frame->height = frame.height;
    nv12_frame->sample_aspect_ratio = frame.sample_aspect_ratio;
    nv12_frame->crop_top = frame.crop_top;
    nv12_frame->crop_bottom = frame.crop_bottom;
    nv12_frame->crop_left = frame.crop_left;
    nv12_frame->crop_right = frame.crop_right;

    av_frame_get_buffer(nv12_frame.get(), 0);

    if (m_sws_context == nullptr) {
        m_sws_context =
            SWSContextPtr(sws_getContext(frame.width, frame.height, AVPixelFormat(frame.format),
                                         nv12_frame->width, nv12_frame->height, AV_PIX_FMT_NV12,
                                         SWS_FAST_BILINEAR, nullptr, nullptr, nullptr),
                          &ReleaseSWSContext);
    }
    const auto res = sws_scale(m_sws_context.get(), frame.data, frame.linesize, 0, frame.height,
                               nv12_frame->data, nv12_frame->linesize);
    if (res < 0) {
        LOG_ERROR(Lib_AvPlayer, "Could not convert to NV12: {}", av_err2str(res));
        return AVFramePtr{nullptr, &ReleaseAVFrame};
    }
    return nv12_frame;
}

static void CopyNV12Data(u8* dst, const AVFrame& src, bool use_vdec2) {
    // Match ImageSizeLinearAligned pitch (tile.h): max(8, 64/Bpp) = 64 for R8.
    const u32 pitch = Common::AlignUp(u32(src.width), 64u);
    const u32 src_stride_y = u32(src.linesize[0]);
    const u32 src_stride_uv = u32(src.linesize[1]);
    const u32 frame_h = u32(src.height);

    if (pitch == src_stride_y) {
        std::memcpy(dst, src.data[0], pitch * frame_h);
    } else {
        for (u32 y = 0; y < frame_h; ++y) {
            std::memcpy(dst + y * pitch, src.data[0] + y * src_stride_y, src.width);
        }
    }

    u8* chroma_dst = dst + pitch * frame_h;
    const u32 uv_h = frame_h / 2;
    if (pitch == src_stride_uv) {
        std::memcpy(chroma_dst, src.data[1], pitch * uv_h);
    } else {
        for (u32 y = 0; y < uv_h; ++y) {
            std::memcpy(chroma_dst + y * pitch, src.data[1] + y * src_stride_uv, src.width);
        }
    }
}

Frame AvPlayerSource::PrepareVideoFrame(GuestBuffer buffer, const AVFrame& frame) {
    ASSERT(frame.format == AV_PIX_FMT_NV12);

    auto p_buffer = buffer.GetBuffer();
    CopyNV12Data(p_buffer, frame, m_use_vdec2);

    // PTS-based timestamp; AV_NOPTS_VALUE handled to avoid u64 overflow.
    s64 raw_ts = frame.best_effort_timestamp;
    if (raw_ts == AV_NOPTS_VALUE) {
        raw_ts = frame.pts != AV_NOPTS_VALUE ? frame.pts : frame.pkt_dts;
    }
    if (raw_ts == AV_NOPTS_VALUE || raw_ts < 0) {
        raw_ts = 0;
    }
    const u64 ts_units = u64(raw_ts) * 1000;
    const auto stream = m_avformat_context->streams[m_video_stream_index.value()];
    const auto time_base = stream->time_base;
    const auto den = time_base.den;
    const auto num = time_base.num;
    const auto timestamp = (num != 0 && den > 1) ? (ts_units * num) / den : ts_units;

    // Throttled diagnostic: log raw timestamp fields once per second so we can
    // see whether the FFmpeg-decoded frame ordering is sane (out-of-order
    // best_effort_timestamp would explain jitter).
    {
        using namespace std::chrono;
        static std::atomic<int64_t> s_next_log_ms{0};
        const auto now_ms =
            duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
        int64_t expected = s_next_log_ms.load(std::memory_order_relaxed);
        if (now_ms >= expected && s_next_log_ms.compare_exchange_strong(expected, now_ms + 1000)) {
            LOG_INFO(Lib_AvPlayer,
                     "PrepareVideoFrame: pts={} dts={} best_effort={} -> ts={}ms tb={}/{}",
                     s64(frame.pts), s64(frame.pkt_dts), s64(frame.best_effort_timestamp),
                     timestamp, num, den);
        }
    }

    const u32 width = u32(frame.width);
    const u32 height = u32(frame.height);
    // Match ImageSizeLinearAligned pitch (tile.h): max(8, 64/Bpp) = 64 for R8.
    const u32 pitch = Common::AlignUp(width, 64u);
    // PS4's sceAvPlayerGetVideoDataEx adjusts crop_right by the row
    // padding so the reported value is relative to the buffer pitch, not
    // the codec's coded width.
    const u32 crop_right_offset = u32(frame.crop_right) + (pitch - width);

    return Frame{
        .buffer = std::move(buffer),
        .info =
            {
                .p_data = p_buffer,
                .timestamp = timestamp,
                .details =
                    {
                        .video =
                            {
                                .width = width,
                                .height = height,
                                .aspect_ratio = (float)av_q2d(frame.sample_aspect_ratio),
                                .crop_left_offset = u32(frame.crop_left),
                                .crop_right_offset = crop_right_offset,
                                .crop_top_offset = u32(frame.crop_top),
                                .crop_bottom_offset = u32(frame.crop_bottom),
                                .pitch = pitch,
                                .luma_bit_depth = 8,
                                .chroma_bit_depth = 8,
                            },
                    },
            },
    };
}

void AvPlayerSource::VideoDecoderThread(std::stop_token stop) {
    using namespace std::chrono;
    Common::SetCurrentThreadName("shadPS4:AvVideoDecoder");

    LOG_INFO(Lib_AvPlayer, "Video Decoder Thread started");
    bool flush_sent = false;
    while ((!m_is_eof || m_video_packets.Size() != 0 || !flush_sent) && !stop.stop_requested()) {
        if (m_video_flush_pending.exchange(false)) {
            avcodec_flush_buffers(m_video_codec_context.get());
            flush_sent = false;
        }
        if (!m_video_packets_cv.Wait(stop,
                                     [this] { return m_video_packets.Size() != 0 || m_is_eof; })) {
            continue;
        }
        const auto packet = m_video_packets.Pop();
        AVPacket* raw_packet = nullptr;
        if (packet.has_value()) {
            raw_packet = packet->get();
        } else if (m_is_eof && !flush_sent) {
            // Feed NULL once to enter drain mode and flush
            // any frames buffered for B-frame reorder. Without this, the last
            // few frames are silently dropped at end of stream.
            flush_sent = true;
        } else {
            continue;
        }

        auto res = avcodec_send_packet(m_video_codec_context.get(), raw_packet);
        if (res < 0 && res != AVERROR(EAGAIN)) {
            m_state.OnError();
            LOG_ERROR(Lib_AvPlayer, "Could not send packet to the video codec. Error = {}",
                      av_err2str(res));
            return;
        }
        while (res >= 0) {
            if (!m_video_buffers_cv.Wait(
                    stop, [this] { return m_video_buffers.Size() != 0 || m_is_eof; })) {
                break;
            }
            if (m_video_buffers.Size() == 0) {
                // m_is_eof is true and the game has stopped recycling buffers
                // (it's waiting for StateStop before its next loop
                // iteration). We cannot drain further; abandon the trailing
                // FFmpeg-internal B-frames and exit. Cleanup transitions state and
                // the decoder threads simply stop.
                return;
            }
            auto up_frame = AVFramePtr(av_frame_alloc(), &ReleaseAVFrame);
            res = avcodec_receive_frame(m_video_codec_context.get(), up_frame.get());
            if (res < 0) {
                if (res == AVERROR_EOF) {
                    LOG_INFO(Lib_AvPlayer, "EOF reached in video decoder");
                    return;
                } else if (res != AVERROR(EAGAIN)) {
                    LOG_ERROR(Lib_AvPlayer,
                              "Could not receive frame from the video codec. Error = {}",
                              av_err2str(res));
                    m_state.OnError();
                    return;
                }
            } else {
                auto buffer = m_video_buffers.Pop();
                if (!buffer.has_value()) {
                    // Video buffers queue was cleared. This means that player was stopped.
                    break;
                }
                if (up_frame->format != AV_PIX_FMT_NV12) {
                    const auto nv12_frame = ConvertVideoFrame(*up_frame);
                    m_video_frames.Push(PrepareVideoFrame(std::move(buffer.value()), *nv12_frame));
                } else {
                    m_video_frames.Push(PrepareVideoFrame(std::move(buffer.value()), *up_frame));
                }
                m_video_frames_cv.Notify();
            }
        }
    }

    LOG_INFO(Lib_AvPlayer, "Video Decoder Thread exited normally");
}

AvPlayerSource::AVFramePtr AvPlayerSource::ConvertAudioFrame(const AVFrame& frame) {
    auto pcm16_frame = AVFramePtr{av_frame_alloc(), &ReleaseAVFrame};
    pcm16_frame->pts = frame.pts;
    pcm16_frame->pkt_dts = frame.pkt_dts < 0 ? 0 : frame.pkt_dts;
    pcm16_frame->format = AV_SAMPLE_FMT_S16;
    pcm16_frame->ch_layout = frame.ch_layout;
    pcm16_frame->sample_rate = frame.sample_rate;

    if (m_swr_context == nullptr) {
        SwrContext* swr_context = nullptr;
        AVChannelLayout in_ch_layout = frame.ch_layout;
        AVChannelLayout out_ch_layout = frame.ch_layout;
        swr_alloc_set_opts2(&swr_context, &out_ch_layout, AV_SAMPLE_FMT_S16, frame.sample_rate,
                            &in_ch_layout, AVSampleFormat(frame.format), frame.sample_rate, 0,
                            nullptr);
        m_swr_context = SWRContextPtr(swr_context, &ReleaseSWRContext);
        swr_init(m_swr_context.get());
    }
    const auto res = swr_convert_frame(m_swr_context.get(), pcm16_frame.get(), &frame);
    if (res < 0) {
        LOG_ERROR(Lib_AvPlayer, "Could not convert audio to S16: {}", av_err2str(res));
        return AVFramePtr{nullptr, &ReleaseAVFrame};
    }
    return pcm16_frame;
}

Frame AvPlayerSource::PrepareAudioFrame(GuestBuffer buffer, const AVFrame& frame) {
    ASSERT(frame.format == AV_SAMPLE_FMT_S16);
    ASSERT(frame.nb_samples <= 1024);

    auto p_buffer = buffer.GetBuffer();
    const auto size = frame.ch_layout.nb_channels * frame.nb_samples * sizeof(u16);
    std::memcpy(p_buffer, frame.data[0], size);

    // Use best_effort_timestamp for the same reasons as video (AV_NOPTS_VALUE
    // safety, PTS-first ordering). For audio dts == pts in normal AAC streams.
    s64 raw_ts = frame.best_effort_timestamp;
    if (raw_ts == AV_NOPTS_VALUE) {
        raw_ts = frame.pts != AV_NOPTS_VALUE ? frame.pts : frame.pkt_dts;
    }
    if (raw_ts == AV_NOPTS_VALUE || raw_ts < 0) {
        raw_ts = 0;
    }
    const u64 ts_units = u64(raw_ts) * 1000;
    const auto stream = m_avformat_context->streams[m_audio_stream_index.value()];
    const auto time_base = stream->time_base;
    const auto den = time_base.den;
    const auto num = time_base.num;
    const auto timestamp = (num != 0 && den > 1) ? (ts_units * num) / den : ts_units;

    return Frame{
        .buffer = std::move(buffer),
        .info =
            {
                .p_data = p_buffer,
                .timestamp = timestamp,
                .details =
                    {
                        .audio =
                            {
                                .channel_count = u16(frame.ch_layout.nb_channels),
                                .sample_rate = u32(frame.sample_rate),
                                .size = u32(size),
                            },
                    },
            },
    };
}

void AvPlayerSource::AudioDecoderThread(std::stop_token stop) {
    using namespace std::chrono;
    Common::SetCurrentThreadName("shadPS4:AvAudioDecoder");

    LOG_INFO(Lib_AvPlayer, "Audio Decoder Thread started");
    bool flush_sent = false;
    while ((!m_is_eof || m_audio_packets.Size() != 0 || !flush_sent) && !stop.stop_requested()) {
        if (m_audio_flush_pending.exchange(false)) {
            avcodec_flush_buffers(m_audio_codec_context.get());
            flush_sent = false;
        }
        if (!m_audio_packets_cv.Wait(stop,
                                     [this] { return m_audio_packets.Size() != 0 || m_is_eof; })) {
            continue;
        }
        const auto packet = m_audio_packets.Pop();
        AVPacket* raw_packet = nullptr;
        if (packet.has_value()) {
            raw_packet = packet->get();
        } else if (m_is_eof && !flush_sent) {
            // Feed NULL once to enter drain mode and flush any
            // frames buffered internally. Without this, the trailing audio
            // samples are silently dropped at end of stream.
            flush_sent = true;
        } else {
            continue;
        }
        auto res = avcodec_send_packet(m_audio_codec_context.get(), raw_packet);
        if (res < 0 && res != AVERROR(EAGAIN)) {
            m_state.OnError();
            LOG_ERROR(Lib_AvPlayer, "Could not send packet to the audio codec. Error = {}",
                      av_err2str(res));
            return;
        }
        while (res >= 0) {
            if (!m_audio_buffers_cv.Wait(
                    stop, [this] { return m_audio_buffers.Size() != 0 || m_is_eof; })) {
                break;
            }
            if (m_audio_buffers.Size() == 0) {
                // See comment on the equivalent path in VideoDecoderThread.
                return;
            }

            auto up_frame = AVFramePtr(av_frame_alloc(), &ReleaseAVFrame);
            res = avcodec_receive_frame(m_audio_codec_context.get(), up_frame.get());
            if (res < 0) {
                if (res == AVERROR_EOF) {
                    LOG_INFO(Lib_AvPlayer, "EOF reached in audio decoder");
                    return;
                } else if (res != AVERROR(EAGAIN)) {
                    m_state.OnError();
                    LOG_ERROR(Lib_AvPlayer,
                              "Could not receive frame from the audio codec. Error = {}",
                              av_err2str(res));
                    return;
                }
            } else {
                auto buffer = m_audio_buffers.Pop();
                if (!buffer.has_value()) {
                    // Audio buffers queue was cleared. This means that player was stopped.
                    break;
                }
                if (up_frame->format != AV_SAMPLE_FMT_S16) {
                    const auto pcm16_frame = ConvertAudioFrame(*up_frame);
                    m_audio_frames.Push(PrepareAudioFrame(std::move(buffer.value()), *pcm16_frame));
                } else {
                    m_audio_frames.Push(PrepareAudioFrame(std::move(buffer.value()), *up_frame));
                }
                m_audio_frames_cv.Notify();
            }
        }
    }

    LOG_INFO(Lib_AvPlayer, "Audio Decoder Thread exited normally");
}

bool AvPlayerSource::HasRunningThreads() const {
    return m_demuxer_thread.Joinable() || m_video_decoder_thread.Joinable() ||
           m_audio_decoder_thread.Joinable();
}

} // namespace Libraries::AvPlayer
