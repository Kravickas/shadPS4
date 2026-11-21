// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class Rasterizer;
struct DrawAhead;
} // namespace Vulkan

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

inline u64 PerfNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Timing of the current command processor cycle (wake to GpuIdle), GPU thread only.
struct CpPerf {
    u64 vo_block_ns;
    u64 vo_spin_ns;
    u64 mem_spin_ns;
    u64 asc_spin_ns;
    u64 draw_ns;
    u64 draw_pipeline_ns;
    u64 draw_bind_ns;
    u64 dispatch_ns;
    u64 dispatch_pipeline_ns;
    u64 copy_ns;
    u64 fence_ns;
    u64 eop_ns;
    u64 flush_ns;
    u32 vo_blocks;
    u32 mem_spins;
    u32 tasks;
    u32 draws;
    u32 dispatches;
    u32 flushes;
    u64 draw_prep_ns;     ///< PrepareRenderState
    u64 draw_begin_ns;    ///< BeginRendering
    u64 bind_buffers_ns;  ///< BindBuffers, draws and dispatches
    u64 bind_textures_ns; ///< BindTextures, draws and dispatches
    u64 dma_sync_ns;      ///< SynchronizeDmaBuffers
    u64 eos_ns;           ///< EVENT_WRITE_EOS handling
    u64 commands_ns;      ///< Commands from other threads run on the GPU thread
    u32 eos;
    u32 commands;
    u64 draw_pre_ns;     ///< Draw: periodic flush, pending ops, filter
    u64 draw_vb_ns;      ///< Draw: vertex and index buffers
    u64 draw_barrier_ns; ///< Draw: FlushBarriers
    u64 draw_desc_ns;    ///< Draw: pipeline descriptor/push updates
    u64 draw_dyn_ns;     ///< Draw: UpdateDynamicState
    u64 draw_rp_ns;      ///< Draw: Scheduler::BeginRendering
    u64 draw_cmd_ns;     ///< Draw: bind pipeline and draw command
    u32 draw_barriers;   ///< Draws that flushed barriers
    u32 render_passes;   ///< Dynamic rendering passes begun
    u32 zpass_dumps;     ///< ZPASS_DONE occlusion counter dumps (results are synthesized)
    u32 obtains;         ///< BufferCache::ObtainBuffer calls
    u32 obtain_stream;   ///< Read-only bindings copied into the stream buffer
    u64 obtain_stream_bytes;
    u32 obtain_stream_reused; ///< ... that reused an identical copy of the same command buffer
    u64 obtain_stream_reused_bytes;
    u32 obtain_written; ///< Writable bindings
    u32 obtain_uploads; ///< Bindings that uploaded CPU-modified pages into the arena
    u64 obtain_upload_bytes;
    u32 obtain_texel;       ///< Read-only texel bindings checked against images
    u32 obtain_texel_image; ///< ... that were filled from metadata or an image
    u64 pipe_key_ns;        ///< Pipeline lookup: RefreshGraphicsKey
    u64 pipe_stages_ns;     ///< Pipeline lookup: RefreshGraphicsStages
    u32 pipe_changes;       ///< Draws whose pipeline differs from the previous draw's
    /// Shader stage sub-steps, timed on every 8th draw and scaled by 8
    u64 stg_params_ns;
    u64 stg_runtime_ns;
    u64 stg_lookup_ns;
    u64 stg_flatbuf_ns;
    u64 stg_spec_ns;
    u64 stg_find_ns;
    u32 stg_calls;       ///< GetProgram calls for an existing program
    u32 stg_same_inputs; ///< Stage lookups with the shader and user data of the previous draw
    /// Time inside the Vulkan calls of a draw, timed on every 8th draw and scaled by 8
    u64 vk_vi_ns;          ///< setVertexInputEXT
    u64 vk_index_ns;       ///< bindIndexBuffer
    u64 vk_desc_ns;        ///< pushConstants and pushDescriptorSet
    u64 vk_dyn_ns;         ///< Dynamic state commit
    u64 vk_draw_ns;        ///< bindPipeline and draw
    u32 stg_perms_scanned; ///< Permutations compared before the match, scaled
    u64 vi_inputs_ns;      ///< Vertex buffers: GetVertexInputs and setVertexInputEXT
    u64 vi_obtain_ns;      ///< Vertex buffers: range merge and ObtainBuffer
    u64 vi_bind_ns;        ///< Vertex buffers: bindVertexBuffers
    u64 index_ns;          ///< BindIndexBuffer
    u64 cmd_record_ns;     ///< Draw offsets, bindPipeline and draw call
    u64 reset_images_ns;   ///< ResetBindings: bound images
    u64 reset_buffers_ns;  ///< ResetBindings: AccessBuffer of bound buffers
    u32 bound_buffers;     ///< Buffers tracked for barriers
    u32 bound_images;
    u32 tex_lookups;
    u32 tex_hits;
    u32 rb_downloads; ///< Buffer readbacks downloaded
    u64 rb_bytes;
    u64 rb_finish_ns; ///< Host GPU drains for buffer readbacks
    u32 img_downloads;
    u64 img_finish_ns; ///< Host GPU drains for linear image readbacks
    u32 rec_cmds;      ///< Commands handed to the recorder thread
    u64 rec_bytes;
    u32 rec_publishes;
    u32 rec_wakes;       ///< Publishes that found the recorder thread asleep
    u32 rec_syncs;       ///< Waits for the recorder thread to run everything recorded
    u64 rec_sync_ns;     ///< Time in those waits
    u32 rec_space_waits; ///< Waits for space in the record ring
    u64 sc_total_ns;     ///< Sampled draws: stream buffer path of ObtainBuffer
    u64 sc_compare_ns;   ///< Sampled draws: comparing guest memory with the kept copy
    u64 sc_guest_ns;     ///< Sampled draws: copying guest memory
    u64 sc_write_ns;     ///< Sampled draws: writing or recording the stream buffer write
    u32 run_n;           ///< Runs of draws and dispatches with only state packets between them
    std::array<u32, 6> run_draws; ///< Draws in runs of 1, 2-3, 4-7, 8-15, 16-31 and 32+ draws
    u32 state_packets;            ///< Packets that only set state
    u32 state_words;
    u32 la_runs;    ///< Runs handed to the lookahead thread
    u32 la_results; ///< Draws whose lookahead result was taken
    u32 la_waits;   ///< Draws that had to wait for their result
    u64 la_wait_ns;
    u32 la_sync_waits; ///< Run ends that had to wait for the lookahead thread to stop
    u64 la_sync_ns;
    u32 la_copies;           ///< Full copies of the state into the lookahead thread's
    u32 la_compares;         ///< Comparisons of the lookahead thread's state with the real one
    u32 la_mismatches;       ///< Comparisons or positions that differed
    u32 la_early;            ///< Runs the lookahead thread stopped before their end
    u32 la_lookups_used;     ///< Draws that used the pipeline looked up ahead
    u32 la_lookups_rejected; ///< Lookups made ahead whose inputs had changed by the draw
    u32 la_lookups_checked;  ///< Lookups made ahead whose inputs were read again at the draw
    u64 la_lookup_ns;        ///< Time the lookahead thread's lookups took
    u64 la_stream_ns;        ///< Time the lookahead thread's reads of binding bytes took
    u32 sa_ahead_known;      ///< Bindings read ahead already read in the same run
    u32 sa_ahead_same;       ///< Bindings read ahead that held the bytes of their last read
    u32 sa_ahead_copied;     ///< Bindings read ahead whose bytes were copied
    u64 sa_ahead_bytes;
    u32 sa_ahead_full;    ///< Bindings not read ahead for lack of space
    u32 sa_used;          ///< Stream buffer bindings served from a read ahead
    u32 sa_stale;         ///< Reads ahead dropped because the GPU wrote the range since
    u32 sa_checks;        ///< Reads ahead compared with guest memory at the draw
    u32 sa_diffs;         ///< Compared reads ahead that differed
    u64 tx_first_ns;      ///< Sampled draws: BindTextures up to the images being found
    u64 tx_second_ns;     ///< Sampled draws: BindTextures views, layouts and descriptors
    u64 tx_samplers_ns;   ///< Sampled draws: BindTextures samplers
    u64 dispatch_hle_ns;  ///< DispatchDirect: shaders run by the host instead
    u64 dispatch_bind_ns; ///< DispatchDirect: BindResources
    u32 dispatch_hles;
    u64 dispatch_bindbuf_ns; ///< DispatchDirect and DispatchIndirect: BindBuffers
    u64 dispatch_bindtex_ns; ///< DispatchDirect and DispatchIndirect: BindTextures
    u64 ow_obtain_ns;        ///< ObtainBuffer of written bindings
    u64 ow_inval_ns;         ///< Invalidating images under written bindings
    // ObtainBuffer of written ranges in the arenas:
    u64 ow_bytes;        ///< Their sizes
    u64 ow_dirty_ns;     ///< Marking images under them
    u64 ow_sync_ns;      ///< SynchronizeMemory, made of:
    u64 ow_lock_ns;      ///< waiting for the lock shared with readbacks
    u64 ow_walk_ns;      ///< the memory tracker walk with its page watcher updates
    u64 ow_mark_ns;      ///< emulated writes, GPU-modified ranges and write ticks
    u64 ow_upload_ns;    ///< copying CPU-modified pages
    u64 ow_upload_bytes; ///< Their size
    u64 ow_watch_pages;  ///< Pages the watcher updates of the walks went over
    u32 ow_uploads;      ///< Written ranges with CPU-modified pages
    u32 ow_protects;     ///< Page protection calls of the walks
    // Checked cycles only. Image bindings of graphics stages:
    u32 bc_images;
    u32 bc_images_same;  ///< Same T# and use as the same slot of the draw before, same run
    u32 bc_images_equal; ///< ...and resolved to the same view and layout
    u32 bc_image_stages;
    u32 bc_image_stages_same;  ///< Stages with all image bindings as in the draw before
    u32 bc_image_stages_quiet; ///< ...equal results, nothing recorded, same registry
    u32 bc_draws;
    u32 bc_draws_state_same; ///< Draws with no context or uconfig register changed since the last
    u32 dispatch_skips;      ///< DispatchDirect: dispatches BindResources handled itself
};
inline CpPerf cp_perf;

/// Work of the command recorder thread.
struct RecorderPerf {
    std::atomic<u64> run_ns;          ///< Time running recorded commands
    std::atomic<u64> sleeps;          ///< Times the thread went to sleep with nothing to run
    std::atomic<u64> stream_write_ns; ///< Sampled draws: stream buffer writes
};
inline RecorderPerf recorder_perf;

/// Work of the lookahead thread.
struct LookaheadPerf {
    std::atomic<u64> scan_ns; ///< Time scanning runs
    std::atomic<u64> sleeps;  ///< Times the thread went to sleep with nothing to scan
};
inline LookaheadPerf lookahead_perf;

/// Packets that ended a run of draws, by opcode. The last entry counts commands of other threads.
inline std::array<u64, 257> perf_run_enders{};
inline u32 perf_run_log_cycles = 0;

/// Time guest threads spend waiting for a buffer readback done by the GPU thread.
struct ReadbackWaitPerf {
    std::atomic<u64> count;
    std::atomic<u64> wait_ns;
    std::atomic<u64> max_ns;
    std::atomic<u64> emulated; ///< Accesses done beside GPU data without a readback
};
inline ReadbackWaitPerf readback_wait;

/// Whether the current draw is one whose Vulkan calls are timed.
inline bool perf_vk_sample = false;
/// Probe, on for one command processor cycle in PerfBindCheckRate: whether the bindings of a
/// draw resolve to what the same bindings of the draw before it resolved to.
/// Cycles with at least this many draws turn the sampling profiler on for the next cycle.
inline constexpr u32 PerfProfiledDraws = 3000;
inline bool perf_bind_check = false;
inline constexpr u32 PerfBindCheckRate = 8;
inline constexpr u64 PerfVkSampleRate = 16;
/// Scale of the timings taken with PerfDrawNs: PerfVkSampleRate inside a sampled draw, 0 inside
/// the other draws, 1 everywhere else. Reading the clock for every draw costs too much.
inline u64 perf_draw_scale = 1;

/// Clock for timings inside draws. Differences come out scaled, and as 0 on draws not sampled.
inline u64 PerfDrawNs() {
    return perf_draw_scale != 0 ? PerfNs() * perf_draw_scale : 0;
}

/// Adds the lifetime of the object to a CpPerf field on sampled draws, scaled by the rate.
struct VkCallPerf {
    explicit VkCallPerf(u64& field_) : field{field_}, t0{perf_vk_sample ? PerfNs() : 0} {}
    ~VkCallPerf() {
        if (perf_vk_sample) {
            field += (PerfNs() - t0) * PerfVkSampleRate;
        }
    }
    u64& field;
    u64 t0;
};

/// Adds the lifetime of the object to a CpPerf field, on the clock for timings inside draws.
struct CpDrawPerfScope {
    explicit CpDrawPerfScope(u64& field_) : field{field_}, t0{PerfDrawNs()} {}
    ~CpDrawPerfScope() {
        field += PerfDrawNs() - t0;
    }
    u64& field;
    u64 t0;
};

/// Adds the lifetime of the object to a CpPerf field.
struct CpPerfScope {
    explicit CpPerfScope(u64& field_) : field{field_}, t0{PerfNs()} {}
    ~CpPerfScope() {
        field += PerfNs() - t0;
    }
    u64& field;
    u64 t0;
};

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    u32 index_buffer_num_indices{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

    /// Runs the commands other threads sent. For the GPU thread while it waits for a thread
    /// that may itself be waiting for one of those commands.
    void RunPendingCommands() {
        ProcessCommands();
    }

    /// What the state packets of the graphics queue set and its draws read.
    struct GfxState {
        Regs& regs;
        ComputeProgram& cs_state;
        std::array<CbDbExtent, NUM_COLOR_BUFFERS>& last_cb_extent;
        CbDbExtent& last_db_extent;
        u32& index_buffer_num_indices;
        VAddr& indirect_args_addr;
    };

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        std::unique_lock lk{submit_mutex};
        submit_cv.wait(lk, [this] { return num_submits == 0; });
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{64};

    std::thread::id GetGpuCommandProcessorThread() {
        return gpu_id;
    }

    /// Names the current run of state packets and draws. For the GPU thread.
    u64 LookaheadRun() const noexcept {
        return lookahead_run;
    }

#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId() {
        return gpu_tid;
    }
#endif

private:
    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    Task ProcessGraphics(std::span<const u32> dcb);
    Task ProcessCeUpdate(std::span<const u32> ccb);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid);

    void ProcessCommands();
    void ProcessCeQueue();
    void Process(std::stop_token stoken);
    void HandleGetLodStats(const PM4CmdGetLodStats* get_lod);

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        std::queue<Task::Handle> submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    std::queue<Task::Handle> ce_submits;
    u32 num_mapped_queues{1u}; // GFX is always available

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};
    u32 vgt_strmout_buffer_filled_size[4]{};

    struct ConstantEngine {
        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    /// The lookahead thread's own copy of the graphics queue state.
    struct GfxStateCopy {
        Regs regs{};
        ComputeProgram cs_state{};
        std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
        CbDbExtent last_db_extent{};
        u32 index_buffer_num_indices{};
        VAddr indirect_args_addr{};

        GfxState Ref() {
            return {regs,
                    cs_state,
                    last_cb_extent,
                    last_db_extent,
                    index_buffer_num_indices,
                    indirect_args_addr};
        }
    };

    struct LookaheadResult {
        const u32* packet;
    };
    static constexpr std::size_t NumLookaheadResults = 4096;

    static constexpr u32 LookaheadCompareInterval = 64;
    static constexpr u32 LookaheadIdleSpins = 4096;

    GfxState LiveGfxState() {
        return {regs,           mapped_queues[GfxQueueId].cs_state, last_cb_extent,
                last_db_extent, index_buffer_num_indices,           indirect_args_addr};
    }

    enum class PacketClass { State, Draw, Other };

    /// Probe: whether a state packet gives a config, context or uconfig register a new value.
    bool ChangesStateRegs(const PM4Header* header) const;

    /// Whether a packet only sets state, is a draw or dispatch, or does anything else.
    static PacketClass ClassifyPacket(const PM4Header* header, bool guest_markers_enabled);

    /// Applies what a state packet or a draw sets in the graphics queue state.
    static void ApplyPacketState(const GfxState& state, const PM4Header* header);

    /// Whether ClassifyPacket and ApplyPacketState can run on a packet that words_available
    /// words are readable from, without reading or writing out of bounds or asserting.
    static bool IsSafePacket(const PM4Header* header, std::size_t words_available);

    /// Lets the lookahead thread scan the packets from the start of dcb up to the first one that
    /// is neither a state packet nor a draw.
    void StartLookaheadRun(std::span<const u32> dcb);

    /// Waits until the lookahead thread has stopped. packet is where the run ends.
    void EndLookaheadRun(const u32* packet);

    /// Takes the lookahead result of the draw at the start of dcb, restarting the run there if
    /// the lookahead thread did not get to it.
    void TakeLookaheadResult(std::span<const u32> dcb);

    void WaitLookaheadStopped();

    /// The lookahead thread can be waiting for this thread to run a command for it (a readback
    /// of memory its lookup reads), so this thread keeps running commands while it waits.
    void PauseForLookahead();

    bool CompareLookaheadState();

    void LookaheadThread(std::stop_token stoken);

    std::unique_ptr<GfxStateCopy> lookahead_state;
    std::array<LookaheadResult, NumLookaheadResults> lookahead_results;
    /// What was prepared for the draw of each result. An entry is not rewritten while it is
    /// the last one taken, so the draw it belongs to can use it.
    std::unique_ptr<Vulkan::DrawAhead[]> lookahead_draws;
    /// Probe timings of the lookahead thread's lookups, read while it is stopped.
    std::unique_ptr<CpPerf> lookahead_cp_perf;
    /// Run the lookahead thread scans. Only changes while it is stopped.
    const u32* lookahead_run_begin{};
    const u32* lookahead_run_end{};
    /// Where the lookahead thread stopped. Valid once lookahead_stopped_run names the run.
    const u32* lookahead_stop{};
    bool is_lookahead_running{};
    bool lookahead_needs_copy{true};
    u32 lookahead_compare_countdown{};
    u32 lookahead_logged_mismatches{};
    u64 lookahead_run{};
    u32 perf_cycles{};
    /// Probe: a context, uconfig or config register changed since the last draw.
    bool perf_state_changed{};
    u64 lookahead_taken{};
    alignas(64) std::atomic<u64> lookahead_started_run{};
    alignas(64) std::atomic<u64> lookahead_stopped_run{};
    alignas(64) std::atomic<u64> lookahead_produced{};
    alignas(64) std::atomic<u64> lookahead_consumed{};
    alignas(64) std::atomic<bool> lookahead_sleeping{};
    std::mutex lookahead_mutex;
    std::condition_variable_any lookahead_cv;

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    const bool guest_markers_enabled;
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
#ifdef __linux__
    u32 gpu_tid;
#endif
    s32 curr_qid{-1};

private:
    std::jthread lookahead_thread;
};

} // namespace AmdGpu
