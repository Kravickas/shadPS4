// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
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

    /// Equal at two calls: no register counted in state_regs_generation changed in between.
    u64 TakeStateRegsGeneration() {
        has_state_regs_change = false;
        return state_regs_generation;
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

    static constexpr u32 LookaheadIdleSpins = 4096;
    static constexpr u32 LookaheadBusySpins = 1 << 20;

    GfxState LiveGfxState() {
        return {regs,           mapped_queues[GfxQueueId].cs_state, last_cb_extent,
                last_db_extent, index_buffer_num_indices,           indirect_args_addr};
    }

    enum class PacketClass { State, Draw, Other };

    /// Whether a packet only sets state, is a draw or dispatch, or does anything else.
    static PacketClass ClassifyPacket(const PM4Header* header, bool guest_markers_enabled);

    /// Applies what a state packet or a draw sets in the graphics queue state.
    static void ApplyPacketState(const GfxState& state, const PM4Header* header);

    /// Whether ClassifyPacket and ApplyPacketState can run on a packet that words_available
    /// words are readable from, without reading or writing out of bounds or asserting.
    static bool IsSafePacket(const PM4Header* header, std::size_t words_available);

    /// Whether the lookahead thread would prepare a draw in the packets from dcb up to where it
    /// stops. It prepares the direct draws only.
    bool HasLookaheadDraw(std::span<const u32> dcb) const;

    /// Called for a state packet before it is applied: counts it in state_regs_generation if
    /// it gives a register another value.
    void NoteStatePacket(const PM4Header* header);
    void NoteStateRegsChange() {
        ++state_regs_generation;
        has_state_regs_change = true;
    }

    /// Applies a packet of a run without the lookahead thread to the lookahead state. The
    /// thread is stopped then.
    void ApplySoloPacketState(const PM4Header* header, std::size_t words_available);

    /// In a run or a stretch without the lookahead thread: applies the packets at the start of
    /// dcb that only set state, whatever their payload, and returns what is behind them.
    /// Commands of other threads run every StatePacketsPerCommandCheck packets.
    std::span<const u32> ApplyStatePackets(const GfxState& gfx_state, std::span<const u32> dcb);
    static constexpr u32 StatePacketsPerCommandCheck = 64;

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

    void LookaheadThread(std::stop_token stoken);
    /// Second stage of the lookahead: reads what the draws bind for the results the lookahead
    /// thread has made, in their order.
    void LookaheadBindingThread(std::stop_token stoken);
    /// Sleeps until a run other than run is started.
    void SleepUntilLookaheadRun(std::stop_token stoken, u64 run);
    /// Whether a lookahead thread that found no run idle_spins times in a row keeps looking.
    bool SpinsForLookaheadRun(u32 idle_spins) const {
        return idle_spins < LookaheadIdleSpins ||
               (idle_spins < LookaheadBusySpins &&
                !lookahead_gpu_idle.load(std::memory_order_relaxed));
    }

    std::unique_ptr<GfxStateCopy> lookahead_state;
    std::array<LookaheadResult, NumLookaheadResults> lookahead_results;
    /// What was prepared for the draw of each result. An entry is not rewritten while it is
    /// the last one taken, so the draw it belongs to can use it.
    std::unique_ptr<Vulkan::DrawAhead[]> lookahead_draws;
    /// Run the lookahead thread scans. Only changes while it is stopped.
    const u32* lookahead_run_begin{};
    const u32* lookahead_run_end{};
    /// Where the lookahead thread stopped. Valid once lookahead_scanned_run names the run.
    const u32* lookahead_stop{};
    bool is_lookahead_running{};
    /// The packets up to the next Other packet hold no draw the lookahead thread prepares, so
    /// the GPU thread runs them without it and applies them to the lookahead state itself.
    bool is_lookahead_solo{};
    /// Counts the packets that give a register another value through SetConfigReg,
    /// SetContextReg, SetUconfigReg or ClearState, and every packet that is neither a state nor
    /// a draw packet. What IndexType, NumInstances, IndexBase and the draw packets write
    /// (index buffer type and base, instance and index counts, draw initiator) is not counted.
    u64 state_regs_generation{1};
    /// A change was counted since the count was last taken: the packets up to the next take
    /// need no compare.
    bool has_state_regs_change{};
    bool lookahead_needs_copy{true};
    u64 lookahead_run{};
    u64 lookahead_taken{};
    /// Working and waiting time of the GPU thread per cycle, averaged over the last cycles. It
    /// has time left for submits inside a frame while the waiting share is at least
    /// MinIdlePercent.
    static constexpr s64 CycleSmoothing = 16;
    static constexpr s64 MaxIdleSampleNs = 50'000'000;
    static constexpr s64 MinIdlePercent = 10;
    std::chrono::steady_clock::time_point last_cycle_end{};
    bool has_last_cycle{};
    s64 busy_avg_ns{};
    s64 idle_avg_ns{};
    alignas(64) std::atomic<u64> lookahead_started_run{};
    /// Run the lookahead thread has scanned to its end.
    alignas(64) std::atomic<u64> lookahead_scanned_run{};
    /// Run both threads are done with.
    alignas(64) std::atomic<u64> lookahead_stopped_run{};
    /// Results the lookahead thread has made, and those the binding thread has finished. The
    /// GPU thread takes finished ones.
    alignas(64) std::atomic<u64> lookahead_produced{};
    alignas(64) std::atomic<u64> lookahead_ready{};
    alignas(64) std::atomic<u64> lookahead_consumed{};
    alignas(64) std::atomic<u32> lookahead_sleepers{};
    /// Set while the GPU thread waits for work. The lookahead threads only sleep then, or
    /// after LookaheadBusySpins without a run.
    std::atomic<bool> lookahead_gpu_idle{true};
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

    /// Last members: they stop before anything they use is destroyed.
    std::jthread lookahead_thread;
    std::jthread lookahead_binding_thread;
};

} // namespace AmdGpu
