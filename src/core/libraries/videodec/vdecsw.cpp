// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// libSceVdecsw (FW 12.02). Function comments name the SPRX function each block was ported from.

#include <bit>
#include <cstdio>
#include <cstring>

#include "common/logging/log.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/libs.h"
#include "core/libraries/videodec/vdecsw.h"
#include "core/libraries/videodec/vdecsw_core.h"
#include "core/libraries/videodec/vdecsw_memory.h"
#include "core/libraries/videodec/videodec_error.h"
#include "core/memory.h"

namespace Libraries::Vdecsw {

using namespace Libraries::Kernel;

struct VdecswInstance {
    u64 magic0;
    u64 input_count;
    u64 unk10;
    void* last_au_data;
    u64 last_au_size;
    void* last_frame_buffer;
    u64 last_frame_buffer_size;
    u32 sync_output_blocking;
    u32 output_buffer_set;
    u64 output_waiting;
    u64 synced_input_count;
    u32 pending_output_count;
    u32 sync_input_busy;
    u32 new_sequence;
    u32 fatal_stream;
    u32 finalizing;
    u32 pipeline_depth;
    u32 unk68;
    u32 check_memory_type;
    u64 magic1;
    void* core;
    PthreadCondT cond[9];
    PthreadMutexT mutex[12];
};
static_assert(sizeof(VdecswInstance) == 0x128);

namespace {

constexpr u64 kInstanceMagic0 = 0x4c5237444a3555;
constexpr u64 kInstanceMagic1 = 0xa824d9799010a455;
constexpr u64 kCoreMemoryOffset = 0x40000;

s32 Query(const void* addr, OrbisVirtualQueryInfo* info) {
    return ::Core::Memory::Instance()->VirtualQuery(std::bit_cast<VAddr>(addr), 0, info);
}

u32 CompiledSdkVersion(s32* ret) {
    s32 ver = 0;
    *ret = sceKernelGetCompiledSdkVersion(&ver);
    return static_cast<u32>(ver);
}

s32 MutexLock(PthreadMutexT* m) {
    return ORBIS(posix_pthread_mutex_lock)(m);
}

s32 MutexUnlock(PthreadMutexT* m) {
    return ORBIS(posix_pthread_mutex_unlock)(m);
}

s32 CondSignal(PthreadCondT* c) {
    return ORBIS(posix_pthread_cond_signal)(c);
}

s32 CondWait(PthreadCondT* c, PthreadMutexT* m) {
    return ORBIS(posix_pthread_cond_wait)(c, m);
}

// FUN_00003330
void DestroySyncObjects(VdecswInstance* dec) {
    for (auto& c : dec->cond) {
        ORBIS(posix_pthread_cond_destroy)(&c);
    }
    for (auto& m : dec->mutex) {
        ORBIS(posix_pthread_mutex_destroy)(&m);
    }
}

// FUN_00001630
s32 MapCoreError(VdecswInstance* dec, s32 core_err) {
    switch (static_cast<u32>(core_err)) {
    case 0x80c0000a:
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_SIZE;
    case 0x80c0000e:
        return ORBIS_VDECSW_ERROR_INPUT_QUEUE_FULL;
    case 0x80c0000f:
        dec->new_sequence = 1;
        return ORBIS_VDECSW_ERROR_NEW_SEQUENCE;
    case 0x80c00010:
        return ORBIS_VDECSW_ERROR_INVALID_SEQUENCE;
    case 0x80c00011:
        return ORBIS_VDECSW_ERROR_OVERSIZE_DECODE;
    case 0x80c00012:
        return ORBIS_VDECSW_ERROR_ACCESS_UNIT;
    case 0x80c00013:
        dec->fatal_stream = 1;
        return ORBIS_VDECSW_ERROR_FATAL_STREAM;
    case 0x80c00019:
        return ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_FULL;
    default:
        LOG_ERROR(Lib_Vdecsw, "core error {:#x}", static_cast<u32>(core_err));
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
}

bool IsValidInstance(VdecswInstance* dec) {
    OrbisVirtualQueryInfo info{};
    return Query(dec, &info) == 0 && dec->magic1 == kInstanceMagic1;
}

// FUN_000019c0
s32 SyncDecodeInput(VdecswInstance* dec, OrbisVdecswInputResult* result, bool blocking) {
    OrbisVirtualQueryInfo info{};
    void* decoded_au = nullptr;
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    if (Query(result, &info) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (result->this_size != sizeof(OrbisVdecswInputResult)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    PthreadMutexT* m = &dec->mutex[0];
    const s32 lock = MutexLock(m);
    if (!blocking) {
        if (lock != 0) {
            return ORBIS_VDECSW_ERROR_FATAL_STATE;
        }
        s32 ret;
        if (dec->input_count < dec->synced_input_count) {
            ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
        } else if (dec->input_count == dec->synced_input_count) {
            ret = ORBIS_VDECSW_ERROR_INPUT_QUEUE_EMPTY;
        } else {
            u32 count = 0;
            const s32 core_ret = VdecCoreTrySyncDecodeWptr(dec->core, &count, &decoded_au);
            if (core_ret == 0 || static_cast<u32>(core_ret) == 0x80c00012) {
                result->output_frame_count = count;
                dec->synced_input_count++;
                result->decoded_au = decoded_au;
                dec->pending_output_count += result->output_frame_count;
                CondSignal(&dec->cond[0]);
                ret = ORBIS_OK;
            } else if (static_cast<u32>(core_ret) == 0x80c00018) {
                ret = ORBIS_VDECSW_ERROR_DECODE_PENDING;
            } else if (static_cast<u32>(core_ret) == 0x80c0001d) {
                ret = ORBIS_VDECSW_ERROR_API_FAIL;
            } else {
                ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
            }
        }
        if (MutexUnlock(m) != 0) {
            return ORBIS_VDECSW_ERROR_FATAL_STATE;
        }
        return ret;
    }
    if (lock != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    s32 ret;
    bool done;
    if (dec->input_count < dec->synced_input_count) {
        ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
        done = true;
    } else if (dec->input_count == dec->synced_input_count) {
        ret = ORBIS_VDECSW_ERROR_INPUT_QUEUE_EMPTY;
        done = true;
    } else if (dec->sync_input_busy == 0) {
        ret = ORBIS_OK;
        dec->sync_input_busy = 1;
        done = false;
    } else {
        ret = ORBIS_VDECSW_ERROR_API_FAIL;
        done = true;
    }
    if (MutexUnlock(m) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    if (done) {
        return ret;
    }
    u32 count = 0;
    const s32 core_ret = VdecCoreSyncDecodeWptr(dec->core, &count, &decoded_au);
    if (static_cast<u32>(core_ret) != 0x80c00012 && core_ret != 0) {
        if (static_cast<u32>(core_ret) == 0x80c0001d) {
            return ORBIS_VDECSW_ERROR_API_FAIL;
        }
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    if (MutexLock(m) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    dec->synced_input_count++;
    dec->sync_input_busy = 0;
    result->output_frame_count = count;
    result->decoded_au = decoded_au;
    dec->pending_output_count += result->output_frame_count;
    CondSignal(&dec->cond[0]);
    if (MutexUnlock(m) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    return ORBIS_OK;
}

// FUN_00001e20
s32 SyncDecodeOutput(VdecswInstance* dec, OrbisVdecswOutputInfo* out, bool try_only) {
    OrbisVirtualQueryInfo info{};
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    if (Query(out, &info) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if ((out->this_size | 8) != sizeof(OrbisVdecswOutputInfo)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    PthreadMutexT* m = &dec->mutex[0];
    out->is_valid = false;
    out->is_error_frame = false;
    if (MutexLock(m) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    VdecCoreOutput core_out{};
    bool failed = true;
    s32 ret;
    if (!try_only) {
        if (dec->sync_output_blocking == 0 || dec->output_buffer_set == 1) {
            dec->output_waiting = 1;
            bool aborted = false;
            if (dec->pending_output_count == 0) {
                ret = ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_EMPTY;
                u64 waiting = 1;
                do {
                    if (waiting == 0) {
                        aborted = true;
                        break;
                    }
                    CondWait(&dec->cond[0], m);
                    waiting = dec->output_waiting;
                } while (dec->pending_output_count == 0);
                if (!aborted && waiting == 0) {
                    aborted = true;
                }
            }
            if (aborted) {
                ret = ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_EMPTY;
            } else {
                const s32 core_ret = VdecCoreSyncDecodeOutputSw(dec->core, &core_out);
                if (static_cast<u32>(core_ret) == 0x80c0000a) {
                    ret = ORBIS_VDECSW_ERROR_FRAME_BUFFER_SIZE;
                } else if (core_ret == 0) {
                    dec->pending_output_count--;
                    failed = false;
                    ret = ORBIS_OK;
                } else if (static_cast<u32>(core_ret) == 0x80c0001a) {
                    ret = ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_EMPTY;
                } else {
                    LOG_ERROR(Lib_Vdecsw, "core sync output error {:#x}",
                              static_cast<u32>(core_ret));
                    ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
                }
            }
        } else {
            ret = ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_EMPTY;
        }
    } else {
        const s32 core_ret = VdecCoreTrySyncDecodeOutputSw(dec->core, &core_out);
        switch (static_cast<u32>(core_ret)) {
        case 0:
            dec->pending_output_count--;
            failed = false;
            ret = ORBIS_OK;
            break;
        case 0x80c0000a:
            ret = ORBIS_VDECSW_ERROR_FRAME_BUFFER_SIZE;
            break;
        case 0x80c0001a:
            ret = ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_EMPTY;
            break;
        case 0x80c0001b:
            ret = ORBIS_VDECSW_ERROR_DECODE_PENDING;
            break;
        case 0x80c0001c:
            ret = ORBIS_VDECSW_ERROR_OUTPUT_PENDING;
            break;
        default:
            LOG_ERROR(Lib_Vdecsw, "core try sync output error {:#x}", static_cast<u32>(core_ret));
            ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
            break;
        }
    }
    if (MutexUnlock(m) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    if (failed) {
        return ret;
    }
    out->frame_buffer = core_out.frame_buffer;
    out->frame_buffer_size = core_out.frame_buffer_size;
    out->is_valid = true;
    out->frame_width = core_out.frame_width;
    out->frame_pitch = core_out.frame_pitch;
    out->frame_height = core_out.frame_height;
    out->is_error_frame = core_out.error_status != 0 || core_out.error_mb_count != 0;
    out->picture_count = core_out.picture_count;
    out->is_discarded_frame = core_out.is_discarded_frame != 0;
    if (out->this_size == sizeof(OrbisVdecswOutputInfo)) {
        out->frame_pitch_in_bytes = core_out.frame_pitch_in_bytes;
        u32 format = core_out.frame_format;
        if (format != 0) {
            if (format != 6) {
                return ORBIS_VDECSW_ERROR_FATAL_STATE;
            }
            format = 0xc24a;
        }
        out->frame_format = format;
    }
    if (core_out.is_last_frame != 0) {
        dec->synced_input_count = 0;
    }
    out->is_last_frame = core_out.is_last_frame != 0;
    if (core_out.codec == 4) {
        out->codec_type = OrbisVdecswCodecType::Hevc;
    } else if (core_out.codec == 0) {
        out->codec_type = OrbisVdecswCodecType::Avc;
    }
    return ORBIS_OK;
}

} // namespace

s32 PS4_SYSV_ABI sceVdecswQueryComputeMemoryInfo(OrbisVdecswComputeMemoryInfo* info) {
    OrbisVirtualQueryInfo vq{};
    if (Query(info, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (info->this_size != sizeof(OrbisVdecswComputeMemoryInfo)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    u64 size = 0;
    u64 base = 0;
    VdecCoreQueryComputeResourceInfo(&size, &base);
    info->cpu_gpu_memory = nullptr;
    info->cpu_gpu_memory_size = size;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVdecswAllocateComputeQueue(const OrbisVdecswComputeConfigInfo* cfg,
                                               const OrbisVdecswComputeMemoryInfo* mem,
                                               OrbisVdecswComputeQueue* queue) {
    OrbisVirtualQueryInfo vq{};
    if (Query(cfg, &vq) != 0 || Query(mem, &vq) != 0 || Query(queue, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (cfg->this_size != sizeof(OrbisVdecswComputeConfigInfo) ||
        mem->this_size != sizeof(OrbisVdecswComputeMemoryInfo)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    if (cfg->reserved0 != 0 || cfg->reserved1 != 0) {
        return ORBIS_VDECSW_ERROR_CONFIG_INFO;
    }
    if (4 < cfg->compute_pipe_id) {
        return ORBIS_VDECSW_ERROR_COMPUTE_PIPE_ID;
    }
    if (7 < cfg->compute_queue_id) {
        return ORBIS_VDECSW_ERROR_COMPUTE_QUEUE_ID;
    }
    u64 size = 0;
    u64 base = 0;
    VdecCoreQueryComputeResourceInfo(&size, &base);
    if (mem->cpu_gpu_memory_size < size) {
        return ORBIS_VDECSW_ERROR_MEMORY_SIZE;
    }
    const u8 check = cfg->check_memory_type;
    const s32 q = Query(mem->cpu_gpu_memory, &vq);
    if (check == 0) {
        if (q != 0) {
            return ORBIS_VDECSW_ERROR_MEMORY_POINTER;
        }
    } else {
        if (q != 0) {
            return ORBIS_VDECSW_ERROR_MEMORY_POINTER;
        }
        if (vq.memory_type != 0) {
            return ORBIS_VDECSW_ERROR_NOT_ONION_MEMORY;
        }
    }
    s32 sdk_ret = 0;
    const u32 sdk = CompiledSdkVersion(&sdk_ret);
    VdecCoreComputeParams params{};
    params.cpu_gpu_memory_size = mem->cpu_gpu_memory_size;
    params.cpu_gpu_memory = mem->cpu_gpu_memory;
    params.compute_pipe_id = cfg->compute_pipe_id;
    params.compute_queue_id = cfg->compute_queue_id;
    void* handle = nullptr;
    if (VdecCoreInitializeComputeResource(params, &handle, sdk) != 0) {
        return ORBIS_VDECSW_ERROR_CONFIG_INFO;
    }
    *queue = handle;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVdecswReleaseComputeQueue(OrbisVdecswComputeQueue queue) {
    OrbisVirtualQueryInfo vq{};
    if (Query(queue, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_COMPUTE_QUEUE;
    }
    if (VdecCoreFinalizeComputeResource(queue) != 0) {
        return ORBIS_VDECSW_ERROR_COMPUTE_QUEUE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVdecswQueryDecoderMemoryInfo(const OrbisVdecswDecoderConfigInfo* cfg,
                                                 OrbisVdecswDecoderMemoryInfo* mem) {
    OrbisVirtualQueryInfo vq{};
    if (Query(cfg, &vq) != 0 || Query(mem, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (mem->this_size != sizeof(OrbisVdecswDecoderMemoryInfo) ||
        (cfg->this_size != 0x48 && cfg->this_size != sizeof(OrbisVdecswDecoderConfigInfo))) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    s32 sdk_ret = 0;
    const u32 sdk = CompiledSdkVersion(&sdk_ret);
    VdecCoreConfig core{};
    const s32 ret =
        VdecswBuildCoreConfig(*cfg, *mem, core, sdk, static_cast<u32>(sceKernelGetCpumode()));
    mem->reserved0 = 0;
    mem->cpu_memory = nullptr;
    mem->gpu_memory = nullptr;
    mem->cpu_gpu_memory = nullptr;
    if (ret != 0) {
        LOG_ERROR(Lib_Vdecsw, "config error {:#x}", static_cast<u32>(ret));
    }
    return ret;
}

s32 PS4_SYSV_ABI sceVdecswCreateDecoder(const OrbisVdecswDecoderConfigInfo* cfg,
                                        const OrbisVdecswDecoderMemoryInfo* mem,
                                        OrbisVdecswDecoder* out) {
    OrbisVirtualQueryInfo vq{};
    if (Query(cfg, &vq) != 0 || Query(mem, &vq) != 0 || Query(out, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (mem->this_size != sizeof(OrbisVdecswDecoderMemoryInfo) ||
        (cfg->this_size != 0x48 && cfg->this_size != sizeof(OrbisVdecswDecoderConfigInfo))) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    s32 sdk_ret = 0;
    const u32 sdk = CompiledSdkVersion(&sdk_ret);
    OrbisVdecswDecoderMemoryInfo required{};
    VdecCoreConfig core_cfg{};
    s32 ret = VdecswBuildCoreConfig(*cfg, required, core_cfg, sdk,
                                    static_cast<u32>(sceKernelGetCpumode()));
    if (ret != 0) {
        LOG_ERROR(Lib_Vdecsw, "config error {:#x}", static_cast<u32>(ret));
        return ret;
    }
    const auto check_memory = [&](void* addr, s32 wanted_type, s32 type_error) -> s32 {
        const s32 q = Query(addr, &vq);
        if (cfg->check_memory_type == 0) {
            return q == 0 ? ORBIS_OK : ORBIS_VDECSW_ERROR_MEMORY_POINTER;
        }
        if (q != 0) {
            return ORBIS_VDECSW_ERROR_MEMORY_POINTER;
        }
        return vq.memory_type == wanted_type ? ORBIS_OK : type_error;
    };
    ret = check_memory(mem->cpu_memory, 0, ORBIS_VDECSW_ERROR_NOT_ONION_MEMORY);
    if (ret != 0) {
        return ret;
    }
    ret = check_memory(mem->gpu_memory, 3, ORBIS_VDECSW_ERROR_NOT_GARLIC_MEMORY);
    if (ret != 0) {
        return ret;
    }
    ret = check_memory(mem->cpu_gpu_memory, 0, ORBIS_VDECSW_ERROR_NOT_ONION_MEMORY);
    if (ret != 0) {
        return ret;
    }
    if (mem->reserved0 != 0) {
        return ORBIS_VDECSW_ERROR_MEMORY_INFO;
    }
    if (mem->cpu_memory_size < required.cpu_memory_size ||
        mem->cpu_gpu_memory_size < required.cpu_gpu_memory_size ||
        mem->gpu_memory_size < required.gpu_memory_size) {
        return ORBIS_VDECSW_ERROR_MEMORY_SIZE;
    }
    if (mem->max_frame_buffer_size < required.max_frame_buffer_size) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_SIZE;
    }
    if (required.frame_buffer_alignment != 0x100) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    if (static_cast<u8>(mem->frame_buffer_alignment) != 0) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_ALIGNMENT;
    }
    if (cfg->resource_type != 1) {
        return ORBIS_VDECSW_ERROR_RESOURCE_TYPE;
    }
    auto* dec = static_cast<VdecswInstance*>(mem->cpu_memory);
    if (Query(cfg->compute_queue, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_COMPUTE_QUEUE;
    }
    dec->unk68 = 1;
    dec->magic0 = kInstanceMagic0;
    u32 created_mutexes = 0;
    u32 created_conds = 0;
    const auto fail = [&]() {
        for (u32 i = 0; i < created_conds; i++) {
            ORBIS(posix_pthread_cond_destroy)(&dec->cond[i]);
        }
        for (u32 i = 0; i < created_mutexes; i++) {
            ORBIS(posix_pthread_mutex_destroy)(&dec->mutex[i]);
        }
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    };
    char name[0x20];
    for (u32 i = 0; i < 12; i++) {
        PthreadMutexAttrT attr{};
        ORBIS(posix_pthread_mutexattr_init)(&attr);
        std::snprintf(name, sizeof(name), "SceVdecswMutex%02d", i);
        ORBIS(posix_pthread_mutexattr_settype)(&attr, PthreadMutexType::Recursive);
        const s32 r = ORBIS(scePthreadMutexInit)(&dec->mutex[i], &attr, name);
        ORBIS(posix_pthread_mutexattr_destroy)(&attr);
        if (r < 0) {
            return fail();
        }
        created_mutexes++;
    }
    for (u32 i = 0; i < 9; i++) {
        PthreadCondAttrT attr{};
        ORBIS(posix_pthread_condattr_init)(&attr);
        std::snprintf(name, sizeof(name), "SceVdecswCond%02d", i);
        const s32 r = ORBIS(scePthreadCondInit)(&dec->cond[i], &attr, name);
        ORBIS(posix_pthread_condattr_destroy)(&attr);
        if (r < 0) {
            return fail();
        }
        created_conds++;
    }
    VdecCoreMemoryParams core_mem{};
    core_mem.cpu_memory_size = mem->cpu_memory_size - kCoreMemoryOffset;
    core_mem.cpu_memory = static_cast<u8*>(mem->cpu_memory) + kCoreMemoryOffset;
    core_mem.cpu_gpu_memory_size = mem->cpu_gpu_memory_size;
    core_mem.cpu_gpu_memory = mem->cpu_gpu_memory;
    core_mem.gpu_memory_size = mem->gpu_memory_size;
    core_mem.gpu_memory = mem->gpu_memory;
    ret = VdecCoreCreateDecoder(core_cfg, core_mem, cfg->compute_queue, &dec->core, sdk);
    if (ret != 0) {
        LOG_ERROR(Lib_Vdecsw, "core create error {:#x}", static_cast<u32>(ret));
        return fail();
    }
    dec->input_count = 0;
    s32 ver = 0;
    const s32 ver_ret = sceKernelGetCompiledSdkVersion(&ver);
    dec->sync_output_blocking = (0xfffffffu < static_cast<u32>(ver) && ver_ret == 0) ? 1 : 0;
    std::memset(&dec->output_buffer_set, 0, 0x18);
    dec->sync_input_busy = 0;
    dec->new_sequence = 0;
    dec->fatal_stream = 0;
    dec->finalizing = 0;
    dec->pipeline_depth = core_cfg.decode_pipeline_depth;
    dec->magic1 = kInstanceMagic1;
    dec->check_memory_type = cfg->check_memory_type;
    *out = dec;
    if (VdecCoreResetDecoder(dec->core) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVdecswDeleteDecoder(OrbisVdecswDecoder dec) {
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    dec->output_buffer_set = 0;
    dec->output_waiting = 0;
    CondSignal(&dec->cond[0]);
    const s32 core_ret = VdecCoreDeleteDecoder(dec->core);
    DestroySyncObjects(dec);
    dec->magic1 = 0;
    return core_ret == 0 ? ORBIS_OK : ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
}

s32 PS4_SYSV_ABI sceVdecswResetDecoder(OrbisVdecswDecoder dec) {
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    if (MutexLock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    const s32 core_ret = VdecCoreResetDecoder(dec->core);
    dec->input_count = 0;
    std::memset(&dec->output_buffer_set, 0, 0x18);
    dec->sync_input_busy = 0;
    dec->new_sequence = 0;
    dec->fatal_stream = 0;
    dec->finalizing = 0;
    CondSignal(&dec->cond[0]);
    const s32 unlock = MutexUnlock(&dec->mutex[0]);
    return (unlock == 0 && core_ret == 0) ? ORBIS_OK : ORBIS_VDECSW_ERROR_FATAL_STATE;
}

s32 PS4_SYSV_ABI sceVdecswSetDecodeInput(OrbisVdecswDecoder dec,
                                         const OrbisVdecswInputData* input) {
    OrbisVirtualQueryInfo vq{};
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    if (Query(input, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (input->this_size != sizeof(OrbisVdecswInputData)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    if (Query(input->au_data, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ACCESS_UNIT_POINTER;
    }
    const u64 au_size = input->au_size;
    if (au_size == 0) {
        return ORBIS_VDECSW_ERROR_ACCESS_UNIT_SIZE;
    }
    VdecCoreInput core_input{};
    core_input.au_data = input->au_data;
    core_input.au_size = au_size;
    core_input.pts_data = input->pts_data;
    core_input.dts_data = input->dts_data;
    core_input.attached_data = input->attached_data;
    if (MutexLock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    dec->last_au_size = au_size;
    dec->last_au_data = input->au_data;
    s32 ret;
    if (dec->finalizing != 0) {
        ret = ORBIS_VDECSW_ERROR_API_FAIL;
    } else if (dec->new_sequence != 0) {
        ret = ORBIS_VDECSW_ERROR_NEW_SEQUENCE;
    } else if (dec->fatal_stream != 0) {
        ret = ORBIS_VDECSW_ERROR_FATAL_STREAM;
    } else {
        const s32 core_ret = VdecCoreSetDecodeInput(dec->core, core_input);
        if (core_ret == 0) {
            dec->input_count++;
            CondSignal(&dec->cond[0]);
            ret = ORBIS_OK;
        } else {
            ret = MapCoreError(dec, core_ret);
        }
    }
    if (MutexUnlock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    return ret;
}

s32 PS4_SYSV_ABI sceVdecswSyncDecodeInput(OrbisVdecswDecoder dec, OrbisVdecswInputResult* result) {
    return SyncDecodeInput(dec, result, true);
}

s32 PS4_SYSV_ABI sceVdecswTrySyncDecodeInput(OrbisVdecswDecoder dec,
                                             OrbisVdecswInputResult* result) {
    return SyncDecodeInput(dec, result, false);
}

s32 PS4_SYSV_ABI sceVdecswSetDecodeOutput(OrbisVdecswDecoder dec, OrbisVdecswFrameBuffer* fb) {
    OrbisVirtualQueryInfo vq{};
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    if (Query(fb, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if (fb->this_size != sizeof(OrbisVdecswFrameBuffer)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    void* buffer = fb->frame_buffer;
    if (Query(buffer, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_POINTER;
    }
    if ((reinterpret_cast<u64>(buffer) & 0xff) != 0) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_ALIGNMENT;
    }
    if (MutexLock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    dec->last_frame_buffer_size = fb->frame_buffer_size;
    dec->last_frame_buffer = fb->frame_buffer;
    const s32 core_ret = VdecCoreSetDecodeOutputSw(dec->core, buffer, fb->frame_buffer_size);
    s32 ret;
    if (static_cast<u32>(core_ret) == 0x80c0000a) {
        ret = ORBIS_VDECSW_ERROR_FRAME_BUFFER_SIZE;
    } else if (core_ret == 0) {
        dec->output_buffer_set = 1;
        ret = ORBIS_OK;
    } else if (static_cast<u32>(core_ret) == 0x80c00019) {
        ret = ORBIS_VDECSW_ERROR_OUTPUT_BUFFER_FULL;
    } else {
        ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    if (MutexUnlock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    return ret;
}

s32 PS4_SYSV_ABI sceVdecswSyncDecodeOutput(OrbisVdecswDecoder dec, OrbisVdecswOutputInfo* out) {
    return SyncDecodeOutput(dec, out, false);
}

s32 PS4_SYSV_ABI sceVdecswTrySyncDecodeOutput(OrbisVdecswDecoder dec, OrbisVdecswOutputInfo* out) {
    return SyncDecodeOutput(dec, out, true);
}

s32 PS4_SYSV_ABI sceVdecswFinalizeDecodeSequence(OrbisVdecswDecoder dec) {
    if (!IsValidInstance(dec)) {
        return ORBIS_VDECSW_ERROR_DECODER_INSTANCE;
    }
    if (MutexLock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    s32 ret = ORBIS_OK;
    bool flush = true;
    if (dec->finalizing == 0) {
        if (dec->input_count == 0 && dec->pending_output_count == 0) {
            flush = false;
        } else {
            dec->finalizing = 1;
        }
    }
    if (flush) {
        if (dec->input_count != 0) {
            s32 count = 0;
            const s32 core_ret = VdecCoreFlushDecodeOutput(dec->core, &count);
            if (core_ret == 0) {
                dec->input_count = 0;
                dec->pending_output_count += static_cast<u32>(count);
                if (dec->pending_output_count == 0) {
                    dec->output_waiting = 0;
                }
                CondSignal(&dec->cond[0]);
            } else {
                ret = ORBIS_VDECSW_ERROR_FATAL_STATE;
            }
        }
        if (ret == ORBIS_OK && dec->input_count == 0 && dec->pending_output_count == 0) {
            dec->finalizing = 0;
        }
    }
    if (MutexUnlock(&dec->mutex[0]) != 0) {
        return ORBIS_VDECSW_ERROR_FATAL_STATE;
    }
    return ret;
}

// FUN_00002420
s32 PS4_SYSV_ABI sceVdecswGetPictureInfo(const OrbisVdecswOutputInfo* info, void* pic0,
                                         void* pic1) {
    OrbisVirtualQueryInfo vq{};
    if (Query(pic0, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    std::memset(static_cast<u8*>(pic0) + 8, 0, 4);
    void* pics[2] = {pic0, nullptr};
    if (Query(pic1, &vq) == 0) {
        std::memset(static_cast<u8*>(pic1) + 8, 0, 4);
        pics[1] = pic1;
    }
    if (Query(info, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_ARGUMENT_POINTER;
    }
    if ((info->this_size | 8) != sizeof(OrbisVdecswOutputInfo)) {
        return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
    }
    auto* fb = static_cast<u8*>(info->frame_buffer);
    if ((reinterpret_cast<u64>(fb) & 0xff) != 0) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_ALIGNMENT;
    }
    if (Query(fb, &vq) != 0) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_POINTER;
    }
    const u8 count = info->picture_count;
    if (static_cast<u8>(count - 1) >= 2) {
        return ORBIS_VDECSW_ERROR_OUTPUT_INFO;
    }
    u64 fb_size = info->frame_buffer_size;
    u64 trailer = static_cast<u64>(count) << 10;
    if (!(trailer < fb_size)) {
        return ORBIS_VDECSW_ERROR_FRAME_BUFFER_SIZE;
    }
    if (count == 2) {
        s32 ver = 0;
        const s32 r = sceKernelGetCompiledSdkVersion(&ver);
        if (r == 0 && 0xfffffffu < static_cast<u32>(ver)) {
            fb_size += 0x400;
        }
    }
    for (u32 i = 0; i < count; i++) {
        auto* p = static_cast<u8*>(pics[i]);
        if (p == nullptr) {
            break;
        }
        u64 size = 0;
        std::memcpy(&size, p, 8);
        const u64 d = size - 0x68;
        const u64 idx = (d << 0x3c) | (d >> 4);
        if (5 < idx || idx == 3) {
            return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
        }
        const u8* t = fb + (fb_size - (trailer & 0xffffffff));
        const auto t32 = [&](u32 off) {
            u32 v;
            std::memcpy(&v, t + off, 4);
            return v;
        };
        const auto put = [&](u32 off, const void* src, u32 n) { std::memcpy(p + off, src, n); };
        const auto put8 = [&](u32 off, u32 v) { p[off] = static_cast<u8>(v); };
        const auto put32 = [&](u32 off, u32 v) { std::memcpy(p + off, &v, 4); };
        if (t32(0x20) != 0) {
            const u32 type = t32(0);
            if (type == 0 && info->codec_type == OrbisVdecswCodecType::Avc) {
                if ((size | 0x10) != 0x78) {
                    return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
                }
                const u32 flags = t32(0x4c);
                p[8] = 1;
                put(0x10, t + 0x10, 0x10);
                put(0x20, t + 0x08, 8);
                put32(0x2c, t32(0x28));
                put32(0x30, t32(0x2c));
                put32(0x58, t32(0x40));
                put32(0x5c, t32(0x44));
                put8(0x34, flags & 1);
                put8(0x35, (flags >> 1) & 1);
                put8(0x55, (flags >> 2) & 1);
                put8(0x63, (flags >> 3) & 1);
                put8(0x48, (flags >> 4) & 1);
                put8(0x50, (flags >> 5) & 1);
                put8(0x51, (flags >> 6) & 1);
                put8(0x61, (flags >> 7) & 1);
                put8(0x4e, (flags >> 15) & 1);
                put8(0x65, (flags >> 18) & 1);
                put8(0x66, (flags >> 19) & 1);
                put8(0x60, (flags >> 23) & 1);
                put8(0x64, (flags >> 8) & 0xf);
                put8(0x4f, (flags >> 12) & 7);
                put(0x4a, t + 0x52, 2);
                put(0x4c, t + 0x54, 2);
                put8(0x29, t[0x56]);
                put8(0x2a, t[0x57]);
                put8(0x62, t[0x58]);
                put8(0x49, t[0x59]);
                put8(0x52, t[0x5a]);
                put8(0x53, t[0x5b]);
                put8(0x54, t[0x5c]);
                if ((flags & 2) == 0) {
                    std::memset(p + 0x38, 0, 8);
                    put32(0x40, 0);
                    put32(0x44, 0);
                } else {
                    put32(0x38, t32(0x30) * 2);
                    put32(0x3c, t32(0x34) * 2);
                    if ((flags & 1) == 0) {
                        put32(0x40, t32(0x38) << 2);
                        put32(0x44, t32(0x3c) << 2);
                    } else {
                        put32(0x40, t32(0x38) * 2);
                        put32(0x44, t32(0x3c) * 2);
                    }
                }
                if (size == 0x78) {
                    const u32 flags2 = t32(0x60);
                    for (u32 b = 0; b < 8; b++) {
                        put8(0x67 + b, (flags2 >> b) & 1);
                    }
                    put8(0x6f, (flags >> 24) & 1);
                    for (u32 b = 0; b < 5; b++) {
                        put8(0x70 + b, (flags >> (25 + b)) & 1);
                    }
                }
            } else if (type == 1 && info->codec_type == OrbisVdecswCodecType::Hevc) {
                if (0x30 < size - 0x88 ||
                    ((0x1000100000001ULL >> ((size - 0x88) & 0x3f)) & 1) == 0) {
                    return ORBIS_VDECSW_ERROR_STRUCT_SIZE;
                }
                const u32 f = t32(0x58);
                const u32 f2 = t32(0x5c);
                const u32 f3 = t32(0x98);
                p[8] = 1;
                put(0x10, t + 0x10, 0x10);
                put(0x20, t + 0x08, 8);
                put32(0x28, t32(0x28));
                put32(0x2c, t32(0x2c));
                put8(0x30, t[0x64]);
                put8(0x31, t[0x65]);
                put8(0x4d, (f >> 0x15) & 1);
                put32(0x54, (f >> 10) & 3);
                put32(0x58, (f >> 0xc) & 1);
                put32(0x34, t32(0x50));
                put32(0x38, t32(0x54));
                put8(0x32, (f >> 3) & 1);
                put32(0x3c, (f >> 4) & 1);
                put8(0x48, (f >> 0x11) & 1);
                put8(0x49, (f >> 0x12) & 1);
                put8(0x46, t[0x5a] & 1);
                put32(0x50, (f >> 6) & 0xf);
                put8(0x47, (f >> 0xd) & 7);
                put(0x42, t + 0x60, 2);
                put(0x44, t + 0x62, 2);
                put8(0x40, t[0x66]);
                put8(0x4a, t[0x67]);
                put8(0x4b, t[0x68]);
                put8(0x4c, t[0x69]);
                put32(0x5c, (f >> 1) & 1);
                put(0x60, t + 0x30, 0x10);
                put32(0x70, (f >> 2) & 1);
                put(0x74, t + 0x40, 0x10);
                const auto extended = [&]() {
                    put8(0x84, (f >> 0x13) & 1);
                    put8(0x85, t[0x6a]);
                    put8(0x86, t[0x6b]);
                    put8(0x87, (f >> 0x14) & 1);
                    for (u32 b = 0; b < 8; b++) {
                        put8(0x88 + b, (f2 >> b) & 1);
                    }
                    put8(0x90, t[0x5d] & 1);
                    put8(0x91, (f2 >> 9) & 1);
                    put8(0x92, (f2 >> 10) & 1);
                    put8(0x93, f3 & 1);
                    put8(0x94, (f3 >> 1) & 1);
                    put8(0x95, t[0x9e]);
                    put8(0x96, (f3 >> 7) & 1);
                    put8(0x97, t[0x9f]);
                    put8(0x98, (f >> 0x19) & 1);
                    put8(0x99, (f >> 0x1a) & 1);
                    put8(0x9a, f & 1);
                    put8(0x9b, t[0xa0]);
                    put8(0x9c, (f3 >> 5) & 1);
                    put8(0x9d, (f3 >> 6) & 1);
                    put8(0x9e, t[0xa1]);
                    put8(0x9f, t[0x99] & 1);
                    put8(0xa0, t[0xa2]);
                    put8(0xa1, t[0xa3]);
                    put8(0xa2, (f3 >> 9) & 1);
                    put8(0xa3, t[0xa4]);
                    put8(0xa4, t[0xa5]);
                };
                if (size == 0xa8) {
                    extended();
                }
                if (size == 0xb8) {
                    s32 ver = 0x6500000;
                    if (sceKernelGetCompiledSdkVersion(&ver) == 0 &&
                        0x74fffffu < static_cast<u32>(ver)) {
                        extended();
                    }
                    u32 conf_flag = 0;
                    std::memcpy(&conf_flag, p + 0x5c, 4);
                    u32 disp_flag = 0;
                    std::memcpy(&disp_flag, p + 0x70, 4);
                    u32 sum[4]{};
                    if (conf_flag == 0) {
                        std::memset(p + 0xa8, 0, 0x10);
                    } else {
                        std::memcpy(sum, p + 0x60, 0x10);
                    }
                    if (conf_flag == 0 && disp_flag == 0) {
                        p[0xa5] = 0;
                    } else {
                        if (disp_flag != 0) {
                            u32 disp[4];
                            std::memcpy(disp, p + 0x74, 0x10);
                            for (u32 k = 0; k < 4; k++) {
                                sum[k] += disp[k];
                            }
                        }
                        for (u32 k = 0; k < 4; k++) {
                            sum[k] += sum[k];
                        }
                        p[0xa5] = 1;
                        std::memcpy(p + 0xa8, sum, 0x10);
                    }
                }
            }
        }
        trailer -= 0x400;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVdecswGetAvcPictureInfo(const OrbisVdecswOutputInfo* info,
                                            OrbisVdecswAvcPictureInfo* pic0,
                                            OrbisVdecswAvcPictureInfo* pic1) {
    return sceVdecswGetPictureInfo(info, pic0, pic1);
}

s32 PS4_SYSV_ABI sceVdecswGetHevcPictureInfo(const OrbisVdecswOutputInfo* info,
                                             OrbisVdecswHevcPictureInfo* pic) {
    return sceVdecswGetPictureInfo(info, pic, nullptr);
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("hIgrg5h4V6s", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswAllocateComputeQueue);
    LIB_FUNCTION("+L5ArV1tPGA", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswCreateDecoder);
    LIB_FUNCTION("ecUtPX+dBYk", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswDeleteDecoder);
    LIB_FUNCTION("5Y6nZqIZvBg", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswFinalizeDecodeSequence);
    LIB_FUNCTION("ihNT-uuEAr4", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswGetAvcPictureInfo);
    LIB_FUNCTION("PzF+L5zXoyg", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswGetHevcPictureInfo);
    LIB_FUNCTION("FzECy3Wxxas", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswGetPictureInfo);
    LIB_FUNCTION("0moTubWCsTM", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswQueryComputeMemoryInfo);
    LIB_FUNCTION("A+2M7EivuOU", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswQueryDecoderMemoryInfo);
    LIB_FUNCTION("fX-zOOefbbs", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswReleaseComputeQueue);
    LIB_FUNCTION("veb-YBrOqo0", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswResetDecoder);
    LIB_FUNCTION("aqMiF0AgUYI", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswSetDecodeInput);
    LIB_FUNCTION("rgtMCOpyBSc", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswSetDecodeOutput);
    LIB_FUNCTION("AAMM-Q1X0g0", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswSyncDecodeInput);
    LIB_FUNCTION("tWiSgXov8GM", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswSyncDecodeOutput);
    LIB_FUNCTION("l4sQYy5wPkc", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswTrySyncDecodeInput);
    LIB_FUNCTION("kMBw37oH8nI", "libSceVdecsw", 1, "libSceVdecsw", sceVdecswTrySyncDecodeOutput);
}

} // namespace Libraries::Vdecsw
