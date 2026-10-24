// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/elf_info.h"
#include "common/error.h"
#include "common/logging/log.h"
#include "common/scope_exit.h"
#include "common/singleton.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/libs.h"
#include "core/linker.h"
#include "core/memory.h"

namespace Libraries::Kernel {

static s32 g_sdk_version = -1;
static bool g_alias_dmem = false;

u64 PS4_SYSV_ABI sceKernelGetDirectMemorySize() {
    LOG_TRACE(Kernel_Vmm, "called");
    const auto* memory = Core::Memory::Instance();
    return memory->GetTotalDirectSize();
}

s32 PS4_SYSV_ABI sceKernelEnableDmemAliasing() {
    LOG_DEBUG(Kernel_Vmm, "called");
    g_alias_dmem = true;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelAllocateDirectMemory(s64 searchStart, s64 searchEnd, u64 len,
                                               u64 alignment, s32 memoryType, s64* physAddrOut) {
    if (searchStart < 0 || searchEnd < 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid parameters!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len <= 0 || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Length {:#x} is invalid!", len);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware: a non-zero alignment must be a power of two of at least 16 KiB.
    if (alignment != 0 && (!std::has_single_bit(alignment) || !Common::Is16KBAligned(alignment))) {
        LOG_ERROR(Kernel_Vmm, "Alignment {:#x} is invalid!", alignment);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (memoryType < 0 || memoryType > 10) {
        LOG_ERROR(Kernel_Vmm, "Memory type {:#x} is invalid!", memoryType);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (physAddrOut == nullptr) {
        LOG_ERROR(Kernel_Vmm, "Result physical address pointer is null!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    const bool is_in_range = searchEnd - searchStart >= len;
    if (searchEnd <= searchStart || searchEnd < len || !is_in_range) {
        LOG_ERROR(Kernel_Vmm,
                  "Provided address range is too small!"
                  " searchStart = {:#x}, searchEnd = {:#x}, length = {:#x}",
                  searchStart, searchEnd, len);
        return ORBIS_KERNEL_ERROR_EAGAIN;
    }

    auto* memory = Core::Memory::Instance();
    PAddr phys_addr = memory->Allocate(searchStart, searchEnd, len, alignment, memoryType);
    if (phys_addr == -1) {
        return ORBIS_KERNEL_ERROR_EAGAIN;
    }

    *physAddrOut = static_cast<s64>(phys_addr);

    LOG_INFO(Kernel_Vmm,
             "searchStart = {:#x}, searchEnd = {:#x}, len = {:#x}, "
             "alignment = {:#x}, memoryType = {:#x}, physAddrOut = {:#x}",
             searchStart, searchEnd, len, alignment, memoryType, phys_addr);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelAllocateMainDirectMemory(u64 len, u64 alignment, s32 memoryType,
                                                   s64* physAddrOut) {
    const auto searchEnd = static_cast<s64>(sceKernelGetDirectMemorySize());
    return sceKernelAllocateDirectMemory(0, searchEnd, len, alignment, memoryType, physAddrOut);
}

s32 PS4_SYSV_ABI sceKernelCheckedReleaseDirectMemory(u64 start, u64 len) {
    LOG_INFO(Kernel_Vmm, "called start = {:#x}, len = {:#x}", start, len);
    if (!Common::Is16KBAligned(start) || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Misaligned start or length, start = {:#x}, length = {:#x}", start,
                  len);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (static_cast<s64>(start) < 0 || static_cast<s64>(len) < 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid range start = {:#x}, len = {:#x}", start, len);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len == 0) {
        return ORBIS_OK;
    }
    auto* memory = Core::Memory::Instance();
    return memory->Free(start, len, true);
}

s32 PS4_SYSV_ABI sceKernelReleaseDirectMemory(u64 start, u64 len) {
    LOG_INFO(Kernel_Vmm, "called start = {:#x}, len = {:#x}", start, len);
    if (!Common::Is16KBAligned(start) || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Misaligned start or length, start = {:#x}, length = {:#x}", start,
                  len);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (static_cast<s64>(start) < 0 || static_cast<s64>(len) < 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid range start = {:#x}, len = {:#x}", start, len);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len == 0) {
        return ORBIS_OK;
    }
    auto* memory = Core::Memory::Instance();
    memory->Free(start, len, false);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelAvailableDirectMemorySize(u64 searchStart, u64 searchEnd, u64 alignment,
                                                    u64* physAddrOut, u64* sizeOut) {
    LOG_INFO(Kernel_Vmm, "called searchStart = {:#x}, searchEnd = {:#x}, alignment = {:#x}",
             searchStart, searchEnd, alignment);

    if (alignment != 0 && !std::has_single_bit(alignment)) {
        LOG_ERROR(Kernel_Vmm, "Alignment {:#x} is invalid!", alignment);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto* memory = Core::Memory::Instance();

    PAddr physAddr{};
    u64 size{};
    s32 result = memory->DirectQueryAvailable(searchStart, searchEnd, alignment, &physAddr, &size);

    if (size == 0) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    if (physAddrOut != nullptr) {
        *physAddrOut = static_cast<u64>(physAddr);
    }
    if (sizeOut != nullptr) {
        *sizeOut = size;
    }

    return result;
}

s32 PS4_SYSV_ABI sceKernelVirtualQuery(const void* addr, s32 flags, OrbisVirtualQueryInfo* info,
                                       u64 infoSize) {
    LOG_INFO(Kernel_Vmm, "called addr = {}, flags = {:#x}", fmt::ptr(addr), flags);
    auto* memory = Core::Memory::Instance();
    // Measured on hardware (DmemTest VQ9-VQ14): addresses from 0x7F0000000 up to 0xFF0000000 are
    // refused with EACCES, mapped or not, with either flag (0x7EFFFC000 and 0xFF0000000 are
    // answered). On the PS4 every module but the executable (system and game-bundled), the main
    // thread's TLS and the GnmDriver area at 0xFE0000000 are in this range; what is placed here
    // (system modules, the GnmDriver area) has its PS4 counterpart in it as well.
    // A search for the next mapping (flag 1) from below the range is refused too when the mapping
    // it finds starts in the range (DmemTest VQ15: a system module, and a page of the process's
    // own at 0x7F0000000).
    constexpr VAddr RefusedStart = 0x7F0000000ULL;
    constexpr VAddr RefusedEnd = 0xFF0000000ULL;
    const VAddr query_addr = std::bit_cast<VAddr>(addr);
    if (query_addr >= RefusedStart && query_addr < RefusedEnd) {
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    const s32 result = memory->VirtualQuery(query_addr, flags, info);
    if (result == ORBIS_OK && flags == 1 && info->start >= RefusedStart &&
        info->start < RefusedEnd) {
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    return result;
}

static s32 ReserveVirtualRangeImpl(void** addr, u64 len, s32 flags, u64 alignment,
                                   std::string_view name) {
    LOG_INFO(Kernel_Vmm, "addr = {}, len = {:#x}, flags = {:#x}, alignment = {:#x}",
             fmt::ptr(*addr), len, flags, alignment);
    if (addr == nullptr) {
        LOG_ERROR(Kernel_Vmm, "Address is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len == 0 || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Map size is either zero or not 16KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (alignment != 0) {
        if (!std::has_single_bit(alignment) || !Common::Is16KBAligned(alignment) ||
            alignment >= 0x8000'0000ULL) {
            LOG_ERROR(Kernel_Vmm, "Alignment value is invalid!");
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
    }

    auto* memory = Core::Memory::Instance();
    const VAddr in_addr = reinterpret_cast<VAddr>(*addr);
    // Measured on hardware for each of the 32 flag bits: only fixed, no-overwrite and no-coalesce
    // are accepted.
    constexpr auto ValidReserveFlags = Core::MemoryMapFlags::Fixed |
                                       Core::MemoryMapFlags::NoOverwrite |
                                       Core::MemoryMapFlags::NoCoalesce;
    if (True(static_cast<Core::MemoryMapFlags>(flags) & ~ValidReserveFlags)) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware (DmemTest M4): a fixed reservation over a direct mapping is EINVAL, with
    // or without no-overwrite.
    if ((flags & static_cast<s32>(Core::MemoryMapFlags::Fixed)) != 0 &&
        memory->ContainsVmaType(in_addr, len, Core::VMAType::Direct)) {
        LOG_ERROR(Kernel_Vmm, "Fixed reservation over a direct mapping");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    auto map_flags = static_cast<Core::MemoryMapFlags>(flags);

    if (True(map_flags & Core::MemoryMapFlags::Fixed) && in_addr == 0) {
        if (Common::ElfInfo::FW_170 <= g_sdk_version) {
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
        map_flags &= ~Core::MemoryMapFlags::Fixed;
    }

    s32 result = memory->MapMemory(addr, in_addr, len, Core::MemoryProt::NoAccess, map_flags,
                                   Core::VMAType::Reserved, name, false, -1, alignment);
    if (result == 0) {
        LOG_INFO(Kernel_Vmm, "out_addr = {}", fmt::ptr(*addr));
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelReserveVirtualRange(void** addr, u64 len, s32 flags, u64 alignment) {
    return ReserveVirtualRangeImpl(addr, len, flags, alignment, "anon");
}

s32 PS4_SYSV_ABI sceKernelMapNamedDirectMemory(void** addr, u64 len, s32 prot, s32 flags,
                                               s64 phys_addr, u64 alignment, const char* name) {
    // Measured on hardware: a null name is EFAULT.
    if (name == nullptr) {
        LOG_ERROR(Kernel_Vmm, "name is invalid!");
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    LOG_INFO(Kernel_Vmm,
             "in_addr = {}, len = {:#x}, prot = {:#x}, flags = {:#x}, "
             "phys_addr = {:#x}, alignment = {:#x}, name = '{}'",
             fmt::ptr(*addr), len, prot, flags, phys_addr, alignment, name);

    if (len == 0 || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Map size is either zero or not 16KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (!Common::Is16KBAligned(phys_addr)) {
        LOG_ERROR(Kernel_Vmm, "Start address is not 16KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    // libkernel requires a power of two that is a multiple of 16 KiB; alignments of 2 GiB and above
    // were measured to fail.
    if (alignment != 0 && (!std::has_single_bit(alignment) || !Common::Is16KBAligned(alignment) ||
                           alignment >= 2_GB)) {
        LOG_ERROR(Kernel_Vmm, "Alignment value is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (std::strlen(name) >= ORBIS_KERNEL_MAXIMUM_NAME_LENGTH) {
        LOG_ERROR(Kernel_Vmm, "name exceeds 32 bytes!");
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    const auto mem_prot = static_cast<Core::MemoryProt>(prot);
    if (True(mem_prot & Core::MemoryProt::CpuExec)) {
        LOG_ERROR(Kernel_Vmm, "Executable permissions are not allowed.");
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    // Measured on hardware for each of the 32 bits: other protection bits are refused, and the
    // flags are limited to fixed, no-overwrite, DmemCompat (0x400) and no-coalesce.
    constexpr auto ValidProt = Core::MemoryProt::CpuRead | Core::MemoryProt::CpuWrite |
                               Core::MemoryProt::CpuExec | Core::MemoryProt::GpuRead |
                               Core::MemoryProt::GpuWrite;
    if (True(mem_prot & ~ValidProt)) {
        LOG_ERROR(Kernel_Vmm, "Invalid protection {:#x}", prot);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware (DmemTest DT6): direct memory of type 10 refuses CPU or GPU write.
    {
        s32 phys_type = -1;
        void* type_start = nullptr;
        void* type_end = nullptr;
        if (True(mem_prot & (Core::MemoryProt::CpuWrite | Core::MemoryProt::GpuWrite)) &&
            Core::Memory::Instance()->GetDirectMemoryType(phys_addr, &phys_type, &type_start,
                                                          &type_end) == ORBIS_OK &&
            phys_type == 10) {
            LOG_ERROR(Kernel_Vmm, "Memory type 10 is not allowed with write access");
            return ORBIS_KERNEL_ERROR_EACCES;
        }
    }
    constexpr auto ValidDirectFlags =
        Core::MemoryMapFlags::Fixed | Core::MemoryMapFlags::NoOverwrite |
        Core::MemoryMapFlags::Stack | Core::MemoryMapFlags::NoCoalesce;
    if (True(static_cast<Core::MemoryMapFlags>(flags) & ~ValidDirectFlags)) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto map_flags = static_cast<Core::MemoryMapFlags>(flags);
    const VAddr in_addr = reinterpret_cast<VAddr>(*addr);

    auto* memory = Core::Memory::Instance();
    bool should_check = false;
    if (g_sdk_version >= Common::ElfInfo::FW_250 &&
        False(map_flags & Core::MemoryMapFlags::Stack)) {
        // Under these conditions, this would normally redirect to sceKernelMapDirectMemory2.
        should_check = !g_alias_dmem;
    }

    if (True(map_flags & Core::MemoryMapFlags::Fixed) && in_addr == 0) {
        if (Common::ElfInfo::FW_170 <= g_sdk_version) {
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
        map_flags &= ~Core::MemoryMapFlags::Fixed;
    }

    const auto ret =
        memory->MapMemory(addr, in_addr, len, mem_prot, map_flags, Core::VMAType::Direct, name,
                          should_check, phys_addr, alignment);

    LOG_INFO(Kernel_Vmm, "out_addr = {}", fmt::ptr(*addr));
    return ret;
}

s32 PS4_SYSV_ABI sceKernelMapDirectMemory(void** addr, u64 len, s32 prot, s32 flags, s64 phys_addr,
                                          u64 alignment) {
    LOG_TRACE(Kernel_Vmm, "called, redirected to sceKernelMapNamedDirectMemory");
    return sceKernelMapNamedDirectMemory(addr, len, prot, flags, phys_addr, alignment, "anon");
}

s32 PS4_SYSV_ABI sceKernelMapDirectMemory2(void** addr, u64 len, s32 type, s32 prot, s32 flags,
                                           s64 phys_addr, u64 alignment) {
    LOG_INFO(Kernel_Vmm,
             "in_addr = {}, len = {:#x}, prot = {:#x}, flags = {:#x}, "
             "phys_addr = {:#x}, alignment = {:#x}",
             fmt::ptr(*addr), len, prot, flags, phys_addr, alignment);

    if (len == 0 || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Map size is either zero or not 16KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (!Common::Is16KBAligned(phys_addr)) {
        LOG_ERROR(Kernel_Vmm, "Start address is not 16KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    // Measured on hardware (DmemTest SF6, BT8, SF7, DT6): the alignment, address and flags are
    // checked as for MapDirectMemory, which uses this call from SDK 2.50 on; type -1 keeps the
    // memory type, other types must be 0 to 10, and type 10 refuses CPU or GPU write.
    if (alignment != 0 && (!std::has_single_bit(alignment) || !Common::Is16KBAligned(alignment) ||
                           alignment >= 2_GB)) {
        LOG_ERROR(Kernel_Vmm, "Alignment value is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (!Common::Is16KBAligned(reinterpret_cast<VAddr>(*addr))) {
        LOG_ERROR(Kernel_Vmm, "Address is not 16KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    constexpr auto ValidDirectFlags =
        Core::MemoryMapFlags::Fixed | Core::MemoryMapFlags::NoOverwrite |
        Core::MemoryMapFlags::Stack | Core::MemoryMapFlags::NoCoalesce;
    if (True(static_cast<Core::MemoryMapFlags>(flags) & ~ValidDirectFlags)) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (type < -1 || type > 10) {
        LOG_ERROR(Kernel_Vmm, "Memory type {} is invalid", type);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (type == 10 && True(static_cast<Core::MemoryProt>(prot) &
                           (Core::MemoryProt::CpuWrite | Core::MemoryProt::GpuWrite))) {
        LOG_ERROR(Kernel_Vmm, "Memory type 10 is not allowed with write access");
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    const auto mem_prot = static_cast<Core::MemoryProt>(prot);
    if (True(mem_prot & Core::MemoryProt::CpuExec)) {
        LOG_ERROR(Kernel_Vmm, "Executable permissions are not allowed.");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    const auto map_flags = static_cast<Core::MemoryMapFlags>(flags);
    const VAddr in_addr = reinterpret_cast<VAddr>(*addr);

    auto* memory = Core::Memory::Instance();
    const auto ret =
        memory->MapMemory(addr, in_addr, len, mem_prot, map_flags, Core::VMAType::Direct, "anon",
                          !g_alias_dmem, phys_addr, alignment);

    if (ret == 0) {
        // If the map call succeeds, set the direct memory type using the output address.
        auto* memory = Core::Memory::Instance();
        const auto out_addr = reinterpret_cast<VAddr>(*addr);
        if (type != -1) {
            memory->SetDirectMemoryType(out_addr, len, type, true);
        }
        LOG_INFO(Kernel_Vmm, "out_addr = {:#x}", out_addr);
    }
    return ret;
}

s32 PS4_SYSV_ABI sceKernelMapNamedFlexibleMemory(void** addr_in_out, u64 len, s32 prot, s32 flags,
                                                 const char* name) {
    // Measured on hardware: a null name is EFAULT.
    if (name == nullptr) {
        LOG_ERROR(Kernel_Vmm, "name is invalid!");
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    LOG_INFO(Kernel_Vmm, "in_addr = {}, len = {:#x}, prot = {:#x}, flags = {:#x}, name = '{}'",
             fmt::ptr(*addr_in_out), len, prot, flags, name);
    if (len == 0 || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "len is 0 or not 16kb multiple");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (std::strlen(name) >= ORBIS_KERNEL_MAXIMUM_NAME_LENGTH) {
        LOG_ERROR(Kernel_Vmm, "name exceeds 32 bytes!");
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    const VAddr in_addr = reinterpret_cast<VAddr>(*addr_in_out);
    const auto mem_prot = static_cast<Core::MemoryProt>(prot);
    auto map_flags = static_cast<Core::MemoryMapFlags>(flags);
    // Measured on hardware for each of the 32 bits: other protection bits are refused, and the
    // flags are limited to fixed, no-overwrite and no-coalesce.
    constexpr auto ValidProt = Core::MemoryProt::CpuRead | Core::MemoryProt::CpuWrite |
                               Core::MemoryProt::CpuExec | Core::MemoryProt::GpuRead |
                               Core::MemoryProt::GpuWrite;
    if (True(mem_prot & ~ValidProt)) {
        LOG_ERROR(Kernel_Vmm, "Invalid protection {:#x}", prot);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    constexpr auto ValidFlexibleFlags = Core::MemoryMapFlags::Fixed |
                                        Core::MemoryMapFlags::NoOverwrite |
                                        Core::MemoryMapFlags::NoCoalesce;
    if (True(map_flags & ~ValidFlexibleFlags)) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    auto* memory = Core::Memory::Instance();

    if (True(map_flags & Core::MemoryMapFlags::Fixed) && in_addr == 0) {
        if (Common::ElfInfo::FW_170 <= g_sdk_version) {
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
        map_flags &= ~Core::MemoryMapFlags::Fixed;
    }

    const auto ret = memory->MapMemory(addr_in_out, in_addr, len, mem_prot, map_flags,
                                       Core::VMAType::Flexible, name);
    LOG_INFO(Kernel_Vmm, "out_addr = {}", fmt::ptr(*addr_in_out));
    return ret;
}

s32 PS4_SYSV_ABI sceKernelMapFlexibleMemory(void** addr_in_out, u64 len, s32 prot, s32 flags) {
    return sceKernelMapNamedFlexibleMemory(addr_in_out, len, prot, flags, "anon");
}

s32 PS4_SYSV_ABI sceKernelMapNamedSystemFlexibleMemory(void** addr_in_out, u64 len, s32 prot,
                                                       s32 flags, const char* name) {
    LOG_INFO(Kernel_Vmm, "in_addr = {}, len = {:#x}, prot = {:#x}, flags = {:#x}, name = '{}'",
             fmt::ptr(*addr_in_out), len, prot, flags, name);
    if (len == 0 || !Common::Is16KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "len is 0 or not 16kb multiple");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (name == nullptr) {
        LOG_ERROR(Kernel_Vmm, "name is invalid!");
        return ORBIS_KERNEL_ERROR_EFAULT;
    }

    auto map_flags = static_cast<Core::MemoryMapFlags>(flags);
    VAddr in_addr = reinterpret_cast<VAddr>(*addr_in_out);
    if (True(map_flags & Core::MemoryMapFlags::Fixed) && in_addr == 0) {
        if (Common::ElfInfo::FW_170 <= g_sdk_version) {
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
        map_flags &= ~Core::MemoryMapFlags::Fixed;
    }

    if (std::strlen(name) >= ORBIS_KERNEL_MAXIMUM_NAME_LENGTH) {
        LOG_ERROR(Kernel_Vmm, "name exceeds 32 bytes!");
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    if (in_addr == 0) {
        in_addr = 0x880000000;
    }
    const auto mem_prot = static_cast<Core::MemoryProt>(prot);
    auto* memory = Core::Memory::Instance();
    const auto ret = memory->MapMemory(addr_in_out, in_addr, len, mem_prot, map_flags,
                                       Core::VMAType::System, name);
    LOG_INFO(Kernel_Vmm, "out_addr = {}", fmt::ptr(*addr_in_out));
    return ret;
}

s32 PS4_SYSV_ABI sceKernelQueryMemoryProtection(void* addr, void** start, void** end, u32* prot) {
    auto* memory = Core::Memory::Instance();
    return memory->QueryProtection(std::bit_cast<VAddr>(addr), start, end, prot);
}

s32 PS4_SYSV_ABI sceKernelMprotect(const void* addr, u64 size, s32 prot) {
    LOG_INFO(Kernel_Vmm, "called addr = {}, size = {:#x}, prot = {:#x}", fmt::ptr(addr), size,
             prot);
    // Align addr and size to the nearest page boundary.
    const VAddr in_addr = reinterpret_cast<VAddr>(addr);
    auto aligned_addr = Common::AlignDown(in_addr, 16_KB);
    auto aligned_size = Common::AlignUp(size + in_addr - aligned_addr, 16_KB);

    if (aligned_size == 0) {
        // Nothing to do.
        return ORBIS_OK;
    }

    Core::MemoryManager* memory_manager = Core::Memory::Instance();
    // Measured on hardware for each of the 32 bits: CPU read/write/execute and GPU read/write are
    // kept, every other bit is dropped and the call still succeeds.
    constexpr auto KeptProt = Core::MemoryProt::CpuRead | Core::MemoryProt::CpuWrite |
                              Core::MemoryProt::CpuExec | Core::MemoryProt::GpuRead |
                              Core::MemoryProt::GpuWrite;
    Core::MemoryProt protection_flags = static_cast<Core::MemoryProt>(prot) & KeptProt;

    // Measured on hardware (DmemTest TP1): write access to direct memory of type 10 is EACCES.
    if (True(protection_flags & (Core::MemoryProt::CpuWrite | Core::MemoryProt::GpuWrite)) &&
        memory_manager->HasDirectMemoryType(aligned_addr, aligned_size, 10)) {
        LOG_ERROR(Kernel_Vmm, "Memory type 10 is not allowed with write access");
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    s32 result = memory_manager->Protect(aligned_addr, aligned_size, protection_flags);
    if (result == ORBIS_OK) {
        memory_manager->InvalidateMemory(aligned_addr, aligned_size);
    }
    return result;
}

s32 PS4_SYSV_ABI posix_mprotect(const void* addr, u64 size, s32 prot) {
    s32 result = sceKernelMprotect(addr, size, prot);
    if (result < 0) {
        ErrSceToPosix(result);
        return -1;
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelMtypeprotect(const void* addr, u64 size, s32 mtype, s32 prot) {
    LOG_INFO(Kernel_Vmm, "called addr = {}, size = {:#x}, prot = {:#x}", fmt::ptr(addr), size,
             prot);
    // Measured on hardware: types 0 to 10 are valid; type 10 is refused with CPU or GPU write.
    if (mtype == 10 && True(static_cast<Core::MemoryProt>(prot) &
                            (Core::MemoryProt::CpuWrite | Core::MemoryProt::GpuWrite))) {
        LOG_ERROR(Kernel_Vmm, "Memory type {} is not allowed with write access", mtype);
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    if (mtype < 0 || mtype > 10) {
        LOG_ERROR(Kernel_Vmm, "Memory type {} is invalid", mtype);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware for each of the 32 bits: 0x8, 0x40 and 0x80 are refused, execute is
    // EACCES, and bits from 0x100 up are dropped.
    if ((prot & (0x8 | 0x40 | 0x80)) != 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid protection {:#x}", prot);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (True(static_cast<Core::MemoryProt>(prot) & Core::MemoryProt::CpuExec)) {
        LOG_ERROR(Kernel_Vmm, "Executable permissions are not allowed.");
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    // Align addr and size to the nearest page boundary.
    const VAddr in_addr = reinterpret_cast<VAddr>(addr);
    auto aligned_addr = Common::AlignDown(in_addr, 16_KB);
    auto aligned_size = Common::AlignUp(size + in_addr - aligned_addr, 16_KB);

    if (aligned_size == 0) {
        // Nothing to do.
        return ORBIS_OK;
    }
    Core::MemoryManager* memory_manager = Core::Memory::Instance();
    Core::MemoryProt protection_flags =
        static_cast<Core::MemoryProt>(prot) &
        (Core::MemoryProt::CpuReadWrite | Core::MemoryProt::GpuReadWrite);

    // Measured on hardware: flexible memory has no memory type to change (measured for a range of
    // flexible memory). The call is refused before anything changes.
    if (memory_manager->ContainsFlexibleMemory(aligned_addr, aligned_size)) {
        return ORBIS_KERNEL_ERROR_EOPNOTSUPP;
    }
    // Measured on hardware (DmemTest TP3): a range from ReserveVirtualRange is EACCES.
    if (memory_manager->ContainsVmaType(aligned_addr, aligned_size, Core::VMAType::Reserved)) {
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    s32 result = memory_manager->Protect(aligned_addr, aligned_size, protection_flags);
    if (result == ORBIS_OK) {
        memory_manager->SetDirectMemoryType(aligned_addr, aligned_size, mtype);
        memory_manager->InvalidateMemory(aligned_addr, aligned_size);
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelDirectMemoryQuery(u64 offset, s32 flags, OrbisQueryInfo* query_info,
                                            u64 infoSize) {
    LOG_INFO(Kernel_Vmm, "called offset = {:#x}, flags = {:#x}", offset, flags);
    if (flags != 0 && flags != 1) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    static_assert(sizeof(OrbisQueryInfo) == 0x18);
    OrbisQueryInfo info;
    std::memset(&info, 0, sizeof(info));
    auto* memory = Core::Memory::Instance();
    const s32 result = memory->DirectMemoryQuery(offset, flags == 1, &info);
    const u64 copy_size = std::min<u64>(infoSize, sizeof(info));
    if (result != ORBIS_OK || copy_size == 0) {
        return result;
    }
    if (query_info == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    // As on hardware, the copy runs in 8-byte words and then single bytes, and stops at the first
    // word or byte that reaches guest memory that is unmapped or not CPU writable.
    const u64 writable =
        memory->GuestWritablePrefix(reinterpret_cast<VAddr>(query_info), copy_size);
    const u64 word_bytes = copy_size & ~7ULL;
    const u64 copied = writable < word_bytes ? writable & ~7ULL : writable;
    std::memcpy(query_info, &info, copied);
    if (copied < copy_size) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelAvailableFlexibleMemorySize(u64* out_size) {
    auto* memory = Core::Memory::Instance();
    // Measured on hardware: the size is reported rounded up to 1 MiB, so it only drops once the
    // flexible memory in use crosses a 1 MiB boundary (64 KiB: 0, 16 x 64 KiB: 1 MiB, 17 x 64 KiB:
    // 1 MiB, 2 MiB + 16 KiB: 2 MiB).
    *out_size = Common::AlignUp(memory->GetAvailableFlexibleSize(), 1_MB);
    LOG_INFO(Kernel_Vmm, "called size = {:#x}", *out_size);
    return ORBIS_OK;
}

void PS4_SYSV_ABI _sceKernelRtldSetApplicationHeapAPI(void* func[]) {
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    linker->SetHeapAPI(func);
}

s32 PS4_SYSV_ABI sceKernelGetDirectMemoryType(u64 addr, s32* directMemoryTypeOut,
                                              void** directMemoryStartOut,
                                              void** directMemoryEndOut) {
    LOG_WARNING(Kernel_Vmm, "called, direct memory addr = {:#x}", addr);
    auto* memory = Core::Memory::Instance();
    return memory->GetDirectMemoryType(addr, directMemoryTypeOut, directMemoryStartOut,
                                       directMemoryEndOut);
}

s32 PS4_SYSV_ABI sceKernelIsStack(void* addr, void** start, void** end) {
    LOG_DEBUG(Kernel_Vmm, "called, addr = {}", fmt::ptr(addr));
    auto* memory = Core::Memory::Instance();
    return memory->IsStack(std::bit_cast<VAddr>(addr), start, end);
}

u32 PS4_SYSV_ABI sceKernelIsAddressSanitizerEnabled() {
    LOG_DEBUG(Kernel, "called");
    return false;
}

s32 PS4_SYSV_ABI sceKernelBatchMap(OrbisKernelBatchMapEntry* entries, s32 numEntries,
                                   s32* numEntriesOut) {
    return sceKernelBatchMap2(entries, numEntries, numEntriesOut,
                              MemoryFlags::ORBIS_KERNEL_MAP_FIXED); // 0x10, 0x410?
}

s32 PS4_SYSV_ABI sceKernelBatchMap2(OrbisKernelBatchMapEntry* entries, s32 numEntries,
                                    s32* numEntriesOut, s32 flags) {
    // Measured on hardware for each of the 32 bits (DmemTest BT26): fixed, no-overwrite, 0x400,
    // no-coalesce and bits 24 to 28 are accepted, every other bit is EINVAL.
    constexpr s32 ValidBatchFlags = 0x10 | 0x80 | 0x400 | 0x400000 | 0x1F000000;
    if ((flags & ~ValidBatchFlags) != 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    s32 result = ORBIS_OK;
    s32 processed = 0;
    for (s32 i = 0; i < numEntries; i++, processed++) {
        if (entries == nullptr || entries[i].length == 0 || entries[i].operation > 4) {
            result = ORBIS_KERNEL_ERROR_EINVAL;
            break; // break and assign a value to numEntriesOut.
        }

        switch (entries[i].operation) {
        case MemoryOpTypes::ORBIS_KERNEL_MAP_OP_MAP_DIRECT: {
            result = sceKernelMapNamedDirectMemory(&entries[i].start, entries[i].length,
                                                   entries[i].protection, flags,
                                                   static_cast<s64>(entries[i].offset), 0, "anon");
            break;
        }
        case MemoryOpTypes::ORBIS_KERNEL_MAP_OP_UNMAP: {
            result = sceKernelMunmap(entries[i].start, entries[i].length);
            break;
        }
        case MemoryOpTypes::ORBIS_KERNEL_MAP_OP_PROTECT: {
            result = sceKernelMprotect(entries[i].start, entries[i].length, entries[i].protection);
            break;
        }
        case MemoryOpTypes::ORBIS_KERNEL_MAP_OP_MAP_FLEXIBLE: {
            result = sceKernelMapNamedFlexibleMemory(&entries[i].start, entries[i].length,
                                                     entries[i].protection, flags, "anon");
            break;
        }
        case MemoryOpTypes::ORBIS_KERNEL_MAP_OP_TYPE_PROTECT: {
            result = sceKernelMtypeprotect(entries[i].start, entries[i].length, entries[i].type,
                                           entries[i].protection);
            break;
        }
        default: {
            UNREACHABLE();
        }
        }

        if (result != ORBIS_OK) {
            LOG_ERROR(Kernel_Vmm, "failed with error code {:#x}", result);
            break;
        }
    }
    if (numEntriesOut != NULL) { // can be zero. do not return an error code.
        *numEntriesOut = processed;
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelSetVirtualRangeName(const void* addr, u64 len, const char* name) {
    if (name == nullptr) {
        LOG_ERROR(Kernel_Vmm, "name is invalid!");
        return ORBIS_KERNEL_ERROR_EFAULT;
    }

    if (std::strlen(name) >= ORBIS_KERNEL_MAXIMUM_NAME_LENGTH) {
        LOG_ERROR(Kernel_Vmm, "name exceeds 32 bytes!");
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    auto* memory = Core::Memory::Instance();
    memory->NameVirtualRange(std::bit_cast<VAddr>(addr), len, name);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelClearVirtualRangeName(const void* addr, u64 len) {
    auto* memory = Core::Memory::Instance();
    memory->NameVirtualRange(std::bit_cast<VAddr>(addr), len, "");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelMemoryPoolExpand(u64 searchStart, u64 searchEnd, u64 len, u64 alignment,
                                           u64* physAddrOut) {
    if (searchStart < 0 || searchEnd <= searchStart) {
        LOG_ERROR(Kernel_Vmm, "Provided address range is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len <= 0 || !Common::Is64KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Provided length {:#x} is invalid!", len);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (alignment != 0 && !Common::Is64KBAligned(alignment)) {
        LOG_ERROR(Kernel_Vmm, "Alignment {:#x} is invalid!", alignment);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (physAddrOut == nullptr) {
        LOG_ERROR(Kernel_Vmm, "Result physical address pointer is null!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    const bool is_in_range = searchEnd - searchStart >= len;
    if (searchEnd <= searchStart || searchEnd < len || !is_in_range) {
        LOG_ERROR(Kernel_Vmm,
                  "Provided address range is too small!"
                  " searchStart = {:#x}, searchEnd = {:#x}, length = {:#x}",
                  searchStart, searchEnd, len);
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    auto* memory = Core::Memory::Instance();
    PAddr phys_addr = memory->PoolExpand(searchStart, searchEnd, len, alignment);
    if (phys_addr == -1) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    *physAddrOut = static_cast<s64>(phys_addr);

    LOG_INFO(Kernel_Vmm,
             "searchStart = {:#x}, searchEnd = {:#x}, len = {:#x}, alignment = {:#x}, physAddrOut "
             "= {:#x}",
             searchStart, searchEnd, len, alignment, phys_addr);
    return ORBIS_OK;
}

static s32 MemoryPoolReserveImpl(void* addr_in, u64 len, u64 alignment, s32 flags, void** addr_out,
                                 std::string_view name) {
    LOG_INFO(Kernel_Vmm, "addr_in = {}, len = {:#x}, alignment = {:#x}, flags = {:#x}",
             fmt::ptr(addr_in), len, alignment, flags);

    // libkernel accepts only the fixed and no-overwrite flags, and an alignment that is a power of
    // two of at most 2 GiB.
    if ((flags & ~0x90) != 0) {
        LOG_ERROR(Kernel_Vmm, "Flags {:#x} are invalid!", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware: a non-zero alignment must be a power of two from 2 MiB to 1 GiB, and a
    // fixed reservation needs a non-zero, 2 MiB aligned address.
    // Measured on hardware: an alignment of 1 is accepted too.
    if (alignment > 1 &&
        (!std::has_single_bit(alignment) || alignment < 2_MB || alignment > 1_GB)) {
        LOG_ERROR(Kernel_Vmm, "Alignment value is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if ((flags & 0x10) != 0 &&
        (addr_in == nullptr || !Common::Is2MBAligned(reinterpret_cast<VAddr>(addr_in)))) {
        LOG_ERROR(Kernel_Vmm, "Fixed reservation address {} is invalid!", fmt::ptr(addr_in));
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len == 0 || !Common::Is2MBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Map size is either zero or not 2MB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto* memory = Core::Memory::Instance();
    const VAddr in_addr = reinterpret_cast<VAddr>(addr_in);
    const auto map_flags = static_cast<Core::MemoryMapFlags>(flags);
    u64 map_alignment = alignment <= 1 ? 2_MB : alignment;

    return memory->PoolReserve(addr_out, in_addr, len, map_flags, map_alignment, name);
}

s32 PS4_SYSV_ABI sceKernelMemoryPoolReserve(void* addr_in, u64 len, u64 alignment, s32 flags,
                                            void** addr_out) {
    return MemoryPoolReserveImpl(addr_in, len, alignment, flags, addr_out, "anon");
}

// Measured on hardware: anonymous mappings are named "anon:" and the return address of the call
// that made them, in 12 hex digits.
static std::string CallerName(VAddr caller) {
    return fmt::format("anon:{:012x}", caller);
}

// Guest entry. Measured on hardware: a reservation is named "anon:" and the return address of the
// call that made it, in 12 hex digits.
static s32 PS4_SYSV_ABI MemoryPoolReserveFromGuest(VAddr caller, void* addr_in, u64 len,
                                                   u64 alignment, s32 flags, void** addr_out) {
    return MemoryPoolReserveImpl(addr_in, len, alignment, flags, addr_out, CallerName(caller));
}

s32 PS4_SYSV_ABI sceKernelMemoryPoolCommit(void* addr, u64 len, s32 type, s32 prot, s32 flags) {
    // Measured on hardware: the address must be 64 KB aligned, and a length of 0 succeeds.
    if (addr == nullptr || !Common::Is64KBAligned(reinterpret_cast<VAddr>(addr))) {
        LOG_ERROR(Kernel_Vmm, "Address is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (!Common::Is64KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Map size is not 64KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    const auto mem_prot = static_cast<Core::MemoryProt>(prot);
    // Measured on hardware for each of the 32 bits: no flag is accepted, and protection bits other
    // than CPU and GPU read/write are refused.
    if (flags != 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (True(mem_prot & ~(Core::MemoryProt::CpuReadWrite | Core::MemoryProt::GpuReadWrite |
                          Core::MemoryProt::CpuExec))) {
        LOG_ERROR(Kernel_Vmm, "Invalid protection {:#x}", prot);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (True(mem_prot & Core::MemoryProt::CpuExec)) {
        LOG_ERROR(Kernel_Vmm, "Executable permissions are not allowed.");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware: pooled memory can be committed as type 0 or 3, or as type 10 without
    // write access.
    if (type == 10 && True(mem_prot & (Core::MemoryProt::CpuWrite | Core::MemoryProt::GpuWrite))) {
        LOG_ERROR(Kernel_Vmm, "Memory type {} is not allowed", type);
        return ORBIS_KERNEL_ERROR_EACCES;
    }
    if (type != 0 && type != 3 && type != 10) {
        LOG_ERROR(Kernel_Vmm, "Memory type {} is invalid", type);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware: a length of 0 succeeds once type and protection are valid.
    if (len == 0) {
        return ORBIS_OK;
    }

    LOG_INFO(Kernel_Vmm, "addr = {}, len = {:#x}, type = {:#x}, prot = {:#x}, flags = {:#x}",
             fmt::ptr(addr), len, type, prot, flags);

    const VAddr in_addr = reinterpret_cast<VAddr>(addr);
    auto* memory = Core::Memory::Instance();
    return memory->PoolCommit(in_addr, len, mem_prot, type);
}

s32 PS4_SYSV_ABI sceKernelMemoryPoolDecommit(void* addr, u64 len, s32 flags) {
    // Measured on hardware: the address must be 64 KB aligned, and a length of 0 succeeds.
    if (addr == nullptr || !Common::Is64KBAligned(reinterpret_cast<VAddr>(addr))) {
        LOG_ERROR(Kernel_Vmm, "Address is invalid!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware for each of the 32 bits: no flag is accepted.
    if (flags != 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (len == 0) {
        return ORBIS_OK;
    }
    if (!Common::Is64KBAligned(len)) {
        LOG_ERROR(Kernel_Vmm, "Map size is not 64KB aligned!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    LOG_INFO(Kernel_Vmm, "addr = {}, len = {:#x}, flags = {:#x}", fmt::ptr(addr), len, flags);

    const VAddr pool_addr = reinterpret_cast<VAddr>(addr);
    auto* memory = Core::Memory::Instance();

    return memory->PoolDecommit(pool_addr, len);
}

s32 PS4_SYSV_ABI sceKernelMemoryPoolBatch(const OrbisKernelMemoryPoolBatchEntry* entries, s32 count,
                                          s32* num_processed, s32 flags) {
    if (entries == nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware for each of the 32 bits: no flag is accepted.
    if (flags != 0) {
        LOG_ERROR(Kernel_Vmm, "Invalid flags {:#x}", flags);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    s32 result = ORBIS_OK;
    s32 processed = 0;

    for (s32 i = 0; i < count; i++, processed++) {
        OrbisKernelMemoryPoolBatchEntry entry = entries[i];
        switch (entry.opcode) {
        case OrbisKernelMemoryPoolOpcode::Commit: {
            result = sceKernelMemoryPoolCommit(entry.commit_params.addr, entry.commit_params.len,
                                               entry.commit_params.type, entry.commit_params.prot,
                                               entry.flags);
            break;
        }
        case OrbisKernelMemoryPoolOpcode::Decommit: {
            result = sceKernelMemoryPoolDecommit(entry.decommit_params.addr,
                                                 entry.decommit_params.len, entry.flags);
            break;
        }
        case OrbisKernelMemoryPoolOpcode::Protect: {
            result = sceKernelMprotect(entry.protect_params.addr, entry.protect_params.len,
                                       entry.protect_params.prot);
            break;
        }
        case OrbisKernelMemoryPoolOpcode::TypeProtect: {
            result = sceKernelMtypeprotect(
                entry.type_protect_params.addr, entry.type_protect_params.len,
                entry.type_protect_params.type, entry.type_protect_params.prot);
            break;
        }
        case OrbisKernelMemoryPoolOpcode::Move: {
            UNREACHABLE_MSG("Unimplemented sceKernelMemoryPoolBatch opcode Move");
        }
        default: {
            // Measured on hardware: an unknown opcode is EOPNOTSUPP.
            result = ORBIS_KERNEL_ERROR_EOPNOTSUPP;
            break;
        }
        }

        if (result != ORBIS_OK) {
            break;
        }
    }

    if (num_processed != nullptr) {
        *num_processed = processed;
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelMemoryPoolGetBlockStats(OrbisKernelMemoryPoolBlockStats* stats,
                                                  u64 size) {
    LOG_WARNING(Kernel_Vmm, "called");
    auto* memory = Core::Memory::Instance();
    OrbisKernelMemoryPoolBlockStats local_stats;
    memory->GetMemoryPoolStats(&local_stats);

    u64 size_to_copy = size < sizeof(OrbisKernelMemoryPoolBlockStats)
                           ? size
                           : sizeof(OrbisKernelMemoryPoolBlockStats);
    // As of firmware 12.02, the kernel does not check if stats is null,
    // this can cause crashes on real hardware, so have an assert for this case.
    ASSERT_MSG(stats != nullptr || size == 0, "Block stats cannot be null");
    std::memcpy(stats, &local_stats, size_to_copy);
    return ORBIS_OK;
}

static void* PosixMmapImpl(void* addr, u64 len, s32 prot, s32 flags, s32 fd, s64 phys_addr,
                           std::string_view name) {
    LOG_INFO(
        Kernel_Vmm,
        "called addr = {}, len = {:#x}, prot = {:#x}, flags = {:#x}, fd = {}, phys_addr = {:#x}",
        fmt::ptr(addr), len, prot, flags, fd, phys_addr);

    if (len == 0) {
        // If length is 0, mmap returns EINVAL.
        ErrSceToPosix(ORBIS_KERNEL_ERROR_EINVAL);
        LOG_ERROR(Kernel_Vmm, "Invalid length");
        return reinterpret_cast<void*>(-1);
    }

    void* addr_out;
    auto* memory = Core::Memory::Instance();
    const auto mem_prot = static_cast<Core::MemoryProt>(prot);
    const auto mem_flags = static_cast<Core::MemoryMapFlags>(flags);
    const auto vaddr = reinterpret_cast<VAddr>(addr);

    // Align address and size here. Both align up to the next page
    const VAddr aligned_addr = Common::AlignUp(vaddr, 16_KB);
    const u64 aligned_size = Common::AlignUp(len, 16_KB);

    if (True(mem_flags & Core::MemoryMapFlags::Fixed) && vaddr != aligned_addr) {
        // If flags Fixed is specified, the input address must be aligned.
        ErrSceToPosix(ORBIS_KERNEL_ERROR_EINVAL);
        LOG_ERROR(Kernel_Vmm, "Misaligned input address");
        return reinterpret_cast<void*>(-1);
    }

    // Measured on hardware (memory_test findings, DmemTest SF2/SF3/FW1/BT9):
    // a null address with Fixed, and the sanitizer flag (0x200000), are EINVAL.
    if ((True(mem_flags & Core::MemoryMapFlags::Fixed) && vaddr == 0) || (flags & 0x200000) != 0) {
        ErrSceToPosix(ORBIS_KERNEL_ERROR_EINVAL);
        return reinterpret_cast<void*>(-1);
    }
    // an anonymous mapping with an offset or a file, and a void mapping with an offset, are EINVAL.
    if ((True(mem_flags & Core::MemoryMapFlags::Anon) && (phys_addr != 0 || fd != -1)) ||
        (True(mem_flags & Core::MemoryMapFlags::Void) && phys_addr != 0)) {
        ErrSceToPosix(ORBIS_KERNEL_ERROR_EINVAL);
        return reinterpret_cast<void*>(-1);
    }
    if (True(mem_flags & Core::MemoryMapFlags::Stack)) {
        // a stack mapping needs CPU read/write and no file; without an address it is ENOMEM, and so
        // is a fixed one over a mapped range.
        if ((mem_prot & Core::MemoryProt::CpuReadWrite) != Core::MemoryProt::CpuReadWrite ||
            fd != -1) {
            ErrSceToPosix(ORBIS_KERNEL_ERROR_EINVAL);
            return reinterpret_cast<void*>(-1);
        }
        if ((False(mem_flags & Core::MemoryMapFlags::Fixed) && vaddr == 0) ||
            (True(mem_flags & Core::MemoryMapFlags::Fixed) &&
             memory->IsRangeMapped(aligned_addr, aligned_size))) {
            ErrSceToPosix(ORBIS_KERNEL_ERROR_ENOMEM);
            return reinterpret_cast<void*>(-1);
        }
    }

    s32 result = ORBIS_OK;
    if (True(mem_flags & Core::MemoryMapFlags::Anon)) {
        // Measured on hardware for each of the 32 bits: an anonymous mapping keeps CPU
        // read/write/execute and GPU read/write and drops every other protection bit.
        constexpr auto KeptProt = Core::MemoryProt::CpuRead | Core::MemoryProt::CpuWrite |
                                  Core::MemoryProt::CpuExec | Core::MemoryProt::GpuRead |
                                  Core::MemoryProt::GpuWrite;
        // Maps flexible or system memory
        const Core::VMAType mapping_type = True(mem_flags & Core::MemoryMapFlags::System)
                                               ? Core::VMAType::System
                                               : Core::VMAType::Flexible;
        result = memory->MapMemory(&addr_out, aligned_addr, aligned_size, mem_prot & KeptProt,
                                   mem_flags, mapping_type, name, false);
    } else if (True(mem_flags & Core::MemoryMapFlags::Stack)) {
        // Maps stack memory
        result = memory->MapMemory(&addr_out, aligned_addr, aligned_size, mem_prot, mem_flags,
                                   Core::VMAType::Stack, "anon", false);
    } else if (True(mem_flags & Core::MemoryMapFlags::Void)) {
        // Reserves memory
        result =
            memory->MapMemory(&addr_out, aligned_addr, aligned_size, Core::MemoryProt::NoAccess,
                              mem_flags, Core::VMAType::Reserved, "anon", false);
    } else {
        // Default to file mapping
        // Measured on hardware (DmemTest FM2): an offset inside a page maps from the start of that
        // page, the size grows by the offset within the page, and the address returned points at
        // the requested offset.
        const u64 page_offset = static_cast<u64>(phys_addr) & (16_KB - 1);
        result = memory->MapFile(&addr_out, aligned_addr, Common::AlignUp(len + page_offset, 16_KB),
                                 mem_prot, mem_flags, fd, phys_addr - page_offset, name);
        if (result == ORBIS_OK) {
            addr_out = static_cast<u8*>(addr_out) + page_offset;
        }
    }

    if (result != ORBIS_OK) {
        // If the memory mappings fail, mmap sets errno to the appropriate error code,
        // then returns (void*)-1;
        ErrSceToPosix(result);
        LOG_ERROR(Kernel_Vmm, "error = {}", *__Error());
        return reinterpret_cast<void*>(-1);
    }

    LOG_INFO(Kernel_Vmm, "addr_out = {}", fmt::ptr(addr_out));
    return addr_out;
}

void* PS4_SYSV_ABI posix_mmap(void* addr, u64 len, s32 prot, s32 flags, s32 fd, s64 phys_addr) {
    return PosixMmapImpl(addr, len, prot, flags, fd, phys_addr, "anon");
}

static s32 KernelMmapImpl(void* addr, u64 len, s32 prot, s32 flags, s32 fd, s64 phys_addr,
                          void** res, std::string_view name) {
    void* addr_out = PosixMmapImpl(addr, len, prot, flags, fd, phys_addr, name);

    if (addr_out == reinterpret_cast<void*>(-1)) {
        // posix_mmap failed, calculate and return the appropriate kernel error code using errno.
        return ErrnoToSceKernelError(*__Error());
    }

    // Set the outputted address
    *res = addr_out;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelMmap(void* addr, u64 len, s32 prot, s32 flags, s32 fd, s64 phys_addr,
                               void** res) {
    return KernelMmapImpl(addr, len, prot, flags, fd, phys_addr, res, "anon");
}

s32 PS4_SYSV_ABI sceKernelConfiguredFlexibleMemorySize(u64* sizeOut) {
    if (sizeOut == nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto* memory = Core::Memory::Instance();
    *sizeOut = memory->GetTotalFlexibleSize();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelMunmap(void* addr, u64 len) {
    LOG_INFO(Kernel_Vmm, "addr = {}, len = {:#x}", fmt::ptr(addr), len);
    if (len == 0) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    auto* memory = Core::Memory::Instance();
    return memory->UnmapMemory(std::bit_cast<VAddr>(addr), len);
}

s32 PS4_SYSV_ABI posix_munmap(void* addr, u64 len) {
    s32 result = sceKernelMunmap(addr, len);
    if (result < 0) {
        LOG_ERROR(Kernel_Pthread, "posix_munmap: error = {}", result);
        ErrSceToPosix(result);
        return -1;
    }
    return result;
}

s32 PS4_SYSV_ABI posix_mlock(const void* addr, u64 len) {
    if (!addr || len == 0) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    const VAddr raw_addr = reinterpret_cast<VAddr>(addr);
    if (raw_addr + len < raw_addr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    // Measured on hardware: a range that is not all mapped is ENOMEM.
    if (!Core::Memory::Instance()->IsRangeMapped(raw_addr, len)) {
        *__Error() = POSIX_ENOMEM;
        return -1;
    }
    LOG_DEBUG(Kernel_Vmm, "(STUBBED) addr = {}, len = {:#x}", fmt::ptr(addr), len);
    return 0;
}

s32 PS4_SYSV_ABI sceKernelMlock(void* addr, u64 len) {
    s32 result = posix_mlock(addr, len);
    if (result < 0) {
        LOG_ERROR(Kernel_Vmm, "addr = {}, len = {:#x}, error = {:#x}", fmt::ptr(addr), len,
                  *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_munlock(const void* addr, u64 len) {
    if (!addr || len == 0) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    const VAddr raw_addr = reinterpret_cast<VAddr>(addr);
    if (raw_addr + len < raw_addr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    // Measured on hardware: a range that is not all mapped is ENOMEM, and a mapped one EINVAL.
    if (!Core::Memory::Instance()->IsRangeMapped(raw_addr, len)) {
        *__Error() = POSIX_ENOMEM;
        return -1;
    }
    *__Error() = POSIX_EINVAL;
    return -1;
}

s32 PS4_SYSV_ABI sceKernelMunlock(void* addr, u64 len) {
    s32 result = posix_munlock(addr, len);
    if (result < 0) {
        LOG_ERROR(Kernel_Vmm, "addr = {}, len = {:#x}, error = {:#x}", fmt::ptr(addr), len,
                  *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelReleaseFlexibleMemory(void* addr, u64 len) {
    const VAddr raw_addr = reinterpret_cast<VAddr>(addr);

    if (!addr || ((raw_addr | len) & 0x3fff)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    // Measured on hardware: a length of 0, or a range without flexible memory (a hole, direct
    // memory), succeeds.
    if (len == 0 || !Core::Memory::Instance()->ContainsFlexibleMemory(raw_addr, len)) {
        return ORBIS_OK;
    }

    LOG_DEBUG(Kernel_Vmm, "addr = {}, len = {:#x}", fmt::ptr(addr), len);

#if defined(_WIN32)
    if (!VirtualAlloc(addr, len, MEM_RESET, PAGE_READWRITE)) {
        LOG_ERROR(Kernel_Vmm, "VirtualAlloc failed: {}", Common::GetLastErrorMsg());
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#elif defined(__APPLE__)
    if (madvise(addr, len, MADV_FREE) != 0) {
        LOG_ERROR(Kernel_Vmm, "madvise failed: {}", Common::GetLastErrorMsg());
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#else
    if (madvise(addr, len, MADV_DONTNEED) != 0) {
        LOG_ERROR(Kernel_Vmm, "madvise failed: {}", Common::GetLastErrorMsg());
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#endif

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_msync(void* addr, u64 len, s32 flags) {
    // Measured on hardware: a range that is not all mapped is EINVAL.
    if (!Core::Memory::Instance()->IsRangeMapped(reinterpret_cast<VAddr>(addr), len)) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    LOG_ERROR(Kernel_Vmm, "(STUBBED) called, addr = {}, len = {:#x}, flags = {}", fmt::ptr(addr),
              len, flags);
    return ORBIS_OK;
}

static constexpr s32 MAX_PRT_APERTURES = 3;
static constexpr VAddr PRT_AREA_START_ADDR = 0x1000000000;
static constexpr u64 PRT_AREA_SIZE = 0xec00000000;
static std::array<std::pair<VAddr, u64>, MAX_PRT_APERTURES> PrtApertures{};

s32 PS4_SYSV_ABI sceKernelSetPrtAperture(s32 id, VAddr address, u64 size) {
    if (id < 0 || id >= MAX_PRT_APERTURES) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (address < PRT_AREA_START_ADDR || address + size > PRT_AREA_START_ADDR + PRT_AREA_SIZE) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (address % 4096 != 0) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    LOG_WARNING(Kernel_Vmm,
                "PRT aperture id = {}, address = {:#x}, size = {:#x} is set but not used", id,
                address, size);

    auto* memory = Core::Memory::Instance();
    memory->SetPrtArea(id, address, size);

    PrtApertures[id] = {address, size};
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetPrtAperture(s32 id, VAddr* address, u64* size) {
    if (id < 0 || id >= MAX_PRT_APERTURES) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    std::tie(*address, *size) = PrtApertures[id];
    return ORBIS_OK;
}

// Guest entries for the calls measured on hardware.
static s32 PS4_SYSV_ABI MapFlexibleMemoryFromGuest(VAddr caller, void** addr_in_out, u64 len,
                                                   s32 prot, s32 flags) {
    return sceKernelMapNamedFlexibleMemory(addr_in_out, len, prot, flags,
                                           CallerName(caller).c_str());
}
static s32 PS4_SYSV_ABI MapDirectMemoryFromGuest(VAddr caller, void** addr, u64 len, s32 prot,
                                                 s32 flags, s64 phys_addr, u64 alignment) {
    return sceKernelMapNamedDirectMemory(addr, len, prot, flags, phys_addr, alignment,
                                         CallerName(caller).c_str());
}
// Measured on hardware: an empty name gives the mapping its caller-based name.
static s32 PS4_SYSV_ABI MapNamedDirectMemoryFromGuest(VAddr caller, void** addr, u64 len, s32 prot,
                                                      s32 flags, s64 phys_addr, u64 alignment,
                                                      const char* name) {
    if (name != nullptr && name[0] == '\0') {
        return sceKernelMapNamedDirectMemory(addr, len, prot, flags, phys_addr, alignment,
                                             CallerName(caller).c_str());
    }
    return sceKernelMapNamedDirectMemory(addr, len, prot, flags, phys_addr, alignment, name);
}
static s32 PS4_SYSV_ABI MapNamedFlexibleMemoryFromGuest(VAddr caller, void** addr_in_out, u64 len,
                                                        s32 prot, s32 flags, const char* name) {
    if (name != nullptr && name[0] == '\0') {
        return sceKernelMapNamedFlexibleMemory(addr_in_out, len, prot, flags,
                                               CallerName(caller).c_str());
    }
    return sceKernelMapNamedFlexibleMemory(addr_in_out, len, prot, flags, name);
}
static s32 PS4_SYSV_ABI ReserveVirtualRangeFromGuest(VAddr caller, void** addr, u64 len, s32 flags,
                                                     u64 alignment) {
    return ReserveVirtualRangeImpl(addr, len, flags, alignment, CallerName(caller));
}
static s32 PS4_SYSV_ABI MmapFromGuest(VAddr caller, void* addr, u64 len, s32 prot, s32 flags,
                                      s32 fd, s64 phys_addr, void** res) {
    return KernelMmapImpl(addr, len, prot, flags, fd, phys_addr, res, CallerName(caller));
}

void RegisterMemory(Core::Loader::SymbolsResolver* sym) {
    ASSERT_MSG(sceKernelGetCompiledSdkVersion(&g_sdk_version) == ORBIS_OK,
               "Failed to get compiled SDK verision.");

    LIB_FUNCTION("usHTMoFoBTM", "libkernel_dmem_aliasing2", 1, "libkernel",
                 sceKernelEnableDmemAliasing);
    LIB_FUNCTION("usHTMoFoBTM", "libkernel", 1, "libkernel", sceKernelEnableDmemAliasing);
    LIB_FUNCTION("rTXw65xmLIA", "libkernel", 1, "libkernel", sceKernelAllocateDirectMemory);
    LIB_FUNCTION("B+vc2AO2Zrc", "libkernel", 1, "libkernel", sceKernelAllocateMainDirectMemory);
    LIB_FUNCTION("C0f7TJcbfac", "libkernel", 1, "libkernel", sceKernelAvailableDirectMemorySize);
    LIB_FUNCTION("hwVSPCmp5tM", "libkernel", 1, "libkernel", sceKernelCheckedReleaseDirectMemory);
    LIB_FUNCTION("rVjRvHJ0X6c", "libkernel", 1, "libkernel", sceKernelVirtualQuery);
    LIB_FUNCTION_WITH_CALLER("7oxv3PPCumo", "libkernel", 1, "libkernel",
                             ReserveVirtualRangeFromGuest);
    LIB_FUNCTION("BC+OG5m9+bw", "libkernel", 1, "libkernel", sceKernelGetDirectMemoryType);
    LIB_FUNCTION("pO96TwzOm5E", "libkernel", 1, "libkernel", sceKernelGetDirectMemorySize);
    LIB_FUNCTION("yDBwVAolDgg", "libkernel", 1, "libkernel", sceKernelIsStack);
    LIB_FUNCTION("jh+8XiK4LeE", "libkernel", 1, "libkernel", sceKernelIsAddressSanitizerEnabled);
    LIB_FUNCTION_WITH_CALLER("NcaWUxfMNIQ", "libkernel", 1, "libkernel",
                             MapNamedDirectMemoryFromGuest);
    LIB_FUNCTION_WITH_CALLER("L-Q3LEjIbgA", "libkernel", 1, "libkernel", MapDirectMemoryFromGuest);
    LIB_FUNCTION("BQQniolj9tQ", "libkernel", 1, "libkernel", sceKernelMapDirectMemory2);
    LIB_FUNCTION("WFcfL2lzido", "libkernel", 1, "libkernel", sceKernelQueryMemoryProtection);
    LIB_FUNCTION("BHouLQzh0X0", "libkernel", 1, "libkernel", sceKernelDirectMemoryQuery);
    LIB_FUNCTION("MBuItvba6z8", "libkernel", 1, "libkernel", sceKernelReleaseDirectMemory);
    LIB_FUNCTION_WITH_CALLER("PGhQHd-dzv8", "libkernel", 1, "libkernel", MmapFromGuest);
    LIB_FUNCTION("cQke9UuBQOk", "libkernel", 1, "libkernel", sceKernelMunmap);
    LIB_FUNCTION_WITH_CALLER("mL8NDH86iQI", "libkernel", 1, "libkernel",
                             MapNamedFlexibleMemoryFromGuest);
    LIB_FUNCTION("kc+LEEIYakc", "libkernel", 1, "libkernel", sceKernelMapNamedSystemFlexibleMemory);
    LIB_FUNCTION("aNz11fnnzi4", "libkernel", 1, "libkernel", sceKernelAvailableFlexibleMemorySize);
    LIB_FUNCTION("aNz11fnnzi4", "libkernel_avlfmem", 1, "libkernel",
                 sceKernelAvailableFlexibleMemorySize);
    LIB_FUNCTION_WITH_CALLER("IWIBBdTHit4", "libkernel", 1, "libkernel",
                             MapFlexibleMemoryFromGuest);
    LIB_FUNCTION("p5EcQeEeJAE", "libkernel", 1, "libkernel", _sceKernelRtldSetApplicationHeapAPI);
    LIB_FUNCTION("2SKEx6bSq-4", "libkernel", 1, "libkernel", sceKernelBatchMap);
    LIB_FUNCTION("kBJzF8x4SyE", "libkernel", 1, "libkernel", sceKernelBatchMap2);
    LIB_FUNCTION("DGMG3JshrZU", "libkernel", 1, "libkernel", sceKernelSetVirtualRangeName);
    LIB_FUNCTION("mkgXxsoxWHg", "libkernel", 1, "libkernel", sceKernelClearVirtualRangeName);
    LIB_FUNCTION("n1-v6FgU7MQ", "libkernel", 1, "libkernel", sceKernelConfiguredFlexibleMemorySize);

    LIB_FUNCTION("vSMAm3cxYTY", "libkernel", 1, "libkernel", sceKernelMprotect);
    LIB_FUNCTION("YQOfxL4QfeU", "libkernel", 1, "libkernel", posix_mprotect);
    LIB_FUNCTION("YQOfxL4QfeU", "libScePosix", 1, "libkernel", posix_mprotect);
    LIB_FUNCTION("9bfdLIyuwCY", "libkernel", 1, "libkernel", sceKernelMtypeprotect);

    // Memory pool
    LIB_FUNCTION("qCSfqDILlns", "libkernel", 1, "libkernel", sceKernelMemoryPoolExpand);
    LIB_FUNCTION_WITH_CALLER("pU-QydtGcGY", "libkernel", 1, "libkernel",
                             MemoryPoolReserveFromGuest);
    LIB_FUNCTION("Vzl66WmfLvk", "libkernel", 1, "libkernel", sceKernelMemoryPoolCommit);
    LIB_FUNCTION("LXo1tpFqJGs", "libkernel", 1, "libkernel", sceKernelMemoryPoolDecommit);
    LIB_FUNCTION("YN878uKRBbE", "libkernel", 1, "libkernel", sceKernelMemoryPoolBatch);
    LIB_FUNCTION("bvD+95Q6asU", "libkernel", 1, "libkernel", sceKernelMemoryPoolGetBlockStats);

    LIB_FUNCTION("BPE9s9vQQXo", "libkernel", 1, "libkernel", posix_mmap);
    LIB_FUNCTION("BPE9s9vQQXo", "libScePosix", 1, "libkernel", posix_mmap);
    LIB_FUNCTION("UqDGjXA5yUM", "libkernel", 1, "libkernel", posix_munmap);
    LIB_FUNCTION("UqDGjXA5yUM", "libScePosix", 1, "libkernel", posix_munmap);
    LIB_FUNCTION("3k6kx-zOOSQ", "libkernel", 1, "libkernel", sceKernelMlock);
    LIB_FUNCTION("mTBZfEal2Bw", "libkernel", 1, "libkernel", posix_mlock);
    LIB_FUNCTION("mTBZfEal2Bw", "libScePosix", 1, "libkernel", posix_mlock);
    LIB_FUNCTION("xQIIfJ860sk", "libkernel", 1, "libkernel", sceKernelMunlock);
    LIB_FUNCTION("OG4RsDwLguo", "libkernel", 1, "libkernel", posix_munlock);
    LIB_FUNCTION("OG4RsDwLguo", "libScePosix", 1, "libkernel", posix_munlock);
    LIB_FUNCTION("teiItL2boFw", "libkernel", 1, "libkernel", sceKernelReleaseFlexibleMemory);
    LIB_FUNCTION("tZY4+SZNFhA", "libkernel", 1, "libkernel", posix_msync);
    LIB_FUNCTION("tZY4+SZNFhA", "libScePosix", 1, "libkernel", posix_msync);

    // PRT memory management
    LIB_FUNCTION("BohYr-F7-is", "libkernel", 1, "libkernel", sceKernelSetPrtAperture);
    LIB_FUNCTION("L0v2Go5jOuM", "libkernel", 1, "libkernel", sceKernelGetPrtAperture);
}

} // namespace Libraries::Kernel
