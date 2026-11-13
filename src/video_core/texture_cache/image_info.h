// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/tile.h"
#include "video_core/texture_cache/types.h"

namespace AmdGpu {
struct ColorBuffer;
struct DepthBuffer;
} // namespace AmdGpu

namespace Libraries::VideoOut {
struct BufferAttributeGroup;
}

namespace Shader {
struct ImageResource;
}

namespace VideoCore {

struct ImageProperties {
    u32 is_volume : 1;
    u32 is_tiled : 1;
    u32 is_pow2 : 1;
    u32 is_block : 1;
    u32 is_depth : 1;
    u32 has_stencil : 1;
};

struct ImageInfo {
    ImageInfo() = default;
    ImageInfo(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address) noexcept;
    ImageInfo(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint) noexcept;
    ImageInfo(const AmdGpu::DepthBuffer& buffer, u32 num_slices, VAddr htile_address,
              AmdGpu::CbDbExtent hint, bool write_buffer = false) noexcept;
    ImageInfo(const AmdGpu::Image& sharp, const Shader::ImageResource& desc) noexcept;

    bool IsTiled() const {
        return tile_mode != AmdGpu::TileMode::DisplayLinearAligned;
    }

    Extent2D BlockDim() const {
        const auto dim = props.is_block ? 2 : 0;
        return Extent2D{pitch >> dim, size.height >> dim};
    }

    // A thin-tiled single-sample colour slice addresses its texels through pitch and tile
    // parameters only; height just sizes the slice (AddrLib: sliceOffset is the only height term
    // and is zero for slice 0 without a tile split, which a 1x colour micro tile never needs).
    // Two such descriptors with the same pitch and padded base level are the same texels even when
    // one states the logical extent and the other the padded one.
    bool IsThinTiledSlice() const {
        return props.is_tiled && !props.is_depth && type == AmdGpu::ImageType::Color2D &&
               num_samples == 1 && resources.layers == 1 && size.depth == 1 &&
               AmdGpu::GetMicroTileThickness(array_mode) == 1;
    }

    bool HasSameBlockLayout(const ImageInfo& info) const {
        const auto lhs = BlockDim();
        const auto rhs = info.BlockDim();
        if (lhs == rhs) {
            return true;
        }
        if (!IsThinTiledSlice() || !info.IsThinTiledSlice() || lhs.width != rhs.width ||
            (resources.levels != 1 && info.resources.levels != 1)) {
            return false;
        }
        const auto lhs_dim = props.is_block ? 2 : 0;
        const auto rhs_dim = info.props.is_block ? 2 : 0;
        return (mips_layout[0].height >> lhs_dim) == (info.mips_layout[0].height >> rhs_dim);
    }

    bool HasSameLevelLayout(const ImageInfo& info, s32 mip) const {
        return resources.levels == 1 && IsThinTiledSlice() && info.IsThinTiledSlice() &&
               !props.is_block && !info.props.is_block &&
               mips_layout[0].pitch == info.mips_layout[mip].pitch &&
               mips_layout[0].height == info.mips_layout[mip].height;
    }

    s32 MipOf(const ImageInfo& info) const;
    s32 SliceOf(const ImageInfo& info, s32 mip) const;

    bool IsCompatible(const ImageInfo& info) const;

    bool IsSubrectOf(const ImageInfo& info) const {
        return (size.width < info.size.width || size.height < info.size.height) &&
               size.width <= info.size.width && size.height <= info.size.height &&
               size.depth == 1 && info.size.depth == 1 && guest_address == info.guest_address &&
               guest_size <= info.guest_size && type == AmdGpu::ImageType::Color2D &&
               type == info.type && resources.levels == 1 && resources.layers == 1 &&
               info.resources.levels == 1 && info.resources.layers == 1 && !props.is_depth &&
               !info.props.is_depth && !props.is_block && !info.props.is_block &&
               pixel_format == info.pixel_format && num_bits == info.num_bits &&
               num_samples == info.num_samples && pitch == info.pitch &&
               tile_mode == info.tile_mode && bank_swizzle == info.bank_swizzle &&
               alt_tile == info.alt_tile && mips_layout[0].pitch == info.mips_layout[0].pitch;
    }
    void UpdateSize() noexcept;

    struct {
        VAddr cmask_addr;
        VAddr fmask_addr;
        VAddr htile_addr;
        s32 htile_clear_mask = -1;
    } meta_info{};

    ImageProperties props;
    vk::Format pixel_format;
    AmdGpu::ImageType type;
    SubresourceExtent resources;
    Extent3D size;
    u32 num_bits;
    u32 num_samples;
    u32 pitch;
    AmdGpu::TileMode tile_mode;
    AmdGpu::ArrayMode array_mode;
    std::array<MipInfo, MAX_MIPS> mips_layout;
    u32 micro_tiled_mips{};
    VAddr guest_address;
    u32 guest_size;
    u8 bank_swizzle{};
    bool alt_tile{};
    VAddr stencil_addr{};
    u32 stencil_size{};
};

} // namespace VideoCore
