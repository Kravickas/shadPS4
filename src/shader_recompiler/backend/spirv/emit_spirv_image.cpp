// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

struct ImageOperands {
    void Add(spv::ImageOperandsMask new_mask, Id value) {
        if (!Sirit::ValidId(value)) {
            return;
        }
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value);
    }
    void Add(spv::ImageOperandsMask new_mask, Id value1, Id value2) {
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value1);
        operands.push_back(value2);
    }

    void AddOffset(EmitContext& ctx, const IR::Value& offset,
                   bool can_use_runtime_offsets = false) {
        if (offset.IsEmpty()) {
            return;
        }
        if (offset.IsImmediate()) {
            const s32 operand = offset.U32();
            Add(spv::ImageOperandsMask::ConstOffset, ctx.ConstS32(operand));
            return;
        }
        IR::Inst* const inst{offset.InstRecursive()};
        if (inst->AreAllArgsImmediates()) {
            switch (inst->GetOpcode()) {
            case IR::Opcode::CompositeConstructU32x2:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32())));
                return;
            case IR::Opcode::CompositeConstructU32x3:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32()),
                                 static_cast<s32>(inst->Arg(2).U32())));
                return;
            default:
                break;
            }
        }
        if (can_use_runtime_offsets) {
            Add(spv::ImageOperandsMask::Offset, ctx.Def(offset));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Runtime offset provided to unsupported image sample instruction");
        }
    }

    void AddDerivatives(EmitContext& ctx, Id derivatives_dx, Id derivatives_dy) {
        if (!Sirit::ValidId(derivatives_dx) || !Sirit::ValidId(derivatives_dy)) {
            return;
        }
        Add(spv::ImageOperandsMask::Grad, derivatives_dx, derivatives_dy);
    }

    spv::ImageOperandsMask mask{};
    boost::container::static_vector<Id, 4> operands;
};

Id EmitBindlessImageIndex(EmitContext& ctx, u32 table_binding, Id index, u32 sampler_binding) {
    // This opcode is consumed by GetImage() — it never produces independent SPIR-V.
    // GetImage() reads the BindlessImageIndex instruction's args directly from the IR.
    // Return the index so the Invoke template has a valid Id to store.
    return index;
}

Id EmitImageSampleRaw(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id address1,
                      Id address2, Id address3, Id address4) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                              Id bias, const IR::Value& offset) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(4);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleImplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return tex.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                              Id lod, const IR::Value& offset) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(4);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return tex.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle,
                                  Id coords, Id dref, Id bias, const IR::Value& offset) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(1);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefImplicitLod(result_type, sampled_image, coords, dref,
                                                       operands.mask, operands.operands);
    const Id sample_typed = tex.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle,
                                  Id coords, Id dref, Id lod, const IR::Value& offset) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(1);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefExplicitLod(result_type, sampled_image, coords, dref,
                                                       operands.mask, operands.operands);
    const Id sample_typed = tex.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageGather(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                   const IR::Value& offset) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(4);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    const u32 comp = inst->Flags<IR::TextureInstInfo>().gather_comp.Value();
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = ctx.OpImageGather(result_type, sampled_image, coords, ctx.ConstU32(comp),
                                        operands.mask, operands.operands);
    return tex.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageGatherDref(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                       const IR::Value& offset, Id dref) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(4);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = ctx.OpImageDrefGather(result_type, sampled_image, coords, dref, operands.mask,
                                            operands.operands);
    return tex.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageQueryDimensions(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id lod,
                            bool has_mips) {
    const auto tex = ctx.GetImage(handle);
    const Id image = tex.id;
    const Id zero = ctx.u32_zero_value;
    const auto mips{[&] { return has_mips ? ctx.OpImageQueryLevels(ctx.U32[1], image) : zero; }};
    const bool uses_lod{tex.view_type != AmdGpu::ImageType::Color2DMsaa && !tex.is_storage};
    const auto query{[&](Id type) {
        return uses_lod ? ctx.OpImageQuerySizeLod(type, image, lod)
                        : ctx.OpImageQuerySize(type, image);
    }};
    switch (tex.view_type) {
    case AmdGpu::ImageType::Color1D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[1]), zero, zero, mips());
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[2]), zero, mips());
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color3D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[3]), mips());
    default:
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

Id EmitImageQueryLod(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords) {
    const auto tex = ctx.GetImage(handle);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    return ctx.OpImageQueryLod(ctx.F32[2], sampled_image, coords);
}

Id EmitImageGradient(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                     Id derivatives_dx, Id derivatives_dy, const IR::Value& offset,
                     const IR::Value& lod_clamp) {
    const auto tex = ctx.GetImage(handle);
    const Id result_type = tex.data_types->Get(4);
    const Id sampler = ctx.GetSampler(handle);
    const Id sampled_image = ctx.OpSampledImage(tex.sampled_type, tex.id, sampler);
    ImageOperands operands;
    operands.AddDerivatives(ctx, derivatives_dx, derivatives_dy);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return tex.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageRead(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords, Id lod,
                 Id ms) {
    const auto tex = ctx.GetImage(handle);
    const Id color_type = tex.data_types->Get(4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Sample, ms);
    Id texel;
    if (!tex.is_storage) {
        operands.Add(spv::ImageOperandsMask::Lod, lod);
        texel = ctx.OpImageFetch(color_type, tex.id, coords, operands.mask, operands.operands);
    } else {
        Id image_ptr = tex.var_id;
        if (ctx.profile.supports_image_load_store_lod) {
            operands.Add(spv::ImageOperandsMask::Lod, lod);
        } else if (Sirit::ValidId(lod)) {
            UNREACHABLE_MSG("Unsupported ImageRead with Lod");
        }
        const Id image = Sirit::ValidId(image_ptr) ? ctx.OpLoad(tex.image_type, image_ptr) : tex.id;
        texel = ctx.OpImageRead(color_type, image, coords, operands.mask, operands.operands);
    }
    return tex.is_integer ? ctx.OpBitcast(ctx.F32[4], texel) : texel;
}

void EmitImageWrite(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords, Id lod,
                    Id ms, Id color) {
    const auto tex = ctx.GetImage(handle);
    Id image_ptr = tex.var_id;
    const Id color_type = tex.data_types->Get(4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Sample, ms);
    if (ctx.profile.supports_image_load_store_lod) {
        operands.Add(spv::ImageOperandsMask::Lod, lod);
    } else if (Sirit::ValidId(lod)) {
        LOG_WARNING(Render, "Fallback for ImageWrite with LOD");
        ASSERT(tex.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
        const Id single_image_ptr_type =
            ctx.TypePointer(spv::StorageClass::UniformConstant, tex.image_type);
        image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
    }
    const Id image = Sirit::ValidId(image_ptr) ? ctx.OpLoad(tex.image_type, image_ptr) : tex.id;
    const Id texel = tex.is_integer ? ctx.OpBitcast(color_type, color) : color;
    ctx.OpImageWrite(image, coords, texel, operands.mask, operands.operands);
}

Id EmitImageAtomicIAdd32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicIAdd(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicSMin32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicSMin(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicUMin32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicUMin(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicSMax32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicSMax(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicUMax32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicUMax(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicFMax32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_f32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpBitcast(ctx.F32[1],
                         ctx.OpAtomicUMax(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value,
                                          ctx.OpBitcast(ctx.U32[1], value)));
}

Id EmitImageAtomicFMin32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                         Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_f32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpBitcast(ctx.F32[1],
                         ctx.OpAtomicUMin(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value,
                                          ctx.OpBitcast(ctx.U32[1], value)));
}

Id EmitImageAtomicInc32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                        Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicIIncrement(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value);
}

Id EmitImageAtomicDec32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                        Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicIDecrement(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value);
}

Id EmitImageAtomicAnd32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                        Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicAnd(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicOr32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                       Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicOr(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicXor32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                        Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicXor(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicExchange32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id coords,
                             Id value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, coords, ctx.u32_zero_value);
    return ctx.OpAtomicExchange(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value, value);
}

Id EmitImageAtomicCmpSwap32(EmitContext& ctx, IR::Inst* inst, const IR::Value& handle, Id address,
                            Id value, Id cmp_value) {
    const auto tex = ctx.GetImage(handle);
    const Id pointer =
        ctx.OpImageTexelPointer(ctx.image_u32, tex.var_id, address, ctx.u32_zero_value);
    return ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, ctx.ConstU32(1u), ctx.u32_zero_value,
                                       ctx.u32_zero_value, value, cmp_value);
}

Id EmitCubeFaceIndex(EmitContext& ctx, IR::Inst* inst, Id cube_coords) {
    if (ctx.profile.supports_native_cube_calc) {
        return ctx.OpCubeFaceIndexAMD(ctx.F32[1], cube_coords);
    } else {
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

} // namespace Shader::Backend::SPIRV
