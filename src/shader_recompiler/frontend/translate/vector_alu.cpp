// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/opcodes.h"
#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/attribute.h"
#include "shader_recompiler/profile.h"

namespace Shader::Gcn {

void Translator::EmitVectorAlu(const GcnInst& inst) {
    switch (inst.opcode) {
        // VOP2
    case Opcode::V_CNDMASK_B32:
        return V_CNDMASK_B32(inst);
    case Opcode::V_READLANE_B32:
        return V_READLANE_B32(inst);
    case Opcode::V_WRITELANE_B32:
        return V_WRITELANE_B32(inst);
    case Opcode::V_ADD_F32:
        return V_ADD_F32(inst);
    case Opcode::V_SUB_F32:
        return V_SUB_F32(inst);
    case Opcode::V_SUBREV_F32:
        return V_SUBREV_F32(inst);
    case Opcode::V_MAC_LEGACY_F32:
        return V_MAC_LEGACY_F32(inst);
    case Opcode::V_MUL_LEGACY_F32:
        return V_MUL_LEGACY_F32(inst);
    case Opcode::V_MUL_F32:
        return V_MUL_F32(inst);
    case Opcode::V_MUL_I32_I24:
        return V_MUL_I32_I24(inst, true);
    case Opcode::V_MUL_U32_U24:
        return V_MUL_I32_I24(inst, false);
    case Opcode::V_MIN_LEGACY_F32:
        return V_MIN_F32(inst, true);
    case Opcode::V_MAX_LEGACY_F32:
        return V_MAX_F32(inst, true);
    case Opcode::V_MIN_F32:
        return V_MIN_F32(inst, false);
    case Opcode::V_MAX_F32:
        return V_MAX_F32(inst);
    case Opcode::V_MIN_I32:
        return V_MIN_I32(inst);
    case Opcode::V_MAX_I32:
        return V_MAX_U32(true, inst);
    case Opcode::V_MIN_U32:
        return V_MIN_U32(inst);
    case Opcode::V_MAX_U32:
        return V_MAX_U32(false, inst);
    case Opcode::V_LSHR_B32:
        return V_LSHR_B32(inst);
    case Opcode::V_LSHRREV_B32:
        return V_LSHRREV_B32(inst);
    case Opcode::V_ASHR_I32:
        return V_ASHR_I32(inst);
    case Opcode::V_ASHRREV_I32:
        return V_ASHRREV_I32(inst);
    case Opcode::V_LSHL_B32:
        return V_LSHL_B32(inst);
    case Opcode::V_LSHLREV_B32:
        return V_LSHLREV_B32(inst);
    case Opcode::V_AND_B32:
        return V_AND_B32(inst);
    case Opcode::V_OR_B32:
        return V_OR_B32(false, inst);
    case Opcode::V_XOR_B32:
        return V_OR_B32(true, inst);
    case Opcode::V_BFM_B32:
        return V_BFM_B32(inst);
    case Opcode::V_MAC_F32:
        return V_MAC_F32(inst);
    case Opcode::V_MADMK_F32:
        return V_MADMK_F32(inst);
    case Opcode::V_MADAK_F32:
        return V_FMA_F32(inst);
    case Opcode::V_BCNT_U32_B32:
        return V_BCNT_U32_B32(inst);
    case Opcode::V_MBCNT_LO_U32_B32:
        return V_MBCNT_U32_B32(false, inst);
    case Opcode::V_MBCNT_HI_U32_B32:
        return V_MBCNT_U32_B32(true, inst);
    case Opcode::V_ADD_I32:
        return V_ADD_I32(inst);
    case Opcode::V_SUB_I32:
        return V_SUB_I32(inst);
    case Opcode::V_SUBREV_I32:
        return V_SUBREV_I32(inst);
    case Opcode::V_ADDC_U32:
        return V_ADDC_U32(inst);
    case Opcode::V_SUBB_U32:
        return V_SUBB_U32(inst);
    case Opcode::V_SUBBREV_U32:
        return V_SUBBREV_U32(inst);
    case Opcode::V_LDEXP_F32:
        return V_LDEXP_F32(inst);
    case Opcode::V_CVT_PKNORM_U16_F32:
        return V_CVT_PKNORM_U16_F32(inst);
    case Opcode::V_CVT_PKNORM_I16_F32:
        return V_CVT_PKNORM_I16_F32(inst);
    case Opcode::V_CVT_PKRTZ_F16_F32:
        return V_CVT_PKRTZ_F16_F32(inst);
    case Opcode::V_ADD_F16:
        return V_ADD_F16(inst);
    case Opcode::V_SUB_F16:
        return V_SUB_F16(inst);
    case Opcode::V_MUL_F16:
        return V_MUL_F16(inst);
    case Opcode::V_MAX_F16:
        return V_MAX_F16(inst);
    case Opcode::V_MIN_F16:
        return V_MIN_F16(inst);

        // VOP1
    case Opcode::V_MOV_B32:
        return V_MOV(inst);
    case Opcode::V_READFIRSTLANE_B32:
        return V_READFIRSTLANE_B32(inst);
    case Opcode::V_CVT_I32_F64:
        return V_CVT_I32_F64(inst);
    case Opcode::V_CVT_F64_I32:
        return V_CVT_F64_I32(inst);
    case Opcode::V_CVT_F64_U32:
        return V_CVT_F64_U32(inst);
    case Opcode::V_CVT_F32_I32:
        return V_CVT_F32_I32(inst);
    case Opcode::V_CVT_F32_U32:
        return V_CVT_F32_U32(inst);
    case Opcode::V_CVT_U32_F32:
        return V_CVT_U32_F32(inst);
    case Opcode::V_CVT_I32_F32:
        return V_CVT_I32_F32(inst);
    case Opcode::V_CVT_F16_F32:
        return V_CVT_F16_F32(inst);
    case Opcode::V_CVT_F32_F16:
        return V_CVT_F32_F16(inst);
    case Opcode::V_CVT_FLR_I32_F32:
        return V_CVT_FLR_I32_F32(inst);
    case Opcode::V_CVT_F32_F64:
        return V_CVT_F32_F64(inst);
    case Opcode::V_CVT_F64_F32:
        return V_CVT_F64_F32(inst);
    case Opcode::V_CVT_RPI_I32_F32:
        return V_CVT_RPI_I32_F32(inst);
    case Opcode::V_CVT_OFF_F32_I4:
        return V_CVT_OFF_F32_I4(inst);
    case Opcode::V_CVT_F32_UBYTE0:
        return V_CVT_F32_UBYTE(0, inst);
    case Opcode::V_CVT_F32_UBYTE1:
        return V_CVT_F32_UBYTE(1, inst);
    case Opcode::V_CVT_F32_UBYTE2:
        return V_CVT_F32_UBYTE(2, inst);
    case Opcode::V_CVT_F32_UBYTE3:
        return V_CVT_F32_UBYTE(3, inst);
    case Opcode::V_TRUNC_F64:
        return V_TRUNC_F64(inst);
    case Opcode::V_FLOOR_F64:
        return V_FLOOR_F64(inst);
    case Opcode::V_FRACT_F32:
        return V_FRACT_F32(inst);
    case Opcode::V_TRUNC_F32:
        return V_TRUNC_F32(inst);
    case Opcode::V_CEIL_F32:
        return V_CEIL_F32(inst);
    case Opcode::V_RNDNE_F32:
        return V_RNDNE_F32(inst);
    case Opcode::V_FLOOR_F32:
        return V_FLOOR_F32(inst);
    case Opcode::V_EXP_F32:
        return V_EXP_F32(inst);
    case Opcode::V_LOG_CLAMP_F32:
        return V_LOG_F32(inst);
    case Opcode::V_LOG_F32:
        return V_LOG_F32(inst);
    case Opcode::V_RCP_F32:
        return V_RCP_F32(inst);
    case Opcode::V_RCP_LEGACY_F32:
        return V_RCP_LEGACY_F32(inst);
    case Opcode::V_RCP_F64:
        return V_RCP_F64(inst);
    case Opcode::V_RCP_IFLAG_F32:
        return V_RCP_F32(inst);
    case Opcode::V_RCP_CLAMP_F32:
        return V_RCP_F32(inst);
    case Opcode::V_RSQ_CLAMP_F32:
        return V_RSQ_F32(inst);
    case Opcode::V_RSQ_LEGACY_F32:
        return V_RSQ_F32(inst);
    case Opcode::V_RSQ_F32:
        return V_RSQ_F32(inst);
    case Opcode::V_SQRT_F32:
        return V_SQRT_F32(inst);
    case Opcode::V_SIN_F32:
        return V_SIN_F32(inst);
    case Opcode::V_COS_F32:
        return V_COS_F32(inst);
    case Opcode::V_NOT_B32:
        return V_NOT_B32(inst);
    case Opcode::V_BFREV_B32:
        return V_BFREV_B32(inst);
    case Opcode::V_FFBH_U32:
        return V_FFBH_U32(inst);
    case Opcode::V_FFBL_B32:
        return V_FFBL_B32(inst);
    case Opcode::V_FFBH_I32:
        return V_FFBH_I32(inst);
    case Opcode::V_FREXP_EXP_I32_F64:
        return V_FREXP_EXP_I32_F64(inst);
    case Opcode::V_FREXP_MANT_F64:
        return V_FREXP_MANT_F64(inst);
    case Opcode::V_FRACT_F64:
        return V_FRACT_F64(inst);
    case Opcode::V_FREXP_EXP_I32_F32:
        return V_FREXP_EXP_I32_F32(inst);
    case Opcode::V_FREXP_MANT_F32:
        return V_FREXP_MANT_F32(inst);
    case Opcode::V_MOVRELD_B32:
        return V_MOVRELD_B32(inst);
    case Opcode::V_MOVRELS_B32:
        return V_MOVRELS_B32(inst);
    case Opcode::V_MOVRELSD_B32:
        return V_MOVRELSD_B32(inst);

        // VOPC
        //     V_CMP_{OP16}_F32
    case Opcode::V_CMP_F_F32:
        return V_CMP_F32(ConditionOp::F, true, false, inst);
    case Opcode::V_CMP_LT_F32:
        return V_CMP_F32(ConditionOp::LT, true, false, inst);
    case Opcode::V_CMP_EQ_F32:
        return V_CMP_F32(ConditionOp::EQ, true, false, inst);
    case Opcode::V_CMP_LE_F32:
        return V_CMP_F32(ConditionOp::LE, true, false, inst);
    case Opcode::V_CMP_GT_F32:
        return V_CMP_F32(ConditionOp::GT, true, false, inst);
    case Opcode::V_CMP_LG_F32:
        return V_CMP_F32(ConditionOp::LG, true, false, inst);
    case Opcode::V_CMP_GE_F32:
        return V_CMP_F32(ConditionOp::GE, true, false, inst);
    case Opcode::V_CMP_U_F32:
        return V_CMP_F32(ConditionOp::U, true, false, inst);
    case Opcode::V_CMP_NGE_F32:
        return V_CMP_F32(ConditionOp::LT, false, false, inst);
    case Opcode::V_CMP_NGT_F32:
        return V_CMP_F32(ConditionOp::LE, false, false, inst);
    case Opcode::V_CMP_NLE_F32:
        return V_CMP_F32(ConditionOp::GT, false, false, inst);
    case Opcode::V_CMP_NEQ_F32:
        return V_CMP_F32(ConditionOp::LG, false, false, inst);
    case Opcode::V_CMP_NLT_F32:
        return V_CMP_F32(ConditionOp::GE, false, false, inst);

        //     V_CMPX_{OP16}_F32
    case Opcode::V_CMPX_F_F32:
        return V_CMP_F32(ConditionOp::F, true, true, inst);
    case Opcode::V_CMPX_LT_F32:
        return V_CMP_F32(ConditionOp::LT, true, true, inst);
    case Opcode::V_CMPX_EQ_F32:
        return V_CMP_F32(ConditionOp::EQ, true, true, inst);
    case Opcode::V_CMPX_LE_F32:
        return V_CMP_F32(ConditionOp::LE, true, true, inst);
    case Opcode::V_CMPX_GT_F32:
        return V_CMP_F32(ConditionOp::GT, true, true, inst);
    case Opcode::V_CMPX_LG_F32:
        return V_CMP_F32(ConditionOp::LG, true, true, inst);
    case Opcode::V_CMPX_GE_F32:
        return V_CMP_F32(ConditionOp::GE, true, true, inst);
    case Opcode::V_CMPX_NGE_F32:
        return V_CMP_F32(ConditionOp::LT, false, true, inst);
    case Opcode::V_CMPX_NLG_F32:
        return V_CMP_F32(ConditionOp::EQ, false, true, inst);
    case Opcode::V_CMPX_NGT_F32:
        return V_CMP_F32(ConditionOp::LE, false, true, inst);
    case Opcode::V_CMPX_NLE_F32:
        return V_CMP_F32(ConditionOp::GT, false, true, inst);
    case Opcode::V_CMPX_NEQ_F32:
        return V_CMP_F32(ConditionOp::LG, false, true, inst);
    case Opcode::V_CMPX_NLT_F32:
        return V_CMP_F32(ConditionOp::GE, false, true, inst);

        //     V_CMP_{OP16}_F64
    case Opcode::V_CMP_F_F64:
        return V_CMP_F64(ConditionOp::F, true, false, inst);
    case Opcode::V_CMP_LT_F64:
        return V_CMP_F64(ConditionOp::LT, true, false, inst);
    case Opcode::V_CMP_EQ_F64:
        return V_CMP_F64(ConditionOp::EQ, true, false, inst);
    case Opcode::V_CMP_LE_F64:
        return V_CMP_F64(ConditionOp::LE, true, false, inst);
    case Opcode::V_CMP_GT_F64:
        return V_CMP_F64(ConditionOp::GT, true, false, inst);
    case Opcode::V_CMP_LG_F64:
        return V_CMP_F64(ConditionOp::LG, true, false, inst);
    case Opcode::V_CMP_GE_F64:
        return V_CMP_F64(ConditionOp::GE, true, false, inst);
    case Opcode::V_CMP_U_F64:
        return V_CMP_F64(ConditionOp::U, true, false, inst);
    case Opcode::V_CMP_NGE_F64:
        return V_CMP_F64(ConditionOp::LT, false, false, inst);
    case Opcode::V_CMP_NGT_F64:
        return V_CMP_F64(ConditionOp::LE, false, false, inst);
    case Opcode::V_CMP_NLE_F64:
        return V_CMP_F64(ConditionOp::GT, false, false, inst);
    case Opcode::V_CMP_NEQ_F64:
        return V_CMP_F64(ConditionOp::LG, false, false, inst);
    case Opcode::V_CMP_NLT_F64:
        return V_CMP_F64(ConditionOp::GE, false, false, inst);

        //     V_CMP_{OP8}_I32
    case Opcode::V_CMP_LT_I32:
        return V_CMP_U32(ConditionOp::LT, true, false, inst);
    case Opcode::V_CMP_EQ_I32:
        return V_CMP_U32(ConditionOp::EQ, true, false, inst);
    case Opcode::V_CMP_LE_I32:
        return V_CMP_U32(ConditionOp::LE, true, false, inst);
    case Opcode::V_CMP_GT_I32:
        return V_CMP_U32(ConditionOp::GT, true, false, inst);
    case Opcode::V_CMP_NE_I32:
        return V_CMP_U32(ConditionOp::LG, true, false, inst);
    case Opcode::V_CMP_GE_I32:
        return V_CMP_U32(ConditionOp::GE, true, false, inst);
    case Opcode::V_CMPX_LE_I32:
        return V_CMP_U32(ConditionOp::LE, true, true, inst);

        //     V_CMPX_{OP8}_I32
    case Opcode::V_CMPX_LT_I32:
        return V_CMP_U32(ConditionOp::LT, true, true, inst);
    case Opcode::V_CMPX_EQ_I32:
        return V_CMP_U32(ConditionOp::EQ, true, true, inst);
    case Opcode::V_CMPX_GT_I32:
        return V_CMP_U32(ConditionOp::GT, true, true, inst);
    case Opcode::V_CMPX_LG_I32:
        return V_CMP_U32(ConditionOp::LG, true, true, inst);
    case Opcode::V_CMPX_GE_I32:
        return V_CMP_U32(ConditionOp::GE, true, true, inst);

        //     V_CMP_{OP8}_U32
    case Opcode::V_CMP_F_U32:
        return V_CMP_U32(ConditionOp::F, false, false, inst);
    case Opcode::V_CMP_LT_U32:
        return V_CMP_U32(ConditionOp::LT, false, false, inst);
    case Opcode::V_CMP_EQ_U32:
        return V_CMP_U32(ConditionOp::EQ, false, false, inst);
    case Opcode::V_CMP_LE_U32:
        return V_CMP_U32(ConditionOp::LE, false, false, inst);
    case Opcode::V_CMP_GT_U32:
        return V_CMP_U32(ConditionOp::GT, false, false, inst);
    case Opcode::V_CMP_NE_U32:
        return V_CMP_U32(ConditionOp::LG, false, false, inst);
    case Opcode::V_CMP_GE_U32:
        return V_CMP_U32(ConditionOp::GE, false, false, inst);
    case Opcode::V_CMP_TRU_U32:
        return V_CMP_U32(ConditionOp::TRU, false, false, inst);

        //     V_CMPX_{OP8}_U32
    case Opcode::V_CMPX_F_U32:
        return V_CMP_U32(ConditionOp::F, false, true, inst);
    case Opcode::V_CMPX_LT_U32:
        return V_CMP_U32(ConditionOp::LT, false, true, inst);
    case Opcode::V_CMPX_EQ_U32:
        return V_CMP_U32(ConditionOp::EQ, false, true, inst);
    case Opcode::V_CMPX_LE_U32:
        return V_CMP_U32(ConditionOp::LE, false, true, inst);
    case Opcode::V_CMPX_GT_U32:
        return V_CMP_U32(ConditionOp::GT, false, true, inst);
    case Opcode::V_CMPX_NE_U32:
        return V_CMP_U32(ConditionOp::LG, false, true, inst);
    case Opcode::V_CMPX_GE_U32:
        return V_CMP_U32(ConditionOp::GE, false, true, inst);
    case Opcode::V_CMPX_TRU_U32:
        return V_CMP_U32(ConditionOp::TRU, false, true, inst);

        //     V_CMP_{OP8}_I64
    case Opcode::V_CMP_F_I64:
        return V_CMP_U64(ConditionOp::F, true, false, inst);
    case Opcode::V_CMP_LT_I64:
        return V_CMP_U64(ConditionOp::LT, true, false, inst);
    case Opcode::V_CMP_EQ_I64:
        return V_CMP_U64(ConditionOp::EQ, true, false, inst);
    case Opcode::V_CMP_LE_I64:
        return V_CMP_U64(ConditionOp::LE, true, false, inst);
    case Opcode::V_CMP_GT_I64:
        return V_CMP_U64(ConditionOp::GT, true, false, inst);
    case Opcode::V_CMP_NE_I64:
        return V_CMP_U64(ConditionOp::LG, true, false, inst);
    case Opcode::V_CMP_GE_I64:
        return V_CMP_U64(ConditionOp::GE, true, false, inst);
    case Opcode::V_CMP_TRU_I64:
        return V_CMP_U64(ConditionOp::TRU, true, false, inst);

        //     V_CMPX_{OP8}_I64
    case Opcode::V_CMPX_EQ_I64:
        return V_CMP_U64(ConditionOp::EQ, true, true, inst);

        //     V_CMP_{OP8}_U64
    case Opcode::V_CMP_EQ_U64:
        return V_CMP_U64(ConditionOp::EQ, false, false, inst);
    case Opcode::V_CMP_NE_U64:
        return V_CMP_U64(ConditionOp::LG, false, false, inst);
    case Opcode::V_CMP_GT_U64:
        return V_CMP_U64(ConditionOp::GT, false, false, inst);
    case Opcode::V_CMP_LT_U64:
        return V_CMP_U64(ConditionOp::LT, false, false, inst);

        //     V_CMPX_{OP8}_U64
    case Opcode::V_CMPX_EQ_U64:
        return V_CMP_U64(ConditionOp::EQ, false, true, inst);
    case Opcode::V_CMPX_LG_U64:
        return V_CMP_U64(ConditionOp::LG, false, true, inst);

    case Opcode::V_CMP_CLASS_F32:
        return V_CMP_CLASS_F32(inst);

        // VOP3a
    case Opcode::V_MAD_LEGACY_F32:
        return V_MAD_LEGACY_F32(inst);
    case Opcode::V_MAD_F32:
        return V_MAD_F32(inst);
    case Opcode::V_MAD_I32_I24:
        return V_MAD_I32_I24(inst);
    case Opcode::V_MAD_U32_U24:
        return V_MAD_U32_U24(inst);
    case Opcode::V_CUBEID_F32:
        return V_CUBEID_F32(inst);
    case Opcode::V_CUBESC_F32:
        return V_CUBESC_F32(inst);
    case Opcode::V_CUBETC_F32:
        return V_CUBETC_F32(inst);
    case Opcode::V_CUBEMA_F32:
        return V_CUBEMA_F32(inst);
    case Opcode::V_BFE_U32:
        return V_BFE_U32(false, inst);
    case Opcode::V_BFE_I32:
        return V_BFE_U32(true, inst);
    case Opcode::V_BFI_B32:
        return V_BFI_B32(inst);
    case Opcode::V_FMA_F32:
        return V_FMA_F32(inst);
    case Opcode::V_FMA_F64:
        return V_FMA_F64(inst);
    case Opcode::V_MIN3_F32:
        return V_MIN3_F32(inst);
    case Opcode::V_MIN3_I32:
        return V_MIN3_U32(true, inst);
    case Opcode::V_MIN3_U32:
        return V_MIN3_U32(false, inst);
    case Opcode::V_MAX3_F32:
        return V_MAX3_F32(inst);
    case Opcode::V_MAX3_I32:
        return V_MAX3_U32(true, inst);
    case Opcode::V_MAX3_U32:
        return V_MAX3_U32(false, inst);
    case Opcode::V_MED3_F32:
        return V_MED3_F32(inst);
    case Opcode::V_MED3_I32:
        return V_MED3_U32(true, inst);
    case Opcode::V_MED3_U32:
        return V_MED3_U32(false, inst);
    case Opcode::V_SAD_U32:
        return V_SAD_U32(inst);
    case Opcode::V_CVT_PK_U16_U32:
        return V_CVT_PK_U16_U32(inst);
    case Opcode::V_CVT_PK_I16_I32:
        return V_CVT_PK_I16_I32(inst);
    case Opcode::V_CVT_PK_U8_F32:
        return V_CVT_PK_U8_F32(inst);
    case Opcode::V_LSHL_B64:
        return V_LSHL_B64(inst);
    case Opcode::V_LSHR_B64:
        return V_LSHR_B64(inst);
    case Opcode::V_ASHR_I64:
        return V_ASHR_I64(inst);
    case Opcode::V_ADD_F64:
        return V_ADD_F64(inst);
    case Opcode::V_ALIGNBIT_B32:
        return V_ALIGNBIT_B32(inst);
    case Opcode::V_ALIGNBYTE_B32:
        return V_ALIGNBYTE_B32(inst);
    case Opcode::V_MUL_F64:
        return V_MUL_F64(inst);
    case Opcode::V_MIN_F64:
        return V_MIN_F64(inst);
    case Opcode::V_MAX_F64:
        return V_MAX_F64(inst);
    case Opcode::V_MUL_LO_U32:
        return V_MUL_LO_U32(inst);
    case Opcode::V_MUL_HI_U32:
        return V_MUL_HI_U32(false, inst);
    case Opcode::V_MUL_LO_I32:
        return V_MUL_LO_U32(inst);
    case Opcode::V_MUL_HI_I32:
        return V_MUL_HI_U32(true, inst);
    case Opcode::V_MAD_U64_U32:
        return V_MAD_U64_U32(inst);
    case Opcode::V_LSHRREV_B16:
        return V_LSHRREV_B16(inst);
    case Opcode::V_ASHRREV_I16:
        return V_ASHRREV_I16(inst);
    case Opcode::V_LSHLREV_B16:
        return V_LSHLREV_B16(inst);
    case Opcode::V_ADD_LSHL_U32:
        return V_ADD_LSHL_U32(inst);
    case Opcode::V_LSHL_ADD_U32:
        return V_LSHL_ADD_U32(inst);
    case Opcode::V_MIN3_F16:
        return V_MIN3_F16(inst);
    case Opcode::V_MAX3_F16:
        return V_MAX3_F16(inst);
    case Opcode::V_MED3_F16:
        return V_MED3_F16(inst);
    case Opcode::V_ADD3_U32:
        return V_ADD3_U32(inst);
    case Opcode::V_LSHL_OR_B32:
        return V_LSHL_OR_B32(inst);
    case Opcode::V_AND_OR_B32:
        return V_AND_OR_B32(inst);
    case Opcode::V_OR3_B32:
        return V_OR3_B32(inst);
    case Opcode::V_NOP:
        return;

    // VOP3P
    case Opcode::V_PK_MUL_LO_U16:
        return V_PK_MUL_LO_U16(inst);
    case Opcode::V_PK_ADD_I16:
        return V_PK_ADD_I16(inst);
    case Opcode::V_PK_SUB_I16:
        return V_PK_SUB_I16(inst);
    case Opcode::V_PK_LSHRREV_B16:
        return V_PK_LSHRREV_B16(inst);
    case Opcode::V_PK_LSHLREV_B16:
        return V_PK_LSHLREV_B16(inst);
    case Opcode::V_PK_MAD_U16:
        return V_PK_MAD_U16(inst);
    case Opcode::V_PK_ADD_U16:
        return V_PK_ADD_U16(inst);
    case Opcode::V_PK_SUB_U16:
        return V_PK_SUB_U16(inst);
    case Opcode::V_PK_MAX_U16:
        return V_PK_MAX_U16(inst);
    case Opcode::V_PK_MIN_U16:
        return V_PK_MIN_U16(inst);
    case Opcode::V_PK_FMA_F16:
        return V_PK_FMA_F16(inst);
    case Opcode::V_PK_ADD_F16:
        return V_PK_ADD_F16(inst);
    case Opcode::V_PK_MUL_F16:
        return V_PK_MUL_F16(inst);
    case Opcode::V_PK_MIN_F16:
        return V_PK_MIN_F16(inst);
    case Opcode::V_PK_MAX_F16:
        return V_PK_MAX_F16(inst);
    case Opcode::V_MAD_MIX_F32:
        return V_MAD_MIX_F32(inst);
    case Opcode::V_MAD_MIXLO_F16:
        return V_MAD_MIXLO_F16(inst);
    case Opcode::V_MAD_MIXHI_F16:
        return V_MAD_MIXHI_F16(inst);

    default:
        LogMissingOpcode(inst);
    }
}

// VOP2

void Translator::V_CNDMASK_B32(const GcnInst& inst) {
    IR::U64 mask;
    if (inst.src[2].field == OperandField::Undefined) {
        mask = ir.PackUint2x32(ir.CompositeConstruct(ir.GetVccLo(), ir.GetVccHi()));
    } else {
        mask = GetSrc64(inst.src[2]);
    }
    const IR::Value result = ir.Select(ir.InverseBallot(mask), GetSrc<IR::F32>(inst.src[1]),
                                       GetSrc<IR::F32>(inst.src[0]));
    SetDst(inst.dst[0], IR::U32F32{result});
}

void Translator::V_ADD_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    SetDst(inst.dst[0], ir.FPAdd(src0, src1));
}

void Translator::V_ADD_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F64 src1{GetSrc64<IR::F64>(inst.src[1])};
    SetDst64(inst.dst[0], ir.FPAdd(src0, src1));
}

void Translator::V_SUB_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    SetDst(inst.dst[0], ir.FPSub(src0, src1));
}

void Translator::V_SUBREV_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    SetDst(inst.dst[0], ir.FPSub(src1, src0));
}

void Translator::V_MUL_F32(const GcnInst& inst) {
    SetDst(inst.dst[0], ir.FPMul(GetSrc<IR::F32>(inst.src[0]), GetSrc<IR::F32>(inst.src[1])));
}

void Translator::V_MUL_LEGACY_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    SetDst(inst.dst[0], LegacyMul(src0, src1));
}

void Translator::V_MAC_LEGACY_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 dst0{GetSrc<IR::F32>(inst.dst[0])};
    SetDst(inst.dst[0], ir.FPAdd(LegacyMul(src0, src1), dst0));
}

void Translator::V_MAD_LEGACY_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 src2{GetSrc<IR::F32>(inst.src[2])};
    SetDst(inst.dst[0], ir.FPAdd(LegacyMul(src0, src1), src2));
}

void Translator::V_MUL_I32_I24(const GcnInst& inst, bool is_signed) {
    const IR::U32 src0{
        ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    const IR::U32 src1{
        ir.BitFieldExtract(GetSrc(inst.src[1]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    SetDst(inst.dst[0], ir.IMul(src0, src1));
}

void Translator::V_MIN_F32(const GcnInst& inst, bool is_legacy) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};

    const IR::F32 fpmin = ir.FPMin(src0, src1);
    const IR::F32 result =
        is_legacy
            ? IR::F32{ir.Select(ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1)), src1, fpmin)}
            : fpmin;
    SetDst(inst.dst[0], result);
}

void Translator::V_MAX_F32(const GcnInst& inst, bool is_legacy) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};

    const IR::F32 fpmax = ir.FPMax(src0, src1);
    const IR::F32 result =
        is_legacy
            ? IR::F32{ir.Select(ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1)), src1, fpmax)}
            : fpmax;
    SetDst(inst.dst[0], result);
}

void Translator::V_MIN_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.SMin(src0, src1));
}

void Translator::V_MIN_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.IMin(src0, src1, false));
}

void Translator::V_MAX_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.IMax(src0, src1, is_signed));
}

void Translator::V_LSHR_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ShiftRightLogical(src0, ir.BitwiseAnd(src1, ir.Imm32(0x1F))));
}

void Translator::V_LSHRREV_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ShiftRightLogical(src1, ir.BitwiseAnd(src0, ir.Imm32(0x1F))));
}

void Translator::V_ASHR_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ShiftRightArithmetic(src0, ir.BitwiseAnd(src1, ir.Imm32(0x1F))));
}

void Translator::V_ASHRREV_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ShiftRightArithmetic(src1, ir.BitwiseAnd(src0, ir.Imm32(0x1F))));
}

void Translator::V_LSHL_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ShiftLeftLogical(src0, ir.BitwiseAnd(src1, ir.Imm32(0x1F))));
}

void Translator::V_LSHLREV_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ShiftLeftLogical(src1, ir.BitwiseAnd(src0, ir.Imm32(0x1F))));
}

void Translator::V_AND_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.BitwiseAnd(src0, src1));
}

void Translator::V_OR_B32(bool is_xor, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0],
           is_xor ? IR::U32{ir.BitwiseXor(src0, src1)} : IR::U32{ir.BitwiseOr(src0, src1)});
}

void Translator::V_BFM_B32(const GcnInst& inst) {
    // bitmask width, S0[4:0]
    const IR::U32 src0{ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0), ir.Imm32(5))};
    // bitmask offset, S1[4:0]
    const IR::U32 src1{ir.BitFieldExtract(GetSrc(inst.src[1]), ir.Imm32(0), ir.Imm32(5))};
    const IR::U32 ones = ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1), src0), ir.Imm32(1));
    SetDst(inst.dst[0], ir.ShiftLeftLogical(ones, src1));
}

void Translator::V_MAC_F32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc<IR::F32>(inst.src[1]);
    const auto dst0 = GetSrc<IR::F32>(inst.dst[0]);
    SetDst(inst.dst[0], ir.FPAdd(ir.FPMul(src0, src1), dst0));
}

void Translator::V_MADMK_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 k{GetSrc<IR::F32>(inst.src[2])};
    SetDst(inst.dst[0], ir.FPAdd(ir.FPMul(src0, k), src1));
}

void Translator::V_BCNT_U32_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.IAdd(ir.BitCount(src0), src1));
}

void Translator::V_MBCNT_U32_B32(bool hi, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.MaskedBitCount(src0, src1, hi));
}

void Translator::V_ADD_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.IAdd(src0, src1)};
    SetDst(inst.dst[0], result);

    SetCarryOut(inst, ir.ILessThan(result, src0, false));
}

void Translator::V_SUB_I32(const GcnInst& inst) {
    // Unsigned components
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.ISub(src0, src1)};
    SetDst(inst.dst[0], result);

    const IR::U1 did_underflow{ir.IGreaterThan(src1, src0, false)};
    SetCarryOut(inst, did_underflow);
}

void Translator::V_SUBREV_I32(const GcnInst& inst) {
    // Unsigned components
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.ISub(src1, src0)};
    SetDst(inst.dst[0], result);

    const IR::U1 did_underflow{ir.IGreaterThan(src0, src1, false)};
    SetCarryOut(inst, did_underflow);
}

void Translator::V_ADDC_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{GetCarryIn(inst)};
    const IR::U32 result1{ir.IAdd(src0, src1)};
    const IR::U32 result2{ir.IAdd(result1, carry)};
    const IR::U1 carry_out1{ir.ILessThan(result1, src0, false)};
    const IR::U1 carry_out2{ir.ILessThan(result2, result1, false)};
    SetDst(inst.dst[0], result2);

    const IR::U1 did_overflow{ir.LogicalOr(carry_out1, carry_out2)};
    SetCarryOut(inst, did_overflow);
}

void Translator::V_SUBB_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{GetCarryIn(inst)};
    const IR::U32 result{ir.ISub(ir.ISub(src0, src1), carry)};
    SetDst(inst.dst[0], result);

    const IR::U1 underflow{ir.IGreaterThan(src1, src0, false)};
    const IR::U32 difference{ir.ISub(src0, src1)};
    const IR::U1 borrow_underflow{ir.IGreaterThan(carry, difference, false)};
    SetCarryOut(inst, ir.LogicalOr(underflow, borrow_underflow));
}

void Translator::V_SUBBREV_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{GetCarryIn(inst)};
    const IR::U32 result{ir.ISub(ir.ISub(src1, src0), carry)};
    SetDst(inst.dst[0], result);

    const IR::U1 underflow{ir.IGreaterThan(src0, src1, false)};
    const IR::U32 difference{ir.ISub(src1, src0)};
    const IR::U1 borrow_underflow{ir.IGreaterThan(carry, difference, false)};
    SetCarryOut(inst, ir.LogicalOr(underflow, borrow_underflow));
}

void Translator::V_LDEXP_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.FPLdexp(src0, src1));
}

namespace {

bool FlushesDenormInput(AmdGpu::FpDenormMode mode) {
    return mode == AmdGpu::FpDenormMode::InOutFlush ||
           mode == AmdGpu::FpDenormMode::InFlushOutAllow;
}

bool FlushesDenormOutput(AmdGpu::FpDenormMode mode) {
    return mode == AmdGpu::FpDenormMode::InOutFlush ||
           mode == AmdGpu::FpDenormMode::InAllowOutFlush;
}

IR::F32 FlushDenorm32(IR::IREmitter& ir, const IR::F32& value) {
    const IR::U32 bits{ir.BitCast<IR::U32>(value)};
    const IR::U1 is_denorm{ir.IEqual(ir.BitwiseAnd(bits, ir.Imm32(0x7F800000U)), ir.Imm32(0U))};
    const IR::U32 zero{ir.BitwiseAnd(bits, ir.Imm32(0x80000000U))};
    return ir.BitCast<IR::F32>(IR::U32{ir.Select(is_denorm, zero, bits)});
}

// value is integral or infinite; NaN source saturates by sign.
IR::U32 SaturateToI32(IR::IREmitter& ir, const IR::F32& src, const IR::F32& value) {
    const IR::U32 converted{ir.ConvertFToS(32, value)};
    const IR::U32 high{ir.Select(ir.FPGreaterThanEqual(value, ir.Imm32(2147483648.f)),
                                 ir.Imm32(0x7FFFFFFFU), converted)};
    const IR::U32 clamped{
        ir.Select(ir.FPLessThan(value, ir.Imm32(-2147483648.f)), ir.Imm32(0x80000000U), high)};
    const IR::U1 negative{ir.ILessThan(ir.BitCast<IR::U32>(src), ir.Imm32(0U), true)};
    const IR::U32 nan_value{ir.Select(negative, ir.Imm32(0x80000000U), ir.Imm32(0x7FFFFFFFU))};
    return IR::U32{ir.Select(ir.FPIsNan(src), nan_value, clamped)};
}

// Low 16 bits hold the f16 result; denormals are kept, NaN payload is truncated.
IR::U32 F32ToF16Bits(IR::IREmitter& ir, const IR::U32& bits, bool round_to_zero) {
    const IR::U32 sign{ir.BitwiseAnd(ir.ShiftRightLogical(bits, ir.Imm32(16U)), ir.Imm32(0x8000U))};
    const IR::U32 abs{ir.BitwiseAnd(bits, ir.Imm32(0x7FFFFFFFU))};
    // NaN: the quiet bit set, the payload's top 10 bits kept
    const IR::U32 nan{
        ir.BitwiseOr(ir.Imm32(0x7E00U),
                     ir.BitwiseAnd(ir.ShiftRightLogical(abs, ir.Imm32(13U)), ir.Imm32(0x3FFU)))};

    const IR::U32 rebiased{ir.ISub(abs, ir.Imm32(0x38000000U))};
    IR::U32 normal;
    if (round_to_zero) {
        normal = ir.ShiftRightLogical(rebiased, ir.Imm32(13U));
    } else {
        const IR::U32 lsb{ir.BitwiseAnd(ir.ShiftRightLogical(abs, ir.Imm32(13U)), ir.Imm32(1U))};
        normal =
            ir.ShiftRightLogical(ir.IAdd(ir.IAdd(rebiased, ir.Imm32(0xFFFU)), lsb), ir.Imm32(13U));
    }

    const IR::U32 exponent{ir.ShiftRightLogical(abs, ir.Imm32(23U))};
    const IR::U32 shift{ir.UClamp(ir.ISub(ir.Imm32(126U), exponent), ir.Imm32(1U), ir.Imm32(31U))};
    const IR::U32 mantissa{
        ir.BitwiseOr(ir.BitwiseAnd(abs, ir.Imm32(0x7FFFFFU)), ir.Imm32(0x800000U))};
    const IR::U32 truncated{ir.ShiftRightLogical(mantissa, shift)};
    IR::U32 denorm{truncated};
    if (!round_to_zero) {
        const IR::U32 rem{ir.BitwiseAnd(
            mantissa, ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1U), shift), ir.Imm32(1U)))};
        const IR::U32 half{ir.ShiftLeftLogical(ir.Imm32(1U), ir.ISub(shift, ir.Imm32(1U)))};
        const IR::U1 odd{ir.INotEqual(ir.BitwiseAnd(truncated, ir.Imm32(1U)), ir.Imm32(0U))};
        const IR::U1 round_up{ir.LogicalOr(ir.IGreaterThan(rem, half, false),
                                           ir.LogicalAnd(ir.IEqual(rem, half), odd))};
        denorm = ir.IAdd(truncated, IR::U32{ir.Select(round_up, ir.Imm32(1U), ir.Imm32(0U))});
    }

    IR::U32 result{
        ir.Select(ir.IGreaterThanEqual(abs, ir.Imm32(0x38800000U), false), normal, denorm)};
    result =
        IR::U32{ir.Select(ir.ILessThan(abs, ir.Imm32(0x33000000U), false), ir.Imm32(0U), result)};
    if (round_to_zero) {
        result = IR::U32{ir.Select(ir.IGreaterThanEqual(abs, ir.Imm32(0x477FE000U), false),
                                   ir.Imm32(0x7BFFU), result)};
        result =
            IR::U32{ir.Select(ir.IEqual(abs, ir.Imm32(0x7F800000U)), ir.Imm32(0x7C00U), result)};
    } else {
        result = IR::U32{ir.Select(ir.IGreaterThanEqual(abs, ir.Imm32(0x477FF000U), false),
                                   ir.Imm32(0x7C00U), result)};
    }
    result = IR::U32{ir.Select(ir.IGreaterThan(abs, ir.Imm32(0x7F800000U), false), nan, result)};
    return ir.BitwiseOr(sign, result);
}

// Rounds clamp(value) * (2^n - 1) to nearest using the exact product.
IR::U32 PackNorm16(IR::IREmitter& ir, const IR::F32& value, bool is_signed) {
    const IR::F32 low{ir.Imm32(is_signed ? -1.f : 0.f)};
    const IR::F32 high{ir.Imm32(1.f)};
    const IR::F32 lower{ir.Select(ir.FPLessThan(value, low), low, value)};
    const IR::F32 bounded{ir.Select(ir.FPGreaterThan(lower, high), high, lower)};
    const IR::F32 clamped{ir.Select(ir.FPIsNan(value), ir.Imm32(0.f), bounded)};

    const IR::F32 product{ir.FPMul(clamped, ir.Imm32(is_signed ? 32767.f : 65535.f))};
    // clamped * 2^n is exact and close to product, so this difference is exact.
    const IR::F32 residual{
        ir.FPSub(ir.FPMul(clamped, ir.Imm32(is_signed ? 32768.f : 65536.f)), product)};

    const IR::F32 floor{ir.FPFloor(product)};
    const IR::F32 mid{ir.FPAdd(floor, ir.Imm32(0.5f))};
    const IR::U32 floor_int{ir.ConvertFToS(32, floor)};
    const IR::U1 odd{ir.INotEqual(ir.BitwiseAnd(floor_int, ir.Imm32(1U)), ir.Imm32(0U))};
    const IR::U1 tie_up{ir.LogicalOr(ir.FPGreaterThan(residual, clamped),
                                     ir.LogicalAnd(ir.FPEqual(residual, clamped), odd))};
    const IR::U1 round_up{ir.LogicalOr(ir.FPGreaterThan(product, mid),
                                       ir.LogicalAnd(ir.FPEqual(product, mid), tie_up))};
    const IR::U32 rounded{
        ir.IAdd(floor_int, IR::U32{ir.Select(round_up, ir.Imm32(1U), ir.Imm32(0U))})};
    return ir.BitwiseAnd(rounded, ir.Imm32(0xFFFFU));
}

} // Anonymous namespace

void Translator::V_CVT_PKNORM_U16_F32(const GcnInst& inst) {
    const IR::U32 lo{PackNorm16(ir, GetSrc<IR::F32>(inst.src[0]), false)};
    const IR::U32 hi{PackNorm16(ir, GetSrc<IR::F32>(inst.src[1]), false)};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, ir.ShiftLeftLogical(hi, ir.Imm32(16U))));
}

void Translator::V_CVT_PKNORM_I16_F32(const GcnInst& inst) {
    const IR::U32 lo{PackNorm16(ir, GetSrc<IR::F32>(inst.src[0]), true)};
    const IR::U32 hi{PackNorm16(ir, GetSrc<IR::F32>(inst.src[1]), true)};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, ir.ShiftLeftLogical(hi, ir.Imm32(16U))));
}

void Translator::V_CVT_PKRTZ_F16_F32(const GcnInst& inst) {
    const IR::U32 lo{F32ToF16Bits(ir, ir.BitCast<IR::U32>(GetSrc<IR::F32>(inst.src[0])), true)};
    const IR::U32 hi{F32ToF16Bits(ir, ir.BitCast<IR::U32>(GetSrc<IR::F32>(inst.src[1])), true)};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, ir.ShiftLeftLogical(hi, ir.Imm32(16U))));
}

void Translator::V_ADD_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);

    const auto result = ir.FPAdd(src0, src1);

    SetDst16(inst.dst[0], result);
}

void Translator::V_SUB_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);

    const auto result = ir.FPSub(src0, src1);

    SetDst16(inst.dst[0], result);
}

void Translator::V_MUL_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);

    const auto result = ir.FPMul(src0, src1);

    SetDst16(inst.dst[0], result);
}

void Translator::V_MAX_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);

    const auto result = ir.FPMax(src0, src1);

    SetDst16(inst.dst[0], result);
}

void Translator::V_MIN_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);

    const auto result = ir.FPMin(src0, src1);

    SetDst16(inst.dst[0], result);
}

// VOP1

void Translator::V_MOV(const GcnInst& inst) {
    SetDst(inst.dst[0], GetSrc<IR::F32>(inst.src[0]));
}

void Translator::V_CVT_I32_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::U32 converted{ir.ConvertFToS(32, src0)};
    const IR::U32 high{ir.Select(ir.FPGreaterThanEqual(src0, ir.Imm64(2147483648.0)),
                                 ir.Imm32(0x7FFFFFFFU), converted)};
    const IR::U32 clamped{
        ir.Select(ir.FPLessThanEqual(src0, ir.Imm64(-2147483649.0)), ir.Imm32(0x80000000U), high)};
    SetDst(inst.dst[0], IR::U32{ir.Select(ir.FPIsNan(src0), ir.Imm32(0U), clamped)});
}

void Translator::V_CVT_F64_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst64(inst.dst[0], ir.ConvertSToF(64, 32, src0));
}

void Translator::V_CVT_F64_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst64(inst.dst[0], ir.ConvertUToF(64, 32, src0));
}

void Translator::V_CVT_F32_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.ConvertSToF(32, 32, src0));
}

void Translator::V_CVT_F32_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.ConvertUToF(32, 32, src0));
}

void Translator::V_CVT_U32_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 converted{ir.ConvertFToU(32, src0)};
    const IR::U32 high{ir.Select(ir.FPGreaterThanEqual(src0, ir.Imm32(4294967296.f)),
                                 ir.Imm32(0xFFFFFFFFU), converted)};
    SetDst(inst.dst[0],
           IR::U32{ir.Select(ir.FPGreaterThan(src0, ir.Imm32(0.f)), high, ir.Imm32(0U))});
}

void Translator::V_CVT_I32_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 converted{ir.ConvertFToS(32, src0)};
    const IR::U32 high{ir.Select(ir.FPGreaterThanEqual(src0, ir.Imm32(2147483648.f)),
                                 ir.Imm32(0x7FFFFFFFU), converted)};
    const IR::U32 clamped{
        ir.Select(ir.FPLessThan(src0, ir.Imm32(-2147483648.f)), ir.Imm32(0x80000000U), high)};
    SetDst(inst.dst[0], IR::U32{ir.Select(ir.FPIsNan(src0), ir.Imm32(0U), clamped)});
}

void Translator::V_CVT_F16_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], F32ToF16Bits(ir, ir.BitCast<IR::U32>(src0), false));
}

void Translator::V_CVT_F32_F16(const GcnInst& inst) {
    // abs / neg act on the half's sign, bit 15.
    InstOperand operand{inst.src[0]};
    operand.input_modifier.abs = false;
    operand.input_modifier.neg = false;
    IR::U32 src0{GetSrc(operand)};
    if (inst.src[0].input_modifier.abs) {
        src0 = ir.BitwiseAnd(src0, ir.Imm32(0xFFFF7FFFU));
    }
    if (inst.src[0].input_modifier.neg) {
        src0 = ir.BitwiseXor(src0, ir.Imm32(0x8000U));
    }
    const IR::U32 sign{ir.ShiftLeftLogical(ir.BitwiseAnd(src0, ir.Imm32(0x8000U)), ir.Imm32(16U))};
    const IR::U32 exponent{
        ir.BitwiseAnd(ir.ShiftRightLogical(src0, ir.Imm32(10U)), ir.Imm32(0x1FU))};
    const IR::U32 mantissa{ir.BitwiseAnd(src0, ir.Imm32(0x3FFU))};
    const IR::U32 shifted_mantissa{ir.ShiftLeftLogical(mantissa, ir.Imm32(13U))};

    const IR::U32 normal{ir.BitwiseOr(
        ir.ShiftLeftLogical(ir.IAdd(exponent, ir.Imm32(112U)), ir.Imm32(23U)), shifted_mantissa)};
    const IR::U32 inf_nan{ir.BitwiseOr(ir.Imm32(0x7F800000U), shifted_mantissa)};
    const IR::U32 denorm{ir.BitCast<IR::U32>(
        IR::F32{ir.FPMul(ir.ConvertUToF(32, 32, mantissa), ir.Imm32(0x1p-24f))})};

    IR::U32 result{ir.Select(ir.IEqual(exponent, ir.Imm32(0U)), denorm, normal)};
    result = IR::U32{ir.Select(ir.IEqual(exponent, ir.Imm32(0x1FU)), inf_nan, result)};
    SetDst(inst.dst[0], ir.BitCast<IR::F32>(IR::U32{ir.BitwiseOr(sign, result)}));
}

void Translator::V_CVT_RPI_I32_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 value{
        FlushesDenormInput(runtime_info.props.fp_denorm_mode32) ? FlushDenorm32(ir, src0) : src0};
    // value - floor is exact, so the half comparison sees the unrounded value + 0.5.
    const IR::F32 floor{ir.FPFloor(value)};
    const IR::F32 fraction{ir.FPSub(value, floor)};
    const IR::F32 rounded{ir.Select(ir.FPGreaterThanEqual(fraction, ir.Imm32(0.5f)),
                                    ir.FPAdd(floor, ir.Imm32(1.f)), floor)};
    SetDst(inst.dst[0], SaturateToI32(ir, src0, rounded));
}

void Translator::V_CVT_FLR_I32_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 value{
        FlushesDenormInput(runtime_info.props.fp_denorm_mode32) ? FlushDenorm32(ir, src0) : src0};
    SetDst(inst.dst[0], SaturateToI32(ir, src0, IR::F32{ir.FPFloor(value)}));
}

void Translator::V_CVT_OFF_F32_I4(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 nibble{ir.BitFieldExtract(src0, ir.Imm32(0), ir.Imm32(4), true)};
    SetDst(inst.dst[0], ir.FPMul(ir.ConvertSToF(32, 32, nibble), ir.Imm32(0.0625f)));
}

void Translator::V_CVT_F32_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F32 value{ir.FPConvert(32, src0)};
    const IR::F32 result{FlushesDenormOutput(runtime_info.props.fp_denorm_mode32)
                             ? FlushDenorm32(ir, value)
                             : value};
    // NaN: the quiet bit set, the payload's top 22 bits kept
    const IR::Value halves{ir.UnpackDouble2x32(src0)};
    const IR::U32 lo{ir.CompositeExtract(halves, 0)};
    const IR::U32 hi{ir.CompositeExtract(halves, 1)};
    const IR::U32 payload{ir.BitwiseOr(ir.ShiftLeftLogical(hi, ir.Imm32(3U)),
                                       ir.ShiftRightLogical(lo, ir.Imm32(29U)))};
    const IR::U32 nan{
        ir.BitwiseOr(ir.BitwiseOr(ir.BitwiseAnd(hi, ir.Imm32(0x80000000U)), ir.Imm32(0x7FC00000U)),
                     ir.BitwiseAnd(payload, ir.Imm32(0x7FFFFFU)))};
    SetDst(inst.dst[0], ir.BitCast<IR::F32>(IR::U32{
                            ir.Select(ir.FPIsNan(src0), nan, ir.BitCast<IR::U32>(result))}));
}

void Translator::V_CVT_F64_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 value{
        FlushesDenormInput(runtime_info.props.fp_denorm_mode32) ? FlushDenorm32(ir, src0) : src0};
    // NaN is widened bit for bit, without quieting.
    const IR::U32 bits{ir.BitCast<IR::U32>(src0)};
    const IR::U32 nan_hi{ir.BitwiseOr(
        ir.BitwiseOr(ir.BitwiseAnd(bits, ir.Imm32(0x80000000U)), ir.Imm32(0x7FF00000U)),
        ir.BitwiseAnd(ir.ShiftRightLogical(bits, ir.Imm32(3U)), ir.Imm32(0xFFFFFU)))};
    const IR::U32 nan_lo{ir.ShiftLeftLogical(bits, ir.Imm32(29U))};
    // Select has no F64: pick each dword.
    const IR::Value converted{ir.UnpackDouble2x32(IR::F64{ir.FPConvert(64, value)})};
    IR::U32 conv_lo{ir.CompositeExtract(converted, 0)};
    IR::U32 conv_hi{ir.CompositeExtract(converted, 1)};
    if (!FlushesDenormInput(runtime_info.props.fp_denorm_mode32)) {
        // A kept denormal widens exactly: mantissa * 2^-149 is normal in f64
        const IR::F64 magnitude{ir.FPMul(
            ir.ConvertUToF(64, 32, ir.BitwiseAnd(bits, ir.Imm32(0x7FFFFFU))), ir.Imm64(0x1p-149))};
        const IR::Value exact{ir.UnpackDouble2x32(magnitude)};
        const IR::U1 is_denorm{ir.IEqual(ir.BitwiseAnd(bits, ir.Imm32(0x7F800000U)), ir.Imm32(0U))};
        conv_lo = IR::U32{ir.Select(is_denorm, IR::U32{ir.CompositeExtract(exact, 0)}, conv_lo)};
        conv_hi = IR::U32{ir.Select(is_denorm,
                                    ir.BitwiseOr(IR::U32{ir.CompositeExtract(exact, 1)},
                                                 ir.BitwiseAnd(bits, ir.Imm32(0x80000000U))),
                                    conv_hi)};
    }
    const IR::U1 is_nan{ir.FPIsNan(src0)};
    const IR::U32 lo{ir.Select(is_nan, nan_lo, conv_lo)};
    const IR::U32 hi{ir.Select(is_nan, nan_hi, conv_hi)};
    SetDst64(inst.dst[0], ir.PackDouble2x32(ir.CompositeConstruct(lo, hi)));
}

void Translator::V_CVT_F32_UBYTE(u32 index, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 byte = ir.BitFieldExtract(src0, ir.Imm32(8 * index), ir.Imm32(8));
    SetDst(inst.dst[0], ir.ConvertUToF(32, 32, byte));
}

void Translator::V_TRUNC_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    SetDst64(inst.dst[0], ir.FPTrunc(src0));
}

void Translator::V_FLOOR_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    SetDst64(inst.dst[0], ir.FPFloor(src0));
}

void Translator::V_FRACT_F32(const GcnInst& inst) {
    // An infinity gives the default NaN 0xFFC00000.
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0],
           IR::F32{ir.Select(ir.FPIsInf(src0), ir.BitCast<IR::F32>(ir.Imm32(0xFFC00000U)),
                             IR::F32{ir.FPFract(src0)})});
}

void Translator::V_TRUNC_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPTrunc(src0));
}

void Translator::V_CEIL_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPCeil(src0));
}

void Translator::V_RNDNE_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPRoundEven(src0));
}

void Translator::V_FLOOR_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPFloor(src0));
}

void Translator::V_EXP_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPExp2(src0));
}

void Translator::V_LOG_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPLog2(src0));
}

void Translator::V_RCP_F32(const GcnInst& inst) {
    // No denormals in any float mode: input and result flushed.
    const IR::F32 src0{FlushDenorm32(ir, GetSrc<IR::F32>(inst.src[0]))};
    SetDst(inst.dst[0], FlushDenorm32(ir, IR::F32{ir.FPRecip(src0)}));
}

void Translator::V_RCP_LEGACY_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const auto result = ir.FPRecip(src0);
    const auto inf = ir.FPIsInf(result);

    const auto raw_result = ir.ConvertFToU(32, result);
    const auto sign_bit = ir.ShiftRightLogical(raw_result, ir.Imm32(31u));
    const auto sign_bit_set = ir.INotEqual(sign_bit, ir.Imm32(0u));
    const IR::F32 inf_result{ir.Select(sign_bit_set, ir.Imm32(-0.0f), ir.Imm32(0.0f))};
    const IR::F32 val{ir.Select(inf, inf_result, result)};

    SetDst(inst.dst[0], val);
}

void Translator::V_RCP_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    SetDst64(inst.dst[0], ir.FPRecip(src0));
}

void Translator::V_RSQ_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPRecipSqrt(src0));
}

void Translator::V_SQRT_F32(const GcnInst& inst) {
    // A denormal input flushes in any float mode; the root of -0 is +0.
    const IR::F32 src0{FlushDenorm32(ir, GetSrc<IR::F32>(inst.src[0]))};
    const IR::F32 root{ir.FPSqrt(src0)};
    SetDst(inst.dst[0], IR::F32{ir.Select(ir.FPEqual(src0, ir.Imm32(0.f)), ir.Imm32(0.f), root)});
}

void Translator::V_SIN_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPSin(src0));
}

void Translator::V_COS_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPCos(src0));
}

void Translator::V_NOT_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.BitwiseNot(src0));
}

void Translator::V_BFREV_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.BitReverse(src0));
}

void Translator::V_FFBH_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    // Gcn wants the MSB position counting from the left, but SPIR-V counts from the rightmost (LSB)
    // position
    const IR::U32 msb_pos = ir.FindUMsb(src0);
    const IR::U32 pos_from_left = ir.ISub(ir.Imm32(31), msb_pos);
    // Select 0xFFFFFFFF if src0 was 0
    const IR::U1 cond = ir.INotEqual(src0, ir.Imm32(0));
    SetDst(inst.dst[0], IR::U32{ir.Select(cond, pos_from_left, ir.Imm32(~0U))});
}

void Translator::V_FFBL_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.FindILsb(src0));
}

void Translator::V_FFBH_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    // Gcn wants the MSB position counting from the left, but SPIR-V counts from the rightmost (LSB)
    // position
    const IR::U32 msb_pos = ir.FindSMsb(src0);
    const IR::U32 pos_from_left = ir.ISub(ir.Imm32(31), msb_pos);
    // Select 0xFFFFFFFF if src0 was 0 or -1
    const IR::U32 minusOne = ir.Imm32(~0U);
    const IR::U1 cond =
        ir.LogicalAnd(ir.INotEqual(src0, ir.Imm32(0)), ir.INotEqual(src0, minusOne));
    SetDst(inst.dst[0], IR::U32{ir.Select(cond, pos_from_left, minusOne)});
}

void Translator::V_FREXP_EXP_I32_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPFrexpExp(src0));
}

void Translator::V_FREXP_MANT_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    SetDst64(inst.dst[0], ir.FPFrexpSig(src0));
}

void Translator::V_FRACT_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    SetDst64(inst.dst[0], ir.FPFract(src0));
}

void Translator::V_FREXP_EXP_I32_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPFrexpExp(src0));
}

void Translator::V_FREXP_MANT_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    SetDst(inst.dst[0], ir.FPFrexpSig(src0));
}

void Translator::V_MOVRELD_B32(const GcnInst& inst) {
    const IR::U32 src_val{GetSrc(inst.src[0])};
    u32 dst_vgprno = inst.dst[0].code - static_cast<u32>(IR::VectorReg::V0);
    IR::U32 m0 = ir.GetM0();

    VMovRelDHelper(dst_vgprno, src_val, m0);
}

void Translator::V_MOVRELS_B32(const GcnInst& inst) {
    u32 src_vgprno = inst.src[0].code - static_cast<u32>(IR::VectorReg::V0);
    const IR::U32 m0 = ir.GetM0();

    const IR::U32 src_val = VMovRelSHelper(src_vgprno, m0);
    SetDst(inst.dst[0], src_val);
}

void Translator::V_MOVRELSD_B32(const GcnInst& inst) {
    u32 src_vgprno = inst.src[0].code - static_cast<u32>(IR::VectorReg::V0);
    u32 dst_vgprno = inst.dst[0].code - static_cast<u32>(IR::VectorReg::V0);
    IR::U32 m0 = ir.GetM0();

    const IR::U32 src_val = VMovRelSHelper(src_vgprno, m0);
    VMovRelDHelper(dst_vgprno, src_val, m0);
}

// VOPC

void Translator::V_CMP_F32(ConditionOp op, bool ordered, bool set_exec, const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::U1 result = [&] {
        switch (op) {
        case ConditionOp::F:
            return ir.Imm1(false);
        case ConditionOp::EQ:
            return ir.FPEqual(src0, src1, ordered);
        case ConditionOp::LG:
            return ir.FPNotEqual(src0, src1, ordered);
        case ConditionOp::GT:
            return ir.FPGreaterThan(src0, src1, ordered);
        case ConditionOp::LT:
            return ir.FPLessThan(src0, src1, ordered);
        case ConditionOp::LE:
            return ir.FPLessThanEqual(src0, src1, ordered);
        case ConditionOp::GE:
            return ir.FPGreaterThanEqual(src0, src1, ordered);
        case ConditionOp::U:
            return ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1));
        default:
            UNREACHABLE();
        }
    }();
    if (set_exec) {
        // V_CMPX evaluates on active lanes only; hardware writes exec & result to both EXEC
        // and the VCC/SDST destination, zeroing inactive lanes' bits.
        const IR::U1 masked{ir.LogicalAnd(ir.GetExec(), result)};
        ir.SetExec(masked);
        SetDst64(inst.dst[1], ir.Ballot(masked));
        return;
    }
    SetDst64(inst.dst[1], ir.Ballot(result));
}

void Translator::V_CMP_F64(ConditionOp op, bool ordered, bool set_exec, const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F64 src1{GetSrc64<IR::F64>(inst.src[1])};
    const IR::U1 result = [&] {
        switch (op) {
        case ConditionOp::F:
            return ir.Imm1(false);
        case ConditionOp::EQ:
            return ir.FPEqual(src0, src1, ordered);
        case ConditionOp::LG:
            return ir.FPNotEqual(src0, src1, ordered);
        case ConditionOp::GT:
            return ir.FPGreaterThan(src0, src1, ordered);
        case ConditionOp::LT:
            return ir.FPLessThan(src0, src1, ordered);
        case ConditionOp::LE:
            return ir.FPLessThanEqual(src0, src1, ordered);
        case ConditionOp::GE:
            return ir.FPGreaterThanEqual(src0, src1, ordered);
        case ConditionOp::U:
            return ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1));
        default:
            UNREACHABLE();
        }
    }();
    if (set_exec) {
        // See the V_CMPX note in V_CMP_F32.
        const IR::U1 masked{ir.LogicalAnd(ir.GetExec(), result)};
        ir.SetExec(masked);
        SetDst64(inst.dst[1], ir.Ballot(masked));
        return;
    }
    SetDst64(inst.dst[1], ir.Ballot(result));
}

void Translator::V_CMP_U32(ConditionOp op, bool is_signed, bool set_exec, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U1 result = [&] {
        switch (op) {
        case ConditionOp::F:
            return ir.Imm1(false);
        case ConditionOp::TRU:
            return ir.Imm1(true);
        case ConditionOp::EQ:
            return ir.IEqual(src0, src1);
        case ConditionOp::LG:
            return ir.INotEqual(src0, src1);
        case ConditionOp::GT:
            return ir.IGreaterThan(src0, src1, is_signed);
        case ConditionOp::LT:
            return ir.ILessThan(src0, src1, is_signed);
        case ConditionOp::LE:
            return ir.ILessThanEqual(src0, src1, is_signed);
        case ConditionOp::GE:
            return ir.IGreaterThanEqual(src0, src1, is_signed);
        default:
            UNREACHABLE();
        }
    }();
    if (set_exec) {
        // See the V_CMPX note in V_CMP_F32.
        const IR::U1 masked{ir.LogicalAnd(ir.GetExec(), result)};
        ir.SetExec(masked);
        SetDst64(inst.dst[1], ir.Ballot(masked));
        return;
    }
    SetDst64(inst.dst[1], ir.Ballot(result));
}

void Translator::V_CMP_U64(ConditionOp op, bool is_signed, bool set_exec, const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    const IR::U1 result = [&] {
        switch (op) {
        case ConditionOp::F:
            return ir.Imm1(false);
        case ConditionOp::TRU:
            return ir.Imm1(true);
        case ConditionOp::EQ:
            return ir.IEqual(src0, src1);
        case ConditionOp::LG:
            return ir.INotEqual(src0, src1);
        case ConditionOp::GT:
            return ir.IGreaterThan(src0, src1, is_signed);
        case ConditionOp::LT:
            return ir.ILessThan(src0, src1, is_signed);
        case ConditionOp::LE:
            return ir.ILessThanEqual(src0, src1, is_signed);
        case ConditionOp::GE:
            return ir.IGreaterThanEqual(src0, src1, is_signed);
        default:
            UNREACHABLE();
        }
    }();
    if (set_exec) {
        // See the V_CMPX note in V_CMP_F32.
        const IR::U1 masked{ir.LogicalAnd(ir.GetExec(), result)};
        ir.SetExec(masked);
        SetDst64(inst.dst[1], ir.Ballot(masked));
        return;
    }
    SetDst64(inst.dst[1], ir.Ballot(result));
}

void Translator::V_CMP_CLASS_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    IR::U1 value;
    if (src1.IsImmediate()) {
        const auto class_mask = static_cast<IR::FloatClassFunc>(src1.U32());
        if ((class_mask & IR::FloatClassFunc::NaN) == IR::FloatClassFunc::NaN) {
            value = ir.FPIsNan(src0);
        } else if ((class_mask & IR::FloatClassFunc::Infinity) == IR::FloatClassFunc::Infinity) {
            value = ir.FPIsInf(src0);
        } else if ((class_mask & IR::FloatClassFunc::Negative) == IR::FloatClassFunc::Negative) {
            value = ir.FPLessThanEqual(src0, ir.Imm32(-0.f));
        } else {
            UNREACHABLE_MSG("Unsupported float class mask: {:#x}", static_cast<u32>(class_mask));
        }
    } else {
        // We don't know the type yet, delay its resolution.
        value = ir.FPCmpClass32(src0, src1);
    }
    SetDst64(inst.dst[1], ir.Ballot(value));
}

// VOP3a

void Translator::V_MAD_F32(const GcnInst& inst) {
    // No denormals in any float mode: inputs and result flushed.
    const IR::F32 src0{FlushDenorm32(ir, GetSrc<IR::F32>(inst.src[0]))};
    const IR::F32 src1{FlushDenorm32(ir, GetSrc<IR::F32>(inst.src[1]))};
    const IR::F32 src2{FlushDenorm32(ir, GetSrc<IR::F32>(inst.src[2]))};
    SetDst(inst.dst[0], FlushDenorm32(ir, IR::F32{ir.FPAdd(ir.FPMul(src0, src1), src2)}));
}

void Translator::V_MAD_I32_I24(const GcnInst& inst, bool is_signed) {
    const IR::U32 src0{
        ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    const IR::U32 src1{
        ir.BitFieldExtract(GetSrc(inst.src[1]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    const IR::U32 src2{GetSrc(inst.src[2])};
    SetDst(inst.dst[0], ir.IAdd(ir.IMul(src0, src1), src2));
}

void Translator::V_MAD_U32_U24(const GcnInst& inst) {
    V_MAD_I32_I24(inst, false);
}

void Translator::V_CUBEID_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);

    SetDst(inst.dst[0], ir.CubeFaceIndex(x, y, z));
}

void Translator::V_CUBESC_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);

    SetDst(inst.dst[0], ir.CubeFaceCoordS(x, y, z));
}

void Translator::V_CUBETC_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);

    SetDst(inst.dst[0], ir.CubeFaceCoordT(x, y, z));
}

void Translator::V_CUBEMA_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);

    SetDst(inst.dst[0], ir.CubeFaceMajorAxis(x, y, z));
}

void Translator::V_BFE_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    IR::U32 src1{GetSrc(inst.src[1])};
    IR::U32 src2{GetSrc(inst.src[2])};
    if (!src1.IsImmediate()) {
        src1 = ir.BitwiseAnd(src1, ir.Imm32(0x1F));
    }
    if (!src2.IsImmediate()) {
        src2 = ir.BitwiseAnd(src2, ir.Imm32(0x1F));
    }
    SetDst(inst.dst[0], ir.BitFieldExtract(src0, src1, src2, is_signed));
}

void Translator::V_BFI_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    SetDst(inst.dst[0],
           ir.BitwiseOr(ir.BitwiseAnd(src0, src1), ir.BitwiseAnd(ir.BitwiseNot(src0), src2)));
}

void Translator::V_FMA_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 src2{GetSrc<IR::F32>(inst.src[2])};
    SetDst(inst.dst[0], ir.FPFma(src0, src1, src2));
}

void Translator::V_FMA_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F64 src1{GetSrc64<IR::F64>(inst.src[1])};
    const IR::F64 src2{GetSrc64<IR::F64>(inst.src[2])};
    SetDst64(inst.dst[0], ir.FPFma(src0, src1, src2));
}

void Translator::V_MIN3_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 src2{GetSrc<IR::F32>(inst.src[2])};
    SetDst(inst.dst[0], ir.FPMinTri(src0, src1, src2));
}

void Translator::V_MIN3_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    SetDst(inst.dst[0], ir.IMinTri(src0, src1, src2, is_signed));
}

void Translator::V_MAX3_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 src2{GetSrc<IR::F32>(inst.src[2])};
    SetDst(inst.dst[0], ir.FPMaxTri(src0, src1, src2));
}

void Translator::V_MAX3_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    SetDst(inst.dst[0], ir.IMaxTri(src0, src1, src2, is_signed));
}

void Translator::V_MED3_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 src2{GetSrc<IR::F32>(inst.src[2])};
    SetDst(inst.dst[0], ir.FPMedTri(src0, src1, src2));
}

void Translator::V_MED3_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    SetDst(inst.dst[0], ir.IMedTri(src0, src1, src2, is_signed));
}

void Translator::V_SAD(const GcnInst& inst) {
    const IR::U32 abs_diff = ir.IAbs(ir.ISub(GetSrc(inst.src[0]), GetSrc(inst.src[1])));
    SetDst(inst.dst[0], ir.IAdd(abs_diff, GetSrc(inst.src[2])));
}

void Translator::V_SAD_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    IR::U32 result;
    if (src0.IsImmediate() && src0.U32() == 0U) {
        result = src1;
    } else if (src1.IsImmediate() && src1.U32() == 0U) {
        result = src0;
    } else {
        const IR::U32 max{ir.IMax(src0, src1, false)};
        const IR::U32 min{ir.IMin(src0, src1, false)};
        result = ir.ISub(max, min);
    }
    SetDst(inst.dst[0], ir.IAdd(result, src2));
}

void Translator::V_CVT_PK_U16_U32(const GcnInst& inst) {
    const IR::U32 lo{ir.UMin(GetSrc<IR::U32>(inst.src[0]), ir.Imm32(0xFFFFU))};
    const IR::U32 hi{ir.UMin(GetSrc<IR::U32>(inst.src[1]), ir.Imm32(0xFFFFU))};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, ir.ShiftLeftLogical(hi, ir.Imm32(16U))));
}

void Translator::V_CVT_PK_I16_I32(const GcnInst& inst) {
    const IR::U32 min{ir.Imm32(-32768)};
    const IR::U32 max{ir.Imm32(32767)};
    const IR::U32 lo{ir.SClamp(GetSrc<IR::U32>(inst.src[0]), min, max)};
    const IR::U32 hi{ir.SClamp(GetSrc<IR::U32>(inst.src[1]), min, max)};
    SetDst(inst.dst[0], ir.BitwiseOr(ir.BitwiseAnd(lo, ir.Imm32(0xFFFFU)),
                                     ir.ShiftLeftLogical(hi, ir.Imm32(16U))));
}

void Translator::V_CVT_PK_U8_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};

    const IR::F32 clamped{ir.FPClamp(ir.FPRoundEven(src0), ir.Imm32(0.0f), ir.Imm32(255.0f))};
    const IR::F32 value{ir.Select(ir.FPIsNan(src0), ir.Imm32(0.0f), clamped)};
    const IR::U32 value_uint{ir.ConvertFToU(32, value)};
    const IR::U32 offset{ir.ShiftLeftLogical(ir.BitwiseAnd(src1, ir.Imm32(3)), ir.Imm32(3))};
    SetDst(inst.dst[0], ir.BitFieldInsert(src2, value_uint, offset, ir.Imm32(8)));
}

void Translator::V_LSHL_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    SetDst64(inst.dst[0], ir.ShiftLeftLogical(src0, ir.BitwiseAnd(src1, ir.Imm64(u64(0x3F)))));
}

void Translator::V_LSHR_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    SetDst64(inst.dst[0], ir.ShiftRightLogical(src0, ir.BitwiseAnd(src1, ir.Imm64(u64(0x3F)))));
}

void Translator::V_ASHR_I64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    SetDst64(inst.dst[0], ir.ShiftRightArithmetic(src0, ir.BitwiseAnd(src1, ir.Imm64(u64(0x3F)))));
}

void Translator::V_ALIGNBIT_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{ir.BitwiseAnd(GetSrc(inst.src[2]), ir.Imm32(0x1F))};
    const IR::U32 lo{ir.ShiftRightLogical(src1, src2)};
    const IR::U32 hi{
        ir.ShiftLeftLogical(ir.ShiftLeftLogical(src0, ir.Imm32(1)), ir.ISub(ir.Imm32(31), src2))};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, hi));
}

void Translator::V_ALIGNBYTE_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{ir.BitwiseAnd(GetSrc(inst.src[2]), ir.Imm32(0x3))};
    const IR::U32 shift{ir.ShiftLeftLogical(src2, ir.Imm32(3))};
    const IR::U32 lo{ir.ShiftRightLogical(src1, shift)};
    const IR::U32 hi{
        ir.ShiftLeftLogical(ir.ShiftLeftLogical(src0, ir.Imm32(1)), ir.ISub(ir.Imm32(31), shift))};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, hi));
}

void Translator::V_MUL_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F64 src1{GetSrc64<IR::F64>(inst.src[1])};
    SetDst64(inst.dst[0], ir.FPMul(src0, src1));
}

void Translator::V_MIN_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F64 src1{GetSrc64<IR::F64>(inst.src[1])};
    SetDst64(inst.dst[0], ir.FPMin(src0, src1));
}

void Translator::V_MAX_F64(const GcnInst& inst) {
    const IR::F64 src0{GetSrc64<IR::F64>(inst.src[0])};
    const IR::F64 src1{GetSrc64<IR::F64>(inst.src[1])};
    SetDst64(inst.dst[0], ir.FPMax(src0, src1));
}

void Translator::V_MUL_LO_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.IMul(src0, src1));
}

void Translator::V_MUL_HI_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 hi{ir.IMulHi(src0, src1, is_signed)};
    SetDst(inst.dst[0], hi);
}

void Translator::V_MAD_U64_U32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc64<IR::U64>(inst.src[2]);

    const IR::U64 mul_result = ir.IMul(ir.UConvert(64, src0), ir.UConvert(64, src1));
    const IR::U64 sum_result = ir.IAdd(mul_result, src2);

    SetDst64(inst.dst[0], sum_result);

    const IR::U1 less_src0 = ir.ILessThan(sum_result, mul_result, false);
    const IR::U1 less_src1 = ir.ILessThan(sum_result, src2, false);
    const IR::U1 did_overflow = ir.LogicalOr(less_src0, less_src1);

    const auto unpacked = ir.UnpackUint2x32(ir.Ballot(did_overflow));
    const IR::U32 lo{ir.CompositeExtract(unpacked, 0U)};
    const IR::U32 hi{ir.CompositeExtract(unpacked, 1U)};
    ir.SetVccLo(lo);
    ir.SetVccHi(hi);
}

void Translator::V_LSHLREV_B16(const GcnInst& inst) {
    const auto shift = GetSrc16<IR::U32>(inst.src[0]);
    const auto src = GetSrc16<IR::U32>(inst.src[1]);

    const auto result = ir.ShiftLeftLogical(src, ir.BitwiseAnd(shift, ir.Imm32(0xF)));

    SetDst16(inst.dst[0], result);
}

void Translator::V_LSHRREV_B16(const GcnInst& inst) {
    const auto shift = GetSrc16<IR::U32>(inst.src[0]);
    const auto src = GetSrc16<IR::U32>(inst.src[1]);

    const auto result = ir.ShiftRightLogical(src, ir.BitwiseAnd(shift, ir.Imm32(0xF)));

    SetDst16(inst.dst[0], result);
}

void Translator::V_ASHRREV_I16(const GcnInst& inst) {
    const auto shift = GetSrc16<IR::U32, true>(inst.src[0]);
    const auto src = GetSrc16<IR::U32, true>(inst.src[1]);

    const auto result = ir.ShiftRightArithmetic(src, ir.BitwiseAnd(shift, ir.Imm32(0xF)));

    SetDst16<true>(inst.dst[0], result);
}

void Translator::V_ADD_LSHL_U32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc<IR::U32>(inst.src[2]);

    const auto shift = ir.BitwiseAnd(src2, ir.Imm32(0x1F));

    const auto result = ir.ShiftLeftLogical(ir.IAdd(src0, src1), shift);

    SetDst(inst.dst[0], result);
}

void Translator::V_LSHL_ADD_U32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc<IR::U32>(inst.src[2]);

    const auto shift = ir.BitwiseAnd(src1, ir.Imm32(0x1F));

    const auto result = ir.IAdd(ir.ShiftLeftLogical(src0, shift), src2);

    SetDst(inst.dst[0], result);
}

void Translator::V_MIN3_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);
    const auto src2 = GetSrc16<IR::F32>(inst.src[2]);

    const auto result = ir.FPMinTri(src0, src1, src2);

    SetDst16(inst.dst[0], result);
}

void Translator::V_MAX3_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);
    const auto src2 = GetSrc16<IR::F32>(inst.src[2]);

    const auto result = ir.FPMaxTri(src0, src1, src2);

    SetDst16(inst.dst[0], result);
}

void Translator::V_MED3_F16(const GcnInst& inst) {
    const auto src0 = GetSrc16<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc16<IR::F32>(inst.src[1]);
    const auto src2 = GetSrc16<IR::F32>(inst.src[2]);

    const auto result = ir.FPMedTri(src0, src1, src2);

    SetDst16(inst.dst[0], result);
}

void Translator::V_ADD3_U32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc<IR::U32>(inst.src[2]);

    SetDst(inst.dst[0], ir.IAdd(src0, ir.IAdd(src1, src2)));
}

void Translator::V_PK_MUL_LO_U16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.IMul(src0.first, src1.first);
    const auto result_hi = ir.IMul(src0.second, src1.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_ADD_I16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32, true>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32, true>(inst.src[1]);

    const auto result_lo = ir.IAdd(src0.first, src1.first);
    const auto result_hi = ir.IAdd(src0.second, src1.second);

    SetDstPk<IR::U32, true>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_SUB_I16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32, true>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32, true>(inst.src[1]);

    const auto result_lo = ir.ISub(src0.first, src1.first);
    const auto result_hi = ir.ISub(src0.second, src1.second);

    SetDstPk<IR::U32, true>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_LSHLREV_B16(const GcnInst& inst) {
    const auto shift = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.ShiftLeftLogical(src.first, shift.first);
    const auto result_hi = ir.ShiftLeftLogical(src.second, shift.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_LSHRREV_B16(const GcnInst& inst) {
    const auto shift = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.ShiftRightLogical(src.first, shift.first);
    const auto result_hi = ir.ShiftRightLogical(src.second, shift.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_MAD_U16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32>(inst.src[1]);
    const auto src2 = GetSrcPk<IR::U32>(inst.src[2]);

    const auto result_lo = ir.IAdd(ir.IMul(src0.first, src1.first), src2.first);
    const auto result_hi = ir.IAdd(ir.IMul(src0.second, src1.second), src2.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_ADD_U16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.IAdd(src0.first, src1.first);
    const auto result_hi = ir.IAdd(src0.second, src1.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_SUB_U16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.ISub(src0.first, src1.first);
    const auto result_hi = ir.ISub(src0.second, src1.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_MAX_U16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.UMax(src0.first, src1.first);
    const auto result_hi = ir.UMax(src0.second, src1.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_MIN_U16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::U32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::U32>(inst.src[1]);

    const auto result_lo = ir.UMin(src0.first, src1.first);
    const auto result_hi = ir.UMin(src0.second, src1.second);

    SetDstPk<IR::U32, false>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_FMA_F16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::F32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::F32>(inst.src[1]);
    const auto src2 = GetSrcPk<IR::F32>(inst.src[2]);

    const auto result_lo = ir.FPFma(src0.first, src1.first, src2.first);
    const auto result_hi = ir.FPFma(src0.second, src1.second, src2.second);

    SetDstPk<IR::F32>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_ADD_F16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::F32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::F32>(inst.src[1]);

    const auto result_lo = ir.FPAdd(src0.first, src1.first);
    const auto result_hi = ir.FPAdd(src0.second, src1.second);

    SetDstPk<IR::F32>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_MUL_F16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::F32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::F32>(inst.src[1]);

    const auto result_lo = ir.FPMul(src0.first, src1.first);
    const auto result_hi = ir.FPMul(src0.second, src1.second);

    SetDstPk<IR::F32>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_MIN_F16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::F32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::F32>(inst.src[1]);

    const auto result_lo = ir.FPMin(src0.first, src1.first);
    const auto result_hi = ir.FPMin(src0.second, src1.second);

    SetDstPk<IR::F32>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_PK_MAX_F16(const GcnInst& inst) {
    const auto src0 = GetSrcPk<IR::F32>(inst.src[0]);
    const auto src1 = GetSrcPk<IR::F32>(inst.src[1]);

    const auto result_lo = ir.FPMax(src0.first, src1.first);
    const auto result_hi = ir.FPMax(src0.second, src1.second);

    SetDstPk<IR::F32>(inst.dst[0], {result_lo, result_hi});
}

void Translator::V_LSHL_OR_B32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc<IR::U32>(inst.src[2]);

    const auto shift = ir.BitwiseAnd(src1, ir.Imm32(0x1F));

    const auto result = ir.BitwiseOr(ir.ShiftLeftLogical(src0, shift), src2);

    SetDst(inst.dst[0], result);
}

void Translator::V_AND_OR_B32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc<IR::U32>(inst.src[2]);

    const auto result = ir.BitwiseOr(ir.BitwiseAnd(src0, src1), src2);

    SetDst(inst.dst[0], result);
}

void Translator::V_OR3_B32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc<IR::U32>(inst.src[2]);

    const auto result = ir.BitwiseOr(ir.BitwiseOr(src0, src1), src2);

    SetDst(inst.dst[0], result);
}

void Translator::V_MAD_MIX_F32(const GcnInst& inst) {
    const auto src0 = GetSrcMix(inst.src[0]);
    const auto src1 = GetSrcMix(inst.src[1]);
    const auto src2 = GetSrcMix(inst.src[2]);

    const IR::F32 result = ir.FPAdd(ir.FPMul(src0, src1), src2);

    SetDst(inst.dst[0], result);
}

void Translator::V_MAD_MIXLO_F16(const GcnInst& inst) {
    const auto src0 = GetSrcMix(inst.src[0]);
    const auto src1 = GetSrcMix(inst.src[1]);
    const auto src2 = GetSrcMix(inst.src[2]);

    const IR::F32 result = ir.FPAdd(ir.FPMul(src0, src1), src2);
    const IR::F16 result_f16 = ir.FPConvert(16, result);
    const IR::U16 result_f16_u16 = ir.BitCast<IR::U16, IR::F16>(result_f16);

    const IR::U32 old_value{GetSrc(inst.dst[0])};
    const IR::U32 new_value{
        ir.BitFieldInsert(old_value, ir.UConvert(32, result_f16_u16), ir.Imm32(0U), ir.Imm32(16U))};
    SetDst(inst.dst[0], new_value);
}

void Translator::V_MAD_MIXHI_F16(const GcnInst& inst) {
    const auto src0 = GetSrcMix(inst.src[0]);
    const auto src1 = GetSrcMix(inst.src[1]);
    const auto src2 = GetSrcMix(inst.src[2]);

    const IR::F32 result = ir.FPAdd(ir.FPMul(src0, src1), src2);
    const IR::F16 result_f16 = ir.FPConvert(16, result);
    const IR::U16 result_f16_u16 = ir.BitCast<IR::U16, IR::F16>(result_f16);

    const IR::U32 old_value{GetSrc(inst.dst[0])};
    const IR::U32 new_value{ir.BitFieldInsert(old_value, ir.UConvert(32, result_f16_u16),
                                              ir.Imm32(16U), ir.Imm32(16U))};
    SetDst(inst.dst[0], new_value);
}

IR::U32 Translator::GetCarryIn(const GcnInst& inst) {
    IR::U64 carry;
    if (inst.src_count == 3) { // VOP3
        carry = GetSrc64(inst.src[2]);
    } else { // VOP2
        carry = ir.PackUint2x32(ir.CompositeConstruct(ir.GetVccLo(), ir.GetVccHi()));
    }

    return IR::U32{ir.Select(ir.InverseBallot(carry), ir.Imm32(1), ir.Imm32(0))};
}

void Translator::SetCarryOut(const GcnInst& inst, const IR::U1& carry) {
    if (inst.dst_count == 2) { // VOP3
        SetDst64(inst.dst[1], ir.Ballot(carry));
    } else { // VOP2
        const auto unpacked = ir.UnpackUint2x32(ir.Ballot(carry));
        const IR::U32 lo{ir.CompositeExtract(unpacked, 0U)};
        const IR::U32 hi{ir.CompositeExtract(unpacked, 1U)};
        ir.SetVccLo(lo);
        ir.SetVccHi(hi);
    }
}

IR::F32 Translator::LegacyMul(const IR::F32& a, const IR::F32& b) {
    // DX9 rules, 0.0 * x = 0.0
    const IR::F32 zero{ir.Imm32(0.0f)};
    const IR::U1 either_zero{ir.LogicalOr(ir.FPEqual(a, zero), ir.FPEqual(b, zero))};
    return IR::F32{ir.Select(either_zero, zero, ir.FPMul(a, b))};
}

// TODO: add range analysis pass to hopefully put an upper bound on m0, and only select one of
// [src_vgprno, src_vgprno + max_m0]. Same for dst regs we may write back to

IR::U32 Translator::VMovRelSHelper(u32 src_vgprno, const IR::U32 m0) {
    // Read from VGPR0 by default when src_vgprno + m0 > num_allocated_vgprs
    IR::U32 src_val = ir.GetVectorReg<IR::U32>(IR::VectorReg::V0);
    for (u32 i = src_vgprno; i < runtime_info.props.num_allocated_vgprs; i++) {
        const IR::U1 cond = ir.IEqual(m0, ir.Imm32(i - src_vgprno));
        src_val =
            IR::U32{ir.Select(cond, ir.GetVectorReg<IR::U32>(IR::VectorReg::V0 + i), src_val)};
    }
    return src_val;
}

void Translator::VMovRelDHelper(u32 dst_vgprno, const IR::U32 src_val, const IR::U32 m0) {
    for (u32 i = dst_vgprno; i < runtime_info.props.num_allocated_vgprs; i++) {
        const IR::U1 cond = ir.IEqual(m0, ir.Imm32(i - dst_vgprno));
        const IR::U32 dst_val =
            IR::U32{ir.Select(cond, src_val, ir.GetVectorReg<IR::U32>(IR::VectorReg::V0 + i))};
        ir.SetVectorReg(IR::VectorReg::V0 + i, dst_val);
    }
}

} // namespace Shader::Gcn
