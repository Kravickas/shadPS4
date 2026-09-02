// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include "common/types.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/attribute.h"
#include "shader_recompiler/profile.h"

namespace Shader {

// Transform feedback record written by the last pre-rasterization stage: float4 clip position
// followed by the vertex index.
constexpr u32 XfbVertexStride = 20;
constexpr u32 XfbVertexIndexOffset = 16;

// Output location for the vertex index, or nullopt when every vertex output location is in use.
inline std::optional<u32> XfbVertexIndexLocation(const Info& info, const Profile& profile) {
    const u32 shift =
        profile.needs_clip_distance_emulation && info.stores.GetAny(IR::Attribute::ClipDistance)
            ? 1
            : 0;
    u32 location = shift;
    for (u32 i = 0; i < IR::NumParams; i++) {
        if (info.stores.GetAny(IR::Attribute::Param0 + i)) {
            location = std::max(location, i + shift + 1);
        }
    }
    if (location >= IR::NumParams) {
        return std::nullopt;
    }
    return location;
}

// True when this shader is the last pre-rasterization stage and is compiled with Xfb outputs.
// Both the SPIR-V backend and the rasterizer must agree on this.
inline bool XfbCaptureEnabled(const Info& info, const Profile& profile, bool tess_emulated) {
    return profile.supports_transform_feedback && info.stage == Stage::Vertex &&
           info.l_stage == LogicalStage::Vertex && !tess_emulated &&
           XfbVertexIndexLocation(info, profile).has_value();
}

} // namespace Shader
