// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

// Transform feedback records are 5 dwords: float4 clip position, uint vertex index.
layout(set = 0, binding = 0, std430) readonly buffer CurCapture {
    uint cur_data[];
};
layout(set = 0, binding = 1, std430) readonly buffer PrevCapture {
    uint prev_data[];
};

layout (push_constant) uniform Constants {
    uint cur_offset;
    uint prev_offset;
    uint prev_max_vertices;
    uint has_prev;
    vec2 viewport_size;
} pc;

layout(location = 0) out vec4 out_cur;
layout(location = 1) out vec4 out_prev;
layout(location = 2) out float out_invalid;

void main() {
    const uint slot = uint(gl_VertexIndex);
    const uint cur_base = pc.cur_offset + slot * 5u;
    const vec4 cur =
        vec4(uintBitsToFloat(cur_data[cur_base + 0u]), uintBitsToFloat(cur_data[cur_base + 1u]),
             uintBitsToFloat(cur_data[cur_base + 2u]), uintBitsToFloat(cur_data[cur_base + 3u]));

    vec4 prev = cur;
    bool valid = pc.has_prev != 0u && slot < pc.prev_max_vertices;
    if (valid) {
        const uint prev_base = pc.prev_offset + slot * 5u;
        prev = vec4(
            uintBitsToFloat(prev_data[prev_base + 0u]), uintBitsToFloat(prev_data[prev_base + 1u]),
            uintBitsToFloat(prev_data[prev_base + 2u]), uintBitsToFloat(prev_data[prev_base + 3u]));
        valid = prev_data[prev_base + 4u] == cur_data[cur_base + 4u];
    }

    gl_Position = cur;
    out_cur = cur;
    out_prev = prev;
    out_invalid = valid ? 0.0 : 1.0;
}
