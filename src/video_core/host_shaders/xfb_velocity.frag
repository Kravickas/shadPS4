// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (push_constant) uniform Constants {
    uint cur_offset;
    uint prev_offset;
    uint prev_max_vertices;
    uint has_prev;
    vec2 viewport_size;
} pc;

layout(location = 0) in vec4 in_cur;
layout(location = 1) in vec4 in_prev;
layout(location = 2) in float in_invalid;

// Motion vector in pixels from the current pixel to its previous-frame position.
layout(location = 0) out vec2 out_motion;
// 1 where the vector cannot be trusted.
layout(location = 1) out float out_mask;

void main() {
    const vec2 cur_ndc = in_cur.xy / in_cur.w;
    const vec2 prev_ndc = in_prev.xy / in_prev.w;
    out_motion = (prev_ndc - cur_ndc) * 0.5 * pc.viewport_size;
    out_mask = in_invalid > 0.0 || in_prev.w <= 0.0 ? 1.0 : 0.0;
}
