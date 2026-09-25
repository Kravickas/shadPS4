// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (push_constant) uniform Constants {
    uint cur_offset;
    uint prev_offset;
    uint prev_max_vertices;
    uint has_prev;
    vec2 viewport_size;
    vec2 viewport_offset;
    vec2 depth_range;
    uint blended;
    uint negative_one_to_one;
} pc;

// Clip to previous clip transform of static geometry, solved by xfb_camera_solve.comp.
layout(set = 0, binding = 2, std430) readonly buffer Camera {
    vec4 clip_to_prev[4];
    vec4 stats;
} camera;

layout(location = 0) in vec4 in_cur;
layout(location = 1) in vec4 in_prev;
layout(location = 2) in float in_invalid;

// Motion vector in pixels from the current pixel to its previous-frame position.
layout(location = 0) out vec2 out_motion;
// 0 per-vertex motion, 0.25 camera reprojection, 0.5 untrusted, 1 (clear) not covered.
layout(location = 1) out float out_mask;

// The fragment passed an EQUAL test, so gl_FragCoord.z is the stored scene depth.
bool CameraMotion(out vec2 motion) {
    motion = vec2(0.0);
    const bool centered = pc.negative_one_to_one != 0u;
    const float depth_scale = (pc.depth_range.y - pc.depth_range.x) * (centered ? 0.5 : 1.0);
    if (camera.stats.x == 0.0 || depth_scale == 0.0) {
        return false;
    }
    const float depth_offset =
        centered ? 0.5 * (pc.depth_range.x + pc.depth_range.y) : pc.depth_range.x;
    const vec2 center = pc.viewport_offset + 0.5 * pc.viewport_size;
    const vec2 ndc = (gl_FragCoord.xy - center) * 2.0 / pc.viewport_size;
    const vec4 cur = vec4(ndc, (gl_FragCoord.z - depth_offset) / depth_scale, 1.0);
    const vec4 prev = vec4(dot(camera.clip_to_prev[0], cur), dot(camera.clip_to_prev[1], cur),
                           dot(camera.clip_to_prev[2], cur), dot(camera.clip_to_prev[3], cur));
    if (prev.w <= 0.0) {
        return false;
    }
    motion = (prev.xy / prev.w - ndc) * 0.5 * pc.viewport_size;
    return true;
}

void main() {
    if (in_invalid > 0.0 || in_prev.w <= 0.0) {
        vec2 motion;
        if (pc.blended == 0u && CameraMotion(motion)) {
            out_motion = motion;
            out_mask = 0.25;
        } else {
            out_motion = vec2(0.0);
            out_mask = 0.5;
        }
        return;
    }
    const vec2 cur_ndc = in_cur.xy / in_cur.w;
    const vec2 prev_ndc = in_prev.xy / in_prev.w;
    out_motion = (prev_ndc - cur_ndc) * 0.5 * pc.viewport_size;
    out_mask = 0.0;
}
