// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;

layout (push_constant) uniform settings {
    float gamma;
    bool hdr;
} pp;

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma_fn(vec3 rgb) {
    return mix(
        a * pow(rgb, vec3(1.0 / (2.4 + 1.0 - pp.gamma))) - b,
        d * rgb / pp.gamma,
        lessThan(rgb, vec3(cutoff))
    );
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    // DIAGNOSTIC: top-left quarter = sampled display buffer, rest = solid red
    // If you see red filling 3/4 of the screen → pp_pass IS writing, display buffer is black
    // If full screen is black → pp_pass pipeline is broken (not writing at all)
    if (uv.x > 0.5 || uv.y > 0.5) {
        color = vec4(1.0, 0.0, 0.0, 1.0); // solid red
    } else {
        color = vec4(gamma_fn(color_linear.rgb), 1.0);
    }
}
