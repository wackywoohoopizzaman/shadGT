// SPDX-License-Identifier: GPL-2.0-or-later
#version 450
#extension GL_GOOGLE_include_directive : require
layout(location=0) in vec2 uv;
layout(location=0) out vec4 color;
layout(binding=0) uniform sampler2D source;
layout(push_constant) uniform Source { uint decode_srgb; uint region; uint easu; uint width; uint height; } source_info;

#define A_GPU 1
#define A_GLSL 1
#include "fsr/ffx_a.h"
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { return textureGather(source, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(source, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(source, p, 2); }
#include "fsr/ffx_fsr1.h"

void main() {
    vec2 coord = uv;
    if (source_info.region != 0) {
        float start = source_info.region == 1 ? 0.0 : 0.5;
        float border = 0.5 / float(textureSize(source, 0).x);
        coord.x = clamp(start + uv.x * 0.5, start + border, start + 0.5 - border);
    }
    vec3 rgb;
    vec2 src = vec2(textureSize(source, 0));
    vec2 dst = vec2(source_info.width, source_info.height);
    if (source_info.easu != 0 && source_info.region == 0 && all(greaterThanEqual(dst, src)) && any(greaterThan(dst, src))) {
        uvec4 c0, c1, c2, c3;
        FsrEasuCon(c0, c1, c2, c3, src.x, src.y, src.x, src.y, dst.x, dst.y);
        FsrEasuF(rgb, uvec2(gl_FragCoord.xy), c0, c1, c2, c3);
    } else {
        rgb = texture(source, coord).rgb;
    }
    if (source_info.decode_srgb != 0) {
        rgb = mix(pow((rgb + 0.055) / 1.055, vec3(2.4)), rgb / 12.92, lessThanEqual(rgb, vec3(0.04045)));
    }
    color = vec4(rgb, 1.0);
}
