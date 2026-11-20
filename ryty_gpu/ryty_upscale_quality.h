// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace Vulkan::RytyUpscale {
enum class Quality { Auto, Native, Quality, Balanced, Performance, UltraPerformance };

constexpr Quality ParseQuality(std::string_view name, Quality fallback) {
    if (name == "native")
        return Quality::Native;
    if (name == "quality")
        return Quality::Quality;
    if (name == "balanced")
        return Quality::Balanced;
    if (name == "performance")
        return Quality::Performance;
    if (name == "ultra_performance")
        return Quality::UltraPerformance;
    return fallback;
}

inline Quality GetQuality() {
    static const Quality quality = [] {
        const char* value = std::getenv("SHADGT_UPSCALE_QUALITY");
        return ParseQuality(value ? value : "", Quality::Auto);
    }();
    return quality;
}

inline Quality VrQuality() {
    static const Quality quality = [] {
        if (const char* value = std::getenv("SHADGT_OPENXR_UPSCALE_QUALITY"))
            return ParseQuality(value, Quality::Quality);
        return std::getenv("SHADGT_OPENXR_OUTPUT_PERCENT") ? Quality::Quality : GetQuality();
    }();
    return quality;
}

constexpr double Scale(Quality quality) {
    switch (quality) {
    case Quality::Native:
        return 1.0;
    case Quality::Quality:
        return 1.5;
    case Quality::Balanced:
        return 1.0 / 0.58;
    case Quality::Performance:
        return 2.0;
    case Quality::UltraPerformance:
        return 3.0;
    default:
        return 0.0;
    }
}

constexpr uint32_t VrInputDimension(uint32_t output, Quality quality) {
    const auto scale = Scale(quality);
    return scale > 1.0 ? std::max(1u, static_cast<uint32_t>(output / scale + 0.5)) : output;
}
} // namespace Vulkan::RytyUpscale
