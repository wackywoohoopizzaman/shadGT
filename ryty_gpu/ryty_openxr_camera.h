// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "ryty_gpu/ryty_openxr.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <utility>

namespace Vulkan::Gt7XrCamera {
inline bool Requested() {
    static const bool requested = [] {
        const auto enabled = [](const char* name) {
            const char* value = std::getenv(name);
            return value && std::string_view(value) == "1";
        };
        return enabled("SHARED_OPENXR") && enabled("SHADGT_OPENXR_COCKPIT");
    }();
    return requested;
}

// Protect the active seat and predictions while the guest selects its next buffer.
inline std::mutex prediction_mutex;
// GT7 allocates one two-layer image before its first tracked frame.
inline vk::Extent2D allocation_size{};
// The title publishes the committed size once it has created all three flip slots.
inline std::atomic<uint64_t> game_eye_extent{};
inline std::atomic<uint64_t> reconstruction_extent{};
inline std::atomic<float> dlss_material_mip_bias{};
inline uint64_t PackExtent(uint32_t width, uint32_t height) {
    return (uint64_t{width} << 32) | height;
}
// Each axis is shared by both layers of GT7's eye texture.
constexpr uint32_t SceneDimension(uint32_t left, uint32_t right, uint32_t left_limit,
                                  uint32_t right_limit, uint32_t percent) {
    const auto pixels = (uint64_t{std::max(left, right)} * percent + 50) / 100;
    return static_cast<uint32_t>(
        std::clamp<uint64_t>(pixels, 1, std::min(left_limit, right_limit)));
}
inline void SetAllocationSize(vk::Extent2D size) {
    std::lock_guard lock(prediction_mutex);
    allocation_size = size;
}
inline vk::Extent2D AllocationSize() {
    std::lock_guard lock(prediction_mutex);
    return allocation_size;
}
inline std::optional<OpenXrOutput::ViewFrame> prediction;
inline std::chrono::steady_clock::time_point updated;
inline std::array<std::shared_ptr<const GameStereoFrame>, 3> buffers;

inline void Publish(const std::optional<OpenXrOutput::ViewFrame>& views) {
    std::lock_guard lock(prediction_mutex);
    prediction = views;
    updated = std::chrono::steady_clock::now();
    if (!views)
        buffers.fill(nullptr);
}

inline std::optional<OpenXrOutput::ViewFrame> Read() {
    std::lock_guard lock(prediction_mutex);
    if (std::chrono::steady_clock::now() - updated > std::chrono::milliseconds(250))
        return {};
    return prediction;
}
inline void TagBuffer(int slot, const GameStereoFrame* game) {
    if (slot < 0 || static_cast<size_t>(slot) >= buffers.size())
        return;
    std::lock_guard lock(prediction_mutex);
    buffers[slot] = game ? std::make_shared<GameStereoFrame>(*game) : nullptr;
}

inline std::shared_ptr<const GameStereoFrame> Claim(int slot) {
    if (!Requested() || slot < 0 || static_cast<size_t>(slot) >= buffers.size())
        return {};
    std::lock_guard lock(prediction_mutex);
    return std::exchange(buffers[slot], nullptr);
}
} // namespace Vulkan::Gt7XrCamera
