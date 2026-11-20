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
// Both layers use a shared extent for the game's post-processing targets.
inline std::atomic<uint64_t> reconstruction_extent{};
// Published after a successful per-eye DLSS evaluation and cleared with the XR session.
inline std::atomic<float> dlss_material_mip_bias{};
inline uint64_t PackExtent(uint32_t width, uint32_t height) {
    return (uint64_t{width} << 32) | height;
}
// Accumulate spare display slots. At 60 game frames and 90 Hz, every other real
// pair gets an intermediate pair. At 60 Hz the same game can use every slot itself.
constexpr unsigned IntermediateSlots(int64_t& credit, int64_t display_period, int64_t source_period,
                                     unsigned limit) {
    if (!limit || display_period <= 0 || source_period <= display_period) {
        credit = 0;
        return 0;
    }
    credit += std::min(source_period - display_period, int64_t(limit) * display_period);
    const unsigned slots = std::min(limit, unsigned((credit + 1000) / display_period));
    credit = std::max<int64_t>(0, credit - slots * display_period);
    return slots;
}
// Distinct game draws can use the same tracking prediction. Interpolated displays
// can also advance its ID between adjacent source frames.
constexpr int64_t SourceInterval(const GameStereoFrame& last, const GameStereoFrame& next) {
    if (!last.source_id || next.source_id != last.source_id + 1 ||
        next.cameras.origin != last.cameras.origin || next.source_time_ns <= last.source_time_ns)
        return 0;
    const auto interval = next.source_time_ns - last.source_time_ns;
    return interval <= 100'000'000 ? interval : 0;
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

struct DrawTracking {
    XrSession session{};
    XrSpace seat{};
    std::chrono::steady_clock::time_point host{};
    XrTime xr_time{};
};
inline DrawTracking tracking;

inline void Publish(const std::optional<OpenXrOutput::ViewFrame>& views, DrawTracking live = {}) {
    std::lock_guard lock(prediction_mutex);
    prediction = views;
    tracking = views ? live : DrawTracking{};
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
// GT7 can start two draws between OpenXR waits. Locate each source at its own
// tracking time and retain those exact views with the submitted display buffer.
inline std::optional<OpenXrOutput::ViewFrame> ReadForDraw() {
    std::lock_guard lock(prediction_mutex);
    const auto now = std::chrono::steady_clock::now();
    if (!prediction || now - updated > std::chrono::milliseconds(250))
        return {};
    if (!tracking.session)
        return prediction;
    auto frame = *prediction;
    frame.display_time =
        tracking.xr_time +
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - tracking.host).count();
    const XrViewLocateInfo where{XR_TYPE_VIEW_LOCATE_INFO, nullptr,
                                 XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, frame.display_time,
                                 tracking.seat};
    XrViewState state{XR_TYPE_VIEW_STATE};
    uint32_t count{};
    constexpr auto valid = XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT;
    if (XR_SUCCEEDED(
            xrLocateViews(tracking.session, &where, &state, 2, &count, frame.views.data())) &&
        count == 2 && (state.viewStateFlags & valid) == valid)
        return frame;
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
