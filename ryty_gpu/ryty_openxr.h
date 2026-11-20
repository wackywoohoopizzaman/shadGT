// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include "video_core/renderer_vulkan/vk_common.h"
#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <openxr/openxr.h>
#include <optional>
#include <vector>

namespace Vulkan {
class Instance;
class Scheduler;
struct Frame;
struct GameStereoFrame;

class OpenXrOutput {
public:
    static std::unique_ptr<OpenXrOutput> Create();
    ~OpenXrOutput();
    void AddInstanceExtensions(std::vector<const char*>& enabled);
    vk::PhysicalDevice PreferredAdapter(vk::Instance instance) const;
    void CheckAdapter(vk::Instance instance, vk::PhysicalDevice gpu,
                      const std::vector<std::string>& available, std::vector<const char*>& enabled);
    void Start(const Instance& graphics);
    bool CanCapture() const;
    static bool HasSession() {
        return bound_sessions.load(std::memory_order_relaxed) != 0;
    }

    static bool IsRunning() {
        return active_sessions.load(std::memory_order_relaxed) != 0;
    }

    struct ViewFrame {
        uint64_t id;
        XrTime display_time;
        std::array<XrView, 2> views;
        std::array<vk::Extent2D, 2> sizes;
        uint64_t origin{};
    };
    struct StereoImages {
        uint64_t frame_id;
        // Left and right color views in GENERAL layout, sampled after the game's
        // draw.
        std::array<vk::ImageView, 2> colors;
        bool display_encoded;
        bool movie{};
        std::array<vk::ImageView, 2> depth{}, motion{};
        std::array<std::array<float, 4>, 2>
            motion_scale{};        // each eye's velocity scale and jitter in motion units
        vk::Extent2D color_size{}; // frame generation consumes the final composited color
    };
    // Called before camera-dependent work and paired with Present on the XR frame
    // owner. Repeated queries before the flip return the same predicted views.
    std::optional<ViewFrame> BeginFrame();
    void Present(Scheduler& scheduler, const Frame& frame, const StereoImages* stereo = nullptr,
                 const GameStereoFrame* game = nullptr, vk::Extent2D game_eye_size = {});

private:
    inline static std::atomic<unsigned> bound_sessions{}, active_sessions{};
    struct State;
    explicit OpenXrOutput(std::unique_ptr<State> state);
    std::unique_ptr<State> state;
};
struct GameStereoFrame {
    OpenXrOutput::ViewFrame cameras;
    uint64_t source_id{};
    int64_t source_time_ns{}; // Producer clock at display-buffer selection.
    uint32_t reflex_token{};
};
} // namespace Vulkan
