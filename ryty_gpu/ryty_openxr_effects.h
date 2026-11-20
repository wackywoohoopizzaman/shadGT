// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "ryty_gpu/ryty_openxr.h"

namespace Vulkan {
struct XrEffectImage {
    vk::Image image;
    vk::ImageView view;
    vk::Format format;
    vk::Extent2D size;
};
struct XrEffectFrame {
    XrEffectImage color, motion, depth, scaled, generated;
    float dt_ms, vertical_fov;
    bool reset;
    uint64_t sequence;
    XrPosef pose;
    std::array<float, 2> jitter_pixels{};
};
namespace XrVendor {
bool Fidelity(vk::PhysicalDevice gpu, vk::Device device, vk::CommandBuffer cmd, void*& context,
              const XrEffectFrame& frame, bool generate);
void DestroyFidelity(void*& context);
bool GenerateDlss(vk::CommandBuffer cmd, const XrEffectFrame& frame, unsigned eye, const XrView&,
                  unsigned count, unsigned index);
bool Dlss(vk::CommandBuffer cmd, const XrEffectFrame& frame, unsigned eye);
} // namespace XrVendor
class XrEyeEffects {
  public:
    enum class Stage { Scene, FrameGeneration };
    XrEyeEffects(const Instance&, Stage);
    ~XrEyeEffects();
    static constexpr unsigned MaxIntermediate = 1;
    unsigned GenerationLimit() const;
    struct Pair {
        std::array<vk::Image, 2> reconstructed{}; // linear HDR in GENERAL layout
        std::array<std::array<vk::ImageView, 2>, MaxIntermediate> generated{};
        unsigned count{};
    };
    Pair Process(vk::CommandBuffer cmd, const OpenXrOutput::StereoImages& source,
                 const OpenXrOutput::ViewFrame& cameras, vk::Extent2D input,
                 const std::array<vk::Extent2D, 2>& output, int64_t source_interval_ns,
                 unsigned slots = 1);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace Vulkan
