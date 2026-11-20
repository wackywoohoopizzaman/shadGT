// SPDX-License-Identifier: GPL-2.0-or-later
#include "ryty_gpu/ryty_openxr_effects.h"
#include "common/logging/log.h"
#include "ryty_gpu/ryty_openxr_camera.h"
#include "ryty_gpu/ryty_upscale.h"
#include "video_core/host_shaders/openxr_inputs_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include <cstdlib>
#include <string_view>

namespace Vulkan {
namespace {
constexpr auto storage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                         vk::ImageUsageFlagBits::eTransferSrc;
constexpr auto compute = vk::ShaderStageFlagBits::eCompute;
constexpr auto general = vk::ImageLayout::eGeneral;
constexpr auto read = vk::ImageLayout::eShaderReadOnlyOptimal;
constexpr vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
struct Image {
    vk::UniqueDeviceMemory memory;
    vk::UniqueImage image;
    vk::UniqueImageView view;
    XrEffectImage desc{};

    bool Make(vk::Device device, vk::PhysicalDevice gpu, vk::Extent2D size, vk::Format format) {
        auto [ir, i] = device.createImageUnique({.imageType = vk::ImageType::e2D,
                                                 .format = format,
                                                 .extent = {size.width, size.height, 1},
                                                 .mipLevels = 1,
                                                 .arrayLayers = 1,
                                                 .samples = vk::SampleCountFlagBits::e1,
                                                 .usage = storage});
        if (ir != vk::Result::eSuccess)
            return false;
        const auto requirements = device.getImageMemoryRequirements(*i);
        const auto props = gpu.getMemoryProperties();
        for (uint32_t n = 0; n < props.memoryTypeCount; ++n) {
            if (!(requirements.memoryTypeBits & (1u << n)) ||
                !(props.memoryTypes[n].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
                continue;
            auto [mr, m] = device.allocateMemoryUnique(
                {.allocationSize = requirements.size, .memoryTypeIndex = n});
            if (mr != vk::Result::eSuccess ||
                device.bindImageMemory(*i, *m, 0) != vk::Result::eSuccess)
                continue;
            auto [vr, v] = device.createImageViewUnique({.image = *i,
                                                         .viewType = vk::ImageViewType::e2D,
                                                         .format = format,
                                                         .subresourceRange = range});
            if (vr != vk::Result::eSuccess)
                return false;
            memory = std::move(m);
            image = std::move(i);
            view = std::move(v);
            desc = {*image, *view, format, size};
            return true;
        }
        return false;
    }
};

void Transition(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from, vk::ImageLayout to) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = from,
        .newLayout = to,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = range};
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}
} // namespace

struct XrEyeEffects::Impl {
    vk::Device device;
    vk::PhysicalDevice gpu;
    vk::UniqueDescriptorSetLayout descriptors;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::UniqueSampler sampler;
    bool dlss{}, fsr{}, failed{}, sr_failed{};
    uint64_t sequence{};
    bool have_history{};
    vk::Extent2D source_size{}, color_size{};
    std::array<vk::Extent2D, 2> output_sizes{};
    struct Eye {
        Image color, depth, motion, scaled;
        void* sr_ctx{};
        ~Eye() {
            XrVendor::DestroyFidelity(sr_ctx);
        }
    };
    std::array<std::unique_ptr<Eye>, 2> eyes;

    bool Pipeline() {
        std::array<vk::DescriptorSetLayoutBinding, 6> b{};
        for (unsigned n = 0; n < b.size(); ++n)
            b[n] = {n,
                    n < 3 ? vk::DescriptorType::eCombinedImageSampler
                          : vk::DescriptorType::eStorageImage,
                    1, compute};
        auto [dr, ds] = device.createDescriptorSetLayoutUnique(
            {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
             .bindingCount = unsigned(b.size()),
             .pBindings = b.data()});
        if (dr != vk::Result::eSuccess)
            return false;
        descriptors = std::move(ds);
        const vk::PushConstantRange pc{compute, 0, 16};
        auto [lr, l] = device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                          .pSetLayouts = &*descriptors,
                                                          .pushConstantRangeCount = 1,
                                                          .pPushConstantRanges = &pc});
        if (lr != vk::Result::eSuccess)
            return false;
        layout = std::move(l);
        auto [mr, mod] = device.createShaderModuleUnique(
            {.codeSize = sizeof(OPENXR_INPUTS_COMP), .pCode = OPENXR_INPUTS_COMP});
        if (mr != vk::Result::eSuccess)
            return false;
        auto [pr, p] = device.createComputePipelineUnique(
            {}, {.stage = {.stage = compute, .module = *mod, .pName = "main"}, .layout = *layout});
        if (pr != vk::Result::eSuccess)
            return false;
        pipeline = std::move(p);
        auto [sr, s] = device.createSamplerUnique({});
        if (sr != vk::Result::eSuccess)
            return false;
        sampler = std::move(s);
        return true;
    }

    bool Allocate(vk::Extent2D input, vk::Extent2D color,
                  const std::array<vk::Extent2D, 2>& output) {
        if (!pipeline && !Pipeline())
            return false;
        // Present waits for all reads of the previous eye resources before returning.
        for (unsigned i = 0; i < 2; ++i) {
            eyes[i] = std::make_unique<Eye>();
            auto& e = *eyes[i];
            constexpr auto rgba = vk::Format::eR16G16B16A16Sfloat;
            if (!e.color.Make(device, gpu, color, rgba) ||
                !e.depth.Make(device, gpu, input, vk::Format::eR32Sfloat) ||
                !e.motion.Make(device, gpu, input, vk::Format::eR16G16Sfloat))
                return false;
            if ((fsr || dlss) && !e.scaled.Make(device, gpu, output[i], rgba))
                return false;
        }
        source_size = input;
        color_size = color;
        output_sizes = output;
        have_history = false;
        return true;
    }

    void Prepare(vk::CommandBuffer cmd, Eye& e, const OpenXrOutput::StereoImages& s, unsigned i) {
        for (auto* im : {&e.color, &e.depth, &e.motion})
            Transition(cmd, *im->image, vk::ImageLayout::eUndefined, general);
        const vk::MemoryBarrier2 b{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                   .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                   .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                                   .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &b});
        const std::array<vk::ImageView, 6> views{s.colors[i],   s.depth[i],    s.motion[i],
                                                 *e.color.view, *e.depth.view, *e.motion.view};
        std::array<vk::DescriptorImageInfo, 6> imgs{};
        std::array<vk::WriteDescriptorSet, 6> writes{};
        for (unsigned n = 0; n < views.size(); ++n) {
            imgs[n] = {n < 3 ? *sampler : vk::Sampler{}, views[n], general};
            writes[n] = {.dstBinding = n,
                         .descriptorCount = 1,
                         .descriptorType = n < 3 ? vk::DescriptorType::eCombinedImageSampler
                                                 : vk::DescriptorType::eStorageImage,
                         .pImageInfo = &imgs[n]};
        }
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *layout, 0, writes);
        cmd.pushConstants(*layout, compute, 0, 16, s.motion_scale[i].data());
        cmd.dispatch((std::max(color_size.width, source_size.width) + 7) / 8,
                     (std::max(color_size.height, source_size.height) + 7) / 8, 1);
        for (auto* im : {&e.color, &e.depth, &e.motion})
            Transition(cmd, *im->image, general, read);
    }
};

XrEyeEffects::XrEyeEffects(const Instance& instance, Stage stage) : impl(std::make_unique<Impl>()) {
    impl->device = instance.GetDevice();
    impl->gpu = instance.GetPhysicalDevice();
    const char* mode = std::getenv("SHADGT_OPENXR_UPSCALER");
    impl->fsr = stage == Stage::Scene && mode && std::string_view{mode} == "fsr";
    impl->dlss = stage == Stage::Scene && mode && std::string_view{mode} == "dlss";

}
XrEyeEffects::~XrEyeEffects() = default;


XrEyeEffects::Pair XrEyeEffects::Process(vk::CommandBuffer cmd,
                                         const OpenXrOutput::StereoImages& source,
                                         const OpenXrOutput::ViewFrame& cameras, vk::Extent2D input,
                                         const std::array<vk::Extent2D, 2>& output,
                                         int64_t source_interval_ns) {
    Pair pair{};
    auto& s = *impl;
    const bool upscale = s.fsr || s.dlss;
    if (s.failed || !upscale || s.sr_failed || !input.width || !input.height)
        return pair;
    for (unsigned i = 0; i < 2; ++i) {
        if (!source.colors[i] || !source.depth[i] || !source.motion[i] ||
            input.width > output[i].width || input.height > output[i].height) {
            s.have_history = false;
            return pair;
        }
    }
    const auto color_size = source.color_size.width ? source.color_size : input;
    if ((s.source_size != input || s.color_size != color_size || s.output_sizes != output) &&
        !s.Allocate(input, color_size, output)) {
        LOG_WARNING(Render_Vulkan, "OpenXR: could not allocate eye processing resources");
        s.failed = true;
        s.eyes = {};
        return pair;
    }
    const bool reset = !s.have_history || source_interval_ns <= 0;
    const float dt = reset ? 16.667f : float(source_interval_ns) / 1e6f;
    s.have_history = true;
    ++s.sequence;
    std::array<XrEffectFrame, 2> frames{};
    for (unsigned i = 0; i < 2; ++i) {
        auto& e = *s.eyes[i];
        s.Prepare(cmd, e, source, i);
        const auto& fov = cameras.views[i].fov;
        const auto& motion = source.motion_scale[i];
        frames[i] = {e.color.desc,
                     e.motion.desc,
                     e.depth.desc,
                     e.scaled.desc,
                     {},
                     dt,
                     fov.angleUp - fov.angleDown,
                     reset,
                     s.sequence,
                     cameras.views[i].pose,
                     {motion[0] * input.width * motion[2], motion[1] * input.height * motion[3]}};
    }

    if ((s.fsr || s.dlss) && !s.sr_failed) {
        for (unsigned i = 0; i < 2; ++i) {
            auto& f = frames[i];
            Transition(cmd, f.scaled.image, vk::ImageLayout::eUndefined, general);
            bool ok = s.dlss
                          ? XrVendor::Dlss(cmd, f, i)
                          : XrVendor::Fidelity(s.gpu, s.device, cmd, s.eyes[i]->sr_ctx, f, false);

            if (!ok) {
                LOG_WARNING(Render_Vulkan,
                            "OpenXR: {} unavailable for eye {}, keeping GT7's resolved stereo pair",
                            s.dlss ? "DLSS" : "FSR 3.1", i);
                s.sr_failed = true;
                pair.reconstructed = {};
                break;
            }
            pair.reconstructed[i] = f.scaled.image;
        }
    }
    if (s.dlss) {
        static const std::string_view detail = [] {
            const char* value = std::getenv("SHADGT_OPENXR_DLSS_TEXTURE_DETAIL");
            return std::string_view{value ? value : "full"};
        }();
        // Apply the material bias only when both histories produced a replacement.
        const float bias = pair.reconstructed[0] && pair.reconstructed[1] && detail != "off"
                               ? RytyUpscale::DlssMaterialMipBias(
                                     input, output[0], detail == "reduced" ? 0.0f : -1.0f)
                               : 0.0f;
        Gt7XrCamera::dlss_material_mip_bias.store(bias, std::memory_order_relaxed);
    }
    return pair;
}
} // namespace Vulkan
