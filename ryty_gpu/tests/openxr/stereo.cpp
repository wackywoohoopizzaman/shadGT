// SPDX-License-Identifier: GPL-2.0-or-later
#include "ryty_gpu/ryty_openxr.h"
#include "ryty_gpu/ryty_openxr_camera.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE
using namespace Vulkan;
static_assert(Gt7XrCamera::SceneDimension(3072, 3072, 4096, 4096, 50) == 1536);
static_assert(Gt7XrCamera::SceneDimension(3264, 3264, 4096, 4096, 100) == 3264);
static_assert(Gt7XrCamera::SceneDimension(1400, 1502, 1600, 1800, 75) == 1127);
static_assert(Gt7XrCamera::SceneDimension(1600, 2000, 1800, 2200, 100) == 1800);
template <class T>
auto must(T result) {
    if (result.result != vk::Result::eSuccess)
        throw std::runtime_error(vk::to_string(result.result));
    return std::move(result.value);
}
struct Tex {
    vk::UniqueDeviceMemory mem;
    vk::UniqueImage image;
    vk::UniqueImageView view;
};
// This fixture uses the simulator file protocol in an isolated LOCALAPPDATA.
int main(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) != "--simulator" ||
        !std::getenv("XR_RUNTIME_JSON") || !std::getenv("LOCALAPPDATA")) {
        std::cerr << "Run using the documented process-local simulator configuration.\n";
        return 1;
    }
    vk::detail::DynamicLoader dl;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
    auto xr = OpenXrOutput::Create();
    if (!xr)
        return 2;
    std::vector<const char*> exts;
    xr->AddInstanceExtensions(exts);
    vk::ApplicationInfo app{.pApplicationName = "shadGT GPU test",
                            .apiVersion = VK_API_VERSION_1_3};
    auto ins = must(vk::createInstanceUnique({.pApplicationInfo = &app,
                                              .enabledExtensionCount = uint32_t(exts.size()),
                                              .ppEnabledExtensionNames = exts.data()}));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*ins);
    auto gpu = xr->PreferredAdapter(*ins);
    if (!gpu)
        return 3;
    std::vector<std::string> available;
    for (auto e : must(gpu.enumerateDeviceExtensionProperties()))
        available.emplace_back(e.extensionName.data());
    exts = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    xr->CheckAdapter(*ins, gpu, available, exts);
    if (!xr->CanCapture())
        return 3;
    auto families = gpu.getQueueFamilyProperties();
    uint32_t family = 0;
    while (family < families.size() &&
           !(families[family].queueFlags & vk::QueueFlagBits::eGraphics))
        ++family;
    if (family == families.size())
        return 3;
    float priority = 1;
    vk::DeviceQueueCreateInfo qc{
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    vk::PhysicalDeviceVulkan12Features f12{.timelineSemaphore = true};
    vk::PhysicalDeviceVulkan13Features f13{
        .pNext = &f12, .synchronization2 = true, .dynamicRendering = true};
    auto dev = must(gpu.createDeviceUnique({.pNext = &f13,
                                            .queueCreateInfoCount = 1,
                                            .pQueueCreateInfos = &qc,
                                            .enabledExtensionCount = uint32_t(exts.size()),
                                            .ppEnabledExtensionNames = exts.data()}));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*dev);
    auto pool = must(dev->createCommandPoolUnique(
        {.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = family}));
    auto buffers =
        must(dev->allocateCommandBuffersUnique({.commandPool = *pool,
                                                .level = vk::CommandBufferLevel::ePrimary,
                                                .commandBufferCount = 1}));
    Scheduler s{*buffers.front(), dev->getQueue(family, 0)};
    if (s.cmd.begin(vk::CommandBufferBeginInfo{}) != vk::Result::eSuccess)
        return 4;
    Instance graphics{*ins, gpu, *dev, family};
    if (OpenXrOutput::HasSession() || OpenXrOutput::IsRunning())
        return 5;
    xr->Start(graphics);
    if (!xr->CanCapture() || !OpenXrOutput::HasSession() || OpenXrOutput::IsRunning())
        return 5;
    constexpr int w = 256, h = 144;
    auto memory = gpu.getMemoryProperties();
    auto choose = [&](auto req, vk::MemoryPropertyFlags flags) {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((req.memoryTypeBits & (1u << i)) &&
                ((memory.memoryTypes[i].propertyFlags & flags) == flags))
                return i;
        throw std::runtime_error("memory type unavailable");
    };
    auto texture = [&](const XrView* left, const XrView* right = nullptr) {
        const uint32_t width = right ? w * 2 : w;
        Tex t;
        const auto format = vk::Format::eR8G8B8A8Unorm;
        t.image = must(dev->createImageUnique(
            {.imageType = vk::ImageType::e2D,
             .format = format,
             .extent = {width, h, 1},
             .mipLevels = 1,
             .arrayLayers = 1,
             .samples = vk::SampleCountFlagBits::e1,
             .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst}));
        auto req = dev->getImageMemoryRequirements(*t.image);
        t.mem = must(dev->allocateMemoryUnique(
            {.allocationSize = req.size, .memoryTypeIndex = choose(req, {})}));
        if (dev->bindImageMemory(*t.image, *t.mem, 0) != vk::Result::eSuccess)
            abort();
        vk::UniqueDeviceMemory staging;
        auto buf = must(dev->createBufferUnique({.size = static_cast<vk::DeviceSize>(width) * h * 4,
                                                 .usage = vk::BufferUsageFlagBits::eTransferSrc}));
        req = dev->getBufferMemoryRequirements(*buf);
        staging = must(dev->allocateMemoryUnique(
            {.allocationSize = req.size,
             .memoryTypeIndex = choose(req, vk::MemoryPropertyFlagBits::eHostVisible |
                                                vk::MemoryPropertyFlagBits::eHostCoherent)}));
        if (dev->bindBufferMemory(*buf, *staging, 0) != vk::Result::eSuccess)
            abort();
        uint32_t* p = (uint32_t*)must(
            dev->mapMemory(*staging, 0, static_cast<vk::DeviceSize>(width) * h * 4));
        for (int y = 0; y < h; y++)
            for (uint32_t x = 0; x < width; x++) {
                const auto* eye = x < w ? left : right;
                if (!eye) {
                    p[y * width + x] = 0xff989898;
                    continue;
                }
                float u = ((x % w) + 0.5f) / w, v = (y + 0.5f) / h;
                const XrVector3f viewRay{
                    std::lerp(std::tan(eye->fov.angleLeft), std::tan(eye->fov.angleRight), u),
                    std::lerp(std::tan(eye->fov.angleUp), std::tan(eye->fov.angleDown), v), -1};
                const auto q = eye->pose.orientation;
                const auto cross = [](XrVector3f a, XrVector3f b) {
                    return XrVector3f{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                                      a.x * b.y - a.y * b.x};
                };
                const XrVector3f axis{q.x, q.y, q.z};
                auto tangent = cross(axis, viewRay);
                auto twice = cross(axis, tangent);
                XrVector3f ray{viewRay.x + 2 * (q.w * tangent.x + twice.x),
                               viewRay.y + 2 * (q.w * tangent.y + twice.y),
                               viewRay.z + 2 * (q.w * tangent.z + twice.z)};
                const auto pos = eye->pose.position;
                // A narrow rear post at +6 cm is hidden by a foreground pillar in just one eye.
                auto hit = [&](float z, float x0, float halfWidth, float halfHeight) {
                    float t = (z - pos.z) / ray.z;
                    return t > 0 && std::abs(pos.x + t * ray.x - x0) < halfWidth &&
                           std::abs(pos.y + t * ray.y) < halfHeight;
                };
                p[y * width + x] = hit(-0.5f, 0, 0.030f, 0.8f)        ? 0xff0000ff
                                   : hit(-2.f, 0.060f, 0.025f, 0.35f) ? 0xff00ff00
                                                                      : 0xff400000;
            }
        dev->unmapMemory(*staging);
        vk::ImageMemoryBarrier2 b{
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *t.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
        s.cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
        s.cmd.copyBufferToImage(
            *buf, *t.image, vk::ImageLayout::eGeneral,
            vk::BufferImageCopy{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                .imageExtent = {width, h, 1}});
        s.Finish();
        t.view = must(dev->createImageViewUnique({.image = *t.image,
                                                  .viewType = vk::ImageViewType::e2D,
                                                  .format = format,
                                                  .subresourceRange = b.subresourceRange}));
        return t;
    };
    auto screen = texture(nullptr);
    Frame frame{w, h, *screen.view, false};
    int pairs = 0;
    for (int i = 0; i < 155 && xr->CanCapture(); ++i) {
        auto viewFrame = xr->BeginFrame();
        if (!viewFrame || i < 25) {
            xr->Present(s, frame);
            continue;
        }
        if (xr->BeginFrame()->id != viewFrame->id ||
            viewFrame->views[0].pose.position.x >= viewFrame->views[1].pose.position.x)
            return 7;
        Tex left = texture(&viewFrame->views[0]), right = texture(&viewFrame->views[1]);
        OpenXrOutput::StereoImages pair{viewFrame->id, {*left.view, *right.view}, false};
        if (i >= 80 && i < 95)
            --pair.frame_id;
        if (i >= 95)
            pair.colors[1] = pair.colors[0];
        if (i == 65 || i == 85 || i == 101 || i == 115 || i == 130 || i == 145) {
            auto dir = std::filesystem::path(std::getenv("LOCALAPPDATA")) / "OpenXR-Simulator";
            const char* id = i == 65    ? "50505050505050505050505050505050"
                             : i == 85  ? "60606060606060606060606060606060"
                             : i == 101 ? "70707070707070707070707070707070"
                             : i == 115 ? "80808080808080808080808080808080"
                             : i == 130 ? "90909090909090909090909090909090"
                                        : "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0";
            std::ofstream o(dir / "request.tmp");
            o << std::format(R"({{"capture_id":"{}","eye":"both","layer":"all","after_sync":0}})",
                             id);
            o.close();
            std::filesystem::rename(dir / "request.tmp", dir / "screenshot_request.json");
        }
        if (i >= 125) {
            // Only the native eye textures contain the test geometry.
            GameStereoFrame game{*viewFrame};
            OpenXrOutput::StereoImages native_eyes{
                viewFrame->id, {*left.view, *right.view}, false, i >= 140};
            xr->Present(s, frame, &native_eyes, &game, {w, h});
        } else if (i >= 110) {
            auto packed = texture(&viewFrame->views[0], &viewFrame->views[1]);
            Frame packedFrame{w * 2, h, *packed.view, false};
            GameStereoFrame game{*viewFrame};
            xr->Present(s, packedFrame, nullptr, &game, {});
        } else {
            xr->Present(s, frame, &pair, nullptr, {});
        }
        if (i < 80)
            ++pairs;
    }
    const bool ok = xr->CanCapture() && OpenXrOutput::IsRunning() && pairs > 20;
    xr.reset();
    return ok && !OpenXrOutput::HasSession() && !OpenXrOutput::IsRunning() ? 0 : 6;
}
