// SPDX-License-Identifier: GPL-2.0-or-later
// A single real Vulkan queue and command buffer stand in for the game scheduler in this test.
#pragma once
#include "video_core/renderer_vulkan/vk_common.h"
#include <cstdio>
#include <format>
#include <mutex>
#include <stdexcept>
#define LOG_WARNING(channel, ...) std::fprintf(stderr, "%s\n", std::format(__VA_ARGS__).c_str())
#define LOG_INFO(channel, ...) std::fprintf(stdout, "%s\n", std::format(__VA_ARGS__).c_str())
namespace Vulkan {
struct Frame {
    uint32_t width, height;
    vk::ImageView image_view;
    bool is_hdr;
};
struct Instance {
    vk::Instance instance;
    vk::PhysicalDevice gpu;
    vk::Device device;
    uint32_t family;
    auto GetInstance() const {
        return instance;
    }
    auto GetPhysicalDevice() const {
        return gpu;
    }
    auto GetDevice() const {
        return device;
    }
    auto GetGraphicsQueueFamilyIndex() const {
        return family;
    }
};
struct Scheduler {
    vk::CommandBuffer cmd;
    vk::Queue queue;
    static inline std::mutex submit_mutex;
    vk::CommandBuffer DirectCommandBuffer() {
        return cmd;
    }
    void Finish() {
        if (cmd.end() != vk::Result::eSuccess)
            throw std::runtime_error("end command buffer");
        vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &cmd};
        if (queue.submit(submit) != vk::Result::eSuccess ||
            queue.waitIdle() != vk::Result::eSuccess || cmd.reset() != vk::Result::eSuccess ||
            cmd.begin(vk::CommandBufferBeginInfo{}) != vk::Result::eSuccess)
            throw std::runtime_error("queue submit/wait");
    }
};
} // namespace Vulkan
