// SPDX-License-Identifier: GPL-2.0-or-later
#include "ryty_gpu/ryty_openxr.h"
#include "ryty_gpu/ryty_openxr_camera.h"
#include "ryty_gpu/openxr_system.h"

#ifdef _WIN32
#include <windows.h>
#include <unknwn.h>
#define XR_USE_PLATFORM_WIN32
#endif
#define XR_USE_GRAPHICS_API_VULKAN
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/host_shaders/openxr_eye_frag.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <openxr/openxr_platform.h>
#include <sstream>

namespace Vulkan {
namespace {
constexpr XrPosef Identity{{0, 0, 0, 1}, {0, 0, 0}};
constexpr auto Stereo = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
constexpr auto Color = vk::ImageAspectFlagBits::eColor;

uint32_t EyePercent(const char* value, uint32_t fallback) {
    if (!value)
        return fallback;
    const std::string_view text{value};
    uint32_t percent{};
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, percent);
    if (parsed.ec == std::errc{} && parsed.ptr == end &&
        ((percent >= 25 && percent <= 100) || (percent == 0 && fallback == 0)))
        return percent;
    LOG_WARNING(Render_Vulkan, "OpenXR: invalid eye percentage '{}', using {}", text, fallback);
    return fallback;
}

bool Success(XrResult result, const char* operation) {
    if (XR_SUCCEEDED(result))
        return true;
    LOG_WARNING(Render_Vulkan, "OpenXR: {} failed ({})", operation, int(result));
    return false;
}


struct EyeTarget {
    XrSwapchain chain{};
    vk::Extent2D size{};
    std::vector<XrSwapchainImageVulkanKHR> images;
    std::vector<vk::UniqueImageView> views;
};

template <typename GetNames>
bool ReadExtensions(GetNames get_names, XrInstance xr, XrSystemId system,
                    std::vector<std::string>& names) {
    uint32_t count{};
    if (!Success(get_names(xr, system, 0, &count, nullptr), "query Vulkan extensions"))
        return false;
    if (!count)
        return true;
    std::vector<char> buf(count);
    if (!Success(get_names(xr, system, count, &count, buf.data()), "read Vulkan extensions"))
        return false;
    std::istringstream input(buf.data());
    for (std::string name; input >> name;)
        names.push_back(std::move(name));
    return true;
}

void Append(const std::vector<std::string>& names, std::vector<const char*>& enabled) {
    for (const auto& name : names) {
        if (std::ranges::find(enabled, name) == enabled.end())
            enabled.push_back(name.c_str());
    }
}
} // namespace

// FEAT-SGT-075: the Vulkan device outlives every resource owned by this session.
struct OpenXrOutput::State {
    XrInstance xr{};
    XrSystemId system{};
    XrSession session{};
    XrSpace local{}, head{}, seat{};
    XrTime recenter_at{};
    XrEnvironmentBlendMode blend{};
    PFN_xrGetVulkanGraphicsDeviceKHR graphics_device{};
    std::vector<std::string> instance_exts, device_exts;
    vk::Device device{};
    vk::Format format{};
    std::array<EyeTarget, 2> eyes{};
    vk::UniqueSampler linear;
    vk::UniqueDescriptorSetLayout descriptors;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    std::atomic<bool> enabled{true};
    bool running{}, focused{}, seat_calibrated{}, hdr_reported{};
#ifdef SHADGT_DIAGNOSTICS
    int last_mode{-1};
#endif
    uint64_t frame_id{};
    uint64_t origin{};
    bool frame_begun{}, screen_ready{};
    XrTime display_time{};
    XrDuration display_period{};
    std::optional<ViewFrame> located;

    void CloseSession() {
        if (Gt7XrCamera::Requested()) {
            Gt7XrCamera::SetAllocationSize({});
            Gt7XrCamera::game_eye_extent = 0;
            Gt7XrCamera::Publish({});
        }
        for (auto& eye : eyes) {
            eye.views.clear();
            if (eye.chain)
                xrDestroySwapchain(eye.chain);
            eye.chain = XR_NULL_HANDLE;
        }
        if (seat)
            xrDestroySpace(seat);
        if (head)
            xrDestroySpace(head);
        if (local)
            xrDestroySpace(local);
        if (session) {
            xrDestroySession(session);
            bound_sessions.fetch_sub(1, std::memory_order_relaxed);
        }
        seat = head = local = XR_NULL_HANDLE;
        focused = seat_calibrated = false;
        session = XR_NULL_HANDLE;
        if (running) {
            active_sessions.fetch_sub(1, std::memory_order_relaxed);
        }
        running = frame_begun = screen_ready = false;
        located.reset();
    }
    ~State() {
        CloseSession();
        if (xr)
            xrDestroyInstance(xr);
    }
    bool Initialize();
    bool Bind(const Instance& graphics);
    bool MakePipeline();
    bool Poll();
    bool Center(XrTime time);
    void Draw(vk::CommandBuffer cmd, const EyeTarget& target, uint32_t index, vk::ImageView source,
              bool encoded, uint32_t region);
};

OpenXrOutput::OpenXrOutput(std::unique_ptr<State> s) : state(std::move(s)) {}
OpenXrOutput::~OpenXrOutput() = default;

std::unique_ptr<OpenXrOutput> OpenXrOutput::Create() {
    const char* value = std::getenv("SHARED_OPENXR");
    if (!value || std::string_view{value} != "1")
        return {};
    auto s = std::make_unique<State>();
    if (!s->Initialize())
        return {};
    return std::unique_ptr<OpenXrOutput>(new OpenXrOutput(std::move(s)));
}

bool OpenXrOutput::State::Initialize() {
    RygtXr::Extensions extensions;
    std::ostringstream report;
    if (!RygtXr::OpenSystem(xr, system, extensions, false, report)) {
        LOG_WARNING(Render_Vulkan, "OpenXR: {}", report.str());
        return false;
    }
#ifdef SHADGT_DIAGNOSTICS
    if (RygtXr::DescribeSystem(xr, system, extensions, false, report)) {
        LOG_INFO(Render_Vulkan, "OpenXR: {}", report.str());
    } else {
        LOG_WARNING(Render_Vulkan, "OpenXR: {}", report.str());
    }
#endif
    auto proc = [&](const char* name, auto& pointer) {
        PFN_xrVoidFunction address{};
        if (!Success(xrGetInstanceProcAddr(xr, name, &address), name) || !address)
            return false;
        pointer = reinterpret_cast<std::remove_reference_t<decltype(pointer)>>(address);
        return true;
    };
    PFN_xrGetVulkanGraphicsRequirementsKHR requirements{};
    PFN_xrGetVulkanInstanceExtensionsKHR instances{};
    PFN_xrGetVulkanDeviceExtensionsKHR devices{};
    if (!proc("xrGetVulkanGraphicsRequirementsKHR", requirements) ||
        !proc("xrGetVulkanGraphicsDeviceKHR", graphics_device) ||
        !proc("xrGetVulkanInstanceExtensionsKHR", instances) ||
        !proc("xrGetVulkanDeviceExtensionsKHR", devices))
        return false;
    XrGraphicsRequirementsVulkanKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
    if (!Success(requirements(xr, system, &req), "Vulkan requirements"))
        return false;
    if (req.minApiVersionSupported > XR_MAKE_VERSION(1, 3, 0)) {
        LOG_WARNING(Render_Vulkan, "OpenXR runtime requires a Vulkan version above 1.3");
        return false;
    }
    return ReadExtensions(instances, xr, system, instance_exts) &&
           ReadExtensions(devices, xr, system, device_exts);
}

void OpenXrOutput::AddInstanceExtensions(std::vector<const char*>& enabled) {
    auto [result, props] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess) {
        state->enabled = false;
        return;
    }
    for (const auto& name : state->instance_exts) {
        if (!std::ranges::any_of(props,
                                 [&](const auto& p) { return name == p.extensionName.data(); })) {
            LOG_WARNING(Render_Vulkan, "OpenXR needs unavailable Vulkan extension {}", name);
            state->enabled = false;
            return;
        }
    }
    Append(state->instance_exts, enabled);
}

vk::PhysicalDevice OpenXrOutput::PreferredAdapter(vk::Instance instance) const {
    VkPhysicalDevice required{};
    if (state->enabled &&
        Success(state->graphics_device(state->xr, state->system, instance, &required),
                "find XR GPU")) {
        return required;
    }
    return {};
}

void OpenXrOutput::CheckAdapter(vk::Instance instance, vk::PhysicalDevice gpu,
                                const std::vector<std::string>& available,
                                std::vector<const char*>& enabled) {
    if (!state->enabled)
        return;
    if (PreferredAdapter(instance) != gpu) {
        LOG_WARNING(Render_Vulkan, "OpenXR requires the game and headset to use the same GPU");
        state->enabled = false;
        return;
    }
    for (const auto& name : state->device_exts) {
        if (std::ranges::find(available, name) == available.end()) {
            LOG_WARNING(Render_Vulkan, "OpenXR needs unavailable GPU extension {}", name);
            state->enabled = false;
            return;
        }
    }
    Append(state->device_exts, enabled);
}

void OpenXrOutput::Start(const Instance& graphics) {
    if (state->enabled && !state->Bind(graphics)) {
        state->enabled = false;
        state->CloseSession();
        LOG_WARNING(Render_Vulkan, "OpenXR setup failed, continuing on desktop");
    }
}
bool OpenXrOutput::CanCapture() const {
    return state->enabled.load(std::memory_order_relaxed);
}

bool OpenXrOutput::State::Bind(const Instance& graphics) {
    device = graphics.GetDevice();
    XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    binding.instance = graphics.GetInstance();
    binding.physicalDevice = graphics.GetPhysicalDevice();
    binding.device = device;
    binding.queueFamilyIndex = graphics.GetGraphicsQueueFamilyIndex();
    XrSessionCreateInfo ci{XR_TYPE_SESSION_CREATE_INFO, &binding};
    ci.systemId = system;
    if (!Success(xrCreateSession(xr, &ci, &session), "create Vulkan session"))
        return false;
    bound_sessions.fetch_add(1, std::memory_order_relaxed);
    XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space.poseInReferenceSpace = Identity;
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    if (!Success(xrCreateReferenceSpace(session, &space, &local), "create local space"))
        return false;
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    if (!Success(xrCreateReferenceSpace(session, &space, &head), "create view space"))
        return false;
    uint32_t n{};
    if (!Success(xrEnumerateEnvironmentBlendModes(xr, system, Stereo, 0, &n, nullptr),
                 "count blend modes") ||
        !n)
        return false;
    std::vector<XrEnvironmentBlendMode> modes(n);
    if (!Success(xrEnumerateEnvironmentBlendModes(xr, system, Stereo, n, &n, modes.data()),
                 "read blend modes"))
        return false;
    blend = std::ranges::find(modes, XR_ENVIRONMENT_BLEND_MODE_OPAQUE) != modes.end()
                ? XR_ENVIRONMENT_BLEND_MODE_OPAQUE
                : modes.front();
    if (!Success(xrEnumerateSwapchainFormats(session, 0, &n, nullptr), "count color formats"))
        return false;
    std::vector<int64_t> formats(n);
    if (!Success(xrEnumerateSwapchainFormats(session, n, &n, formats.data()), "read color formats"))
        return false;
    for (auto f : formats) {
        if (f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB) {
            format = static_cast<vk::Format>(f);
            break;
        }
    }
    if (format == vk::Format::eUndefined) {
        LOG_WARNING(Render_Vulkan, "OpenXR runtime offers no supported sRGB color format");
        return false;
    }
    std::array<XrViewConfigurationView, 2> configs{};
    for (auto& cfg : configs)
        cfg.type = XR_TYPE_VIEW_CONFIGURATION_VIEW;
    if (!Success(xrEnumerateViewConfigurationViews(xr, system, Stereo, 2, &n, configs.data()),
                 "read stereo views") ||
        n != 2)
        return false;
    const char* output_setting = std::getenv("SHADGT_OPENXR_OUTPUT_PERCENT");
    const bool use_output_size = output_setting != nullptr;
    const auto output_percent = EyePercent(output_setting, 100);
    const auto shared_size = [&](uint32_t percent) -> vk::Extent2D {
        const auto& l = configs[0];
        const auto& r = configs[1];
        return {
            Gt7XrCamera::SceneDimension(l.recommendedImageRectWidth, r.recommendedImageRectWidth,
                                        l.maxImageRectWidth, r.maxImageRectWidth, percent),
            Gt7XrCamera::SceneDimension(l.recommendedImageRectHeight, r.recommendedImageRectHeight,
                                        l.maxImageRectHeight, r.maxImageRectHeight, percent)};
    };
    const auto output = shared_size(output_percent);
    for (size_t i = 0; i < eyes.size(); ++i) {
        auto& eye = eyes[i];
        eye.size = use_output_size ? output
                                   : vk::Extent2D{configs[i].recommendedImageRectWidth,
                                                  configs[i].recommendedImageRectHeight};
        XrSwapchainCreateInfo sc{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        sc.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
        sc.format = static_cast<int64_t>(format);
        sc.width = eye.size.width;
        sc.height = eye.size.height;
        sc.sampleCount = sc.faceCount = sc.arraySize = sc.mipCount = 1;
        const auto created = xrCreateSwapchain(session, &sc, &eye.chain);
        if (!Success(created, "create eye swapchain") ||
            !Success(xrEnumerateSwapchainImages(eye.chain, 0, &n, nullptr), "count eye images") ||
            !n)
            return false;
        eye.images.resize(n, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        if (!Success(xrEnumerateSwapchainImages(
                         eye.chain, n, &n,
                         reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())),
                     "read eye images"))
            return false;
        for (const auto& image : eye.images) {
            auto [res, view] =
                device.createImageViewUnique({.image = image.image,
                                              .viewType = vk::ImageViewType::e2D,
                                              .format = format,
                                              .subresourceRange = {Color, 0, 1, 0, 1}});
            if (res != vk::Result::eSuccess)
                return false;
            eye.views.push_back(std::move(view));
        }
    }
    if (!MakePipeline())
        return false;
    if (Gt7XrCamera::Requested()) {
        if (use_output_size) {
            Gt7XrCamera::SetAllocationSize(output);
        } else if (const auto percent =
                       EyePercent(std::getenv("SHADGT_OPENXR_RENDER_PERCENT"), 0)) {
            Gt7XrCamera::SetAllocationSize(shared_size(percent));
        }
    }
    return true;
}

bool OpenXrOutput::State::MakePipeline() {
    const auto make_sampler = [&](vk::Filter filter) {
        return device.createSamplerUnique({.magFilter = filter,
                                           .minFilter = filter,
                                           .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                                           .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                                           .addressModeW = vk::SamplerAddressMode::eClampToEdge});
    };
    auto [lr, ls] = make_sampler(vk::Filter::eLinear);
    if (lr != vk::Result::eSuccess)
        return false;
    linear = std::move(ls);
    constexpr auto fragment = vk::ShaderStageFlagBits::eFragment;
    const vk::DescriptorSetLayoutBinding binding{0, vk::DescriptorType::eCombinedImageSampler, 1,
                                                 fragment};
    auto [dr, ds] = device.createDescriptorSetLayoutUnique(
        {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
         .bindingCount = 1,
         .pBindings = &binding});
    if (dr != vk::Result::eSuccess)
        return false;
    descriptors = std::move(ds);
    const vk::PushConstantRange range{fragment, 0, 5 * sizeof(uint32_t)};
    auto [pr, pl] = device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                       .pSetLayouts = &*descriptors,
                                                       .pushConstantRangeCount = 1,
                                                       .pPushConstantRanges = &range});
    if (pr != vk::Result::eSuccess)
        return false;
    layout = std::move(pl);
    auto [vr, vs] =
        device.createShaderModuleUnique({.codeSize = sizeof(FS_TRI_VERT), .pCode = FS_TRI_VERT});
    auto [fr, fs] = device.createShaderModuleUnique(
        {.codeSize = sizeof(OPENXR_EYE_FRAG), .pCode = OPENXR_EYE_FRAG});
    if (vr != vk::Result::eSuccess || fr != vk::Result::eSuccess)
        return false;
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex, .module = *vs, .pName = "main"},
        vk::PipelineShaderStageCreateInfo{.stage = fragment, .module = *fs, .pName = "main"}};
    const vk::PipelineVertexInputStateCreateInfo vi{};
    const vk::PipelineInputAssemblyStateCreateInfo ia{.topology =
                                                          vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineViewportStateCreateInfo vp{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo rs{.lineWidth = 1.0f};
    const vk::PipelineMultisampleStateCreateInfo ms{.rasterizationSamples =
                                                        vk::SampleCountFlagBits::e1};
    const vk::PipelineColorBlendAttachmentState attachment{.colorWriteMask =
                                                               vk::ColorComponentFlags{0xf}};
    const vk::PipelineColorBlendStateCreateInfo cb{.attachmentCount = 1,
                                                   .pAttachments = &attachment};
    const std::array dynamic{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dyn{.dynamicStateCount = 2,
                                                 .pDynamicStates = dynamic.data()};
    const vk::PipelineRenderingCreateInfo rendering{.colorAttachmentCount = 1,
                                                    .pColorAttachmentFormats = &format};
    const vk::GraphicsPipelineCreateInfo ci{.pNext = &rendering,
                                            .stageCount = 2,
                                            .pStages = stages.data(),
                                            .pVertexInputState = &vi,
                                            .pInputAssemblyState = &ia,
                                            .pViewportState = &vp,
                                            .pRasterizationState = &rs,
                                            .pMultisampleState = &ms,
                                            .pColorBlendState = &cb,
                                            .pDynamicState = &dyn,
                                            .layout = *layout};
    auto [result, p] = device.createGraphicsPipelineUnique({}, ci);
    if (result != vk::Result::eSuccess)
        return false;
    pipeline = std::move(p);
    return true;
}

bool OpenXrOutput::State::Poll() {
    for (;;) {
        XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
        auto result = xrPollEvent(xr, &event);
        if (result == XR_EVENT_UNAVAILABLE)
            return true;
        if (!Success(result, "poll events") ||
            event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING)
            return false;
        if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
            const auto& change =
                reinterpret_cast<const XrEventDataReferenceSpaceChangePending&>(event);
            if (change.session == session &&
                change.referenceSpaceType == XR_REFERENCE_SPACE_TYPE_LOCAL)
                recenter_at = change.changeTime;
        }
        if (event.type != XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
            continue;
        const auto& change = reinterpret_cast<const XrEventDataSessionStateChanged&>(event);
        if (change.session != session)
            continue;
        focused = change.state == XR_SESSION_STATE_FOCUSED;
        if (change.state == XR_SESSION_STATE_READY && !running) {
            const XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO, nullptr, Stereo};
            if (!Success(xrBeginSession(session, &begin), "begin session"))
                return false;
            running = true;
            active_sessions.fetch_add(1, std::memory_order_relaxed);
        } else if (change.state == XR_SESSION_STATE_STOPPING && running) {
            if (Gt7XrCamera::Requested())
                Gt7XrCamera::Publish({});
            if (!Success(xrEndSession(session), "end session"))
                return false;
            running = false;
            active_sessions.fetch_sub(1, std::memory_order_relaxed);
            if (seat) {
                xrDestroySpace(seat);
                seat = XR_NULL_HANDLE;
            }
        } else if (change.state == XR_SESSION_STATE_EXITING ||
                   change.state == XR_SESSION_STATE_LOSS_PENDING) {
            LOG_WARNING(Render_Vulkan, "OpenXR session closed, continuing on desktop");
            return false;
        }
    }
}

bool OpenXrOutput::State::Center(XrTime time) {
    if (seat && (seat_calibrated || !focused) && (!recenter_at || time < recenter_at))
        return true;
    XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
    constexpr auto required =
        XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT;
    if (!Success(xrLocateSpace(head, local, time, &location), "locate seated origin") ||
        (location.locationFlags & required) != required)
        return seat != XR_NULL_HANDLE;
    // A runtime can supply a valid estimated pose before the worn headset is
    // tracked. Keep a provisional origin for menus and calibrate once tracking
    // and input focus arrive. Keep the driver's existing seat on tracking loss.
    constexpr auto tracked =
        XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT;
    const bool calibrated = focused && (location.locationFlags & tracked) == tracked;
    if (seat && !calibrated)
        return true;
    // Keep the horizon level. Only the driver's initial heading and seat position
    // set the origin.
    const auto q = location.pose.orientation;
    float yaw = std::atan2(2.f * (q.x * q.z + q.w * q.y), 1.f - 2.f * (q.x * q.x + q.y * q.y));
    XrReferenceSpaceCreateInfo ci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    ci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    ci.poseInReferenceSpace = {{0, std::sin(yaw / 2), 0, std::cos(yaw / 2)},
                               location.pose.position};
    XrSpace next{};
    if (!Success(xrCreateReferenceSpace(session, &ci, &next), "create seated origin"))
        return false;
    if (seat) {
        if (Gt7XrCamera::Requested())
            Gt7XrCamera::Publish({});
        xrDestroySpace(seat);
    }
    seat = next;
    seat_calibrated = calibrated;
    ++origin;
    if (time >= recenter_at)
        recenter_at = 0;
    return true;
}

void OpenXrOutput::State::Draw(vk::CommandBuffer cmd, const EyeTarget& eye, uint32_t index,
                               vk::ImageView source, bool encoded, uint32_t region) {
    const vk::ImageMemoryBarrier2 output{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = eye.images[index].image,
        .subresourceRange = {Color, 0, 1, 0, 1}};
    const vk::MemoryBarrier2 input{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                   .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                   .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                                   .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1,
                          .pMemoryBarriers = &input,
                          .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &output});
    const vk::Rect2D area{{0, 0}, eye.size};
    const vk::RenderingAttachmentInfo color{.imageView = *eye.views[index],
                                            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
                                            .loadOp = vk::AttachmentLoadOp::eDontCare,
                                            .storeOp = vk::AttachmentStoreOp::eStore};
    cmd.beginRendering({.renderArea = area,
                        .layerCount = 1,
                        .colorAttachmentCount = 1,
                        .pColorAttachments = &color});
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    cmd.setViewport(0, vk::Viewport{0, 0, float(eye.size.width), float(eye.size.height), 0, 1});
    cmd.setScissor(0, area);
    const vk::DescriptorImageInfo image{*linear, source, vk::ImageLayout::eGeneral};
    const vk::WriteDescriptorSet write{.dstBinding = 0,
                                       .descriptorCount = 1,
                                       .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &image};
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *layout, 0, write);
    const std::array<uint32_t, 5> sampling{encoded, region, 0, eye.size.width,
                                           eye.size.height};
    cmd.pushConstants(*layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(sampling),
                      sampling.data());
    cmd.draw(3, 1, 0, 0);
    cmd.endRendering();
}

std::optional<OpenXrOutput::ViewFrame> OpenXrOutput::BeginFrame() {
    auto& s = *state;
    if (!s.enabled)
        return {};
    if (s.frame_begun)
        return s.located;
    if (!s.Poll()) {
        s.enabled = false;
        s.CloseSession();
        return {};
    }
    if (!s.running)
        return {};
    XrFrameState timing{XR_TYPE_FRAME_STATE};
    const XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
    const XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
    bool ok = Success(xrWaitFrame(s.session, &wait, &timing), "wait frame");
    if (ok) {
        std::scoped_lock lock{Scheduler::submit_mutex};
        ok = Success(xrBeginFrame(s.session, &begin), "begin frame");
    }
    if (!ok) {
        s.enabled = false;
        s.CloseSession();
        return {};
    }
    s.frame_begun = true;
    s.display_time = timing.predictedDisplayTime;
    s.display_period = timing.predictedDisplayPeriod;
    ++s.frame_id;
    s.screen_ready = timing.shouldRender && s.Center(s.display_time);
    s.located.reset();
    if (s.screen_ready) {
        ViewFrame result{
            s.frame_id, s.display_time, {}, {s.eyes[0].size, s.eyes[1].size}, s.origin};
        for (auto& view : result.views)
            view.type = XR_TYPE_VIEW;
        XrViewState flags{XR_TYPE_VIEW_STATE};
        const XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO, nullptr, Stereo, s.display_time,
                                      s.seat};
        uint32_t count{};
        constexpr auto valid =
            XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT;
        if (Success(xrLocateViews(s.session, &locate, &flags, 2, &count, result.views.data()),
                    "locate stereo views") &&
            count == 2 && (flags.viewStateFlags & valid) == valid)
            s.located = result;
    }
    if (Gt7XrCamera::Requested()) {
        Gt7XrCamera::Publish(s.located);
    }
    return s.located;
}

void OpenXrOutput::Present(Scheduler& scheduler, const Frame& frame, const StereoImages* stereo,
                           const GameStereoFrame* game, vk::Extent2D game_eye_size) {
    auto& s = *state;
    // A projection must correspond to the prediction already supplied to its
    // producer.
    const bool two_images =
        stereo && stereo->colors[0] && stereo->colors[1] && stereo->colors[0] != stereo->colors[1];
    const bool separate =
        !game && s.frame_begun && s.located && two_images && stereo->frame_id == s.frame_id;
    if (!s.frame_begun)
        BeginFrame();
    if (!s.frame_begun)
        return;
    const auto time = s.display_time;
    // Game images can wait in the flip queue. Reproject from the poses used to draw this buffer.
    // Accept or reject the source once. Its last display slot may fall beyond the age limit.
    const bool packed = game && s.located && game->cameras.origin == s.origin &&
                        game->cameras.display_time <= time &&
                        time - game->cameras.display_time <= 150'000'000 && frame.width >= 2 &&
                        (two_images || !(frame.width & 1));
    const bool game_eyes = packed && two_images && stereo->frame_id == game->cameras.id;
    // A seated movie uses the same complete source for both eyes and the desktop.
    const bool movie = game_eyes && stereo->movie;
    const bool paired = separate || (packed && !movie);
    if (frame.is_hdr && !s.hdr_reported)
        LOG_WARNING(Render_Vulkan, "OpenXR requires SDR output");
    s.hdr_reported = frame.is_hdr;
    using Views = std::array<XrView, 2>;
    const Views views = packed ? game->cameras.views : (s.located ? s.located->views : Views{});
    const auto submit = [&] {
        XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
        XrCompositionLayerProjection projection{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        std::array<XrCompositionLayerProjectionView, 2> projection_views{};
        const XrCompositionLayerBaseHeader* layer{};
        if (s.screen_ready && !frame.is_hdr && frame.width && frame.height &&
            (!game || (packed && s.located && game->cameras.origin == s.origin))) {
            const size_t needed = paired ? 2 : 1;
            std::array<uint32_t, 2> indices{};
            size_t ready = 0;
            const XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            const XrSwapchainImageWaitInfo await{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, nullptr,
                                                 XR_INFINITE_DURATION};
            const XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            for (; ready < needed; ++ready) {
                auto chain = s.eyes[ready].chain;
                XrResult acquired;
                {
                    std::scoped_lock lock{Scheduler::submit_mutex};
                    acquired = xrAcquireSwapchainImage(chain, &acquire, &indices[ready]);
                }
                if (!Success(acquired, "acquire eye image") ||
                    !Success(xrWaitSwapchainImage(chain, &await), "wait eye image")) {
                    s.enabled = false;
                    break;
                }
            }
            if (ready == needed) {
                auto cmd = scheduler.DirectCommandBuffer();
                for (size_t i = 0; i < needed; ++i) {
                    auto& eye = s.eyes[i];
                    const bool use_eye = separate || game_eyes;
                    s.Draw(cmd, eye, indices[i],
                           use_eye ? stereo->colors[i] : frame.image_view,
                           use_eye ? stereo->display_encoded : true,
                           packed && !game_eyes ? static_cast<uint32_t>(i + 1) : 0);
                    const XrSwapchainSubImage sub{
                        eye.chain,
                        {{0, 0}, {int32_t(eye.size.width), int32_t(eye.size.height)}},
                        0};
                    if (paired) {
                        const auto& view = views[i];
                        projection_views[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW, nullptr,
                                               view.pose, view.fov, sub};
                    } else {
                        quad.subImage = sub;
                    }
                }
                // A released swapchain image must contain this submission's pixels.
                // A late GPU copy can otherwise expose the previous use of the same image.
                scheduler.Finish();
                if (paired) {
                    projection.space = s.seat;
                    projection.viewCount = 2;
                    projection.views = projection_views.data();
                    layer = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&projection);
                } else {
                    quad.space = s.seat;
                    quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
                    quad.pose = Identity;
                    quad.pose.position.z = -2.0f;
                    quad.size = {2.4f, movie ? 2.4f * 9.f / 16.f
                                             : 2.4f * float(frame.height) / float(frame.width)};
                    layer = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&quad);
                }
            }
            for (size_t i = 0; i < ready; ++i) {
                std::scoped_lock lock{Scheduler::submit_mutex};
                if (!Success(xrReleaseSwapchainImage(s.eyes[i].chain, &release),
                             "release eye image")) {
                    s.enabled = false;
                    layer = nullptr;
                }
            }
        }
        const XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO, nullptr,
                                 s.display_time,         s.blend,
                                 layer ? 1u : 0u,        layer ? &layer : nullptr};
        {
            std::scoped_lock lock{Scheduler::submit_mutex};
            if (!Success(xrEndFrame(s.session, &end), "end frame"))
                s.enabled = false;
        }
        s.frame_begun = false;
        s.located.reset();
        return layer != nullptr && s.enabled;
    };
    [[maybe_unused]] const bool submitted = s.frame_begun && submit();
#ifdef SHADGT_DIAGNOSTICS
    const int mode = movie ? 3 : game_eyes ? 2 : int(paired);
    if (submitted && s.last_mode != mode) {
        s.last_mode = mode;
        if (movie) {
            LOG_INFO(Render_Vulkan, "OpenXR: seated movie screen");
        } else if (game_eyes) {
            LOG_WARNING(
                Render_Vulkan,
                "OpenXR: native eye images {}x{} to {}x{} per eye",
                game_eye_size.width, game_eye_size.height, s.eyes[0].size.width,
                s.eyes[0].size.height);
        } else {
            LOG_INFO(Render_Vulkan, "OpenXR: {}", paired ? "native stereo" : "virtual screen");
        }
    }
#endif
    if (!s.enabled)
        s.CloseSession();
    else if (Gt7XrCamera::Requested())
        BeginFrame();
}
} // namespace Vulkan
