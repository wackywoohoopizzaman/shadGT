// SPDX-License-Identifier: GPL-2.0-or-later
#include "openxr_system.h"

#include <algorithm>
#include <cstring>
#include <ostream>
#include <vector>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <unknwn.h>
#define XR_USE_PLATFORM_WIN32
#endif
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr_platform.h>

// FEAT-SGT-043: discover the active runtime and its supported headset views.
namespace RygtXr {
namespace {
bool Check(XrResult result, const char* operation, std::ostream& out) {
    if (XR_SUCCEEDED(result))
        return true;
    out << operation << " failed (" << result << ")\n";
    return false;
}
constexpr auto Stereo = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
constexpr auto Inset = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET;
} // namespace

bool OpenSystem(XrInstance& instance, XrSystemId& system, Extensions& enabled, bool query_gaze,
                std::ostream& out) {
    uint32_t count{};
    if (!Check(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr),
               "Enumerate extensions", out))
        return false;
    std::vector<XrExtensionProperties> available(count, {XR_TYPE_EXTENSION_PROPERTIES});
    if (!Check(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, available.data()),
               "Read extensions", out))
        return false;
    const auto advertised = [&](const char* name) {
        return std::ranges::any_of(
            available, [&](const auto& ext) { return std::strcmp(ext.extensionName, name) == 0; });
    };
    // Keep the application's identity and Vulkan binding identical in the probe
    // and game.
    std::vector<const char*> extensions;
    if (advertised(XR_KHR_VULKAN_ENABLE_EXTENSION_NAME)) {
        extensions.push_back(XR_KHR_VULKAN_ENABLE_EXTENSION_NAME);
    } else {
        out << "shadGT Vulkan binding unavailable\n";
        if (!query_gaze)
            return false;
    }
#ifdef _WIN32
    enabled.performance_clock =
        advertised(XR_KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME);
    if (enabled.performance_clock)
        extensions.push_back(XR_KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME);
#endif
    enabled.quad_views = advertised(XR_VARJO_QUAD_VIEWS_EXTENSION_NAME);
    enabled.gaze = query_gaze && advertised(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME);
    enabled.rendering_gaze = query_gaze && advertised(XR_VARJO_FOVEATED_RENDERING_EXTENSION_NAME);
    if (enabled.quad_views)
        extensions.push_back(XR_VARJO_QUAD_VIEWS_EXTENSION_NAME);
    if (enabled.gaze)
        extensions.push_back(XR_EXT_EYE_GAZE_INTERACTION_EXTENSION_NAME);
    if (enabled.rendering_gaze)
        extensions.push_back(XR_VARJO_FOVEATED_RENDERING_EXTENSION_NAME);
    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strcpy(ci.applicationInfo.applicationName, "shadGT");
    ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 1, 0);
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.enabledExtensionNames = extensions.data();
    auto created = xrCreateInstance(&ci, &instance);
    if (created == XR_ERROR_API_VERSION_UNSUPPORTED) {
        ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
        // On 1.0, the VARJO foveation extension depends on the legacy quad
        // extension.
        if (enabled.rendering_gaze && !enabled.quad_views) {
            enabled.rendering_gaze = false;
            --ci.enabledExtensionCount;
        }
        created = xrCreateInstance(&ci, &instance);
    }
    if (!Check(created, "Create shadGT Vulkan instance", out))
        return false;
    XrSystemGetInfo get{XR_TYPE_SYSTEM_GET_INFO};
    get.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    return Check(xrGetSystem(instance, &get, &system), "Select headset system", out);
}

bool DescribeSystem(XrInstance instance, XrSystemId system, const Extensions& enabled,
                    bool query_gaze, std::ostream& out) {
    XrInstanceProperties runtime{XR_TYPE_INSTANCE_PROPERTIES};
    XrSystemProperties props{XR_TYPE_SYSTEM_PROPERTIES};
    XrSystemEyeGazeInteractionPropertiesEXT gaze{
        XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT};
    XrSystemFoveatedRenderingPropertiesVARJO foveated{
        XR_TYPE_SYSTEM_FOVEATED_RENDERING_PROPERTIES_VARJO};
    if (enabled.gaze)
        props.next = &gaze;
    if (enabled.rendering_gaze) {
        foveated.next = props.next;
        props.next = &foveated;
    }
    if (!Check(xrGetInstanceProperties(instance, &runtime), "Identify runtime", out) ||
        !Check(xrGetSystemProperties(instance, system, &props), "Identify headset", out))
        return false;
    out << "Runtime: " << runtime.runtimeName << ' ' << XR_VERSION_MAJOR(runtime.runtimeVersion)
        << '.' << XR_VERSION_MINOR(runtime.runtimeVersion) << '.'
        << XR_VERSION_PATCH(runtime.runtimeVersion) << "\nHeadset: " << props.systemName << '\n';
    if (query_gaze) {
        const auto status = [](bool queried, XrBool32 supported) {
            return !queried    ? "extension unavailable"
                   : supported ? "supported"
                               : "unsupported on this system";
        };
        out << "Interaction gaze (EXT): " << status(enabled.gaze, gaze.supportsEyeGazeInteraction)
            << "\nFoveated rendering (VARJO): "
            << status(enabled.rendering_gaze, foveated.supportsFoveatedRendering) << '\n';
    }
    uint32_t count{};
    if (!Check(xrEnumerateViewConfigurations(instance, system, 0, &count, nullptr),
               "Count view configurations", out))
        return false;
    std::vector<XrViewConfigurationType> types(count);
    if (!Check(xrEnumerateViewConfigurations(instance, system, count, &count, types.data()),
               "Read view configurations", out))
        return false;
    for (auto type : {Stereo, Inset}) {
        const char* name = type == Stereo ? "Stereo" : "Stereo with foveated insets";
        if (std::ranges::find(types, type) == types.end()) {
            out << name << ": not exposed\n";
            continue;
        }
        if (!Check(xrEnumerateViewConfigurationViews(instance, system, type, 0, &count, nullptr),
                   "Count views", out))
            return false;
        std::vector<XrViewConfigurationView> views(count, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
        if (!Check(xrEnumerateViewConfigurationViews(instance, system, type, count, &count,
                                                     views.data()),
                   "Read view sizes", out))
            return false;
        out << name << ": " << count << " views\n";
        for (uint32_t i = 0; i < count; ++i) {
            const auto& view = views[i];
            out << "  [" << i << "] " << view.recommendedImageRectWidth << 'x'
                << view.recommendedImageRectHeight << ", recommended samples "
                << view.recommendedSwapchainSampleCount << '\n';
        }
    }
    return true;
}
} // namespace RygtXr
