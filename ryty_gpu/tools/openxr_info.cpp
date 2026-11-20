// SPDX-License-Identifier: GPL-2.0-or-later
#include "../openxr_system.h"
#include <iostream>

int main() {
    XrInstance instance{};
    XrSystemId system{};
    RygtXr::Extensions enabled;
    std::cout << "shadGT OpenXR headset capabilities\n";
    const bool ok = RygtXr::OpenSystem(instance, system, enabled, true, std::cout) &&
                    RygtXr::DescribeSystem(instance, system, enabled, true, std::cout);
    if (instance)
        xrDestroyInstance(instance);
    return ok ? 0 : 1;
}
