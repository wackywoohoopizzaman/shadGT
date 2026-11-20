// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <iosfwd>
#include <openxr/openxr.h>

namespace RygtXr {
struct Extensions {
    bool quad_views{}, gaze{}, rendering_gaze{}, performance_clock{};
};

// The caller owns the returned instance, including when system selection fails.
bool OpenSystem(XrInstance& instance, XrSystemId& system, Extensions& enabled, bool query_gaze,
                std::ostream& output);
bool DescribeSystem(XrInstance instance, XrSystemId system, const Extensions& enabled,
                    bool query_gaze, std::ostream& output);
} // namespace RygtXr
