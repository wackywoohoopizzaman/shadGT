// SPDX-License-Identifier: GPL-2.0-or-later
// FEAT-GT7-210: tracked cameras and framebuffer reservations for CUSA24769 01.00.
namespace Gt7Camera {
using Signal = s32(PS4_SYSV_ABI*)(void*, s32);
using Tracker = bool(PS4_SYSV_ABI*)(void*, s32, u64);
using SelectBuffer = void(PS4_SYSV_ABI*)(void*, s32, s32);
static Signal original_signal;
static Tracker original_tracker;
static SelectBuffer original_select;
static std::optional<Vulkan::GameStereoFrame> drawing;
static s32 drawing_slot = -1;
static u8* image;
static bool installed, requested_mode;
using AllocateEyes = void(PS4_SYSV_ABI*)(float);
static AllocateEyes allocate_eyes;
static float* eye_axes;

// The title creates three stereo color buffers, then reserves a separate pool
// for scene color, depth, history and the Work surface used by its render passes.
constexpr u32 GameEyeWidth = 1344, GameEyeHeight = 1512;
constexpr u32 MaxTextureDimension = 1 << 14; // GCN T# width/height are 14-bit fields.
struct BufferPool {
    u64 bytes;
    u32 work_width, work_height;
};
static BufferPool ReadPool(u8* state) {
    return {*reinterpret_cast<u64*>(state + 0x200), *reinterpret_cast<u32*>(state + 0x308),
            *reinterpret_cast<u32*>(state + 0x30c)};
}
static void WritePool(u8* state, const BufferPool& pool) {
    *reinterpret_cast<u64*>(state + 0x200) = pool.bytes;
    *reinterpret_cast<u32*>(state + 0x308) = pool.work_width;
    *reinterpret_cast<u32*>(state + 0x30c) = pool.work_height;
}
static thread_local u8* resizing_pool;
static thread_local BufferPool original_pool;
static std::atomic<bool> seat_screen_only{};
using ConfigureBuffers = void(PS4_SYSV_ABI*)(void*);
static ConfigureBuffers configure_buffers;

static void PS4_SYSV_ABI ConfigureRaceBuffers(void* config) {
    configure_buffers(config);
    resizing_pool = nullptr;
    const auto extent = Vulkan::Gt7XrCamera::game_eye_extent.load();
    const vk::Extent2D size{u32(extent >> 32), u32(extent)};
    constexpr u64 original_pixels = u64{GameEyeWidth} * GameEyeHeight;
    const u64 pixels = u64{size.width} * size.height;
    if (!installed || seat_screen_only.load() || pixels <= original_pixels)
        return;
    auto* state = *static_cast<u8**>(config);
    original_pool = ReadPool(state);
    BufferPool scaled{};
    scaled.bytes = (original_pool.bytes * pixels / original_pixels + 0x1fffff) & ~0x1fffffull;
    scaled.work_width = static_cast<u32>(
        (u64{original_pool.work_width} * size.width + GameEyeWidth - 1) / GameEyeWidth);
    scaled.work_height = static_cast<u32>(
        (u64{original_pool.work_height} * size.height + GameEyeHeight - 1) / GameEyeHeight);
    if (scaled.work_width > MaxTextureDimension || scaled.work_height > MaxTextureDimension) {
        seat_screen_only = true;
        std::fprintf(stderr, "[openxr-camera] work surface exceeds the guest texture limit, using "
                             "the seated screen\n");
        return;
    }
    resizing_pool = state;
    WritePool(state, scaled);
    std::fprintf(stderr, "[openxr-camera] framebuffer pool %llu MiB, Work %ux%u\n",
                 scaled.bytes >> 20, scaled.work_width, scaled.work_height);
}

static u64 PS4_SYSV_ABI ReserveRacePool(void* allocator, u64 bytes, u64 alignment) {
    using Reserve = u64(PS4_SYSV_ABI*)(void*, u64, u64);
    const auto reserve = (*static_cast<Reserve**>(allocator))[2];
    const auto address = reserve(allocator, bytes, alignment);
    auto* state = std::exchange(resizing_pool, nullptr);
    if (address || !state)
        return address;
    WritePool(state, original_pool);
    seat_screen_only = true;
    std::fprintf(
        stderr,
        "[openxr-camera] guest could not reserve %llu MiB for stereo, using the seated screen\n",
        bytes >> 20);
    return reserve(allocator, original_pool.bytes, alignment);
}

// GT7's vrscale initializer uses these two axes for the eye buffers and
// its camera-dependent render passes. The desktop pack has a separate size.
static void PS4_SYSV_ABI AllocateGameEyes(float game_scale) {
    const auto size = Vulkan::Gt7XrCamera::AllocationSize();
    eye_axes[0] = 960.f;
    eye_axes[1] = 1080.f;
    const bool selected = installed && size.width && size.height;
    // The GNM helpers also return the byte size of one two-layer color image as a u32.
    if (selected && (size.width > MaxTextureDimension || size.height > MaxTextureDimension ||
                     u64{size.width} * size.height * 8 > UINT32_MAX)) {
        Vulkan::Gt7XrCamera::SetAllocationSize({});
        std::fprintf(
            stderr,
            "[openxr-camera] requested eye size exceeds guest image limits, using game default\n");
    } else if (selected) {
        eye_axes[0] = float(size.width);
        eye_axes[1] = float(size.height);
        game_scale = 1.f;
    }
    allocate_eyes(game_scale);
    const auto width = *reinterpret_cast<const u32*>(image + 0x6475e20);
    const auto height = *reinterpret_cast<const u32*>(image + 0x6475e24);
    for (size_t slot = 0; slot < 3; ++slot) {
        if (!reinterpret_cast<const u64*>(image + 0x7119c70)[slot]) {
            seat_screen_only = true;
            Vulkan::Gt7XrCamera::game_eye_extent = 0;
            std::fprintf(stderr, "[openxr-camera] guest could not allocate stereo color buffers, "
                                 "using the seated screen\n");
            return;
        }
    }
    Vulkan::Gt7XrCamera::game_eye_extent = Vulkan::Gt7XrCamera::PackExtent(width, height);
    std::fprintf(stderr, "[openxr-camera] allocated game eyes %ux%u\n", width, height);
}

// CUSA24769 01.00. This is the input of CameraHMDAdapter's update.
// The game keeps position and orientation in separate 16-byte vectors.
struct Eye {
    XrQuaternionf orientation;
    XrVector3f position;
    float padding;
};
struct CameraInput {
    Eye eyes[2];
    float frustum[2][4]; // Left, right, up, down tangents in guest order.
    float vertical_half_span;
    u8 tracking_valid;
    u8 tracking_enabled;
};
static_assert(offsetof(CameraInput, frustum) == 0x40);
static_assert(offsetof(CameraInput, vertical_half_span) == 0x60);
static_assert(offsetof(CameraInput, tracking_valid) == 0x64);

static void PS4_SYSV_ABI SelectGameBuffer(void* commands, s32 port, s32 slot) {
    original_select(commands, port, slot);
    if (!installed)
        return;
    // VideoOutDriver accepts only handle 1 (the main display).
    drawing_slot = port == 1 ? slot : -1;
    drawing.reset();
    if (drawing_slot >= 0) {
        if (const auto views = Vulkan::Gt7XrCamera::Read()) {
            drawing = Vulkan::GameStereoFrame{*views};
        }
    }
    Vulkan::Gt7XrCamera::TagBuffer(drawing_slot, nullptr);
    if (requested_mode && image[0x7119c8a]) {
        // The display worker evaluates tracking for this slot immediately after this call.
        auto& input = *reinterpret_cast<CameraInput*>(image + 0x7011840);
        input.tracking_enabled = input.tracking_valid = drawing.has_value();
    }
}

static s32 PS4_SYSV_ABI OnGraphicsTick(void* semaphore, s32 increment) {
    // The guest polls the mode-change request immediately after this call.
    const auto result = original_signal(semaphore, increment);
    if (!installed)
        return result;
    auto* display = *reinterpret_cast<u8**>(image + 0x6feeec8);
    auto* state = display ? *reinterpret_cast<u8**>(display + 0xc8) : nullptr;
    if (!state)
        return result;
    const auto views = seat_screen_only.load() ? std::nullopt : Vulkan::Gt7XrCamera::Read();
    auto& command = *reinterpret_cast<s32*>(state + 0x17c);
    if (command == -1 && bool(views) != requested_mode) {
        command = views ? 1 : 0;
        state[0x168] = 1;
        requested_mode = bool(views);
    }
    return result;
}

static bool PS4_SYSV_ABI OnTracker(void* output, s32 slot, u64 time) {
    if (!installed || !requested_mode || !image[0x7119c8a])
        return original_tracker(output, slot, time);
    const auto& frame = drawing;
    auto& input = *reinterpret_cast<CameraInput*>(image + 0x7011840);
    if (!frame) {
        input.tracking_valid = 0;
        return false;
    }
    for (size_t eye = 0; eye < 2; ++eye) {
        const auto& view = frame->cameras.views[eye];
        input.eyes[eye] = {view.pose.orientation, view.pose.position, 0.f};
        auto& bounds = input.frustum[eye];
        bounds[0] = std::tan(view.fov.angleLeft);
        bounds[1] = std::tan(view.fov.angleRight);
        bounds[2] = std::tan(view.fov.angleUp);
        bounds[3] = std::tan(view.fov.angleDown);
    }
    // GT7 derives this shared camera input as half the span between the upper
    // and lower tangent at 0x3280073. Cover the taller eye if they differ.
    input.vertical_half_span = 0.5f * std::max(input.frustum[0][2] - input.frustum[0][3],
                                               input.frustum[1][2] - input.frustum[1][3]);
    input.tracking_valid = 1;
    input.tracking_enabled = 1;
    if (drawing_slot == slot && *reinterpret_cast<const s32*>(image + 0x70118a8) == 2) {
        Vulkan::Gt7XrCamera::TagBuffer(slot, &*frame);
    }
    // The guest's false branch preserves the camera input already written above.
    return false;
}

static Patch FloatOperand(u64 instruction, std::initializer_list<u8> original, u8* literal,
                          const char* name, float factor = 1.f) {
    Patch patch{instruction, original, original, name};
    s32 disp;
    std::memcpy(&disp, patch.original.data() + 4, sizeof(disp));
    const auto next = reinterpret_cast<intptr_t>(image + instruction + 8);
    float value;
    std::memcpy(&value, reinterpret_cast<const void*>(next + disp), sizeof(value));
    value *= factor;
    std::memcpy(literal, &value, sizeof(value));
    // AllocNear places the literal within range of the RIP-relative operand.
    disp = static_cast<s32>(reinterpret_cast<intptr_t>(literal) - next);
    std::memcpy(patch.replacement.data() + 4, &disp, sizeof(disp));
    return patch;
}

static void Install(u8* base) {
    if (!Vulkan::Gt7XrCamera::Requested())
        return;
    auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->OptionalHeader.SizeOfImage != 0x7de7000)
        return;
    image = base + kTextRvaShift;
    const auto match = [](size_t offset, std::initializer_list<u8> bytes) {
        return std::equal(bytes.begin(), bytes.end(), image + offset);
    };
    if (!match(0x81fc19,
               {0x48, 0x8b, 0x0d, 0xa8, 0xf2, 0x7c, 0x06, 0x48, 0x8b, 0x89, 0xc8, 0, 0, 0}) ||
        !match(0x336976f, {0x83, 0x3d, 0x32, 0x81, 0xca, 0x03, 0x02}) ||
        !match(0x327d426, {0xe8, 0x25, 0x23, 0xf9, 0x00}) ||
        !match(0x32800a9,
               {0xe8, 0x12, 0xad, 0x8e, 0x00, 0x84, 0xc0, 0x0f, 0x84, 0x98, 0xda, 0xff, 0xff}) ||
        !match(0x328055a, {0x88, 0x05, 0x44, 0x13, 0xd9, 0x03}) ||
        !match(0x327b5a9, {0xe8, 0x62, 0x77, 0x8d, 0}) ||
        !match(0x33e8c58, {0xb8, 0, 0, 0x60, 0x0e, 0xb9, 0, 0, 0xb0, 0x1b}) ||
        !match(0x33e8c7f, {0xe8, 0x1c, 3, 0, 0}) ||
        !match(0x33e8ce0, {0xe8, 0x5b, 4, 0xf5, 0xff}) ||
        !match(0x327dad9, {0xe8, 0x72, 0x17, 0x56, 0xfd}) ||
        !match(0x33f29ee, {0x80, 0x3d, 0x93, 0x72, 0xd2, 0x03, 0, 0xc5, 0xf8, 0x29, 0x44, 0x24,
                           0x30, 0x74, 0x0e}))
        return;
    auto* stubs = AllocNear(base);
    if (!stubs)
        return;
    original_signal = reinterpret_cast<Signal>(image + 0x420f750);
    original_tracker = reinterpret_cast<Tracker>(image + 0x3b6adc0);
    original_select = reinterpret_cast<SelectBuffer>(image + 0x7df250);
    allocate_eyes = reinterpret_cast<AllocateEyes>(image + 0x3b52d10);
    configure_buffers = reinterpret_cast<ConfigureBuffers>(image + 0x33e8fa0);
    eye_axes = reinterpret_cast<float*>(stubs + 80);
    std::array viewport = {
        // Both allocator implementations go through the original virtual method in ReserveRacePool.
        Patch{0x33e8cda,
              {0x0f, 0x85, 0xa7, 0, 0, 0},
              {0x90, 0x90, 0x90, 0x90, 0x90, 0x90},
              "GT7 XR pool allocation"},
        // Cover the corners of a rectangular OpenXR view with both game masks.
        FloatOperand(0x32ebe56, {0xc5, 0xfa, 0x5e, 0x05, 0xaa, 0x86, 0x1c, 0x01}, stubs + 48,
                     "GT7 XR depth mask", 1.f / std::sqrt(2.f)),
        FloatOperand(0x32ed535, {0xc5, 0xea, 0x5e, 0x15, 0xef, 0x6f, 0x1c, 0x01}, stubs + 52,
                     "GT7 XR post-processing bounds", 1.f / std::sqrt(2.f)),
        FloatOperand(0x3b52d21, {0xc5, 0xfa, 0x59, 0x0d, 0x87, 0x41, 0x96, 0}, stubs + 80,
                     "GT7 XR eye width"),
        FloatOperand(0x3b52d29, {0xc5, 0xfa, 0x59, 0x05, 0x83, 0x41, 0x96, 0}, stubs + 84,
                     "GT7 XR eye height"),
        // AutoExposurePS4Renderer initializes a full-surface metering rectangle.
        // Keep it instead of reading the original headset's calibration record.
        Patch{0x33f29fb, {0x74, 0x0e}, {0xeb, 0x0e}, "GT7 XR exposure metering"},
    };
    for (const auto& patch : viewport) {
        if (!std::equal(patch.original.begin(), patch.original.end(), image + patch.rva)) {
            VirtualFree(stubs, 0, MEM_RELEASE);
            return;
        }
    }
    struct CallSite {
        size_t stub_offset;
        u64 call;
        u64 target;
        void* replacement;
        const char* name;
    };
    const std::array calls = {
        CallSite{0, 0x327d426, 0x420f750, reinterpret_cast<void*>(&OnGraphicsTick),
                 "GT7 XR mode request"},
        CallSite{16, 0x32800a9, 0x3b6adc0, reinterpret_cast<void*>(&OnTracker),
                 "GT7 XR eye cameras"},
        CallSite{32, 0x327dad9, 0x7df250, reinterpret_cast<void*>(&SelectGameBuffer),
                 "GT7 XR render buffer"},
        CallSite{64, 0x327b5a9, 0x3b52d10, reinterpret_cast<void*>(&AllocateGameEyes),
                 "GT7 XR resolution"},
        CallSite{96, 0x33e8c7f, 0x33e8fa0, reinterpret_cast<void*>(&ConfigureRaceBuffers),
                 "GT7 XR framebuffer pool"},
        CallSite{112, 0x33e8ce0, 0x3339140, reinterpret_cast<void*>(&ReserveRacePool),
                 "GT7 XR reserve pool"},
    };

    // The calls and viewport operands form one version-specific layout. If a page
    // is unwritable, unwind everything installed so far before returning to flat play.
    std::vector<Patch> undo;
    undo.reserve(calls.size() + viewport.size());
    auto rollback = [&] {
        bool restored = true;
        for (auto it = undo.rbegin(); it != undo.rend(); ++it) {
            restored &= Apply(image, *it);
        }
        // Keep the stubs and their literals alive if a reader could still reach them.
        if (restored) {
            VirtualFree(stubs, 0, MEM_RELEASE);
        }
    };
    for (const auto& site : calls) {
        const auto* instruction = image + site.call;
        Patch previous{site.call, {}, {instruction, instruction + 5}, site.name};
        if (!RedirectCall(base, stubs + site.stub_offset, site.call, site.target, site.replacement,
                          site.name)) {
            rollback();
            return;
        }
        previous.original.assign(instruction, instruction + 5);
        undo.push_back(std::move(previous));
    }
    for (const auto& patch : viewport) {
        if (!Apply(image, patch)) {
            rollback();
            return;
        }
        undo.push_back({patch.rva, patch.replacement, patch.original, patch.name});
    }
    installed = true;
}
} // namespace Gt7Camera
