# GT7 cockpit VR on Windows

The unified renderer can submit a seated screen to an SDR OpenXR headset. On the PS4 CUSA24769 01.00 build of Gran Turismo 7, it can also drive the game's two cockpit cameras. The camera adapter checks the executable layout before installing any hooks.

## Driving

1. Connect the headset and select its OpenXR runtime. Run `rygt_openxr_info.exe` to check that the runtime can see a headset and report two recommended eye sizes.
2. In the unified GT7 launcher, enable `SHARED_OPENXR` in **Advanced**. Enable `SHADGT_OPENXR_COCKPIT` for native stereo. The launcher selects CPU occlusion for the cockpit so track geometry stays visible in both eyes.
3. Open **Graphics → OpenXR headset**. Start with reconstruction and frame generation off to check native stereo. Restart the game after changing settings. Recenter using the headset runtime while seated and facing forward.
4. To test reconstruction, select DLSS or FSR and choose the input quality. Output percentage sets the submitted dimensions per eye. Frame generation uses at most one intermediate stereo pair per real frame and needs valid color, depth and motion from both eyes. NVIDIA generation requires a compatible `ryGT_dlss.dll` and the vendor libraries. Test the desktop mirror separately using left, right, both or off.

## How flat changes reach VR

| Code changed upstream | What reaches the headset | What to exercise |
| --- | --- | --- |
| Shader translation, pipeline compilation, guest memory, texture decoding and ordinary draw recording | Both cameras render through these same components. A cache or compiler improvement can benefit both eyes. | Flat GT Sport, flat GT7 and a native cockpit race. Include the road, instruments and rear-view mirror. |
| GT7 render targets, resolves, TAA, exposure, bloom, tone mapping or HUD draws | The game still draws these passes. The native adapter also recognizes particular shaders, attachment sizes and array layers to preserve each eye and capture reconstruction inputs. A change to those contracts needs stereo verification. | Drive with reconstruction off and on. Compare both eyes, a movie, the countdown and results. Provider refusal must retain the game's resolved picture. |
| DLSS, FidelityFX or the NVIDIA plugin ABI | The paths reuse provider loading, Vulkan negotiation and quality settings. The headset supplies a distinct history, projection, jitter, depth and motion for each eye. | Check a camera cut and recenter, both eyes after a resolution change, and a run without the provider. Keep the declarations in the renderer and runtime plugin identical. |
| Desktop swapchain, window scaling, HDR or frame pacing | These control the monitor. The headset has its own swapchains, prediction, poses and display cadence. Mirror framing applies after eye rendering. | Test flat output with VR off, the desktop fallback when the runtime has no headset, mirror off, and a headset stop or disconnect during a race. |

The VideoOut flip carries the cameras used to render that buffer. A presenter change should preserve that association across both direct and end-of-pipe flips. Synthesizing the second eye from the desktop frame would lose disocclusion. When the renderer lacks a valid stereo pair it can present a seated screen. Terminal session loss restores desktop presentation.

## Build and verify

Use the renderer's Windows clang-cl environment and initialized submodules. Install the Khronos OpenXR loader and headers first. The companion RyGT build pins OpenXR-SDK 1.1.54 and builds its static loader. Set `OpenXR_DIR` to the directory containing `OpenXRConfig.cmake` when CMake cannot discover the package.

```powershell
cmake -S . -B Build/vr -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DHAVE_ENDIAN_H=OFF -DSHADGT_RYTY_GPU=ON -DRYGT_BUILD_OPENXR=ON
cmake --build Build/vr --target ryty_gpu rygt_openxr_info rygt_openxr_stereo_test
python ryty_gpu/tests/openxr/run_simulator.py Build/vr/rygt_openxr_stereo_test.exe path/to/openxr_simulator.json
```

The test uses the Windows OpenXR Simulator screenshot protocol. It checks separate views, packed views, native game views, a seated movie and two rejected pairs in each mirror mode. These captures exercise the compositor. In-game camera hooks, vendor performance and physical headset pacing require their own runs. For a standard flat build, leave `RYGT_BUILD_OPENXR` at its default of `OFF`.
