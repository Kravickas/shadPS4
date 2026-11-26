// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "common/types.h"
#include "video_core/renderer_vulkan/host_passes/dlss_nr_components.h"
#include "video_core/renderer_vulkan/host_passes/xfb_velocity_pass.h"

namespace VideoCore {
struct Image;
}

namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;
} // namespace Vulkan

namespace Vulkan::HostPasses {

// Drives NVIDIA's Neural Rendering model (nvngx_dlssnr.dll, NGX feature 18) over the presented
// image, with depth and motion vectors from the velocity pass. Windows, MSVC ABI only; the model
// and the forwarder nvngx.dll_dlssnr.dll sit beside the executable.
class DlssNrPass {
public:
    // Defaults are the model's own values for an unset parameter. Preset 0 is not in the weight
    // table, so the model falls back to its shipping preset 2; it has three styles.
    struct Settings {
        bool enable{true};
        u32 preset{0};
        u32 style{0};
        float intensity{1.0f};
        float local_structure{1.0f};
        float local_tone{1.0f};
        float skin_structure{-1.0f};
        bool auto_mask{false};
        // How many times the model runs on each frame, each pass on the previous one's output.
        u32 passes{1};
        DlssNrComponents components;
    };

    DlssNrPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime);
    ~DlssNrPass();

    static bool ModelPresent();

    // CUDA reserves its own address space when it starts, so it has to start before the guest
    // address space is reserved.
    static void InitCuda();

    // Diagnoses a device that failed to create with the CUDA interop extensions.
    static void LogCudaState();

    // Replaces the contents of color with the model's output.
    void Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs,
                Settings& settings);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::HostPasses
