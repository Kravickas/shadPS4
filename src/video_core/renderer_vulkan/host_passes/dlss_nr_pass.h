// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
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
    DlssNrPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime);
    ~DlssNrPass();

    static bool ModelPresent();

    // Replaces the contents of color with the model's output.
    void Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::HostPasses
