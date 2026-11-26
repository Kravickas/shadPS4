// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "common/types.h"
#include "video_core/xfb_capture.h"

namespace Vulkan::HostPasses {

// How much of the DLSS 5 output each part of the picture takes, per game. The model blends its
// output over the game image by saturate(intensity * strength) for each pixel. Draws are tagged
// by their guest programs; component 0 covers every pixel no tagged draw owns.
class DlssNrComponents {
public:
    struct Component {
        std::string name;
        float strength{1.0f};
    };

    void Load(std::string_view game_serial);
    void Save();

    // The members below are guarded by mutex.
    std::mutex mutex;
    std::vector<Component> components;
    std::unordered_map<VideoCore::XfbShaders, u32, VideoCore::XfbShadersHash> tags;

    // The menu arms a pick, a click on the game image stores its position (0 to 1) and the next
    // DLSS 5 frame stores the programs of the draw that owns that pixel.
    bool pick_armed{};
    std::optional<std::array<float, 2>> pick_position;
    bool pick_done{};
    std::optional<VideoCore::XfbShaders> picked;

    // The caller holds mutex.
    [[nodiscard]] bool Active() const;
    [[nodiscard]] float StrengthOf(const VideoCore::XfbShaders& shaders) const;
    void RemoveComponent(u32 index);

private:
    std::string serial;
};

} // namespace Vulkan::HostPasses
