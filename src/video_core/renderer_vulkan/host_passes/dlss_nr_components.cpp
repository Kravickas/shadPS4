// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "video_core/renderer_vulkan/host_passes/dlss_nr_components.h"

namespace Vulkan::HostPasses {

namespace {

std::filesystem::path FilePath(std::string_view serial) {
    return Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "dlss5" /
           (std::string{serial} + ".json");
}

u64 ParseHash(const nlohmann::json& value) {
    return std::stoull(value.get<std::string>(), nullptr, 16);
}

} // Anonymous namespace

void DlssNrComponents::Load(std::string_view game_serial) {
    std::scoped_lock lock{mutex};
    serial = game_serial;
    components = {{"Untagged", 1.0f}, {"Hair", 1.0f}, {"Crowd", 1.0f}, {"Ice", 1.0f}};
    tags.clear();
    if (serial.empty()) {
        return;
    }
    const auto path = FilePath(serial);
    std::ifstream file{path};
    if (!file) {
        return;
    }
    try {
        const auto json = nlohmann::json::parse(file);
        std::vector<Component> loaded;
        for (const auto& entry : json.at("components")) {
            loaded.push_back({
                .name = entry.at("name").get<std::string>(),
                .strength = std::clamp(entry.at("strength").get<float>(), 0.0f, 1.0f),
            });
        }
        if (loaded.empty()) {
            return;
        }
        std::unordered_map<VideoCore::XfbShaders, u32, VideoCore::XfbShadersHash> loaded_tags;
        for (const auto& entry : json.at("tags")) {
            const u32 index = entry.at("component").get<u32>();
            if (index == 0 || index >= loaded.size()) {
                continue;
            }
            loaded_tags[{.vs = ParseHash(entry.at("vs")), .ps = ParseHash(entry.at("ps"))}] = index;
        }
        components = std::move(loaded);
        tags = std::move(loaded_tags);
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "DLSS NR: {} is not a valid component file: {}",
                  Common::FS::PathToUTF8String(path), e.what());
    }
}

void DlssNrComponents::Save() {
    nlohmann::json json;
    std::filesystem::path path;
    {
        std::scoped_lock lock{mutex};
        if (serial.empty()) {
            return;
        }
        path = FilePath(serial);
        json["components"] = nlohmann::json::array();
        for (const auto& component : components) {
            json["components"].push_back({
                {"name", component.name},
                {"strength", component.strength},
            });
        }
        std::vector<std::pair<VideoCore::XfbShaders, u32>> sorted(tags.begin(), tags.end());
        std::ranges::sort(sorted, [](const auto& lhs, const auto& rhs) {
            return std::tie(lhs.second, lhs.first.vs, lhs.first.ps) <
                   std::tie(rhs.second, rhs.first.vs, rhs.first.ps);
        });
        json["tags"] = nlohmann::json::array();
        for (const auto& [shaders, index] : sorted) {
            json["tags"].push_back({
                {"vs", fmt::format("{:#018x}", shaders.vs)},
                {"ps", fmt::format("{:#018x}", shaders.ps)},
                {"component", index},
            });
        }
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream file{path};
    if (!file) {
        LOG_ERROR(Render_Vulkan, "DLSS NR: could not write {}", Common::FS::PathToUTF8String(path));
        return;
    }
    file << json.dump(4);
}

bool DlssNrComponents::Active() const {
    return std::ranges::any_of(
        components, [](const Component& component) { return component.strength != 1.0f; });
}

float DlssNrComponents::StrengthOf(const VideoCore::XfbShaders& shaders) const {
    const auto it = tags.find(shaders);
    return components[it != tags.end() ? it->second : 0].strength;
}

void DlssNrComponents::RemoveComponent(u32 index) {
    if (index == 0 || index >= components.size()) {
        return;
    }
    components.erase(components.begin() + index);
    std::erase_if(tags, [index](const auto& tag) { return tag.second == index; });
    for (auto& [shaders, component] : tags) {
        if (component > index) {
            --component;
        }
    }
}

} // namespace Vulkan::HostPasses
