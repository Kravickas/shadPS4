// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

// ABI-compatible declarations of the few NGX types the Neural Rendering pass touches. The layouts
// and the virtual order are NVIDIA's; nothing here is NVIDIA code.

struct ID3D11Resource;
struct ID3D12Resource;

namespace Vulkan::Ngx {

// NVSDK_NGX_Result: 1 is success, failures are 0xBAD000xx.
using Result = int;
constexpr Result Success = 1;
constexpr int VersionApi = 0x15;
constexpr int FeatureNeuralRendering = 18;

// The virtual order is the ABI: eight setters, eight getters, then Reset, each group in NVIDIA's
// declaration order. Compiled for the MSVC ABI this yields the slots the driver's block uses.
struct Parameter {
    virtual void Set(const char* name, unsigned long long value) = 0;
    virtual void Set(const char* name, float value) = 0;
    virtual void Set(const char* name, double value) = 0;
    virtual void Set(const char* name, unsigned int value) = 0;
    virtual void Set(const char* name, int value) = 0;
    virtual void Set(const char* name, ID3D11Resource* value) = 0;
    virtual void Set(const char* name, ID3D12Resource* value) = 0;
    virtual void Set(const char* name, void* value) = 0;
    virtual Result Get(const char* name, unsigned long long* value) const = 0;
    virtual Result Get(const char* name, float* value) const = 0;
    virtual Result Get(const char* name, double* value) const = 0;
    virtual Result Get(const char* name, unsigned int* value) const = 0;
    virtual Result Get(const char* name, int* value) const = 0;
    virtual Result Get(const char* name, ID3D11Resource** value) const = 0;
    virtual Result Get(const char* name, ID3D12Resource** value) const = 0;
    virtual Result Get(const char* name, void** value) const = 0;
    virtual void Reset() = 0;
};

struct ImageViewInfoVk {
    VkImageView image_view;
    VkImage image;
    VkImageSubresourceRange subresource_range;
    VkFormat format;
    unsigned int width;
    unsigned int height;
};

struct BufferInfoVk {
    VkBuffer buffer;
    unsigned int size_in_bytes;
};

enum class ResourceTypeVk : int {
    ImageView = 0,
    Buffer = 1,
};

struct ResourceVk {
    union {
        ImageViewInfoVk image_view;
        BufferInfoVk buffer;
    } resource;
    ResourceTypeVk type;
    bool read_write;
};

} // namespace Vulkan::Ngx
