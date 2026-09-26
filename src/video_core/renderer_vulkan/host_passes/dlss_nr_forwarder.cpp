// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Built as nvngx.dll_dlssnr.dll. The Neural Rendering model rejects any caller whose module path
// does not contain "nvngx.dll", so every call into it is made from here. No NVIDIA code.
//
// Each result is stored before it is returned: a tail call would turn into a jump, and the model
// would see this function's caller as its own.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace {

using PfnInit = int(__cdecl*)(unsigned long long, const wchar_t*, void*, void*, void*, const void*,
                              int);
using PfnCreate = int(__cdecl*)(void*, int, void*, void**);
using PfnEvaluate = int(__cdecl*)(void*, void*, void*, void*);
using PfnRelease = int(__cdecl*)(void*);

HMODULE g_model{};
PfnInit g_init{};
PfnCreate g_create{};
PfnEvaluate g_evaluate{};
PfnRelease g_release{};

} // Anonymous namespace

extern "C" {

// One bit per Vulkan entry point found; 15 when the model's surface is complete.
__declspec(dllexport) int ShadNrLoad(const wchar_t* model_path) {
    if (!g_model) {
        g_model = LoadLibraryExW(model_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!g_model) {
            return 0;
        }
        g_init = reinterpret_cast<PfnInit>(GetProcAddress(g_model, "NVSDK_NGX_VULKAN_Init_Ext"));
        g_create =
            reinterpret_cast<PfnCreate>(GetProcAddress(g_model, "NVSDK_NGX_VULKAN_CreateFeature"));
        g_evaluate = reinterpret_cast<PfnEvaluate>(
            GetProcAddress(g_model, "NVSDK_NGX_VULKAN_EvaluateFeature"));
        g_release = reinterpret_cast<PfnRelease>(
            GetProcAddress(g_model, "NVSDK_NGX_VULKAN_ReleaseFeature"));
    }
    return (g_init ? 1 : 0) | (g_create ? 2 : 0) | (g_evaluate ? 4 : 0) | (g_release ? 8 : 0);
}

__declspec(dllexport) int ShadNrInit(const wchar_t* data_path, void* instance,
                                     void* physical_device, void* device, int sdk_version) {
    if (!g_init) {
        return -1;
    }
    volatile int result =
        g_init(0, data_path, instance, physical_device, device, nullptr, sdk_version);
    return result;
}

__declspec(dllexport) int ShadNrCreate(void* command_buffer, int feature, void* parameters,
                                       void** handle) {
    if (!g_create) {
        return -1;
    }
    volatile int result = g_create(command_buffer, feature, parameters, handle);
    return result;
}

__declspec(dllexport) int ShadNrEvaluate(void* command_buffer, void* handle, void* parameters) {
    if (!g_evaluate) {
        return -1;
    }
    volatile int result = g_evaluate(command_buffer, handle, parameters, nullptr);
    return result;
}

__declspec(dllexport) int ShadNrRelease(void* handle) {
    if (!g_release) {
        return -1;
    }
    volatile int result = g_release(handle);
    return result;
}

} // extern "C"
