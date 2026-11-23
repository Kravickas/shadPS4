// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/host_passes/dlss_nr_pass.h"

#if defined(_WIN32) && defined(_MSC_VER)

#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "video_core/host_shaders/nr_decode_comp.h"
#include "video_core/host_shaders/nr_encode_comp.h"
#include "video_core/renderer_vulkan/host_passes/ngx_abi.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::HostPasses {

namespace {

constexpr vk::Format WorkingFormat = vk::Format::eR16G16B16A16Sfloat;
// NVSDK_NGX_Result_Fail.
constexpr Ngx::Result NgxFail = static_cast<Ngx::Result>(0xBAD00000u);

using PfnLoad = int (*)(const wchar_t*);
using PfnInit = int (*)(const wchar_t*, void*, void*, void*, int);
using PfnCreate = int (*)(void*, int, void*, void**);
using PfnEvaluate = int (*)(void*, void*, void*);
using PfnRelease = int (*)(void*);
using PfnCoreInit = Ngx::Result (*)(unsigned long long, const wchar_t*, VkInstance,
                                    VkPhysicalDevice, VkDevice, int, const void*);
using PfnCoreParameters = Ngx::Result (*)(Ngx::Parameter**);
using PfnCoreDestroy = Ngx::Result (*)(Ngx::Parameter*);

std::filesystem::path ExecutableDir() {
    std::array<wchar_t, MAX_PATH> path{};
    GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    return std::filesystem::path{path.data()}.parent_path();
}

std::filesystem::path NgxCoreDir() {
    std::array<wchar_t, MAX_PATH> path{};
    DWORD size = static_cast<DWORD>(path.size() * sizeof(wchar_t));
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\NGXCore",
                     L"NGXPath", RRF_RT_REG_SZ, nullptr, path.data(), &size) != ERROR_SUCCESS) {
        return {};
    }
    return std::filesystem::path{path.data()};
}

bool IsSrgb(vk::Format format) {
    switch (format) {
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return true;
    default:
        return false;
    }
}

// Parameter block used when the driver's NGX core cannot provide one.
class OwnParameters final : public Ngx::Parameter {
public:
    void Set(const char* name, unsigned long long value) override {
        values[name] = {Kind::Integer, value, double(value), nullptr};
    }
    void Set(const char* name, float value) override {
        Set(name, static_cast<double>(value));
    }
    void Set(const char* name, double value) override {
        values[name] = {Kind::Real, static_cast<unsigned long long>(static_cast<long long>(value)),
                        value, nullptr};
    }
    void Set(const char* name, unsigned int value) override {
        values[name] = {Kind::Integer, value, double(value), nullptr};
    }
    void Set(const char* name, int value) override {
        values[name] = {Kind::Integer, static_cast<unsigned long long>(value), double(value),
                        nullptr};
    }
    void Set(const char* name, ID3D11Resource* value) override {
        values[name] = {Kind::Pointer, 0, 0.0, value};
    }
    void Set(const char* name, ID3D12Resource* value) override {
        values[name] = {Kind::Pointer, 0, 0.0, value};
    }
    void Set(const char* name, void* value) override {
        values[name] = {Kind::Pointer, 0, 0.0, value};
    }
    Ngx::Result Get(const char* name, unsigned long long* value) const override {
        return GetNumber(name, [&](const Value& v) { *value = v.integer; });
    }
    Ngx::Result Get(const char* name, float* value) const override {
        return GetNumber(name, [&](const Value& v) { *value = static_cast<float>(v.real); });
    }
    Ngx::Result Get(const char* name, double* value) const override {
        return GetNumber(name, [&](const Value& v) { *value = v.real; });
    }
    Ngx::Result Get(const char* name, unsigned int* value) const override {
        return GetNumber(name, [&](const Value& v) { *value = static_cast<unsigned>(v.integer); });
    }
    Ngx::Result Get(const char* name, int* value) const override {
        return GetNumber(name, [&](const Value& v) { *value = static_cast<int>(v.integer); });
    }
    Ngx::Result Get(const char* name, ID3D11Resource** value) const override {
        return GetPointer(name, reinterpret_cast<void**>(value));
    }
    Ngx::Result Get(const char* name, ID3D12Resource** value) const override {
        return GetPointer(name, reinterpret_cast<void**>(value));
    }
    Ngx::Result Get(const char* name, void** value) const override {
        return GetPointer(name, value);
    }
    void Reset() override {
        values.clear();
    }

private:
    enum class Kind { Integer, Real, Pointer };
    struct Value {
        Kind kind;
        unsigned long long integer;
        double real;
        void* pointer;
    };

    template <typename Store>
    Ngx::Result GetNumber(const char* name, Store&& store) const {
        const auto it = values.find(name);
        if (it == values.end() || it->second.kind == Kind::Pointer) {
            return NgxFail;
        }
        store(it->second);
        return Ngx::Success;
    }

    Ngx::Result GetPointer(const char* name, void** value) const {
        const auto it = values.find(name);
        if (it == values.end() || it->second.kind != Kind::Pointer) {
            return NgxFail;
        }
        *value = it->second.pointer;
        return Ngx::Success;
    }

    std::unordered_map<std::string, Value> values;
};

} // Anonymous namespace

struct DlssNrPass::Impl {
    Impl(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_)
        : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
          input{instance.GetDevice(), instance.GetAllocator()},
          output{instance.GetDevice(), instance.GetAllocator()},
          staging{instance.GetDevice(), instance.GetAllocator()} {}

    ~Impl() {
        const vk::Device device = instance.GetDevice();
        if (feature && release) {
            // Never release under the GPU.
            (void)device.waitIdle();
            release(feature);
        }
        if (core_parameters && core_destroy) {
            core_destroy(core_parameters);
        }
        encode_pipeline.reset();
        decode_pipeline.reset();
        device.destroyShaderModule(encode_module);
        device.destroyShaderModule(decode_module);
    }

    void Fail(const char* reason) {
        failed = true;
        LOG_ERROR(Render_Vulkan, "DLSS NR: {}; the pass is off for this session", reason);
    }

    bool Initialise();
    void CreatePipelines();
    void ResizeTargets(vk::Extent2D extent);
    void Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs);
    void SetTuning();

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;

    HMODULE forwarder{};
    PfnInit init{};
    PfnCreate create{};
    PfnEvaluate evaluate{};
    PfnRelease release{};
    HMODULE core{};
    PfnCoreDestroy core_destroy{};
    Ngx::Parameter* core_parameters{};
    OwnParameters own_parameters;
    Ngx::Parameter* parameters{};
    void* feature{};
    bool initialised{};
    bool failed{};
    bool reset{true};
    bool size_mismatch_logged{};
    u64 frames{};

    vk::ShaderModule encode_module;
    vk::ShaderModule decode_module;
    vk::UniqueDescriptorSetLayout encode_set_layout;
    vk::UniqueDescriptorSetLayout decode_set_layout;
    vk::UniquePipelineLayout encode_layout;
    vk::UniquePipelineLayout decode_layout;
    vk::UniquePipeline encode_pipeline;
    vk::UniquePipeline decode_pipeline;

    vk::Extent2D size{};
    VideoCore::UniqueImage input;
    vk::UniqueImageView input_view;
    VideoCore::UniqueImage output;
    vk::UniqueImageView output_view;
    VideoCore::UniqueImage staging;
    vk::UniqueImageView staging_view;
};

bool DlssNrPass::Impl::Initialise() {
    initialised = true;
    const auto dir = ExecutableDir();
    forwarder = LoadLibraryW((dir / L"nvngx.dll_dlssnr.dll").c_str());
    if (!forwarder) {
        Fail("nvngx.dll_dlssnr.dll did not load from the executable's folder");
        return false;
    }
    const auto load = reinterpret_cast<PfnLoad>(GetProcAddress(forwarder, "ShadNrLoad"));
    init = reinterpret_cast<PfnInit>(GetProcAddress(forwarder, "ShadNrInit"));
    create = reinterpret_cast<PfnCreate>(GetProcAddress(forwarder, "ShadNrCreate"));
    evaluate = reinterpret_cast<PfnEvaluate>(GetProcAddress(forwarder, "ShadNrEvaluate"));
    release = reinterpret_cast<PfnRelease>(GetProcAddress(forwarder, "ShadNrRelease"));
    if (!load || !init || !create || !evaluate || !release) {
        Fail("the forwarder is missing exports");
        return false;
    }
    const int surface = load((dir / L"nvngx_dlssnr.dll").c_str());
    LOG_INFO(Render_Vulkan, "DLSS NR: model Vulkan entry points {:#x} of 0xf", surface);
    if (surface != 0xf) {
        Fail("nvngx_dlssnr.dll is missing or does not expose the Vulkan entry points");
        return false;
    }

    const auto data_path = Common::FS::GetUserPath(Common::FS::PathType::LogDir).wstring();
    const auto vk_instance = static_cast<VkInstance>(instance.GetInstance());
    const auto physical_device = static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice());
    const auto device = static_cast<VkDevice>(instance.GetDevice());

    // The driver's NGX core supplies the parameter block a DLSS game would use.
    if (const auto core_dir = NgxCoreDir(); !core_dir.empty()) {
        core = LoadLibraryW((core_dir / L"_nvngx.dll").c_str());
        if (!core) {
            core = LoadLibraryW((core_dir / L"nvngx.dll").c_str());
        }
    }
    if (core) {
        const auto core_init =
            reinterpret_cast<PfnCoreInit>(GetProcAddress(core, "NVSDK_NGX_VULKAN_Init_Ext"));
        const auto core_allocate = reinterpret_cast<PfnCoreParameters>(
            GetProcAddress(core, "NVSDK_NGX_VULKAN_AllocateParameters"));
        core_destroy = reinterpret_cast<PfnCoreDestroy>(
            GetProcAddress(core, "NVSDK_NGX_VULKAN_DestroyParameters"));
        if (core_init && core_allocate) {
            const Ngx::Result core_result =
                core_init(0, data_path.c_str(), vk_instance, physical_device, device,
                          Ngx::VersionApi, nullptr);
            LOG_INFO(Render_Vulkan, "DLSS NR: NGX core init {:#x}", static_cast<u32>(core_result));
            if (core_result == Ngx::Success) {
                const Ngx::Result allocate_result = core_allocate(&core_parameters);
                LOG_INFO(Render_Vulkan, "DLSS NR: NGX core parameter block {:#x}",
                         static_cast<u32>(allocate_result));
                if (allocate_result != Ngx::Success) {
                    core_parameters = nullptr;
                }
            }
        }
    } else {
        LOG_WARNING(Render_Vulkan, "DLSS NR: NGX core not found through NGXCore\\NGXPath");
    }
    parameters = core_parameters ? core_parameters : &own_parameters;
    LOG_INFO(Render_Vulkan, "DLSS NR: using the {} parameter block",
             core_parameters ? "driver's" : "pass's own");

    const int model_result =
        init(data_path.c_str(), vk_instance, physical_device, device, Ngx::VersionApi);
    LOG_INFO(Render_Vulkan, "DLSS NR: model init {:#x}", static_cast<u32>(model_result));
    if (model_result != Ngx::Success) {
        Fail("the model would not initialise on this device");
        return false;
    }
    CreatePipelines();
    return true;
}

void DlssNrPass::Impl::CreatePipelines() {
    const vk::Device device = instance.GetDevice();
    const auto set_layout = [&](vk::DescriptorType first) {
        const std::array bindings = {
            vk::DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = first,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
            vk::DescriptorSetLayoutBinding{
                .binding = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            },
        };
        return Check<"create dlss nr descriptor set layout">(
            device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
                .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                .bindingCount = static_cast<u32>(bindings.size()),
                .pBindings = bindings.data(),
            }));
    };
    const vk::PushConstantRange constants{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(u32),
    };
    const auto pipeline_layout = [&](vk::DescriptorSetLayout layout) {
        return Check<"create dlss nr pipeline layout">(
            device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
                .setLayoutCount = 1,
                .pSetLayouts = &layout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &constants,
            }));
    };
    const auto pipeline = [&](vk::ShaderModule module, vk::PipelineLayout layout) {
        return Check<"create dlss nr pipeline">(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage =
                        {
                            .stage = vk::ShaderStageFlagBits::eCompute,
                            .module = module,
                            .pName = "main",
                        },
                    .layout = layout,
                }));
    };
    encode_module = CompileSPV(NR_ENCODE_COMP, device);
    ASSERT(encode_module);
    decode_module = CompileSPV(NR_DECODE_COMP, device);
    ASSERT(decode_module);
    encode_set_layout = set_layout(vk::DescriptorType::eSampledImage);
    decode_set_layout = set_layout(vk::DescriptorType::eStorageImage);
    encode_layout = pipeline_layout(*encode_set_layout);
    decode_layout = pipeline_layout(*decode_set_layout);
    encode_pipeline = pipeline(encode_module, *encode_layout);
    decode_pipeline = pipeline(decode_module, *decode_layout);
}

void DlssNrPass::Impl::ResizeTargets(vk::Extent2D extent) {
    if (size == extent) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    scheduler.Finish();
    if (feature) {
        // The model's feature is sized; a new one is built on the next evaluate.
        release(feature);
        feature = nullptr;
    }
    input_view.reset();
    output_view.reset();
    staging_view.reset();
    input.Destroy();
    output.Destroy();
    staging.Destroy();

    size = extent;
    reset = true;
    vk::ImageCreateInfo image_ci{
        .imageType = vk::ImageType::e2D,
        .format = WorkingFormat,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    input.Create(image_ci);
    output.Create(image_ci);
    staging.Create(image_ci);
    SetObjectName(device, static_cast<vk::Image>(input), "DLSS NR Input");
    SetObjectName(device, static_cast<vk::Image>(output), "DLSS NR Output");
    SetObjectName(device, static_cast<vk::Image>(staging), "DLSS NR Staging");

    vk::ImageViewCreateInfo view_ci{
        .viewType = vk::ImageViewType::e2D,
        .format = WorkingFormat,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    view_ci.image = input;
    input_view = Check<"create dlss nr view">(device.createImageViewUnique(view_ci));
    view_ci.image = output;
    output_view = Check<"create dlss nr view">(device.createImageViewUnique(view_ci));
    view_ci.image = staging;
    staging_view = Check<"create dlss nr view">(device.createImageViewUnique(view_ci));
}

// Defaults of the model's controls; read once when the feature is built, and again each evaluate.
void DlssNrPass::Impl::SetTuning() {
    parameters->Set("DLSSNR.Intensity", 1.0f);
    parameters->Set("DLSSNR.Style", 0u);
    parameters->Set("DLSSNR.LocalStructureStrength", 1.0f);
    parameters->Set("DLSSNR.LocalToneStrength", 1.0f);
    parameters->Set("DLSSNR.SkinStructureStrength", -1.0f);
    parameters->Set("DLSSNR.UseAutoMask", 1u);
}

void DlssNrPass::Impl::Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs) {
    if (failed || (!initialised && !Initialise())) {
        return;
    }
    const vk::Extent2D extent{color.info.size.width, color.info.size.height};
    if (extent != inputs.size) {
        if (!size_mismatch_logged) {
            LOG_WARNING(Render_Vulkan, "DLSS NR: presented image {}x{} differs from depth {}x{}",
                        extent.width, extent.height, inputs.size.width, inputs.size.height);
            size_mismatch_logged = true;
        }
        return;
    }
    ResizeTargets(extent);

    const vk::CommandBuffer cmdbuf = scheduler.CommandBuffer();
    scheduler.EndRendering();
    runtime.Transit(&color, vk::ImageLayout::eShaderReadOnlyOptimal,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderSampledRead);
    runtime.FlushBarriers();

    const vk::ImageSubresourceRange color_range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto barrier = [&](vk::Image image, vk::ImageLayout old_layout,
                             vk::ImageLayout new_layout, vk::PipelineStageFlags2 src_stage,
                             vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
                             vk::AccessFlags2 dst_access) {
        return vk::ImageMemoryBarrier2{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .image = image,
            .subresourceRange = color_range,
        };
    };
    const auto all = vk::PipelineStageFlagBits2::eAllCommands;
    const auto any_access = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    const std::array start = {
        barrier(input, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, all, any_access,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageWrite),
        barrier(output, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, all, any_access,
                all, any_access),
        barrier(staging, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, all, any_access,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageWrite),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(start.size()),
        .pImageMemoryBarriers = start.data(),
    });

    const auto dispatch = [&](vk::Pipeline pipeline, vk::PipelineLayout layout,
                              vk::DescriptorType first_type, vk::DescriptorImageInfo first,
                              vk::DescriptorImageInfo second, u32 flag) {
        const std::array writes = {
            vk::WriteDescriptorSet{
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = first_type,
                .pImageInfo = &first,
            },
            vk::WriteDescriptorSet{
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eStorageImage,
                .pImageInfo = &second,
            },
        };
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(flag), &flag);
        cmdbuf.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
    };

    VideoCore::ImageViewInfo color_view_info{};
    color_view_info.format = color.info.pixel_format;
    const vk::ImageView color_view = *color.FindView(color_view_info, false).image_view;
    const bool srgb = IsSrgb(color.info.pixel_format);
    dispatch(*encode_pipeline, *encode_layout, vk::DescriptorType::eSampledImage,
             {.imageView = color_view, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
             {.imageView = *input_view, .imageLayout = vk::ImageLayout::eGeneral}, srgb ? 1u : 0u);

    const std::array to_model = {
        barrier(input, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageWrite, all, any_access),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(to_model.size()),
        .pImageMemoryBarriers = to_model.data(),
    });

    const auto command_buffer = static_cast<VkCommandBuffer>(cmdbuf);
    if (!feature) {
        parameters->Set("DLSSNR.Enabled", 1u);
        parameters->Set("DLSSNR.Width", size.width);
        parameters->Set("DLSSNR.Height", size.height);
        parameters->Set("CreationNodeMask", 1u);
        parameters->Set("VisibilityNodeMask", 1u);
        parameters->Set("DLSSNR.Hint.Render.Preset", 0u);
        SetTuning();
        parameters->Set("DLSSNR.UICorrection", 1u);
        const int result =
            create(command_buffer, Ngx::FeatureNeuralRendering, parameters, &feature);
        LOG_INFO(Render_Vulkan, "DLSS NR: feature create {:#x} at {}x{}", static_cast<u32>(result),
                 size.width, size.height);
        if (result != Ngx::Success || !feature) {
            feature = nullptr;
            Fail("the model would not build a feature");
            return;
        }
        reset = true;
    }

    const auto resource = [](vk::Image image, vk::ImageView view, vk::Format format,
                             vk::Extent2D extent, bool read_write) {
        Ngx::ResourceVk out{};
        out.resource.image_view = {
            .image_view = static_cast<VkImageView>(view),
            .image = static_cast<VkImage>(image),
            .subresource_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            .format = static_cast<VkFormat>(format),
            .width = extent.width,
            .height = extent.height,
        };
        out.type = Ngx::ResourceTypeVk::ImageView;
        out.read_write = read_write;
        return out;
    };
    Ngx::ResourceVk color_resource = resource(input, *input_view, WorkingFormat, size, false);
    Ngx::ResourceVk depth_resource =
        resource(inputs.depth, inputs.depth_view, vk::Format::eR32Sfloat, size, false);
    Ngx::ResourceVk motion_resource =
        resource(inputs.motion, inputs.motion_view, vk::Format::eR16G16Sfloat, size, false);
    Ngx::ResourceVk output_resource = resource(output, *output_view, WorkingFormat, size, true);

    parameters->Set("DLSSNR.Color", static_cast<void*>(&color_resource));
    parameters->Set("DLSSNR.Depth", static_cast<void*>(&depth_resource));
    parameters->Set("DLSSNR.MVec", static_cast<void*>(&motion_resource));
    parameters->Set("DLSSNR.Output", static_cast<void*>(&output_resource));
    parameters->Set("DLSSNR.Enabled", 1u);
    parameters->Set("DLSSNR.Width", size.width);
    parameters->Set("DLSSNR.Height", size.height);
    // The velocity pass sees clears at 1.0 as far in the games checked so far.
    parameters->Set("DLSSNR.DepthInverted", 0u);
    parameters->Set("DLSSNR.Reset", reset ? 1u : 0u);
    for (const char* rect : {"DLSSNR.ColorSubrect", "DLSSNR.OutputSubrect", "DLSSNR.DepthSubrect",
                             "DLSSNR.MVecSubrect"}) {
        const std::string prefix{rect};
        parameters->Set((prefix + "BaseX").c_str(), 0u);
        parameters->Set((prefix + "BaseY").c_str(), 0u);
        parameters->Set((prefix + "Width").c_str(), size.width);
        parameters->Set((prefix + "Height").c_str(), size.height);
    }
    // Vectors are already in pixels, current to previous.
    parameters->Set("DLSSNR.MVecScaleX", 1.0f);
    parameters->Set("DLSSNR.MVecScaleY", 1.0f);
    SetTuning();

    const int evaluated = evaluate(command_buffer, feature, parameters);
    reset = false;
    if (evaluated != Ngx::Success) {
        LOG_ERROR(Render_Vulkan, "DLSS NR: evaluate {:#x}", static_cast<u32>(evaluated));
        Fail("the model refused to evaluate");
        return;
    }
    if (frames++ == 0) {
        LOG_INFO(Render_Vulkan, "DLSS NR: first frame evaluated at {}x{}", size.width, size.height);
    }

    const std::array from_model = {
        barrier(output, vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral, all, any_access,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageRead),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(from_model.size()),
        .pImageMemoryBarriers = from_model.data(),
    });
    dispatch(*decode_pipeline, *decode_layout, vk::DescriptorType::eStorageImage,
             {.imageView = *output_view, .imageLayout = vk::ImageLayout::eGeneral},
             {.imageView = *staging_view, .imageLayout = vk::ImageLayout::eGeneral},
             srgb ? 1u : 0u);

    const std::array to_blit = {
        barrier(staging, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
                vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlagBits2::eShaderStorageWrite, vk::PipelineStageFlagBits2::eTransfer,
                vk::AccessFlagBits2::eTransferRead),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(to_blit.size()),
        .pImageMemoryBarriers = to_blit.data(),
    });
    runtime.Transit(&color, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    runtime.FlushBarriers();

    const vk::ImageSubresourceLayers layers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const std::array<vk::Offset3D, 2> bounds = {
        vk::Offset3D{0, 0, 0},
        vk::Offset3D{static_cast<s32>(size.width), static_cast<s32>(size.height), 1},
    };
    const vk::ImageBlit region{
        .srcSubresource = layers,
        .srcOffsets = bounds,
        .dstSubresource = layers,
        .dstOffsets = bounds,
    };
    cmdbuf.blitImage(staging, vk::ImageLayout::eTransferSrcOptimal, color.GetImage(),
                     vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eNearest);
}

DlssNrPass::DlssNrPass(const Instance& instance, Scheduler& scheduler, Runtime& runtime)
    : impl{std::make_unique<Impl>(instance, scheduler, runtime)} {}

DlssNrPass::~DlssNrPass() = default;

bool DlssNrPass::ModelPresent() {
    return std::filesystem::exists(ExecutableDir() / L"nvngx_dlssnr.dll");
}

void DlssNrPass::Render(VideoCore::Image& color, const XfbVelocityPass::Outputs& inputs) {
    impl->Render(color, inputs);
}

} // namespace Vulkan::HostPasses

#else

namespace Vulkan::HostPasses {

struct DlssNrPass::Impl {};

DlssNrPass::DlssNrPass(const Instance&, Scheduler&, Runtime&) {}

DlssNrPass::~DlssNrPass() = default;

bool DlssNrPass::ModelPresent() {
    return false;
}

void DlssNrPass::Render(VideoCore::Image&, const XfbVelocityPass::Outputs&) {}

} // namespace Vulkan::HostPasses

#endif
