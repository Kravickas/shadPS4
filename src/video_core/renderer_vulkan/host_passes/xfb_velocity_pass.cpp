// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "common/assert.h"
#include "common/enum.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "shader_recompiler/xfb_layout.h"
#include "video_core/host_shaders/xfb_camera_solve_comp.h"
#include "video_core/host_shaders/xfb_camera_sums_comp.h"
#include "video_core/host_shaders/xfb_velocity_frag.h"
#include "video_core/host_shaders/xfb_velocity_resolve_comp.h"
#include "video_core/host_shaders/xfb_velocity_vert.h"
#include "video_core/renderer_vulkan/host_passes/xfb_velocity_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/xfb_capture.h"

namespace Vulkan::HostPasses {

// Frames between summary lines.
constexpr u32 ReportInterval = 300;

// Doubles per region in camera_sums: 28 sums and 4 slots of solver scratch.
constexpr u64 CameraSumsStride = 32 * sizeof(double);
// clip_to_prev rows and stats.
constexpr u64 CameraResultSize = 5 * 4 * sizeof(float);

constexpr vk::Format MotionFormat = vk::Format::eR16G16Sfloat;
constexpr vk::Format MaskFormat = vk::Format::eR8Unorm;

namespace {

// The scene depth is the target drawn with the largest viewport sharing the output aspect ratio.
// Image size is not enough: a pass can render into a sub-rectangle of a full-size target.
const VideoCore::XfbRegion* SelectSceneDepth(std::span<const VideoCore::XfbRegion> regions,
                                             vk::Extent2D output) {
    struct Candidate {
        const VideoCore::XfbRegion* region{};
        u64 weight{};
        u32 viewport_width{};
        u32 viewport_height{};
    };
    std::unordered_map<u32, Candidate> candidates;
    for (const auto& region : regions) {
        auto& candidate = candidates[region.depth_id.index];
        if (!candidate.region) {
            candidate.region = &region;
        }
        candidate.weight += region.max_vertices;
        const u32 width = static_cast<u32>(std::abs(region.state.viewport.width));
        const u32 height = static_cast<u32>(std::abs(region.state.viewport.height));
        if (u64(width) * height > u64(candidate.viewport_width) * candidate.viewport_height) {
            candidate.viewport_width = width;
            candidate.viewport_height = height;
        }
    }
    const VideoCore::XfbRegion* best = nullptr;
    bool best_matches = false;
    u64 best_area = 0;
    u64 best_weight = 0;
    for (const auto& [index, candidate] : candidates) {
        const auto& region = *candidate.region;
        const bool matches = VideoCore::XfbMatchesAspect(candidate.viewport_width,
                                                         candidate.viewport_height, output);
        const u64 area = matches ? u64(candidate.viewport_width) * candidate.viewport_height : 0;
        if (!best || std::tie(matches, area, candidate.weight) >
                         std::tie(best_matches, best_area, best_weight)) {
            best = &region;
            best_matches = matches;
            best_area = area;
            best_weight = candidate.weight;
        }
    }
    return best;
}

} // Anonymous namespace

size_t XfbVelocityPass::PipelineKeyHash::operator()(const PipelineKey& key) const noexcept {
    u64 hash = static_cast<u64>(key.depth_format);
    hash = HashCombine(hash, static_cast<u64>(key.samples));
    hash = HashCombine(hash, u64(key.depth_clamp) | (u64(key.depth_clip) << 1) |
                                 (u64(key.negative_one_to_one) << 2));
    return static_cast<size_t>(hash);
}

XfbVelocityPass::XfbVelocityPass(const Instance& instance_, Scheduler& scheduler_,
                                 Runtime& runtime_, VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, texture_cache{texture_cache_},
      region_table{instance, 0, sizeof(RegionEntry) * VideoCore::XfbCapture::MaxRegions,
                   VideoCore::MemoryType::DeviceLocal, "XFB Velocity Regions"},
      camera_sums{instance, 0, CameraSumsStride * VideoCore::XfbCapture::MaxRegions,
                  VideoCore::MemoryType::DeviceLocal, "XFB Camera Sums"},
      camera_result{instance, 0, CameraResultSize, VideoCore::MemoryType::DeviceLocal,
                    "XFB Camera"},
      motion_image{instance.GetDevice(), instance.GetAllocator()},
      mask_image{instance.GetDevice(), instance.GetAllocator()},
      motion_ms_image{instance.GetDevice(), instance.GetAllocator()},
      mask_ms_image{instance.GetDevice(), instance.GetAllocator()} {
    const vk::Device device = instance.GetDevice();

    vertex_module = CompileSPV(XFB_VELOCITY_VERT, device);
    ASSERT(vertex_module);
    SetObjectName(device, vertex_module, "xfb_velocity.vert");

    fragment_module = CompileSPV(XFB_VELOCITY_FRAG, device);
    ASSERT(fragment_module);
    SetObjectName(device, fragment_module, "xfb_velocity.frag");

    const std::array bindings = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eVertex,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
    };
    desc_set_layout = Check<"create xfb velocity descriptor set layout">(
        device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));

    const vk::PushConstantRange push_constants{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    pipeline_layout = Check<"create xfb velocity pipeline layout">(
        device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*desc_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constants,
        }));

    resolve_module = CompileSPV(XFB_VELOCITY_RESOLVE_COMP, device);
    ASSERT(resolve_module);
    SetObjectName(device, resolve_module, "xfb_velocity_resolve.comp");

    const std::array resolve_bindings = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };
    resolve_set_layout = Check<"create xfb resolve descriptor set layout">(
        device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(resolve_bindings.size()),
            .pBindings = resolve_bindings.data(),
        }));
    resolve_layout = Check<"create xfb resolve pipeline layout">(
        device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*resolve_set_layout,
        }));
    resolve_pipeline = Check<"create xfb resolve pipeline">(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage =
                    {
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = resolve_module,
                        .pName = "main",
                    },
                .layout = *resolve_layout,
            }));
    SetObjectName(device, *resolve_pipeline, "xfb velocity resolve pipeline");

    // The solve accumulates in doubles; without them there is no camera fallback.
    if (instance.IsShaderFloat64Supported()) {
        CreateCameraPipelines();
    }
}

XfbVelocityPass::~XfbVelocityPass() {
    const vk::Device device = instance.GetDevice();
    pipelines.clear();
    resolve_pipeline.reset();
    camera_sums_pipeline.reset();
    camera_solve_pipeline.reset();
    device.destroyShaderModule(vertex_module);
    device.destroyShaderModule(fragment_module);
    device.destroyShaderModule(resolve_module);
    device.destroyShaderModule(camera_sums_module);
    device.destroyShaderModule(camera_solve_module);
}

void XfbVelocityPass::CreateCameraPipelines() {
    const vk::Device device = instance.GetDevice();
    const auto storage_bindings = [](u32 count) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings(count);
        for (u32 i = 0; i < count; ++i) {
            bindings[i] = vk::DescriptorSetLayoutBinding{
                .binding = i,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = vk::ShaderStageFlagBits::eCompute,
            };
        }
        return bindings;
    };
    const auto create_set_layout =
        [&](const std::vector<vk::DescriptorSetLayoutBinding>& bindings) {
            return Check<"create xfb camera descriptor set layout">(
                device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{
                    .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
                    .bindingCount = static_cast<u32>(bindings.size()),
                    .pBindings = bindings.data(),
                }));
        };
    const auto create_pipeline = [&](vk::ShaderModule module, vk::PipelineLayout layout) {
        return Check<"create xfb camera pipeline">(device.createComputePipelineUnique(
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

    camera_sums_module = CompileSPV(XFB_CAMERA_SUMS_COMP, device);
    ASSERT(camera_sums_module);
    SetObjectName(device, camera_sums_module, "xfb_camera_sums.comp");
    camera_solve_module = CompileSPV(XFB_CAMERA_SOLVE_COMP, device);
    ASSERT(camera_solve_module);
    SetObjectName(device, camera_solve_module, "xfb_camera_solve.comp");

    camera_sums_set_layout = create_set_layout(storage_bindings(5));
    camera_solve_set_layout = create_set_layout(storage_bindings(2));

    camera_sums_layout = Check<"create xfb camera sums pipeline layout">(
        device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*camera_sums_set_layout,
        }));
    const vk::PushConstantRange solve_constants{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(u32),
    };
    camera_solve_layout = Check<"create xfb camera solve pipeline layout">(
        device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &*camera_solve_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &solve_constants,
        }));

    camera_sums_pipeline = create_pipeline(camera_sums_module, *camera_sums_layout);
    SetObjectName(device, *camera_sums_pipeline, "xfb camera sums pipeline");
    camera_solve_pipeline = create_pipeline(camera_solve_module, *camera_solve_layout);
    SetObjectName(device, *camera_solve_pipeline, "xfb camera solve pipeline");
    camera_supported = true;
}

void XfbVelocityPass::SolveCamera(vk::CommandBuffer cmdbuf, const VideoCore::XfbCapture& capture,
                                  u32 count) {
    const auto to_fragment = [&] {
        const vk::BufferMemoryBarrier2 barrier{
            .srcStageMask =
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask =
                vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = camera_result.Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier,
        });
    };
    if (!camera_supported || count == 0) {
        // stats.x == 0 disables the fallback in the fragment shader.
        cmdbuf.fillBuffer(camera_result.Handle(), 0, vk::WholeSize, 0);
        to_fragment();
        return;
    }

    const std::array sums_infos = {
        vk::DescriptorBufferInfo{capture.CurrentBuffer(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{capture.PreviousBuffer(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{capture.CurrentCounters(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{region_table.Handle(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{camera_sums.Handle(), 0, vk::WholeSize},
    };
    std::array<vk::WriteDescriptorSet, 5> sums_writes{};
    for (u32 i = 0; i < sums_writes.size(); ++i) {
        sums_writes[i] = vk::WriteDescriptorSet{
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &sums_infos[i],
        };
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *camera_sums_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *camera_sums_layout, 0,
                                sums_writes);
    cmdbuf.dispatch(count, 1, 1);

    const vk::BufferMemoryBarrier2 sums_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        .buffer = camera_sums.Handle(),
        .offset = 0,
        .size = vk::WholeSize,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &sums_barrier,
    });

    const std::array solve_infos = {
        vk::DescriptorBufferInfo{camera_sums.Handle(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{camera_result.Handle(), 0, vk::WholeSize},
    };
    std::array<vk::WriteDescriptorSet, 2> solve_writes{};
    for (u32 i = 0; i < solve_writes.size(); ++i) {
        solve_writes[i] = vk::WriteDescriptorSet{
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &solve_infos[i],
        };
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *camera_solve_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *camera_solve_layout, 0,
                                solve_writes);
    cmdbuf.pushConstants(*camera_solve_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(count),
                         &count);
    cmdbuf.dispatch(1, 1, 1);
    to_fragment();
}

vk::Pipeline XfbVelocityPass::GetPipeline(const PipelineKey& key) {
    if (const auto it = pipelines.find(key); it != pipelines.end()) {
        return *it->second;
    }

    const std::array stages = {
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertex_module,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragment_module,
            .pName = "main",
        },
    };

    const std::array color_formats = {MotionFormat, MaskFormat};
    const vk::PipelineRenderingCreateInfo rendering_ci{
        .colorAttachmentCount = static_cast<u32>(color_formats.size()),
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = key.depth_format,
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    const vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control{
        .negativeOneToOne = key.negative_one_to_one,
    };
    const vk::PipelineViewportStateCreateInfo viewport_info{
        .pNext = instance.IsDepthClipControlSupported() ? &clip_control : nullptr,
    };

    vk::StructureChain raster_chain = {
        vk::PipelineRasterizationStateCreateInfo{
            .depthClampEnable =
                key.depth_clamp && (!key.depth_clip || instance.IsDepthClipEnableSupported()),
            .rasterizerDiscardEnable = false,
            .polygonMode = vk::PolygonMode::eFill,
            .lineWidth = 1.0f,
        },
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT{
            .depthClipEnable = key.depth_clip,
        },
    };
    if (!instance.IsDepthClipEnableSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>();
    }

    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = key.samples,
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_stencil{
        .depthTestEnable = true,
        .depthWriteEnable = false,
        .depthCompareOp = vk::CompareOp::eEqual,
        .stencilTestEnable = false,
    };

    const std::array blend_attachments = {
        vk::PipelineColorBlendAttachmentState{
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG,
        },
        vk::PipelineColorBlendAttachmentState{
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR,
        },
    };
    const vk::PipelineColorBlendStateCreateInfo color_blend{
        .attachmentCount = static_cast<u32>(blend_attachments.size()),
        .pAttachments = blend_attachments.data(),
    };

    const std::array dynamic_states = {
        vk::DynamicState::eViewportWithCount, vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eDepthBiasEnable,   vk::DynamicState::eDepthBias,
        vk::DynamicState::eCullMode,          vk::DynamicState::eFrontFace,
    };
    const vk::PipelineDynamicStateCreateInfo dynamic_info{
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::GraphicsPipelineCreateInfo pipeline_ci{
        .pNext = &rendering_ci,
        .stageCount = static_cast<u32>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_chain.get(),
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic_info,
        .layout = *pipeline_layout,
    };

    auto pipeline = Check<"create xfb velocity pipeline">(
        instance.GetDevice().createGraphicsPipelineUnique({}, pipeline_ci));
    SetObjectName(instance.GetDevice(), *pipeline, "xfb velocity pipeline");
    const vk::Pipeline handle = *pipeline;
    pipelines.emplace(key, std::move(pipeline));
    return handle;
}

void XfbVelocityPass::ResizeTargets(u32 width, u32 height, vk::SampleCountFlagBits samples_) {
    if (targets_initialized && size.width == width && size.height == height &&
        samples == samples_) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    scheduler.Finish();

    motion_view.reset();
    mask_view.reset();
    motion_ms_view.reset();
    mask_ms_view.reset();
    motion_image.Destroy();
    mask_image.Destroy();
    motion_ms_image.Destroy();
    mask_ms_image.Destroy();

    size = vk::Extent2D{width, height};
    samples = samples_;
    targets_initialized = true;
    const bool multisampled = samples != vk::SampleCountFlagBits::e1;

    vk::ImageCreateInfo image_ci{
        .imageType = vk::ImageType::e2D,
        .format = MotionFormat,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc |
                 (multisampled ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlags{}),
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    motion_image.Create(image_ci);
    SetObjectName(device, static_cast<vk::Image>(motion_image), "XFB Velocity Motion");

    image_ci.format = MaskFormat;
    mask_image.Create(image_ci);
    SetObjectName(device, static_cast<vk::Image>(mask_image), "XFB Velocity Mask");

    vk::ImageViewCreateInfo view_ci{
        .image = motion_image,
        .viewType = vk::ImageViewType::e2D,
        .format = MotionFormat,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    motion_view = Check<"create xfb motion view">(device.createImageViewUnique(view_ci));
    SetObjectName(device, *motion_view, "XFB Velocity Motion View");

    view_ci.image = mask_image;
    view_ci.format = MaskFormat;
    mask_view = Check<"create xfb mask view">(device.createImageViewUnique(view_ci));
    SetObjectName(device, *mask_view, "XFB Velocity Mask View");

    if (!multisampled) {
        return;
    }
    image_ci.samples = samples;
    image_ci.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
    image_ci.format = MotionFormat;
    motion_ms_image.Create(image_ci);
    SetObjectName(device, static_cast<vk::Image>(motion_ms_image), "XFB Velocity Motion MS");
    image_ci.format = MaskFormat;
    mask_ms_image.Create(image_ci);
    SetObjectName(device, static_cast<vk::Image>(mask_ms_image), "XFB Velocity Mask MS");

    view_ci.image = motion_ms_image;
    view_ci.format = MotionFormat;
    motion_ms_view = Check<"create xfb motion ms view">(device.createImageViewUnique(view_ci));
    view_ci.image = mask_ms_image;
    view_ci.format = MaskFormat;
    mask_ms_view = Check<"create xfb mask ms view">(device.createImageViewUnique(view_ci));
}

void XfbVelocityPass::Resolve(vk::CommandBuffer cmdbuf) {
    const vk::ImageSubresourceRange color_range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto to_sampled = [&](vk::Image image) {
        return vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = image,
            .subresourceRange = color_range,
        };
    };
    const auto to_storage = [&](vk::Image image) {
        return vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = image,
            .subresourceRange = color_range,
        };
    };
    const std::array barriers = {
        to_sampled(motion_ms_image),
        to_sampled(mask_ms_image),
        to_storage(motion_image),
        to_storage(mask_image),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });

    const std::array image_infos = {
        vk::DescriptorImageInfo{
            .imageView = *motion_ms_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        },
        vk::DescriptorImageInfo{
            .imageView = *mask_ms_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        },
        vk::DescriptorImageInfo{
            .imageView = *motion_view,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
        vk::DescriptorImageInfo{
            .imageView = *mask_view,
            .imageLayout = vk::ImageLayout::eGeneral,
        },
    };
    std::array<vk::WriteDescriptorSet, 4> writes{};
    for (u32 i = 0; i < writes.size(); ++i) {
        writes[i] = vk::WriteDescriptorSet{
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType =
                i < 2 ? vk::DescriptorType::eSampledImage : vk::DescriptorType::eStorageImage,
            .pImageInfo = &image_infos[i],
        };
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *resolve_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *resolve_layout, 0, writes);
    cmdbuf.dispatch((size.width + 7) / 8, (size.height + 7) / 8, 1);
}

void XfbVelocityPass::Render(const VideoCore::XfbCapture& capture, vk::Extent2D output_extent) {
    const auto cur_regions = capture.CurrentRegions();
    if (cur_regions.empty()) {
        return;
    }

    const VideoCore::XfbRegion* main = SelectSceneDepth(cur_regions, output_extent);
    if (!main || !main->depth_view) {
        return;
    }
    // The texture cache may have deleted or recycled the depth target since capture. An
    // unregistered image is already queued for destruction and must not be recorded against.
    if (!texture_cache.IsImageAllocated(main->depth_id)) {
        return;
    }
    if (const auto& image = texture_cache.GetImage(main->depth_id);
        image.GetImage() != main->depth_image ||
        False(image.flags & VideoCore::ImageFlagBits::Registered)) {
        return;
    }

    ResizeTargets(main->width, main->height, main->samples);
    const bool multisampled = samples != vk::SampleCountFlagBits::e1;
    const vk::Image motion_target = multisampled ? static_cast<vk::Image>(motion_ms_image)
                                                 : static_cast<vk::Image>(motion_image);
    const vk::Image mask_target =
        multisampled ? static_cast<vk::Image>(mask_ms_image) : static_cast<vk::Image>(mask_image);
    const vk::ImageView motion_target_view = multisampled ? *motion_ms_view : *motion_view;
    const vk::ImageView mask_target_view = multisampled ? *mask_ms_view : *mask_view;

    // Only previous draws against the previous scene depth are candidates. The same mesh drawn
    // into a reflection or shadow pass shares the key, and its occurrence would otherwise shift
    // every match after it onto the wrong pass.
    const auto prev_regions = capture.PreviousRegions();
    const VideoCore::XfbRegion* prev_main = SelectSceneDepth(prev_regions, output_extent);
    std::unordered_map<u64, std::vector<const VideoCore::XfbRegion*>> prev_by_key;
    if (prev_main) {
        for (const auto& region : prev_regions) {
            if (region.depth_id == prev_main->depth_id) {
                prev_by_key[region.key].push_back(&region);
            }
        }
    }

    const vk::CommandBuffer cmdbuf = scheduler.CommandBuffer();
    scheduler.EndRendering();

    auto& depth_image = texture_cache.GetImage(main->depth_id);
    const vk::ImageLayout depth_layout = main->has_stencil
                                             ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                             : vk::ImageLayout::eDepthReadOnlyOptimal;
    runtime.Transit(&depth_image, depth_layout,
                    vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                        vk::PipelineStageFlagBits2::eLateFragmentTests,
                    vk::AccessFlagBits2::eDepthStencilAttachmentRead);
    runtime.FlushBarriers();

    // Match each scene draw to its previous-frame counterpart and build its constants.
    const bool clip_control = instance.IsDepthClipControlSupported();
    std::unordered_map<u64, u32> occurrence;
    std::vector<const VideoCore::XfbRegion*> draws;
    entries.clear();
    u64 matched = 0;
    for (const auto& region : cur_regions) {
        if (region.depth_id != main->depth_id) {
            continue;
        }
        const u32 index = occurrence[region.key]++;
        const VideoCore::XfbRegion* prev = nullptr;
        if (const auto it = prev_by_key.find(region.key); it != prev_by_key.end()) {
            if (index < it->second.size()) {
                prev = it->second[index];
            }
        }
        matched += prev != nullptr;
        const auto& viewport = region.state.viewport;
        entries.push_back(RegionEntry{
            .constants =
                {
                    .cur_offset = region.offset / 4u,
                    .prev_offset = prev ? prev->offset / 4u : 0u,
                    .prev_max_vertices = prev ? prev->max_vertices : 0u,
                    .has_prev = prev && !region.blended ? 1u : 0u,
                    .viewport_size = {viewport.width, viewport.height},
                    .viewport_offset = {viewport.x, viewport.y},
                    .depth_range = {viewport.minDepth, viewport.maxDepth},
                    .blended = region.blended ? 1u : 0u,
                    .negative_one_to_one =
                        region.state.negative_one_to_one && clip_control ? 1u : 0u,
                },
            .counter_offset = region.counter_offset,
            .pad = {},
        });
        draws.push_back(&region);
    }
    const u64 scene_draws = draws.size();

    // Last frame's reads of the region table, camera sums and camera result come first.
    const vk::MemoryBarrier2 reuse_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eFragmentShader,
        .srcAccessMask =
            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask =
            vk::PipelineStageFlagBits2::eTransfer | vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite |
                         vk::AccessFlagBits2::eShaderStorageRead |
                         vk::AccessFlagBits2::eShaderStorageWrite,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &reuse_barrier,
    });

    // vkCmdUpdateBuffer takes at most 65536 bytes per call.
    constexpr size_t EntriesPerUpdate = 65536 / sizeof(RegionEntry);
    for (size_t first = 0; first < entries.size(); first += EntriesPerUpdate) {
        const size_t count = std::min(EntriesPerUpdate, entries.size() - first);
        cmdbuf.updateBuffer(region_table.Handle(), first * sizeof(RegionEntry),
                            count * sizeof(RegionEntry),
                            static_cast<const void*>(entries.data() + first));
    }

    const std::array buffer_barriers = {
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader |
                            vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = capture.CurrentBuffer(),
            .offset = 0,
            .size = vk::WholeSize,
        },
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader |
                            vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = capture.PreviousBuffer(),
            .offset = 0,
            .size = vk::WholeSize,
        },
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect |
                            vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eTransformFeedbackCounterReadEXT |
                             vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = capture.CurrentCounters(),
            .offset = 0,
            .size = vk::WholeSize,
        },
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = region_table.Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        },
    };
    const auto target_barrier = [](vk::Image image) {
        return vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits2::eFragmentShader |
                            vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eShaderSampledRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        };
    };
    const std::array image_barriers = {target_barrier(motion_target), target_barrier(mask_target)};
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers.size()),
        .pBufferMemoryBarriers = buffer_barriers.data(),
        .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
        .pImageMemoryBarriers = image_barriers.data(),
    });

    SolveCamera(cmdbuf, capture, static_cast<u32>(entries.size()));

    const std::array color_attachments = {
        vk::RenderingAttachmentInfo{
            .imageView = motion_target_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                vk::ClearValue{
                    .color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
        },
        vk::RenderingAttachmentInfo{
            .imageView = mask_target_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                vk::ClearValue{
                    .color = vk::ClearColorValue{.float32 = std::array{1.0f, 0.0f, 0.0f, 0.0f}}},
        },
    };
    const vk::RenderingAttachmentInfo depth_attachment{
        .imageView = main->depth_view,
        .imageLayout = depth_layout,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eNone,
    };
    cmdbuf.beginRendering(vk::RenderingInfo{
        .renderArea = {.offset = {0, 0}, .extent = size},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<u32>(color_attachments.size()),
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = &depth_attachment,
    });

    const std::array buffer_infos = {
        vk::DescriptorBufferInfo{capture.CurrentBuffer(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{capture.PreviousBuffer(), 0, vk::WholeSize},
        vk::DescriptorBufferInfo{camera_result.Handle(), 0, vk::WholeSize},
    };
    std::array<vk::WriteDescriptorSet, 3> set_writes{};
    for (u32 i = 0; i < set_writes.size(); ++i) {
        set_writes[i] = vk::WriteDescriptorSet{
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &buffer_infos[i],
        };
    }

    vk::Pipeline bound_pipeline{};
    for (size_t i = 0; i < draws.size(); ++i) {
        const auto& region = *draws[i];
        const PipelineKey key{
            .depth_format = region.depth_format,
            .samples = region.samples,
            .depth_clamp = region.state.depth_clamp,
            .depth_clip = region.state.depth_clip,
            .negative_one_to_one = region.state.negative_one_to_one,
        };
        const vk::Pipeline pipeline = GetPipeline(key);
        if (pipeline != bound_pipeline) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0,
                                        set_writes);
            bound_pipeline = pipeline;
        }

        const auto& state = region.state;
        cmdbuf.setViewportWithCount(state.viewport);
        cmdbuf.setScissorWithCount(state.scissor);
        cmdbuf.setCullMode(state.cull_mode);
        cmdbuf.setFrontFace(state.front_face);
        cmdbuf.setDepthBiasEnable(state.depth_bias_enabled);
        cmdbuf.setDepthBias(state.depth_bias_constant, state.depth_bias_clamp,
                            state.depth_bias_slope);
        cmdbuf.pushConstants(*pipeline_layout,
                             vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                             0, sizeof(PushConstants), &entries[i].constants);

        // Counter is relative to the bound range, which already starts at the region.
        cmdbuf.drawIndirectByteCountEXT(1, 0, capture.CurrentCounters(), region.counter_offset, 0,
                                        Shader::XfbVertexStride);
    }

    cmdbuf.endRendering();
    if (multisampled) {
        Resolve(cmdbuf);
    }
    scheduler.GetDynamicState().Invalidate();

    window_draws += scene_draws;
    window_matched += matched;
    if (++frames >= ReportInterval) {
        LOG_INFO(Render_Vulkan,
                 "xfb velocity: frames={} scene_draws/frame={:.1f} matched={:.2f}% depth={}x{} "
                 "output={}x{}",
                 frames, double(window_draws) / double(frames),
                 100.0 * double(window_matched) / double(std::max<u64>(window_draws, 1)),
                 size.width, size.height, output_extent.width, output_extent.height);
        frames = 0;
        window_draws = 0;
        window_matched = 0;
    }
}

} // namespace Vulkan::HostPasses
