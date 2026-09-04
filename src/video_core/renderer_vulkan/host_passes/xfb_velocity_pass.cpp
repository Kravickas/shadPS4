// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>
#include "common/assert.h"
#include "common/enum.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "shader_recompiler/xfb_layout.h"
#include "video_core/host_shaders/xfb_velocity_frag.h"
#include "video_core/host_shaders/xfb_velocity_vert.h"
#include "video_core/renderer_vulkan/host_passes/xfb_velocity_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/xfb_capture.h"

namespace Vulkan::HostPasses {

// Frames between summary lines.
constexpr u32 ReportInterval = 300;

constexpr vk::Format MotionFormat = vk::Format::eR16G16Sfloat;
constexpr vk::Format MaskFormat = vk::Format::eR8Unorm;

size_t XfbVelocityPass::PipelineKeyHash::operator()(const PipelineKey& key) const noexcept {
    u64 hash = static_cast<u64>(key.depth_format);
    hash = HashCombine(hash, u64(key.depth_clamp) | (u64(key.depth_clip) << 1) |
                                 (u64(key.negative_one_to_one) << 2));
    return static_cast<size_t>(hash);
}

XfbVelocityPass::XfbVelocityPass(const Instance& instance_, Scheduler& scheduler_,
                                 VideoCore::TextureCache& texture_cache_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_},
      motion_image{instance.GetDevice(), instance.GetAllocator()},
      mask_image{instance.GetDevice(), instance.GetAllocator()} {
    const vk::Device device = instance.GetDevice();

    vertex_module =
        Compile(HostShaders::XFB_VELOCITY_VERT, vk::ShaderStageFlagBits::eVertex, device);
    ASSERT(vertex_module);
    SetObjectName(device, vertex_module, "xfb_velocity.vert");

    fragment_module =
        Compile(HostShaders::XFB_VELOCITY_FRAG, vk::ShaderStageFlagBits::eFragment, device);
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
}

XfbVelocityPass::~XfbVelocityPass() {
    const vk::Device device = instance.GetDevice();
    pipelines.clear();
    device.destroyShaderModule(vertex_module);
    device.destroyShaderModule(fragment_module);
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
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
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

void XfbVelocityPass::ResizeTargets(u32 width, u32 height) {
    if (targets_initialized && size.width == width && size.height == height) {
        return;
    }
    const vk::Device device = instance.GetDevice();
    scheduler.Finish();

    motion_view.reset();
    mask_view.reset();
    motion_image.Destroy();
    mask_image.Destroy();

    size = vk::Extent2D{width, height};
    targets_initialized = true;

    vk::ImageCreateInfo image_ci{
        .imageType = vk::ImageType::e2D,
        .format = MotionFormat,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc,
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
}

void XfbVelocityPass::Render(const VideoCore::XfbCapture& capture) {
    const auto cur_regions = capture.CurrentRegions();
    if (cur_regions.empty()) {
        return;
    }

    // The scene depth is the target most captured geometry was drawn against.
    std::unordered_map<u32, u64> depth_weights;
    for (const auto& region : cur_regions) {
        depth_weights[region.depth_id.index] += region.max_vertices;
    }
    const VideoCore::XfbRegion* main = nullptr;
    u64 best_weight = 0;
    for (const auto& region : cur_regions) {
        const u64 weight = depth_weights[region.depth_id.index];
        if (weight > best_weight) {
            best_weight = weight;
            main = &region;
        }
    }
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

    ResizeTargets(main->width, main->height);

    std::unordered_map<u64, std::vector<const VideoCore::XfbRegion*>> prev_by_key;
    for (const auto& region : capture.PreviousRegions()) {
        prev_by_key[region.key].push_back(&region);
    }

    const vk::CommandBuffer cmdbuf = scheduler.CommandBuffer();
    scheduler.EndRendering();

    auto& depth_image = texture_cache.GetImage(main->depth_id);
    const vk::ImageLayout depth_layout = main->has_stencil
                                             ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                             : vk::ImageLayout::eDepthReadOnlyOptimal;
    depth_image.Transit(depth_layout, vk::AccessFlagBits2::eDepthStencilAttachmentRead, {}, cmdbuf);

    const std::array buffer_barriers = {
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = capture.CurrentBuffer().Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        },
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackWriteEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead,
            .buffer = capture.PreviousBuffer().Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        },
        vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransformFeedbackEXT,
            .srcAccessMask = vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT,
            .dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect,
            .dstAccessMask = vk::AccessFlagBits2::eTransformFeedbackCounterReadEXT,
            .buffer = capture.CurrentCounters().Handle(),
            .offset = 0,
            .size = vk::WholeSize,
        },
    };
    const std::array image_barriers = {
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits2::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eShaderSampledRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = motion_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
                            vk::PipelineStageFlagBits2::eFragmentShader,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eShaderSampledRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = mask_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = static_cast<u32>(buffer_barriers.size()),
        .pBufferMemoryBarriers = buffer_barriers.data(),
        .imageMemoryBarrierCount = static_cast<u32>(image_barriers.size()),
        .pImageMemoryBarriers = image_barriers.data(),
    });

    const std::array color_attachments = {
        vk::RenderingAttachmentInfo{
            .imageView = *motion_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                vk::ClearValue{
                    .color = vk::ClearColorValue{.float32 = std::array{0.0f, 0.0f, 0.0f, 0.0f}}},
        },
        vk::RenderingAttachmentInfo{
            .imageView = *mask_view,
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
        vk::DescriptorBufferInfo{
            .buffer = capture.CurrentBuffer().Handle(),
            .offset = 0,
            .range = vk::WholeSize,
        },
        vk::DescriptorBufferInfo{
            .buffer = capture.PreviousBuffer().Handle(),
            .offset = 0,
            .range = vk::WholeSize,
        },
    };
    const std::array set_writes = {
        vk::WriteDescriptorSet{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &buffer_infos[0],
        },
        vk::WriteDescriptorSet{
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &buffer_infos[1],
        },
    };

    std::unordered_map<u64, u32> occurrence;
    vk::Pipeline bound_pipeline{};
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

        const PipelineKey key{
            .depth_format = region.depth_format,
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

        const PushConstants constants{
            .cur_offset = region.offset / 4u,
            .prev_offset = prev ? prev->offset / 4u : 0u,
            .prev_max_vertices = prev ? prev->max_vertices : 0u,
            .has_prev = prev ? 1u : 0u,
            .viewport_size = {state.viewport.width, state.viewport.height},
        };
        cmdbuf.pushConstants(*pipeline_layout,
                             vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
                             0, sizeof(constants), &constants);

        cmdbuf.drawIndirectByteCountEXT(1, 0, capture.CurrentCounters().Handle(),
                                        region.counter_offset, region.offset,
                                        Shader::XfbVertexStride);
    }

    cmdbuf.endRendering();
    scheduler.GetDynamicState().Invalidate();

    window_draws += cur_regions.size();
    window_matched += matched;
    if (++frames >= ReportInterval) {
        LOG_INFO(Render_Vulkan, "xfb velocity: frames={} draws/frame={:.1f} matched={:.2f}% {}x{}",
                 frames, double(window_draws) / double(frames),
                 100.0 * double(window_matched) / double(std::max<u64>(window_draws, 1)),
                 size.width, size.height);
        frames = 0;
        window_draws = 0;
        window_matched = 0;
    }
}

} // namespace Vulkan::HostPasses
