// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>

#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_ /*= false*/)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

namespace {

/// Kept out of IsBufferDescriptor so that it stays small enough to be inlined.
[[noreturn]] void FailDescriptorType(vk::DescriptorType type) {
    UNREACHABLE_MSG("Unexpected descriptor type {}", vk::to_string(type));
}

bool IsBufferDescriptor(vk::DescriptorType type) {
    switch (type) {
    case vk::DescriptorType::eStorageBuffer:
        return true;
    case vk::DescriptorType::eSampler:
    case vk::DescriptorType::eSampledImage:
    case vk::DescriptorType::eStorageImage:
        return false;
    default:
        FailDescriptorType(type);
    }
}

std::size_t DescriptorInfoSize(bool is_buffer) {
    return is_buffer ? sizeof(vk::DescriptorBufferInfo) : sizeof(vk::DescriptorImageInfo);
}

} // Anonymous namespace

void Pipeline::BindResources(DescriptorWrites& set_writes,
                             const Shader::PushData& push_data) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const AmdGpu::VkCallPerf vk_perf{AmdGpu::cp_perf.vk_desc_ns};
    scheduler.Record([layout = *pipeline_layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
    });

    // Bind descriptor set.
    if (set_writes.empty()) {
        return;
    }

    if (uses_push_descriptors) {
        // The writes are followed by the descriptor infos they point to.
        const std::size_t writes_size = set_writes.size() * sizeof(vk::WriteDescriptorSet);
        std::size_t data_size = writes_size;
        for (const auto& set_write : set_writes) {
            data_size += set_write.descriptorCount *
                         DescriptorInfoSize(IsBufferDescriptor(set_write.descriptorType));
        }
        scheduler.Record(
            data_size,
            [&](u8* data) {
                auto* const writes = reinterpret_cast<vk::WriteDescriptorSet*>(data);
                std::memcpy(writes, set_writes.data(), writes_size);
                u8* infos = data + writes_size;
                for (std::size_t i = 0; i < set_writes.size(); ++i) {
                    const auto& set_write = set_writes[i];
                    if (set_write.descriptorCount == 0) {
                        continue;
                    }
                    const bool is_buffer = IsBufferDescriptor(set_write.descriptorType);
                    const std::size_t size =
                        set_write.descriptorCount * DescriptorInfoSize(is_buffer);
                    if (is_buffer) {
                        std::memcpy(infos, set_write.pBufferInfo, size);
                        writes[i].pBufferInfo =
                            reinterpret_cast<const vk::DescriptorBufferInfo*>(infos);
                    } else {
                        std::memcpy(infos, set_write.pImageInfo, size);
                        writes[i].pImageInfo =
                            reinterpret_cast<const vk::DescriptorImageInfo*>(infos);
                    }
                    infos += size;
                }
            },
            [bind_point, layout = *pipeline_layout, count = static_cast<u32>(set_writes.size())](
                vk::CommandBuffer cmdbuf, const u8* data) {
                cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, count,
                                            reinterpret_cast<const vk::WriteDescriptorSet*>(data));
            });
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    scheduler.Record([bind_point, layout = *pipeline_layout, desc_set](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
    });
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

} // namespace Vulkan
