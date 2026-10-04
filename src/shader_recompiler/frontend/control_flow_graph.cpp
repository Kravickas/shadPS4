// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/condition.h"

namespace Shader::Gcn {

struct Compare {
    bool operator()(const Block& lhs, u32 rhs) const noexcept {
        return lhs.begin < rhs;
    }

    bool operator()(u32 lhs, const Block& rhs) const noexcept {
        return lhs < rhs.begin;
    }

    bool operator()(const Block& lhs, const Block& rhs) const noexcept {
        return lhs.begin < rhs.begin;
    }
};

static IR::Condition MakeCondition(const GcnInst& inst) {
    if (inst.IsCmpx()) {
        return IR::Condition::Execnz;
    }

    switch (inst.opcode) {
    case Opcode::S_CBRANCH_SCC0:
        return IR::Condition::Scc0;
    case Opcode::S_CBRANCH_SCC1:
        return IR::Condition::Scc1;
    case Opcode::S_CBRANCH_VCCZ:
        return IR::Condition::Vccz;
    case Opcode::S_CBRANCH_VCCNZ:
        return IR::Condition::Vccnz;
    case Opcode::S_CBRANCH_EXECZ:
        return IR::Condition::ExecWaveZ;
    case Opcode::S_CBRANCH_EXECNZ:
        return IR::Condition::ExecWaveNz;
    default:
        return IR::Condition::True;
    }
}

static bool IgnoresExecMask(const GcnInst& inst) {
    // EXEC mask does not affect scalar instructions or branches. Exports run in every lane, a
    // fragment export discards the lanes with EXEC clear.
    switch (inst.category) {
    case InstCategory::ScalarALU:
    case InstCategory::ScalarMemory:
    case InstCategory::FlowControl:
    case InstCategory::Export:
        return true;
    default:
        break;
    }
    // Compares run in every lane, they clear the bits of the lanes with EXEC clear themselves.
    if (inst.opcode >= Opcode::V_CMP_F_F32 && inst.opcode <= Opcode::V_CMPX_T_U64) {
        return true;
    }
    // Read/Write Lane instructions are not affected either.
    switch (inst.opcode) {
    case Opcode::V_READLANE_B32:
    case Opcode::V_WRITELANE_B32:
    case Opcode::V_READFIRSTLANE_B32:
        return true;
    default:
        break;
    }
    return false;
}

static std::optional<u32> ResolveSetPcTarget(std::span<const GcnInst> list, u32 setpc_index,
                                             std::span<const u32> pc_map) {
    if (setpc_index < 3) {
        return std::nullopt;
    }

    const auto& getpc = list[setpc_index - 3];
    const auto& arith = list[setpc_index - 2];
    const auto& setpc = list[setpc_index];

    if (getpc.opcode != Opcode::S_GETPC_B64 ||
        !(arith.opcode == Opcode::S_ADD_U32 || arith.opcode == Opcode::S_SUB_U32) ||
        setpc.opcode != Opcode::S_SETPC_B64)
        return std::nullopt;

    if (getpc.dst[0].code != setpc.src[0].code || arith.dst[0].code != setpc.src[0].code)
        return std::nullopt;

    if (arith.src_count < 2 || arith.src[1].field != OperandField::LiteralConst)
        return std::nullopt;

    const u32 imm = arith.src[1].code;

    const s32 signed_offset =
        (arith.opcode == Opcode::S_ADD_U32) ? static_cast<s32>(imm) : -static_cast<s32>(imm);

    const u32 base_pc = pc_map[setpc_index - 3] + getpc.length;

    const u32 result_pc = static_cast<u32>(static_cast<s32>(base_pc) + signed_offset);
    LOG_DEBUG(Render_Recompiler, "SetPC target: {} + {} = {}", base_pc, signed_offset, result_pc);
    return result_pc & ~0x3u;
}

static constexpr size_t LabelReserveSize = 32;

CFG::CFG(Common::ObjectPool<Block>& block_pool_, std::span<const GcnInst> inst_list_)
    : block_pool{block_pool_}, inst_list{inst_list_} {
    index_to_pc.resize(inst_list.size() + 1);
    labels.reserve(LabelReserveSize);
    EmitLabels();
    EmitBlocks();
    LinkBlocks();
    SplitDivergenceScopes();
    RemoveUnreachableBlocks();
}

void CFG::EmitLabels() {
    // Always set a label at entry point.
    u32 pc = 0;
    AddLabel(pc);

    // Iterate instruction list and add labels to branch targets.
    for (u32 i = 0; i < inst_list.size(); i++) {
        index_to_pc[i] = pc;
        const GcnInst inst = inst_list[i];
        if (inst.IsUnconditionalBranch()) {
            u32 target = inst.BranchTarget(pc);
            if (inst.opcode == Opcode::S_SETPC_B64) {
                if (auto t = ResolveSetPcTarget(inst_list, i, index_to_pc)) {
                    target = *t;
                } else {
                    ASSERT_MSG(
                        false,
                        "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}): Involved "
                        "instructions not recognized or invalid pattern",
                        pc, i);
                }
            }
            AddLabel(target);
            // Emit this label so that the block ends with the branching instruction
            AddLabel(pc + inst.length);
        } else if (inst.IsConditionalBranch()) {
            const u32 true_label = inst.BranchTarget(pc);
            const u32 false_label = pc + inst.length;
            if (true_label != false_label) {
                AddLabel(true_label);
                AddLabel(false_label);
            }
        } else if (inst.opcode == Opcode::S_ENDPGM) {
            const u32 next_label = pc + inst.length;
            AddLabel(next_label);
        }

        pc += inst.length;
    }
    index_to_pc[inst_list.size()] = pc;

    // Sort labels to make sure block insertion is correct.
    std::ranges::sort(labels);
}

void CFG::SplitDivergenceScopes() {
    // EXEC masks the vector instructions only. Each run of them goes into its own block, which
    // only the lanes with EXEC set enter; scalar instructions and branches stay outside and run
    // in every lane.
    for (auto blk = blocks.begin(); blk != blocks.end(); blk++) {
        if (blk->is_exec_scope) {
            continue;
        }
        u32 first = blk->begin_index;
        while (first <= blk->end_index && IgnoresExecMask(inst_list[first])) {
            ++first;
        }
        if (first > blk->end_index) {
            continue;
        }
        u32 last = first;
        while (last < blk->end_index && !IgnoresExecMask(inst_list[last + 1])) {
            ++last;
        }
        const auto next_blk = std::next(blk);

        // Create a new block for the run.
        Block* block = block_pool.Create();
        block->begin = index_to_pc[first];
        block->end = index_to_pc[last + 1];
        block->begin_index = first;
        block->end_index = last;
        block->end_inst = inst_list[last];
        block->num_predecessors = 1;
        block->is_exec_scope = true;
        block->cond = IR::Condition::True;
        block->end_class = EndClass::Branch;
        blocks.insert_before(next_blk, *block);

        if (last != blk->end_index) {
            // The rest of the parent block follows the run, the lanes that skipped it join there.
            Block* epi_block = block_pool.Create();
            epi_block->begin = index_to_pc[last + 1];
            epi_block->end = blk->end;
            epi_block->begin_index = last + 1;
            epi_block->end_index = blk->end_index;
            epi_block->end_inst = blk->end_inst;
            epi_block->cond = blk->cond;
            epi_block->end_class = blk->end_class;
            epi_block->branch_true = blk->branch_true;
            epi_block->branch_false = blk->branch_false;
            epi_block->num_predecessors = 2;
            blocks.insert_before(next_blk, *epi_block);
            block->branch_true = epi_block;
            blk->branch_false = epi_block;
        } else {
            // The run ends the parent block, both paths continue at its successor.
            ASSERT(blk->cond == IR::Condition::True && blk->branch_true);
            block->branch_true = blk->branch_true;
            blk->branch_false = blk->branch_true;
            blk->branch_true->num_predecessors++;
        }

        // The parent block ends right before the run (it is empty when the run starts it) and
        // enters the run with the lane's EXEC bit.
        blk->end = index_to_pc[first];
        blk->end_index = first - 1;
        if (first != blk->begin_index) {
            blk->end_inst = inst_list[first - 1];
        }
        blk->cond = IR::Condition::Execnz;
        blk->end_class = EndClass::Branch;
        blk->branch_true = block;
    }
}

void CFG::EmitBlocks() {
    for (auto it = labels.cbegin(); it != labels.cend(); ++it) {
        const Label start = *it;
        const auto next_it = std::next(it);
        const bool is_last = (next_it == labels.cend());
        if (is_last) {
            // Last label is special.
            return;
        }
        // The end label is the start instruction of next block.
        // The end instruction of this block is the previous one.
        const Label end = *next_it;
        const size_t end_index = GetIndex(end) - 1;
        const auto& end_inst = inst_list[end_index];

        // Insert block between the labels using the last instruction
        // as an indicator for branching type.
        Block* block = block_pool.Create();
        block->begin = start;
        block->end = end;
        block->begin_index = GetIndex(start);
        block->end_index = end_index;
        block->end_inst = end_inst;
        block->cond = MakeCondition(end_inst);
        blocks.insert(*block);
    }
}

void CFG::LinkBlocks() {
    const auto get_block = [this](u32 address) {
        auto it = blocks.find(address, Compare{});
        ASSERT_MSG(it != blocks.cend() && it->begin == address);
        return &*it;
    };

    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        auto& block = *it;
        const auto end_inst{block.end_inst};

        // If the block doesn't end with a branch we simply
        // need to link with the next block.
        if (!end_inst.IsTerminateInstruction()) {
            auto* next_block = get_block(block.end);
            block.branch_true = next_block;
            block.end_class = EndClass::Branch;
            next_block->num_predecessors++;
            continue;
        }

        // Find the branch targets from the instruction and link the blocks.
        // Note: Block end address is one instruction after end_inst.
        const u32 branch_pc = block.end - end_inst.length;
        u32 target_pc = 0;
        if (end_inst.opcode == Opcode::S_SETPC_B64) {
            auto tgt = ResolveSetPcTarget(inst_list, block.end_index, index_to_pc);
            ASSERT_MSG(tgt,
                       "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}): Involved "
                       "instructions not recognized or invalid pattern",
                       branch_pc, block.end_index);
            target_pc = *tgt;
        } else {
            target_pc = end_inst.BranchTarget(branch_pc);
        }

        if (end_inst.IsUnconditionalBranch()) {
            auto* target_block = get_block(target_pc);
            block.branch_true = target_block;
            block.end_class = EndClass::Branch;
            target_block->num_predecessors++;
        } else if (end_inst.IsConditionalBranch()) {
            auto* end_block = get_block(block.end);
            block.end_class = EndClass::Branch;
            if (target_pc != block.end) {
                auto* target_block = get_block(target_pc);
                block.branch_true = target_block;
                block.branch_false = end_block;
                target_block->num_predecessors++;
            } else {
                block.branch_true = end_block;
                block.cond = IR::Condition::True;
            }
            end_block->num_predecessors++;
        } else if (end_inst.opcode == Opcode::S_ENDPGM) {
            block.end_class = EndClass::Exit;
        } else {
            UNREACHABLE();
        }
    }
}

void CFG::RemoveUnreachableBlocks() {
    for (auto it = std::next(blocks.begin()); it != blocks.end();) {
        if (it->num_predecessors == 0) {
            LOG_WARNING(Render_Recompiler, "Removing unreachable block begin={:#x}", it->begin);
            it = blocks.erase(it);
        } else {
            it++;
        }
    }
}

std::string CFG::Dot() const {
    int node_uid{0};

    const auto name_of = [](const Block& block) { return fmt::format("\"{:#x}\"", block.begin); };

    std::string dot{"digraph shader {\n"};
    dot += fmt::format("\tsubgraph cluster_{} {{\n", 0);
    dot += fmt::format("\t\tnode [style=filled];\n");
    for (const Block& block : blocks) {
        const std::string name{name_of(block)};
        const auto add_branch = [&](Block* branch, bool add_label) {
            dot += fmt::format("\t\t{}->{}", name, name_of(*branch));
            if (add_label && block.cond != IR::Condition::True &&
                block.cond != IR::Condition::False) {
                dot += fmt::format(" [label=\"{}\"]", block.cond);
            }
            dot += '\n';
        };
        dot += fmt::format("\t\t{};\n", name);
        switch (block.end_class) {
        case EndClass::Branch:
            if (block.cond != IR::Condition::False) {
                add_branch(block.branch_true, true);
            }
            if (block.cond != IR::Condition::True) {
                add_branch(block.branch_false, false);
            }
            break;
        case EndClass::Exit:
            dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
            dot +=
                fmt::format("\t\tN{} [label=\"Exit\"][shape=square][style=stripped];\n", node_uid);
            ++node_uid;
            break;
        }
    }
    dot += "\t\tlabel = \"main\";\n\t}\n";
    if (blocks.empty()) {
        dot += "Start;\n";
    } else {
        dot += fmt::format("\tStart -> {};\n", name_of(*blocks.begin()));
    }
    dot += fmt::format("\tStart [shape=diamond];\n");
    dot += "}\n";
    return dot;
}

} // namespace Shader::Gcn
