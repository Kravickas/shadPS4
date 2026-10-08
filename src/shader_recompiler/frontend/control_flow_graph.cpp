// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <map>
#include <optional>
#include <unordered_map>
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
        return IR::Condition::Execz;
    case Opcode::S_CBRANCH_EXECNZ:
        return IR::Condition::Execnz;
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

// Instructions without any effect, they can stay inside an EXEC scope.
static bool IsExecNeutral(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_NOP:
    case Opcode::S_WAITCNT:
    case Opcode::S_SETPRIO:
        return true;
    default:
        return false;
    }
}

static bool IsSaveExec(const GcnInst& inst) {
    return inst.opcode >= Opcode::S_AND_SAVEEXEC_B64 && inst.opcode <= Opcode::S_XNOR_SAVEEXEC_B64;
}

static bool WritesExec(const GcnInst& inst) {
    // S_WQM_B64 is translated as a no-op, it does not change the EXEC of an invocation.
    if (inst.opcode == Opcode::S_WQM_B64) {
        return false;
    }
    return IsSaveExec(inst) || inst.IsCmpx() ||
           (inst.dst_count > 0 && (inst.dst[0].field == OperandField::ExecLo ||
                                   inst.dst[0].field == OperandField::ExecHi));
}

// Registers written by a scalar memory instruction.
static u32 ScalarLoadWidth(Opcode opcode) {
    switch (opcode) {
    case Opcode::S_LOAD_DWORD:
    case Opcode::S_BUFFER_LOAD_DWORD:
        return 1;
    case Opcode::S_LOAD_DWORDX2:
    case Opcode::S_BUFFER_LOAD_DWORDX2:
    case Opcode::S_MEMTIME:
        return 2;
    case Opcode::S_LOAD_DWORDX4:
    case Opcode::S_BUFFER_LOAD_DWORDX4:
        return 4;
    case Opcode::S_LOAD_DWORDX8:
    case Opcode::S_BUFFER_LOAD_DWORDX8:
        return 8;
    default:
        return 16;
    }
}

namespace {
// What is known about EXEC at a program point: whether every lane that executes it has its EXEC
// bit set, and which registers hold a half of a copy of such a full mask. A tag is
// (copy id << 1) | half; the copy id is the index of the instruction that saved it.
struct ExecState {
    bool full{};
    std::array<u32, 128> sgpr_tag{}; // SGPRs and VCC (106, 107)
    std::map<u32, u32> lane_tag{};   // VGPR lanes holding a spilled half: vgpr * 64 + lane

    bool operator==(const ExecState&) const = default;

    bool HoldsFullCopy(u32 reg) const {
        return reg + 1 < sgpr_tag.size() && sgpr_tag[reg] != 0 && (sgpr_tag[reg] & 1) == 0 &&
               sgpr_tag[reg + 1] == sgpr_tag[reg] + 1;
    }

    static ExecState Merge(const ExecState& a, const ExecState& b) {
        ExecState merged{.full = a.full && b.full};
        for (u32 i = 0; i < merged.sgpr_tag.size(); ++i) {
            merged.sgpr_tag[i] = a.sgpr_tag[i] == b.sgpr_tag[i] ? a.sgpr_tag[i] : 0;
        }
        for (const auto& [key, tag] : a.lane_tag) {
            const auto it = b.lane_tag.find(key);
            if (it != b.lane_tag.end() && it->second == tag) {
                merged.lane_tag.emplace(key, tag);
            }
        }
        return merged;
    }
};
} // Anonymous namespace

// The value of an inline integer constant operand, if it is one.
static std::optional<u32> InlineInt(const InstOperand& op) {
    if (op.field == OperandField::ConstZero) {
        return 0;
    }
    if (op.field == OperandField::SignedConstIntPos) {
        return op.code - static_cast<u32>(OperandField::ConstZero);
    }
    return std::nullopt;
}

static void ExecTransfer(const GcnInst& inst, u32 index, ExecState& state) {
    const bool full_before = state.full;
    const auto is_full_copy = [&](const InstOperand& op) {
        return (op.field == OperandField::ScalarGPR || op.field == OperandField::VccLo) &&
               state.HoldsFullCopy(op.code);
    };
    if (WritesExec(inst)) {
        // Restoring a copy of the full mask, s_mov_b64 exec, -1 and the s_or_b64 exec, exec, copy
        // that ends an if/else make EXEC full again; any other write can leave lanes off.
        bool full = false;
        if (inst.opcode == Opcode::S_MOV_B64) {
            full =
                is_full_copy(inst.src[0]) ||
                (inst.src[0].field == OperandField::SignedConstIntNeg && inst.src[0].code == 193);
        } else if (inst.opcode == Opcode::S_OR_B64) {
            full = is_full_copy(inst.src[0]) || is_full_copy(inst.src[1]);
        } else if (inst.opcode == Opcode::S_ORN2_SAVEEXEC_B64) {
            // EXEC = src | ~EXEC: with src = EXEC every lane is on.
            full = inst.src[0].field == OperandField::ExecLo || is_full_copy(inst.src[0]);
        } else if (inst.opcode == Opcode::S_AND_B64) {
            // Ending whole quad mode: a full EXEC and a copy of the full mask stay full.
            const auto is_exec = [](const InstOperand& op) {
                return op.field == OperandField::ExecLo;
            };
            full = full_before && ((is_exec(inst.src[0]) && is_full_copy(inst.src[1])) ||
                                   (is_exec(inst.src[1]) && is_full_copy(inst.src[0])));
        }
        state.full = full;
    }

    // A half of a copy moved through a VGPR lane: v_writelane / v_readlane with a constant lane.
    if (inst.opcode == Opcode::V_READLANE_B32 && inst.dst[0].field == OperandField::ScalarGPR &&
        inst.dst[0].code < 128) {
        const auto lane = InlineInt(inst.src[1]);
        u32 tag = 0;
        if (lane && inst.src[0].field == OperandField::VectorGPR) {
            const auto it = state.lane_tag.find(inst.src[0].code * 64 + *lane);
            tag = it != state.lane_tag.end() ? it->second : 0;
        }
        state.sgpr_tag[inst.dst[0].code] = tag;
        return;
    }
    if (inst.opcode == Opcode::V_WRITELANE_B32 && inst.dst[0].field == OperandField::VectorGPR) {
        const auto lane = InlineInt(inst.src[1]);
        if (!lane) {
            std::erase_if(state.lane_tag,
                          [&](const auto& entry) { return entry.first / 64 == inst.dst[0].code; });
            return;
        }
        const u32 key = inst.dst[0].code * 64 + *lane;
        const bool from_sgpr = inst.src[0].field == OperandField::ScalarGPR ||
                               inst.src[0].field == OperandField::VccLo ||
                               inst.src[0].field == OperandField::VccHi;
        const u32 tag = from_sgpr && inst.src[0].code < 128 ? state.sgpr_tag[inst.src[0].code] : 0;
        if (tag != 0) {
            state.lane_tag[key] = tag;
        } else {
            state.lane_tag.erase(key);
        }
        return;
    }

    // Registers written here no longer hold a half of a copy. Every destination slot is checked:
    // an implicit VCC write is stored in dst[1] without being counted in dst_count.
    for (const InstOperand& dst : inst.dst) {
        if (dst.field == OperandField::VccLo || dst.field == OperandField::VccHi) {
            state.sgpr_tag[static_cast<u32>(OperandField::VccLo)] = 0;
            state.sgpr_tag[static_cast<u32>(OperandField::VccHi)] = 0;
            continue;
        }
        if (dst.field == OperandField::VectorGPR) {
            // Vector memory writes up to four VGPRs (dwordx4, four image components, read2_b64).
            u32 vwidth = 1;
            if (inst.category == InstCategory::VectorMemory ||
                inst.category == InstCategory::DataShare) {
                vwidth = 4;
            } else if (dst.type == ScalarType::Uint64 || dst.type == ScalarType::Sint64 ||
                       dst.type == ScalarType::Float64) {
                vwidth = 2;
            }
            std::erase_if(state.lane_tag, [&](const auto& entry) {
                return entry.first / 64 >= dst.code && entry.first / 64 < dst.code + vwidth;
            });
            continue;
        }
        if (dst.field != OperandField::ScalarGPR) {
            continue;
        }
        u32 width = 2;
        if (inst.category == InstCategory::ScalarMemory) {
            width = ScalarLoadWidth(inst.opcode);
        } else if (dst.type == ScalarType::Uint32 || dst.type == ScalarType::Sint32 ||
                   dst.type == ScalarType::Float32 || dst.type == ScalarType::Uint16 ||
                   dst.type == ScalarType::Sint16 || dst.type == ScalarType::Float16) {
            width = 1;
        }
        for (u32 reg = dst.code; reg < std::min<u32>(dst.code + width, 128); ++reg) {
            state.sgpr_tag[reg] = 0;
        }
    }
    // A save of the full mask: saveexec stores the old EXEC, as does s_mov_b64 sdst, exec.
    if (full_before && inst.dst_count > 0 &&
        (inst.dst[0].field == OperandField::ScalarGPR ||
         inst.dst[0].field == OperandField::VccLo) &&
        inst.dst[0].code + 1 < 128 &&
        (IsSaveExec(inst) ||
         (inst.opcode == Opcode::S_MOV_B64 && inst.src[0].field == OperandField::ExecLo))) {
        state.sgpr_tag[inst.dst[0].code] = (index + 1) << 1;
        state.sgpr_tag[inst.dst[0].code + 1] = ((index + 1) << 1) | 1;
    }
}

void CFG::SplitDivergenceScopes() {
    // Which instructions can run with lanes off in EXEC: forward over the CFG from the entry,
    // where EXEC is full; paths merge to "partial" when any of them is.
    std::unordered_map<const Block*, ExecState> block_in;
    block_in[&*blocks.begin()] = ExecState{.full = true};
    for (bool changed = true; changed;) {
        changed = false;
        for (const Block& blk : blocks) {
            const auto it = block_in.find(&blk);
            if (it == block_in.end()) {
                continue;
            }
            ExecState state = it->second;
            for (u32 i = blk.begin_index; i <= blk.end_index; ++i) {
                ExecTransfer(inst_list[i], i, state);
            }
            for (const Block* succ : {blk.branch_true, blk.branch_false}) {
                if (!succ) {
                    continue;
                }
                // A per-lane EXEC branch sends the lanes with EXEC set one way: EXEC is set in
                // every lane on that edge, and clear in every lane on the other.
                ExecState edge = state;
                if (blk.cond == IR::Condition::Execz || blk.cond == IR::Condition::Execnz) {
                    const bool exec_set_edge =
                        (blk.cond == IR::Condition::Execnz) == (succ == blk.branch_true);
                    edge.full = exec_set_edge;
                }
                const auto [succ_it, inserted] = block_in.try_emplace(succ, edge);
                if (inserted) {
                    changed = true;
                    continue;
                }
                const ExecState merged = ExecState::Merge(succ_it->second, edge);
                if (merged != succ_it->second) {
                    succ_it->second = merged;
                    changed = true;
                }
            }
        }
    }
    std::vector<bool> exec_partial(inst_list.size(), false);
    for (const Block& blk : blocks) {
        const auto it = block_in.find(&blk);
        if (it == block_in.end()) {
            continue;
        }
        ExecState state = it->second;
        for (u32 i = blk.begin_index; i <= blk.end_index; ++i) {
            exec_partial[i] = !state.full;
            ExecTransfer(inst_list[i], i, state);
        }
    }
    const auto is_masked = [&](u32 i) { return exec_partial[i] && !IgnoresExecMask(inst_list[i]); };

    // EXEC masks the vector instructions only. Each run of them that can execute with lanes off
    // goes into its own block, which only the lanes with EXEC set enter; scalar instructions and
    // branches stay outside and run in every lane.
    for (auto blk = blocks.begin(); blk != blocks.end(); blk++) {
        if (blk->is_exec_scope) {
            continue;
        }
        u32 first = blk->begin_index;
        while (first <= blk->end_index && !is_masked(first)) {
            ++first;
        }
        if (first > blk->end_index) {
            continue;
        }
        u32 last = first;
        while (last < blk->end_index &&
               (is_masked(last + 1) || IsExecNeutral(inst_list[last + 1]))) {
            ++last;
        }
        while (IsExecNeutral(inst_list[last])) {
            --last;
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
