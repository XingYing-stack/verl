# -*- coding: utf-8 -*-
"""
Precheck: Attention Rollout heatmap (R) for a given conversation trajectory.

Usage (example):
  python rollout_qwen.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --alpha 0.9 \
    --out_prefix ./rollout_demo

You must provide `conversations` in the code below or load from a JSON file.
"""

import argparse
import json
import math
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
from typing import List, Tuple, Dict


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


import json
from uuid import uuid4
from typing import List, Dict, Any

# monkey_patch_qwen_rope.py
import types, threading
import torch

# 1) 兼容导入（Qwen3 优先，退化到 Qwen2）
try:
    import transformers.models.qwen3.modeling_qwen3 as qwen_mod
except Exception:
    import transformers.models.qwen2.modeling_qwen2 as qwen_mod

# 原函数备份
_ORIG_APPLY_ROPE = qwen_mod.apply_rotary_pos_emb

# 线程本地状态：当前 attention 的 layer_idx（由轻量 forward 包装写入）
_tls = threading.local()
_tls.layer_idx = None

class QKCache:
    def __init__(self, keep_all_layers=False, target_last_layer=None):
        self.q = []   # list of (layer_idx, tensor[B,H,T,Dh])
        self.k = []
        self.keep_all_layers = keep_all_layers
        self.target_last_layer = target_last_layer  # 若给出，只保留等于该层号的
    def _accept(self, layer_idx):
        if self.keep_all_layers:
            return True
        if self.target_last_layer is None:
            return True
        return layer_idx == self.target_last_layer

def _to_bhtd(x):
    # 统一到 [B,H,T,Dh]
    if x.dim() == 4:
        # 常见是 [B, H, T, Dh] 或 [B, T, H, Dh]
        if x.shape[1] <= 256 and x.shape[2] >= 1 and x.shape[1] != x.shape[2]:
            # 猜测 [B,H,T,Dh]
            return x
        # 认为 [B,T,H,Dh]
        return x.permute(0, 2, 1, 3).contiguous()
    elif x.dim() == 3:
        # [B, T, H*Dh]
        B,T,D = x.shape
        # 无法知道 H，只能不动；HF 调用 apply_rope 通常已是 4D
        raise RuntimeError(f"Unexpected 3D shape at apply_rope: {x.shape}")
    raise RuntimeError(f"Unexpected rank at apply_rope: {x.shape}")

def make_patcher(cache: QKCache):
    """
    返回两个可调用对象：patch() / restore()
    - patch(): 猴补 apply_rotary_pos_emb + 轻量 forward 包装（仅写 layer_idx）
    - restore(): 复原
    """
    originals = {"apply": _ORIG_APPLY_ROPE}
    forward_wrapped = []

    def wrapped_apply(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # 调原实现
        q_rope, k_rope = originals["apply"](q, k, cos, sin, position_ids, unsqueeze_dim)
        try:
            layer_idx = getattr(_tls, "layer_idx", None)
            if cache._accept(layer_idx):
                cache.q.append((layer_idx, _to_bhtd(q_rope.detach())))
                cache.k.append((layer_idx, _to_bhtd(k_rope.detach())))
        except Exception:
            # 静默失败，不影响正常前向
            pass
        return q_rope, k_rope

    def wrap_forward(mod_cls):
        """
        只包一层最小逻辑：在进入 attention.forward 前把 layer_idx 写到 thread-local，
        退出时清空。签名完全不改，兼容未来版本。
        """
        orig_fwd = mod_cls.forward
        def fwd(self, *args, **kwargs):
            prev = getattr(_tls, "layer_idx", None)
            try:
                # 大多实现会把 layer_idx 挂在 attention 或 decoder layer 上
                _tls.layer_idx = getattr(self, "layer_idx", getattr(getattr(self, "config", None), "layer_idx", None))
                return orig_fwd(self, *args, **kwargs)
            finally:
                _tls.layer_idx = prev
        mod_cls.forward = fwd
        forward_wrapped.append((mod_cls, orig_fwd))

    def patch():
        # 1) 猴补 apply_rotary_pos_emb（模块级自由函数）
        qwen_mod.apply_rotary_pos_emb = wrapped_apply

        # 2) 找到所有 Attention 类，把 forward 轻量包一下（只写 layer_idx）
        # Qwen3: Qwen3Attention；Qwen2: Qwen2Attention
        # 有些分支会把名字改掉；稳妥些：挑选带有 q_proj/k_proj 且有 head 维度配置的模块
        candidates = []
        for name, obj in qwen_mod.__dict__.items():
            if not isinstance(obj, type):
                continue
            nm = name.lower()
            if ("attention" in nm) and hasattr(obj, "forward"):
                candidates.append(obj)
        for cls in candidates:
            wrap_forward(cls)

    def restore():
        # 还原自由函数
        qwen_mod.apply_rotary_pos_emb = originals["apply"]
        # 还原 forward
        for cls, orig in forward_wrapped:
            cls.forward = orig
        forward_wrapped.clear()

    return patch, restore



def to_openai_messages(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a conversation list into OpenAI Chat Completions 'messages' format.

    Rules:
    - Keep only standard keys for messages (role, content).
    - Normalize assistant tool_calls to OpenAI format:
        assistant: { role, content, tool_calls: [{id, type:'function', function:{name, arguments(str)}}] }
    - Normalize tool messages to:
        { role: 'tool', tool_call_id: <id>, name: <function-name>, content: <str> }
    - Drop non-standard keys (e.g., 'thought', 'server_id', etc.).
    - JSON-serialize non-string 'content'.
    - Try to infer/clean function name for tool messages (e.g., 'x.y.z' -> 'z').
    - If a tool message lacks tool_call_id, attach it to the most recent assistant tool_call (best-effort).
    """
    messages: List[Dict[str, Any]] = []
    # Map tool_call_id -> function name, from preceding assistant tool_calls
    id_to_funcname: Dict[str, str] = {}

    def _as_str(x: Any) -> str:
        return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)

    def _clean_func_name(name: str) -> str:
        if not isinstance(name, str) or not name:
            return name
        # Prefer last path segment after dot
        return name.split(".")[-1]

    # Keep a stack of recent assistant tool_call ids for fallback pairing
    recent_tool_call_ids: List[str] = []

    for item in conversations:
        role = item.get("role")
        if role in ("user", "assistant", "system"):
            msg = {
                "role": role,
                "content": _as_str(item.get("content", "")).strip()
            }

            # Normalize assistant tool calls if present
            tool_calls = item.get("tool_calls")
            if role == "assistant" and tool_calls:
                normalized_calls = []
                recent_tool_call_ids.clear()
                for tc in tool_calls:
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name")
                    args = fn.get("arguments")

                    # arguments must be a string
                    args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)

                    # ensure an id exists
                    tcid = tc.get("id") or str(uuid4())

                    id_to_funcname[tcid] = name or id_to_funcname.get(tcid, "")
                    recent_tool_call_ids.append(tcid)

                    normalized_calls.append({
                        "id": tcid,
                        "type": "function",
                        "function": {
                            "name": name or "",
                            "arguments": args_str
                        }
                    })
                msg["tool_calls"] = normalized_calls

            if role == 'assistant' and item.get('thought'):
                msg['reasoning_content'] = item.get('thought')
            messages.append(msg)

        elif role == "tool":
            # Standardize tool message
            content = _as_str(item.get("content", ""))
            tool_call_id = item.get("tool_call_id")
            name = item.get("name")

            # Try to clean name like "search-fusion-mcp.search" -> "search"
            name_clean = _clean_func_name(name) if name else None

            # If tool_call_id is missing, best-effort: attach to last assistant tool_call
            if not tool_call_id:
                # Find last assistant message with tool_calls
                for prev in reversed(messages):
                    if prev.get("role") == "assistant" and prev.get("tool_calls"):
                        # attach to the last tool_call in that assistant message
                        tool_call_id = prev["tool_calls"][-1]["id"]
                        if not name_clean:
                            # Try to get function name from mapping
                            name_clean = id_to_funcname.get(tool_call_id, name_clean)
                        break

            # If still missing, create a synthetic id (not ideal, but keeps format valid)
            if not tool_call_id:
                tool_call_id = str(uuid4())

            if not name_clean and tool_call_id in id_to_funcname:
                name_clean = id_to_funcname[tool_call_id]

            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content
            }
            if name_clean:
                tool_msg["name"] = name_clean

            messages.append(tool_msg)

        else:
            # Unknown/unsupported role -> skip
            continue

    return messages


def build_chat_prompt(tokenizer, conversations: List[Dict[str, str]]) -> str:
    """
    conversations: [{"role": "system/user/assistant", "content": "..."} , ...]
    Prefer tokenizer.apply_chat_template if available.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=False
        )
    # Fallback: naive concatenation
    role_tag = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines = []
    for turn in conversations:
        r = turn["role"].lower()
        lines.append(f"[{role_tag.get(r, r).title()}]: {turn['content']}")
    return "\n".join(lines)

import torch
import torch.nn.functional as F

import torch

@torch.no_grad()
def accumulate_last_layer_stats_noTT_turns(
    q_last: torch.Tensor,  # [1,Hq,T,Dh]
    k_last: torch.Tensor,  # [1,Hkv,T,Dh]
    turns: Dict[str, List[List[Tuple[int,int]]]],
    g_abs: torch.Tensor | None = None,  # [T]
    alpha: float = 1.0,
    chunk_cols: int = 1024,
    causal: bool = True,
):
    """
    与 credit_from_attn_matrix_turns 同公式，但用 QK 流式重算最后一层注意力的“行”，仅累计所需的列和。
    - 查询行（rows）：始终取 assistant 的行集合
    - 被查询列（cols）：这一轮的 assistant∪tool
    返回：s: [K], A: [K,K]（上三角 j<k）
    """
    device = q_last.device
    q = q_last[0]  # [Hq,T,Dh]
    k = k_last[0]  # [Hkv,T,Dh]
    Hq, T, Dh = q.shape
    Hkv = k.shape[0]
    scale = (Dh ** -0.5)

    # GQA：把 K/V 头复制到 Q 头数
    assert Hq % Hkv == 0
    group = Hq // Hkv
    k_for_q = k.repeat_interleave(group, dim=0)  # [Hq,T,Dh]

    if g_abs is None:
        g_abs = torch.ones(T, device=device, dtype=torch.float32)
    else:
        g_abs = g_abs.to(device=device, dtype=torch.float32)
        assert g_abs.shape[0] == T

    asst_turns = turns["assistant_turns"][0]
    tool_turns = turns.get("tool_turns", [[]])[0] if turns.get("tool_turns", None) else []
    K = len(asst_turns)
    assert K >= 1

    # 构建列掩码 C_k、行掩码 A_k
    mask_A = [torch.zeros(T, dtype=torch.bool, device=device) for _ in range(K)]
    mask_C = [torch.zeros(T, dtype=torch.bool, device=device) for _ in range(K)]
    for k_idx in range(K):
        s_k, e_k = asst_turns[k_idx]
        if e_k > s_k:
            mask_A[k_idx][s_k:e_k] = True
            mask_C[k_idx][s_k:e_k] = True
        if k_idx < len(tool_turns):
            st, et = tool_turns[k_idx]
            if et > st:
                mask_C[k_idx][st:et] = True

    # 预先把每个块内各轮的列索引存起来（相对索引，避免重复构造）
    n_blocks = (T + chunk_cols - 1) // chunk_cols
    cols_rel = [[None]*n_blocks for _ in range(K)]
    for b in range(n_blocks):
        c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
        cols_full = torch.arange(c0, c1, device=device)
        for k_idx in range(K):
            sel = mask_C[k_idx][c0:c1]  # [C] bool
            rel = cols_full[sel] - c0
            cols_rel[k_idx][b] = rel

    # ---- s_attn：答案行 = 最后一轮 assistant 的行 ----
    s_attn = torch.zeros(K, device=device, dtype=torch.float32)
    s_I    = torch.zeros(K, device=device, dtype=torch.float32) if alpha < 1.0 else None

    sL, eL = asst_turns[-1]
    rows_ans = torch.arange(sL, eL, device=device)
    Ta = int(rows_ans.numel())
    assert Ta > 0

    # 两遍流式 softmax（仅对答案行）
    q_ans = q[:, rows_ans, :]  # [Hq,Ta,Dh]
    row_max = torch.full((Hq, Ta), -float("inf"), device=device)
    for b in range(n_blocks):
        c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
        k_blk = k_for_q[:, c0:c1, :]                                  # [Hq,C,Dh]
        scores = torch.einsum("hAd,hCd->hAC", q_ans, k_blk) * scale   # [Hq,Ta,C]
        if causal:
            cols = torch.arange(c0, c1, device=device).view(1,1,-1)
            rpos = rows_ans.view(1,-1,1)
            scores = scores.masked_fill(cols > rpos, float("-inf"))
        row_max = torch.maximum(row_max, scores.max(dim=-1).values)

    row_sumexp = torch.zeros(Hq, Ta, device=device)
    for b in range(n_blocks):
        c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
        k_blk = k_for_q[:, c0:c1, :]
        scores = torch.einsum("hAd,hCd->hAC", q_ans, k_blk) * scale
        if causal:
            cols = torch.arange(c0, c1, device=device).view(1,1,-1)
            rpos = rows_ans.view(1,-1,1)
            scores = scores.masked_fill(cols > rpos, float("-inf"))
        exp_blk = torch.exp(scores - row_max.unsqueeze(-1))           # [Hq,Ta,C]
        row_sumexp += exp_blk.sum(dim=-1)

    inv_denom = 1.0 / (row_sumexp + 1e-12)

    # 第三遍：按块把 Σ_{t∈A_last} A[t,·] 累计出来（先 head-mean 再行求和）
    sum_last_cols = torch.zeros(T, device=device, dtype=torch.float32)
    for b in range(n_blocks):
        c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
        C = c1 - c0
        k_blk = k_for_q[:, c0:c1, :]
        scores = torch.einsum("hAd,hCd->hAC", q_ans, k_blk) * scale
        if causal:
            cols = torch.arange(c0, c1, device=device).view(1,1,-1)
            rpos = rows_ans.view(1,-1,1)
            scores = scores.masked_fill(cols > rpos, float("-inf"))
        exp_blk = torch.exp(scores - row_max.unsqueeze(-1))
        attn_blk = exp_blk * inv_denom.unsqueeze(-1)                  # [Hq,Ta,C]
        A_blk_mean = attn_blk.mean(dim=0)                             # [Ta,C]
        sum_last_cols[c0:c1] += A_blk_mean.sum(dim=0)                 # [C] 累到全局

    for k_idx in range(K):
        for b in range(n_blocks):
            rel = cols_rel[k_idx][b]
            if rel.numel() == 0:
                continue
            c0 = b*chunk_cols
            sum_cols = sum_last_cols[c0:c0+rel.numel()*0 + (chunk_cols if c0+chunk_cols<=T else T-c0)]  # 不用这个值，避免误会
            # 直接索引相对列
            s_attn[k_idx] += (sum_last_cols[c0:c0+chunk_cols].index_select(0, rel) *
                              g_abs[c0:c0+chunk_cols].index_select(0, rel)).sum()

    if alpha < 1.0:
        inter_last = mask_A[-1]  # [T]
        for k_idx in range(K):
            inter = inter_last & mask_C[k_idx]
            if inter.any():
                s_I[k_idx] = g_abs[inter].sum()

    # ---- A_attn：对子回合 k 的 assistant 行求和；父列用 C_j ----
    A_attn = torch.zeros(K, K, device=device, dtype=torch.float32)
    A_I    = torch.zeros(K, K, device=device, dtype=torch.float32) if alpha < 1.0 else None

    for k_idx in range(K):
        rows_k = mask_A[k_idx]
        Tk = int(rows_k.sum().item())
        if Tk == 0:
            continue
        q_k = q[:, rows_k, :]                                        # [Hq,Tk,Dh]

        # 两遍：行最大值、分母
        row_max = torch.full((Hq, Tk), -float("inf"), device=device)
        rows_idx = torch.nonzero(rows_k, as_tuple=False).squeeze(-1)
        for b in range(n_blocks):
            c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
            k_blk = k_for_q[:, c0:c1, :]
            scores = torch.einsum("hkd,hcd->hkc", q_k, k_blk) * scale
            if causal:
                cols = torch.arange(c0, c1, device=device).view(1,1,-1)
                rpos = rows_idx.view(1,-1,1)
                scores = scores.masked_fill(cols > rpos, float("-inf"))
            row_max = torch.maximum(row_max, scores.max(dim=-1).values)

        row_sumexp = torch.zeros(Hq, Tk, device=device)
        for b in range(n_blocks):
            c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
            k_blk = k_for_q[:, c0:c1, :]
            scores = torch.einsum("hkd,hcd->hkc", q_k, k_blk) * scale
            if causal:
                cols = torch.arange(c0, c1, device=device).view(1,1,-1)
                rpos = rows_idx.view(1,-1,1)
                scores = scores.masked_fill(cols > rpos, float("-inf"))
            exp_blk = torch.exp(scores - row_max.unsqueeze(-1))
            row_sumexp += exp_blk.sum(dim=-1)

        inv_denom = 1.0 / (row_sumexp + 1e-12)

        # 第三遍：写 Σ_{i'∈A_k} A[i',·]
        sum_child_cols = torch.zeros(T, device=device, dtype=torch.float32)
        for b in range(n_blocks):
            c0, c1 = b*chunk_cols, min(T, (b+1)*chunk_cols)
            k_blk = k_for_q[:, c0:c1, :]
            scores = torch.einsum("hkd,hcd->hkc", q_k, k_blk) * scale
            if causal:
                cols = torch.arange(c0, c1, device=device).view(1,1,-1)
                rpos = rows_idx.view(1,-1,1)
                scores = scores.masked_fill(cols > rpos, float("-inf"))
            exp_blk = torch.exp(scores - row_max.unsqueeze(-1))
            attn_blk = exp_blk * inv_denom.unsqueeze(-1)             # [Hq,Tk,C]
            A_blk_mean = attn_blk.mean(dim=0)                        # [Tk,C]
            sum_child_cols[c0:c1] += A_blk_mean.sum(dim=0)

        # 聚合到 A_attn[:, k_idx]（父列用 C_j）
        for j_idx in range(k_idx):
            val = 0.0
            for b in range(n_blocks):
                rel = cols_rel[j_idx][b]
                if rel.numel() == 0:
                    continue
                c0 = b*chunk_cols
                val += (sum_child_cols[c0:c0+chunk_cols].index_select(0, rel) *
                        g_abs[c0:c0+chunk_cols].index_select(0, rel)).sum()
            A_attn[j_idx, k_idx] = val

            if alpha < 1.0:
                inter = mask_A[k_idx] & mask_C[j_idx]
                if inter.any():
                    A_I[j_idx, k_idx] = g_abs[inter].sum()

    s_final = (alpha * s_attn) if alpha == 1.0 else (alpha * s_attn + (1.0 - alpha) * s_I)
    A_final = (alpha * A_attn) if alpha == 1.0 else (alpha * A_attn + (1.0 - alpha) * A_I)

    return s_final, A_final

@torch.no_grad()
def credit_from_attn_matrix_turns(
    attn_last: torch.Tensor,                 # [B, H, T, T] 最后一层各 head 注意力（已 softmax & causal）
    turns_list: List[Dict[str, List[List[Tuple[int,int]]]]],  # len=B；每个元素：{'assistant_turns': [[(s,e),...]], 'tool_turns': [[(s,e),...]]}
    g_abs_list: List[torch.Tensor] | None = None,  # len=B；每个样本的 |g_i|, shape [T]
    alpha: float = 1.0,
):
    """
    计算：
      s[k]      = Σ_{t∈A_last} Σ_{i∈C_k} A[t,i] |g_i|
      A[j,k]    = Σ_{i'∈A_k}   Σ_{i∈C_j} A[i',i] |g_i|, 仅 j<k
    其中 A_k 是第 k 轮 assistant 的 token 区间；C_k := A_k ∪ (tool_k), 若最后一轮无 tool 则 C_k=A_k。
    残差项 I 的贡献（当 alpha<1）：
      s_I[k]    = Σ_{t∈A_last∩C_k} |g_t|
      A_I[j,k]  = Σ_{i∈A_k ∩ C_j} |g_i|
    最终：
      s = α·s_attn + (1-α)·s_I
      A = α·A_attn + (1-α)·A_I
    """
    assert attn_last.dim() == 4
    B, H, T, T2 = attn_last.shape
    assert T == T2
    device = attn_last.device

    A_mean = attn_last.mean(dim=1).to(torch.float32)  # [B,T,T]
    if g_abs_list is None:
        g_abs_list = [torch.ones(T, device=device, dtype=torch.float32) for _ in range(B)]

    s_batch, A_batch = [], []

    for b in range(B):
        A = A_mean[b]                        # [T,T]
        g_abs = g_abs_list[b].to(device=device, dtype=torch.float32)
        assert g_abs.shape[0] == T

        turns = turns_list[b]
        asst_turns = turns["assistant_turns"][0]  # List[(s,e)]
        tool_turns = turns.get("tool_turns", [[]])[0] if turns.get("tool_turns", None) else []
        K = len(asst_turns)
        assert K >= 1, "need at least one assistant turn"

        # --- 构建每轮的 mask ---
        mask_A = [torch.zeros(T, dtype=torch.bool, device=device) for _ in range(K)]  # rows: assistant
        mask_C = [torch.zeros(T, dtype=torch.bool, device=device) for _ in range(K)]  # cols: assistant ∪ tool
        for k in range(K):
            s_k, e_k = asst_turns[k]
            if e_k > s_k:
                mask_A[k][s_k:e_k] = True
                mask_C[k][s_k:e_k] = True
            # tool 第 k 轮（存在则合并到列）
            if k < len(tool_turns):
                st, et = tool_turns[k]
                if et > st:
                    mask_C[k][st:et] = True

        # --- s_attn：答案行为最后一轮的 assistant ---
        s_attn = torch.zeros(K, device=device, dtype=torch.float32)
        s_I    = torch.zeros(K, device=device, dtype=torch.float32) if alpha < 1.0 else None

        s_last, e_last = asst_turns[-1]
        assert e_last > s_last, "last assistant span empty?"
        rows_last = torch.arange(s_last, e_last, device=device)
        sum_last_cols = A.index_select(0, rows_last).sum(dim=0)   # [T] = Σ_{t∈A_last} A[t, :]
        for k in range(K):
            cols_k = mask_C[k]
            if cols_k.any():
                s_attn[k] = (sum_last_cols[cols_k] * g_abs[cols_k]).sum()
            if alpha < 1.0:
                # s_I[k] = Σ_{t∈A_last∩C_k} |g_t|
                inter = mask_A[-1] & mask_C[k]
                if inter.any():
                    s_I[k] = g_abs[inter].sum()

        # --- A_attn：对子回合 k 的 assistant 行求和；父列用 C_j ---
        A_attn = torch.zeros(K, K, device=device, dtype=torch.float32)
        A_I    = torch.zeros(K, K, device=device, dtype=torch.float32) if alpha < 1.0 else None
        for k in range(K):
            rows_k = mask_A[k]
            if not rows_k.any():
                continue
            sum_child_cols = A[rows_k, :].sum(dim=0)             # [T]
            for j in range(k):                                   # 只计 j<k
                cols_j = mask_C[j]
                if cols_j.any():
                    A_attn[j, k] = (sum_child_cols[cols_j] * g_abs[cols_j]).sum()
                if alpha < 1.0:
                    # A_I[j,k] = Σ_{i∈A_k ∩ C_j} |g_i|
                    inter = rows_k & cols_j
                    if inter.any():
                        A_I[j, k] = g_abs[inter].sum()

        s_final = (alpha * s_attn) if alpha == 1.0 else (alpha * s_attn + (1.0 - alpha) * s_I)
        A_final = (alpha * A_attn) if alpha == 1.0 else (alpha * A_attn + (1.0 - alpha) * A_I)

        s_batch.append(s_final)
        A_batch.append(A_final)

    return s_batch, A_batch, A_mean

def _pick_kth_layer(tuples, k):
    # tuples: list of (layer_idx, tensor[B,H,T,Dh])
    for layer_idx, ten in reversed(tuples):
        if layer_idx == k:
            return ten
    raise RuntimeError("未在 cache 中找到最后一层的 Q/K；请确认补丁是否生效。")

def attention_rollout(attentions: List[torch.Tensor], alpha: float = 0.9) -> torch.Tensor:
    """
    attentions: list of tensors [1, H, T, T], causal-masked & row-stochastic-ish after softmax.
    Returns:
      R: Tensor[T, T] = ∏_l ( alpha * mean_head(A_l) + (1-alpha)*I )
         By decoder causality, R should be (approximately) upper-triangular (no future->past flow).
    """
    # Sanity: batch size must be 1
    assert all(att.shape[0] == 1 for att in attentions), "Batch > 1 not supported here."

    # Head-average for each layer: (T, T)
    heads_mean = [att[0].mean(dim=0) for att in attentions]  # list of [T, T]

    T = heads_mean[0].shape[-1]
    I = torch.eye(T, device=heads_mean[0].device, dtype=heads_mean[0].dtype)

    R = I.clone()
    for A in heads_mean:
        # Clamp small negatives (shouldn't happen after softmax, but numerical safety)
        A = torch.clamp(A, min=0.0)
        # Optional row-normalization to stabilize (comment out if undesired)
        row_sum = A.sum(dim=-1, keepdim=True) + 1e-12
        A = A / row_sum

        A_tilde = alpha * A + (1.0 - alpha) * I
        R = torch.matmul(R, A_tilde)  # accumulate rollout
    return R  # [T, T]


def plot_rollout(R: torch.Tensor, tokens: List[str], out_png: str):
    """
    Simple matplotlib heatmap (no seaborn).
    Rows: source tokens (earlier), Cols: target tokens (later).
    In decoder-only, influence should respect upper-triangular structure.
    """
    R_np = R.detach().cpu().float().numpy()

    plt.figure(figsize=(8, 6))
    plt.imshow(R_np, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar()
    # Keep token labels light; for long sequences it's better to skip to avoid clutter.
    max_labels = 80
    if len(tokens) <= max_labels:
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=6)
        plt.yticks(range(len(tokens)), tokens, fontsize=6)
    else:
        plt.xticks([])
        plt.yticks([])
    plt.title("Attention Rollout R (token-to-token influence)")
    plt.xlabel("Target token index (later)")
    plt.ylabel("Source token index (earlier)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _find_all_subseq(hay: List[int], needle: List[int]) -> List[int]:
    """在 token 序列 hay 中找到子序列 needle 的所有起始下标。"""
    H, N = len(hay), len(needle)
    if N == 0 or H < N:
        return []
    hits = []
    for i in range(H - N + 1):
        if hay[i:i+N] == needle:
            hits.append(i)
    return hits

def _next_occurrence(hay: List[int], needle: List[int], start: int) -> int:
    """从 start 起查找 needle 的下一次出现的起始下标，不存在则返回 -1。"""
    H, N = len(hay), len(needle)
    for i in range(start, H - N + 1):
        if hay[i:i+N] == needle:
            return i
    return -1

@torch.no_grad()
def get_turns(
    input_ids: torch.Tensor,          # [B, T]
    attention_mask: torch.Tensor,     # [B, T]
    tokenizer
) -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    解析 ChatML 风格 (<|im_start|>role ... <|im_end|>) 批量轨迹，返回：
      - 'assistant_turns': List[ List[(s,e)] ]  —— 每个样本的 assistant 消息 token 闭开区间 [s,e)
      - 'tool_turns':      List[ List[(s,e)] ]  —— 每个样本的 <tool_response> 内容区间 [s,e)
    约定：
      * role ∈ {assistant, user}，role 之后可选一个换行 token。
      * <tool_response>...</tool_response> 仅在 user 片段内部查找。
      * 索引均是样本内的 token 位置（相对于该样本的 input_ids）。
    """
    assert input_ids.dim() == 2 and attention_mask.dim() == 2
    B, T = input_ids.shape
    device = input_ids.device

    # 预编码若干标记（不添加 special tokens）
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    im_start = enc("<|im_start|>")
    im_end   = enc("<|im_end|>")
    tok_nl   = enc("\n")  # 常见模板 role 后跟一个换行
    role_assistant = enc("assistant")
    role_user      = enc("user")
    tool_l = enc("<tool_response>")
    tool_r = enc("</tool_response>")

    all_assistant_turns: List[List[Tuple[int,int]]] = []
    all_tool_turns: List[List[Tuple[int,int]]] = []

    for b in range(B):
        # 截到有效长度
        Tb = int(attention_mask[b].sum().item())
        ids = input_ids[b, :Tb].tolist()

        assistant_turns: List[Tuple[int,int]] = []
        tool_turns: List[Tuple[int,int]] = []

        starts = _find_all_subseq(ids, im_start)
        i = 0

        def _match_role_at(pos: int, role_tok: List[int]) -> Tuple[bool, int]:
            """匹配 [role][\n]?；返回 (是否匹配, 内容起点)"""
            rlen = len(role_tok)
            if ids[pos:pos+rlen] == role_tok:
                after = pos + rlen
                if ids[after:after+len(tok_nl)] == tok_nl:
                    after += len(tok_nl)
                return True, after
            return False, -1

        while i < len(starts):
            s0 = starts[i]
            role_pos = s0 + len(im_start)

            is_assistant, content_start = _match_role_at(role_pos, role_assistant)
            is_user, cs_user = (False, -1)
            if not is_assistant:
                is_user, cs_user = _match_role_at(role_pos, role_user)

            # 找到对应的 <|im_end|>
            search_from = content_start if is_assistant else (cs_user if is_user else role_pos)
            e0 = _next_occurrence(ids, im_end, start=search_from)
            if e0 == -1:
                # 残缺片段，终止
                break

            if is_assistant:
                # assistant 内容 [content_start, e0)
                if content_start < e0:
                    assistant_turns.append((content_start, e0))

            elif is_user:
                content_start = cs_user
                # 仅在该 user 片段内部查找 tool_response
                seg = ids[content_start:e0]
                tl_rel = _find_all_subseq(seg, tool_l)
                tr_rel = _find_all_subseq(seg, tool_r)
                tl_rel.sort()
                tr_rel.sort()
                si = sj = 0
                while si < len(tl_rel) and sj < len(tr_rel):
                    tl = tl_rel[si]
                    tr = tr_rel[sj]
                    if tr <= tl:
                        sj += 1
                        continue
                    abs_l = content_start + tl + len(tool_l)
                    abs_r = content_start + tr
                    if abs_l < abs_r:
                        tool_turns.append((abs_l, abs_r))
                    si += 1
                    sj += 1

            # 下一个 im_start
            i += 1

        all_assistant_turns.append(assistant_turns)
        all_tool_turns.append(tool_turns)

    return {
        "assistant_turns": all_assistant_turns,  # List[ List[(s,e)] ]，长度 = B
        "tool_turns": all_tool_turns,            # List[ List[(s,e)] ]，长度 = B
    }



@torch.no_grad()
def get_counterfactual_input_ids(input_ids, attention_mask, turns_list, tokenizer):
    device = input_ids.device
    conterfactual_input_ids = input_ids.clone()
    for index in range(input_ids.shape[0]):
        for interval_list in turns_list.values():
            for start, end in interval_list[index]:
                conterfactual_input_ids[index][start:end] = torch.LongTensor(tokenizer.encode('\t', add_special_tokens=False) * (end-start)).to(device)
    return {'input_ids':conterfactual_input_ids,
            'attention_mask':attention_mask}





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/share_data/data4/workhome/fanshengda/Qwen/Qwen3-8B",
                        help="Replace with your Qwen3 HF id if needed.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Residual interpolation weight.")
    parser.add_argument("--out_prefix", type=str, default="./rollout_demo")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--trust_remote_code", action="store_true", help="Some Qwen variants require this.")
    args = parser.parse_args()

    # -------- conversations: put your trajectory here if not loading from file --------
    # Example demo; replace with your real trajectory
    conversations = [
  {
    "role": "user",
    "content": "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists?"
  },
  {
    "role": "assistant",
    "content": "\n\n",
    "thought": "\nFirst, I need to find information about the Malko Competition recipients from the 20th century, specifically after 1977. The recipient's nationality should be a country that no longer exists.\n\nI should start by searching for information about the Malko Competition. I'll use the search tool to look it up.\n",
    "tool_calls": [
      {
        "id": "call_3fd344889bd44e20918d42d5",
        "function": {
          "name": "search",
          "arguments": "{\"query\": \"Malko Competition recipients 20th century\"}"
        },
        "server_id": "search-fusion-mcp"
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_3fd344889bd44e20918d42d5",
    "name": "search-fusion-mcp.search",
    "content": {
      "query": "Malko Competition recipients 20th century",
      "engine": "auto",
      "time_ms": 243,
      "num_results": 10,
      "results": [
        {
          "title": "Malko Competition - Wikipedia",
          "link": "https://en.wikipedia.org/wiki/Malko_Competition",
          "snippet": "The Malko Competition is an international competition for young conductors. It is held every three years by the Danish Radio Symphony Orchestra, ...",
          "source": "google",
          "metadata": {
            "display_link": "en.wikipedia.org"
          }
        },
        {
          "title": "All Winners | Malko Competition",
          "link": "https://malkocompetition.dk/winners/all",
          "snippet": "... 20, 5-3 0999 København C Denmark mrrb@dr.dk. Project Manager: Marie Rørbech. Newsletter. Subscribe to the Malko Competition newsletter! Facebook. Cookies. We ...",
          "source": "google",
          "metadata": {
            "display_link": "malkocompetition.dk"
          }
        },
        {
          "title": "langchain_mistral_test.ipynb · felixmortas ...",
          "link": "https://huggingface.co/spaces/felixmortas/Hf_Aagent_Course_Final_Assignment/blob/main/langchain_mistral_test.ipynb",
          "snippet": "Jul 4, 2025 ... ... Malko Competition recipient from the 20th Century (after 1977) whose ... **Identify the Malko Competition recipients from the 20th Century ...",
          "source": "google",
          "metadata": {
            "display_link": "huggingface.co"
          }
        },
        {
          "title": "International Conducting Competition Rotterdam names three ...",
          "link": "https://askonasholt.com/artist/miguel-sepulveda/press/international-conducting-competition-rotterdam-names-three-askonas-holt-fellows-as-designate-winners",
          "snippet": "He was previously a semi-finalist in the 2024 Malko Conducting Competition. ... Following last week's semi-finals, the six Designate Winners will next year ...",
          "source": "google",
          "metadata": {
            "display_link": "askonasholt.com"
          }
        },
        {
          "title": "ninja-x/deepresearch · Datasets at Hugging Face",
          "link": "https://huggingface.co/datasets/ninja-x/deepresearch/viewer",
          "snippet": "What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record is a country that no longer exists ...",
          "source": "google",
          "metadata": {
            "display_link": "huggingface.co"
          }
        },
        {
          "title": "Schedule 2022-24 | Malko Competition",
          "link": "https://www.malkocompetition.dk/academy/schedule",
          "snippet": "The schedule shows the date and time when students of the Academy are required to be present in DR Byen, Emil Holms Kanal 20, Copenhagen, Denmark. 2022.",
          "source": "google",
          "metadata": {
            "display_link": "www.malkocompetition.dk"
          }
        },
        {
          "title": "andrewrreed/agents-benchmark-eval-results · Datasets at Hugging ...",
          "link": "https://huggingface.co/datasets/andrewrreed/agents-benchmark-eval-results/viewer/default/train",
          "snippet": "[{'tool': 'Search', 'tool_input': 'only malko competition recipient from the 20th century after 1977 nationality on record is a country that no longer exists ...",
          "source": "google",
          "metadata": {
            "display_link": "huggingface.co"
          }
        },
        {
          "title": "Schedule | Malko Competition",
          "link": "https://www.malkocompetition.dk/competition/schedule",
          "snippet": "Round 3 Thursday, APRIL 18 10:00-13:00 + 13:50-16:20 The 3rd round is divided into two parts: – Morning: Debussy: La Mer, Ravel: La Valse, Bartok: Concert for ...",
          "source": "google",
          "metadata": {
            "display_link": "www.malkocompetition.dk"
          }
        },
        {
          "title": "Update agent.py · mralamdari/Final_Assignment_Agent at 16ea15b",
          "link": "https://huggingface.co/spaces/mralamdari/Final_Assignment_Agent/commit/16ea15b280a2d87eb4f2438b55c53bca088b6354",
          "snippet": "May 31, 2025 ... - question8 = \"What is the first name of the only Malko Competition recipient from the 20th Century (after 1977) whose nationality on record ...",
          "source": "google",
          "metadata": {
            "display_link": "huggingface.co"
          }
        },
        {
          "title": "Daniel Gyuseo Lee | Malko Competition",
          "link": "https://www.malkocompetition.dk/conductor/2021/daniel-gyuseo-lee",
          "snippet": "He made his conducting debut at the Seoul Arts Centre at the age of 20 and conducted the Incheon Philharmonic at 22. He is the recipient ... C Denmark mrrb@dr.dk",
          "source": "google",
          "metadata": {
            "display_link": "www.malkocompetition.dk"
          }
        }
      ],
      "timestamp": "2025-09-03T10:00:17.675500",
      "success": True
    },
    "server_id": "default"
  },
  {
    "role": "assistant",
    "content": "\n\n",
    "thought": "\nFirst, I need to find the Malko Competition recipients from the 20th century after 1977. The competition is held every three years, so I should look for winners between 1978 and 2000.\n\nThe search results point to the Malko Competition website, specifically the \"All Winners\" page. I'll try to fetch that page to get a list of winners.\n",
    "tool_calls": [
      {
        "id": "call_e4ceab286f6446afa670a338",
        "function": {
          "name": "fetch_url",
          "arguments": "{\"url\": \"https://malkocompetition.dk/winners/all\"}"
        },
        "server_id": "search-fusion-mcp"
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_e4ceab286f6446afa670a338",
    "name": "search-fusion-mcp.fetch_url",
    "content": {
      "url": "https://malkocompetition.dk/winners/all",
      "method": "jina",
      "format": "markdown",
      "success": True,
      "content": "{\"code\":200,\"status\":20000,\"data\":{\"links\":[[\"Skip to main content\",\"https://malkocompetition.dk/winners/all#main-content\"],[\"Malko Competition for Young Conductors\",\"https://malkocompetition.dk/\"],[\"\",\"javascript:void(0)\"],[\"Home\",\"https://malkocompetition.dk/\"],[\"Participants\",\"https://malkocompetition.dk/conductor/2024\"],[\"Schedule\",\"https://malkocompetition.dk/competition/schedule\"],[\"Performances\",\"https://malkocompetition.dk/performances/2024\"],[\"Articles\",\"https://malkocompetition.dk/article/2024\"],[\"Videos\",\"https://malkocompetition.dk/video/2024\"],[\"Photos\",\"https://malkocompetition.dk/photos/2024\"],[\"Repertoire\",\"https://malkocompetition.dk/competition/repertoire\"],[\"Prizes\",\"https://malkocompetition.dk/competition/prizes\"],[\"Jury\",\"https://malkocompetition.dk/competition/jury\"],[\"Jury Rules\",\"https://malkocompetition.dk/competition/rules\"],[\"DANISH NATIONAL SYMPHONY ORCHESTRA\",\"https://malkocompetition.dk/about/danish-national-symphony-orchestra\"],[\"Fabio Luisi\",\"https://malkocompetition.dk/about/fabio-luisi\"],[\"The competition\",\"https://malkocompetition.dk/about/competition\"],[\"Bancroft on Malko\",\"https://malkocompetition.dk/about/bancroft-malko\"],[\"Fondation Caris\",\"https://malkocompetition.dk/about/foundation-caris\"],[\"Winners\",\"https://malkocompetition.dk/winners/2021\"],[\"Participants\",\"https://malkocompetition.dk/conductor/2021\"],[\"Performances\",\"https://malkocompetition.dk/performances/2021\"],[\"Videos\",\"https://malkocompetition.dk/video/2021\"],[\"Articles\",\"https://malkocompetition.dk/article/2021\"],[\"Photos\",\"https://malkocompetition.dk/photos/2021\"],[\"Winners\",\"https://malkocompetition.dk/winners/2018\"],[\"Participants\",\"https://malkocompetition.dk/conductor/2018\"],[\"Performances\",\"https://malkocompetition.dk/performances/2018\"],[\"Video\",\"https://malkocompetition.dk/video/2018\"],[\"Articles\",\"https://malkocompetition.dk/article/2018\"],[\"All Winners\",\"https://malkocompetition.dk/winners/all\"],[\"International Academy\",\"https://malkocompetition.dk/academy\"],[\"International Class 2022\",\"https://malkocompetition.dk/academy/2022\"],[\"Who can apply?\",\"https://malkocompetition.dk/academy/who-can-apply\"],[\"Lessons\",\"https://malkocompetition.dk/academy/courses\"],[\"Schedule\",\"https://malkocompetition.dk/academy/schedule\"],[\"Practical Information\",\"https://malkocompetition.dk/academy/practical\"],[\"Application\",\"https://malkocompetition.dk/academy/application\"],[\"Danish Academy\",\"https://malkocompetition.dk/danish-academy\"],[\"Artists & Contact\",\"https://malkocompetition.dk/academy/staff\"],[\"-\",\"https://malkocompetition.dk/academy/webform/succes\"],[\"-\",\"https://malkocompetition.dk/academy/webform\"],[\"-\",\"https://malkocompetition.dk/academy/kampagne\"],[\"Contact\",\"javascript:void(0)\"],[\"Newsletter\",\"javascript:void(0)\"],[\"Facebook\",\"https://www.facebook.com/malkocompetition/\"],[\"\",\"https://drkoncerthuset.dk/dr-symfoni-orkestret/danish-national-symphony-orchestra/\"],[\"See our cookie policy\",\"https://malkocompetition.dk/cookies\"]],\"title\":\"All Winners\",\"description\":\"\",\"url\":\"https://malkocompetition.dk/winners/all\",\"content\":\"All Winners | Malko Competition\\n\\n===============\\n[Skip to main content](https://malkocompetition.dk/winners/all#main-content)\\n\\n[Malko Competition for Young Conductors](https://malkocompetition.dk/)\\n\\n[](https://malkocompetition.dk/winners/all)\\n\\nAll Winners\\n===========\\n\\n[2024 Samuel Seungwon Lee ------------------- South Korea](https://malkocompetition.dk/winners/all)[2021 Dmitry Matvienko ---------------- Belarus](https://malkocompetition.dk/winners/all)[2018 Ryan Bancroft ------------- United States](https://malkocompetition.dk/winners/all)[2015 Tung-Chieh Chuang ----------------- Taiwan](https://malkocompetition.dk/winners/all)[2012 Rafael Payare ------------- Venezuela](https://malkocompetition.dk/winners/all)[2009 Joshua Weilerstein ------------------ United States](https://malkocompetition.dk/winners/all)[2005 Mei-Ann Chen ------------ United States](https://malkocompetition.dk/winners/all)[1998 Seikyo Kim ---------- Japan](https://malkocompetition.dk/winners/all)[1995 Jan Wagner ---------- Venezuela](https://malkocompetition.dk/winners/all)[1992 Jin Wang -------- Austria](https://malkocompetition.dk/winners/all)[1989 Fabio Mechetti -------------- Brasil](https://malkocompetition.dk/winners/all)[1986 Kazufumi Yamashita ------------------ Japan](https://malkocompetition.dk/winners/all)[1983 Claus Peter Flor ---------------- Germany](https://malkocompetition.dk/winners/all)[1980 Maximiano Valdes ---------------- Chile](https://malkocompetition.dk/winners/all)[1977 Philip Greenberg ---------------- United States](https://malkocompetition.dk/winners/all)[1974 Gotthard Lienicke -----------------](https://malkocompetition.dk/winners/all)[1971 Winston Dan Vogel ----------------- United States](https://malkocompetition.dk/winners/all)[1968 Avi Ostrowsky ------------- Israel](https://malkocompetition.dk/winners/all)[1965 Ralf Weikert ------------ Austria](https://malkocompetition.dk/winners/all)\\n\\n[](javascript:void(0))\\nCompetition\\n-----------\\n\\n*   [Home](https://malkocompetition.dk/)\\n*   [Participants](https://malkocompetition.dk/conductor/2024)\\n*   [Schedule](https://malkocompetition.dk/competition/schedule)\\n*   [Performances](https://malkocompetition.dk/performances/2024)\\n*   [Articles](https://malkocompetition.dk/article/2024)\\n*   [Videos](https://malkocompetition.dk/video/2024)\\n*   [Photos](https://malkocompetition.dk/photos/2024)\\n*   [Repertoire](https://malkocompetition.dk/competition/repertoire)\\n*   [Prizes](https://malkocompetition.dk/competition/prizes)\\n*   [Jury](https://malkocompetition.dk/competition/jury)\\n*   [Jury Rules](https://malkocompetition.dk/competition/rules)\\n\\nAbout\\n-----\\n\\n*   [DANISH NATIONAL SYMPHONY ORCHESTRA](https://malkocompetition.dk/about/danish-national-symphony-orchestra)\\n*   [Fabio Luisi](https://malkocompetition.dk/about/fabio-luisi)\\n*   [The competition](https://malkocompetition.dk/about/competition)\\n*   [Bancroft on Malko](https://malkocompetition.dk/about/bancroft-malko)\\n*   [Fondation Caris](https://malkocompetition.dk/about/foundation-caris)\\n\\nPrevious\\n--------\\n\\n*   2021\\n    *   [Winners](https://malkocompetition.dk/winners/2021)\\n    *   [Participants](https://malkocompetition.dk/conductor/2021)\\n    *   [Performances](https://malkocompetition.dk/performances/2021)\\n    *   [Videos](https://malkocompetition.dk/video/2021)\\n    *   [Articles](https://malkocompetition.dk/article/2021)\\n    *   [Photos](https://malkocompetition.dk/photos/2021)\\n\\n*   2018\\n    *   [Winners](https://malkocompetition.dk/winners/2018)\\n    *   [Participants](https://malkocompetition.dk/conductor/2018)\\n    *   [Performances](https://malkocompetition.dk/performances/2018)\\n    *   [Video](https://malkocompetition.dk/video/2018)\\n    *   [Articles](https://malkocompetition.dk/article/2018)\\n\\n*   [All Winners](https://malkocompetition.dk/winners/all)\\n\\nAcademy\\n-------\\n\\n*   [International Academy](https://malkocompetition.dk/academy)\\n*   [International Class 2022](https://malkocompetition.dk/academy/2022)\\n*   [Who can apply?](https://malkocompetition.dk/academy/who-can-apply)\\n*   [Lessons](https://malkocompetition.dk/academy/courses)\\n*   [Schedule](https://malkocompetition.dk/academy/schedule)\\n*   [Practical Information](https://malkocompetition.dk/academy/practical)\\n*   [Application](https://malkocompetition.dk/academy/application)\\n*   [Danish Academy](https://malkocompetition.dk/danish-academy)\\n*   [Artists & Contact](https://malkocompetition.dk/academy/staff)\\n*   [-](https://malkocompetition.dk/academy/webform/succes)\\n*   [-](https://malkocompetition.dk/academy/webform)\\n*   [-](https://malkocompetition.dk/academy/kampagne)\\n\\n*   [Contact](javascript:void(0))\\nMALKO COMPETITION\\n\\nFOR YOUNG CONDUCTORS\\n\\n Danish National Symphony Orchestra\\n\\n Att.: Marie Rørbech\\n\\n Emil Holms Kanal 20, 5-3\\n\\n 0999 København C\\n\\n Denmark\\n\\n mrrb@dr.dk\\n\\n Project Manager: Marie Rørbech\\n\\n*   [Newsletter](javascript:void(0))Subscribe to the Malko Competition newsletter!         \\n*   [Facebook](https://www.facebook.com/malkocompetition/)\\n\\n[](https://drkoncerthuset.dk/dr-symfoni-orkestret/danish-national-symphony-orchestra/)\\n\\nCookies\\n-------\\n\\nWe use cookies to analyze how visitors use this site.\\n\\n[See our cookie policy](https://malkocompetition.dk/cookies)\\n\\nOK, I agree Decline\",\"metadata\":{\"lang\":\"en\",\"og:site_name\":\"Malkocompetition for Young Conductors\",\"og:type\":\"Page\",\"og:url\":\"https://malkocompetition.dk/winners/all\",\"og:title\":\"All Winners\",\"twitter:card\":\"summary_large_image\",\"Generator\":\"Drupal 10 (https://www.drupal.org)\",\"MobileOptimized\":\"width\",\"HandheldFriendly\":\"true\",\"viewport\":\"width=device-width, initial-scale=1.0\",\"msapplication-TileColor\":\"#da532c\",\"theme-color\":\"#ffffff\"},\"external\":{\"canonical\":{\"https://malkocompetition.dk/winners/all\":{}},\"icon\":{\"https://malkocompetition.dk/core/misc/favicon.ico\":{\"type\":\"image/vnd.microsoft.icon\"},\"https://malkocompetition.dk/favicon-32x32.png\":{\"type\":\"image/png\",\"sizes\":\"32x32\"},\"https://malkocompetition.dk/favicon-16x16.png\":{\"type\":\"image/png\",\"sizes\":\"16x16\"}},\"apple-touch-icon\":{\"https://malkocompetition.dk/apple-touch-icon.png\":{\"sizes\":\"180x180\"}},\"mask-icon\":{\"https://malkocompetition.dk/safari-pinned-tab.svg\":{\"color\":\"#5bbad5\"}}},\"usage\":{\"tokens\":1451}},\"meta\":{\"usage\":{\"tokens\":1451}}}",
      "total_length": 9420,
      "is_paginated": False,
      "pages": 1,
      "current_page": 1,
      "page_info": "Complete content (9420 characters)",
      "time_ms": 315,
      "timestamp": "2025-09-03T10:00:20.736627"
    },
    "server_id": "default"
  },
  {
    "role": "assistant",
    "content": "\n\n",
    "thought": "\nFirst, I need to find the Malko Competition recipients from the 20th century after 1977. The search results show that the competition started in 1965, but I need to focus on the 20th century, which means up to 2000. The winners listed on the page are:\n\n- 1998: Seikyo Kim (Japan)\n- 1995: Jan Wagner (Venezuela)\n- 1992: Jin Wang (Austria)\n- 1989: Fabio Mechetti (Brasil)\n- 1986: Kazufumi Yamashita (Japan)\n- 1983: Claus Peter Flor (Germany)\n- 1980: Maximiano Valdes (Chile)\n- 1977: Philip Greenberg (United States)\n\nNow, I need to check which of these nationalities are countries that no longer exist. Let me list them:\n\n- Japan: Still exists\n- Venezuela: Still exists\n- Austria: Still exists\n- Brasil: Still exists (Brazil)\n- Japan: Again, still exists\n- Germany: Still exists\n- Chile: Still exists\n- United States: Still exists\n\nAll these countries still exist today. So, none of them fit the criteria of \"a country that no longer exists.\" I might need to look for more detailed information about each winner's nationality or check if there's any other winner from the 20th century that I missed.\n\nThe question specifies \"after 1977,\" so I should include winners from 1977 onwards in the 20th century. The winners listed are from 1977 to 1998.\n\nI should also check if there are any other winners between 1965 and 1977 that might have nationalities that no longer exist, but the question says \"after 1977,\" so it might mean after 1977, but the 20th century ends in 2000, so it could be from 1977 to 2000.\n\nLet me double-check the list. The winners are:\n\n- 1998: Seikyo Kim (Japan)\n- 1995: Jan Wagner (Venezuela)\n- 1992: Jin Wang (Austria)\n- 1989: Fabio Mechetti (Brasil)\n- 1986: Kazufumi Yamashita (Japan)\n- 1983: Claus Peter Flor (Germany)\n- 1980: Maximiano Valdes (Chile)\n- 1977: Philip Greenberg (United States)\n\nI need to find if any of these nationalities are countries that no longer exist. For example, Brazil is still Brazil, so that's fine. Austria, Germany, Chile, Japan, Venezuela, United States—all exist.\n\nMaybe I need to look up each winner's nationality more carefully. Perhaps \"Brasil\" is written as \"Brazil\" in English, but it's the same country. Or maybe one of them had a different nationality at the time.\n\nThe question says \"nationality on record is a country that no longer exists.\" So, perhaps the country they were from at the time of winning no longer exists.\n\nFor example, East Germany or West Germany, but Germany is unified now. Or something like that.\n\nLet me check each winner:\n\n- Seikyo Kim: Japan (still exists)\n- Jan Wagner: Venezuela (still exists)\n- Jin Wang: Austria (still exists)\n- Fabio Mechetti: Brazil (still exists)\n- Kazufumi Yamashita: Japan (still exists)\n- Claus Peter Flor: Germany (still exists)\n- Maximiano Valdes: Chile (still exists)\n- Philip Greenberg: United States (still exists)\n\nAll seem to be current countries. Maybe I need to look for winners from 1965 to 1977, but the question says \"after 1977,\" so it might be from 1977 onwards.\n\nLet me see if there are any other winners. The list shows up to 1998, but maybe there are winners from 1965-1977 that I should consider.\n\nThe question says \"from the 20th Century (after 1977)\", so it means after 1977 in the 20th century, which is 1978-2000.\n\nSo, winners from 1977 might be included if it's \"after 1977\" could be interpreted as \"after the year 1977\", but 1977 is part of the 20th century, and after 1977 would be 1978-2000. However, the winner in 1977 is Philip Greenberg, and it says \"after 1977\", so perhaps it means after 1977, so from 1978 onwards.\n\nBut the list includes 1977, so maybe the question is a bit ambiguous. Let me read the question again: \"Malko Competition recipient from the 20th Century (after 1977)\". So, 20th century means up to 2000, and after 1977 means from 1978 to 2000.\n\nTherefore, the winners after 1977 are:\n\n- 1998: Seikyo Kim\n- 1995: Jan Wagner\n- 1992: Jin Wang\n- 1989: Fabio Mechetti\n- 1986: Kazufumi Yamashita\n- 1983: Claus Peter Flor\n- 1980: Maximiano Valdes\n- 1977: Philip Greenberg might not be included if \"after 1977\" means strictly after, but the question says \"(after 1977)\", which might imply after 1977, so perhaps excluding 1977.\n\nTo be safe, let's include all from 1977 onwards since the 20th century ends in 2000.\n\nNow, I need to find which of these nationalities are countries that no longer exist.\n\nLet me think about historical contexts:\n\n- Brazil: Still exists\n- Japan: Still exists\n- Venezuela: Still exists\n- Austria: Still exists\n- Germany: Still exists (unified)\n- Chile: Still exists\n- United States: Still exists\n\nPerhaps one of them had a different nationality at the time. For example, perhaps someone was from a country that no longer exists, like the Soviet Union, but I don't see any obvious candidates.\n\nLet me check if there are any other winners. The list might not be complete. The page says \"All Winners\" but it might not list all, or I might need to look at individual pages.\n\nI should check the Wikipedia page for more information.\n",
    "tool_calls": [
      {
        "id": "call_f140e46b7eed41a4a09da391",
        "function": {
          "name": "search",
          "arguments": "{\"query\": \"Malko Competition winners list complete\"}"
        },
        "server_id": "search-fusion-mcp"
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_f140e46b7eed41a4a09da391",
    "name": "search-fusion-mcp.search",
    "content": {
      "query": "Malko Competition winners list complete",
      "engine": "auto",
      "time_ms": 198,
      "num_results": 10,
      "results": [
        {
          "title": "All Winners | Malko Competition",
          "link": "https://malkocompetition.dk/winners/all",
          "snippet": "Samuel Seungwon Lee · Dmitry Matvienko · Ryan Bancroft · Tung-Chieh Chuang · Rafael Payare · Joshua Weilerstein · Mei-Ann Chen · Seikyo Kim.",
          "source": "google",
          "metadata": {
            "display_link": "malkocompetition.dk"
          }
        },
        {
          "title": "Malko Competition - Wikipedia",
          "link": "https://en.wikipedia.org/wiki/Malko_Competition",
          "snippet": "The Malko Competition is an international competition for young conductors. It is held every three years by the Danish Radio Symphony Orchestra, ...",
          "source": "google",
          "metadata": {
            "display_link": "en.wikipedia.org"
          }
        },
        {
          "title": "Malko Competition: The Malko Way – The competition that changed ...",
          "link": "https://malkocompetition.dk/",
          "snippet": "Previous. 2021. Winners · Participants · Performances · Videos · Articles · Photos. 2018. Winners · Participants · Performances · Video · Articles · All Winners ...",
          "source": "google",
          "metadata": {
            "display_link": "malkocompetition.dk"
          }
        },
        {
          "title": "Winners Announced at 2024 Malko Competition",
          "link": "https://theviolinchannel.com/winners-announced-at-2024-malko-competition/",
          "snippet": "Apr 22, 2024 ... Danish National Symphony Orchestra's chief conductor and Malko Competition jury chairman, Fabio Luisi, said, \"Samuel has a fantastic way of ...",
          "source": "google",
          "metadata": {
            "display_link": "theviolinchannel.com"
          }
        },
        {
          "title": "Guest Conductor Biography: Mei-Ann Chen - Springfield Symphony ...",
          "link": "https://www.springfieldsymphony.org/guest-conductor-biography-mei-ann-chen/",
          "snippet": "Jan 31, 2025 ... ... Winner of the Malko Competition (she remains as the only woman in the competition history since 1965 to have won First Prize), and ASCAP ...",
          "source": "google",
          "metadata": {
            "display_link": "www.springfieldsymphony.org"
          }
        },
        {
          "title": "Valentin Egel | Malko Competition",
          "link": "https://www.malkocompetition.dk/conductor/2021/valentin-egel",
          "snippet": "Valentin Egel is first prize winner of the 7th International Competition for young Conductors Lovro von Matačić, of the MDR Sinfonieorchester Conducting ...",
          "source": "google",
          "metadata": {
            "display_link": "www.malkocompetition.dk"
          }
        },
        {
          "title": "MEN'S TENNIS PICKS UP FIRST WIN OF THE SEASON WITH 6-1 ...",
          "link": "https://hornetsports.com/news/2013/1/19/1_19_2013_3169.aspx",
          "snippet": "Jan 19, 2013 ... Sacramento State&rsquo;s Marek Marksoo, Aliaksandr Malko, Roy Brandys, Tom Miller and Niranjan Ram each posted singles victories to lead the ...",
          "source": "google",
          "metadata": {
            "display_link": "hornetsports.com"
          }
        },
        {
          "title": "Samuel Lee joins HarrisonParrott for Worldwide Management ...",
          "link": "https://www.harrisonparrott.com/news/2024-06-27/samuel-lee-joins-harrisonparrott-for-worldwide-management",
          "snippet": "Jun 27, 2024 ... Winner of the recent Malko Competition, he looks forward to an extensive list of guest engagements worldwide in the coming seasons, building ...",
          "source": "google",
          "metadata": {
            "display_link": "www.harrisonparrott.com"
          }
        },
        {
          "title": "langchain_mistral_test.ipynb · felixmortas ...",
          "link": "https://huggingface.co/spaces/felixmortas/Hf_Aagent_Course_Final_Assignment/blob/main/langchain_mistral_test.ipynb",
          "snippet": "Jul 4, 2025 ... Report your thoughts, and finish your answer with ... **Research and Verification:** - Upon reviewing the list of Malko Competition winners ...",
          "source": "google",
          "metadata": {
            "display_link": "huggingface.co"
          }
        },
        {
          "title": "Two young artists make UGA Presents debuts with Atlanta ...",
          "link": "https://news.uga.edu/two-young-artists-make-uga-presents-debuts-with-atlanta-symphony-orchestra/",
          "snippet": "Apr 26, 2024 ... ... winners at the International Tchaikovsky Competition ... Matvienko is the winner of the 2021 edition of the prestigious Malko Competition for ...",
          "source": "google",
          "metadata": {
            "display_link": "news.uga.edu"
          }
        }
      ],
      "timestamp": "2025-09-03T10:00:50.175547",
      "success": True
    },
    "server_id": "default"
  },
  {
    "role": "assistant",
    "content": "\n\n",
    "thought": "\nFirst, I need to find the complete list of Malko Competition winners, especially from the 20th century after 1977. The initial search gave me a list, but I need more detailed information about each winner, including their nationalities.\n\nLet me try to get the complete winners list from the official website. I'll fetch the page that lists all winners.\n",
    "tool_calls": [
      {
        "id": "call_d1b4cb4176514bdbb7160289",
        "function": {
          "name": "fetch_url",
          "arguments": "{\"url\": \"https://malkocompetition.dk/winners/all\"}"
        },
        "server_id": "search-fusion-mcp"
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_d1b4cb4176514bdbb7160289",
    "name": "search-fusion-mcp.fetch_url",
    "content": {
      "url": "https://malkocompetition.dk/winners/all",
      "method": "jina",
      "format": "markdown",
      "success": True,
      "content": "{\"code\":200,\"status\":20000,\"data\":{\"links\":[[\"Skip to main content\",\"https://malkocompetition.dk/winners/all#main-content\"],[\"Malko Competition for Young Conductors\",\"https://malkocompetition.dk/\"],[\"\",\"javascript:void(0)\"],[\"Home\",\"https://malkocompetition.dk/\"],[\"Participants\",\"https://malkocompetition.dk/conductor/2024\"],[\"Schedule\",\"https://malkocompetition.dk/competition/schedule\"],[\"Performances\",\"https://malkocompetition.dk/performances/2024\"],[\"Articles\",\"https://malkocompetition.dk/article/2024\"],[\"Videos\",\"https://malkocompetition.dk/video/2024\"],[\"Photos\",\"https://malkocompetition.dk/photos/2024\"],[\"Repertoire\",\"https://malkocompetition.dk/competition/repertoire\"],[\"Prizes\",\"https://malkocompetition.dk/competition/prizes\"],[\"Jury\",\"https://malkocompetition.dk/competition/jury\"],[\"Jury Rules\",\"https://malkocompetition.dk/competition/rules\"],[\"DANISH NATIONAL SYMPHONY ORCHESTRA\",\"https://malkocompetition.dk/about/danish-national-symphony-orchestra\"],[\"Fabio Luisi\",\"https://malkocompetition.dk/about/fabio-luisi\"],[\"The competition\",\"https://malkocompetition.dk/about/competition\"],[\"Bancroft on Malko\",\"https://malkocompetition.dk/about/bancroft-malko\"],[\"Fondation Caris\",\"https://malkocompetition.dk/about/foundation-caris\"],[\"Winners\",\"https://malkocompetition.dk/winners/2021\"],[\"Participants\",\"https://malkocompetition.dk/conductor/2021\"],[\"Performances\",\"https://malkocompetition.dk/performances/2021\"],[\"Videos\",\"https://malkocompetition.dk/video/2021\"],[\"Articles\",\"https://malkocompetition.dk/article/2021\"],[\"Photos\",\"https://malkocompetition.dk/photos/2021\"],[\"Winners\",\"https://malkocompetition.dk/winners/2018\"],[\"Participants\",\"https://malkocompetition.dk/conductor/2018\"],[\"Performances\",\"https://malkocompetition.dk/performances/2018\"],[\"Video\",\"https://malkocompetition.dk/video/2018\"],[\"Articles\",\"https://malkocompetition.dk/article/2018\"],[\"All Winners\",\"https://malkocompetition.dk/winners/all\"],[\"International Academy\",\"https://malkocompetition.dk/academy\"],[\"International Class 2022\",\"https://malkocompetition.dk/academy/2022\"],[\"Who can apply?\",\"https://malkocompetition.dk/academy/who-can-apply\"],[\"Lessons\",\"https://malkocompetition.dk/academy/courses\"],[\"Schedule\",\"https://malkocompetition.dk/academy/schedule\"],[\"Practical Information\",\"https://malkocompetition.dk/academy/practical\"],[\"Application\",\"https://malkocompetition.dk/academy/application\"],[\"Danish Academy\",\"https://malkocompetition.dk/danish-academy\"],[\"Artists & Contact\",\"https://malkocompetition.dk/academy/staff\"],[\"-\",\"https://malkocompetition.dk/academy/webform/succes\"],[\"-\",\"https://malkocompetition.dk/academy/webform\"],[\"-\",\"https://malkocompetition.dk/academy/kampagne\"],[\"Contact\",\"javascript:void(0)\"],[\"Newsletter\",\"javascript:void(0)\"],[\"Facebook\",\"https://www.facebook.com/malkocompetition/\"],[\"\",\"https://drkoncerthuset.dk/dr-symfoni-orkestret/danish-national-symphony-orchestra/\"],[\"See our cookie policy\",\"https://malkocompetition.dk/cookies\"]],\"title\":\"All Winners\",\"description\":\"\",\"url\":\"https://malkocompetition.dk/winners/all\",\"content\":\"All Winners | Malko Competition\\n\\n===============\\n[Skip to main content](https://malkocompetition.dk/winners/all#main-content)\\n\\n[Malko Competition for Young Conductors](https://malkocompetition.dk/)\\n\\n[](https://malkocompetition.dk/winners/all)\\n\\nAll Winners\\n===========\\n\\n[2024 Samuel Seungwon Lee ------------------- South Korea](https://malkocompetition.dk/winners/all)[2021 Dmitry Matvienko ---------------- Belarus](https://malkocompetition.dk/winners/all)[2018 Ryan Bancroft ------------- United States](https://malkocompetition.dk/winners/all)[2015 Tung-Chieh Chuang ----------------- Taiwan](https://malkocompetition.dk/winners/all)[2012 Rafael Payare ------------- Venezuela](https://malkocompetition.dk/winners/all)[2009 Joshua Weilerstein ------------------ United States](https://malkocompetition.dk/winners/all)[2005 Mei-Ann Chen ------------ United States](https://malkocompetition.dk/winners/all)[1998 Seikyo Kim ---------- Japan](https://malkocompetition.dk/winners/all)[1995 Jan Wagner ---------- Venezuela](https://malkocompetition.dk/winners/all)[1992 Jin Wang -------- Austria](https://malkocompetition.dk/winners/all)[1989 Fabio Mechetti -------------- Brasil](https://malkocompetition.dk/winners/all)[1986 Kazufumi Yamashita ------------------ Japan](https://malkocompetition.dk/winners/all)[1983 Claus Peter Flor ---------------- Germany](https://malkocompetition.dk/winners/all)[1980 Maximiano Valdes ---------------- Chile](https://malkocompetition.dk/winners/all)[1977 Philip Greenberg ---------------- United States](https://malkocompetition.dk/winners/all)[1974 Gotthard Lienicke -----------------](https://malkocompetition.dk/winners/all)[1971 Winston Dan Vogel ----------------- United States](https://malkocompetition.dk/winners/all)[1968 Avi Ostrowsky ------------- Israel](https://malkocompetition.dk/winners/all)[1965 Ralf Weikert ------------ Austria](https://malkocompetition.dk/winners/all)\\n\\n[](javascript:void(0))\\nCompetition\\n-----------\\n\\n*   [Home](https://malkocompetition.dk/)\\n*   [Participants](https://malkocompetition.dk/conductor/2024)\\n*   [Schedule](https://malkocompetition.dk/competition/schedule)\\n*   [Performances](https://malkocompetition.dk/performances/2024)\\n*   [Articles](https://malkocompetition.dk/article/2024)\\n*   [Videos](https://malkocompetition.dk/video/2024)\\n*   [Photos](https://malkocompetition.dk/photos/2024)\\n*   [Repertoire](https://malkocompetition.dk/competition/repertoire)\\n*   [Prizes](https://malkocompetition.dk/competition/prizes)\\n*   [Jury](https://malkocompetition.dk/competition/jury)\\n*   [Jury Rules](https://malkocompetition.dk/competition/rules)\\n\\nAbout\\n-----\\n\\n*   [DANISH NATIONAL SYMPHONY ORCHESTRA](https://malkocompetition.dk/about/danish-national-symphony-orchestra)\\n*   [Fabio Luisi](https://malkocompetition.dk/about/fabio-luisi)\\n*   [The competition](https://malkocompetition.dk/about/competition)\\n*   [Bancroft on Malko](https://malkocompetition.dk/about/bancroft-malko)\\n*   [Fondation Caris](https://malkocompetition.dk/about/foundation-caris)\\n\\nPrevious\\n--------\\n\\n*   2021\\n    *   [Winners](https://malkocompetition.dk/winners/2021)\\n    *   [Participants](https://malkocompetition.dk/conductor/2021)\\n    *   [Performances](https://malkocompetition.dk/performances/2021)\\n    *   [Videos](https://malkocompetition.dk/video/2021)\\n    *   [Articles](https://malkocompetition.dk/article/2021)\\n    *   [Photos](https://malkocompetition.dk/photos/2021)\\n\\n*   2018\\n    *   [Winners](https://malkocompetition.dk/winners/2018)\\n    *   [Participants](https://malkocompetition.dk/conductor/2018)\\n    *   [Performances](https://malkocompetition.dk/performances/2018)\\n    *   [Video](https://malkocompetition.dk/video/2018)\\n    *   [Articles](https://malkocompetition.dk/article/2018)\\n\\n*   [All Winners](https://malkocompetition.dk/winners/all)\\n\\nAcademy\\n-------\\n\\n*   [International Academy](https://malkocompetition.dk/academy)\\n*   [International Class 2022](https://malkocompetition.dk/academy/2022)\\n*   [Who can apply?](https://malkocompetition.dk/academy/who-can-apply)\\n*   [Lessons](https://malkocompetition.dk/academy/courses)\\n*   [Schedule](https://malkocompetition.dk/academy/schedule)\\n*   [Practical Information](https://malkocompetition.dk/academy/practical)\\n*   [Application](https://malkocompetition.dk/academy/application)\\n*   [Danish Academy](https://malkocompetition.dk/danish-academy)\\n*   [Artists & Contact](https://malkocompetition.dk/academy/staff)\\n*   [-](https://malkocompetition.dk/academy/webform/succes)\\n*   [-](https://malkocompetition.dk/academy/webform)\\n*   [-](https://malkocompetition.dk/academy/kampagne)\\n\\n*   [Contact](javascript:void(0))\\nMALKO COMPETITION\\n\\nFOR YOUNG CONDUCTORS\\n\\n Danish National Symphony Orchestra\\n\\n Att.: Marie Rørbech\\n\\n Emil Holms Kanal 20, 5-3\\n\\n 0999 København C\\n\\n Denmark\\n\\n mrrb@dr.dk\\n\\n Project Manager: Marie Rørbech\\n\\n*   [Newsletter](javascript:void(0))Subscribe to the Malko Competition newsletter!         \\n*   [Facebook](https://www.facebook.com/malkocompetition/)\\n\\n[](https://drkoncerthuset.dk/dr-symfoni-orkestret/danish-national-symphony-orchestra/)\\n\\nCookies\\n-------\\n\\nWe use cookies to analyze how visitors use this site.\\n\\n[See our cookie policy](https://malkocompetition.dk/cookies)\\n\\nOK, I agree Decline\",\"metadata\":{\"lang\":\"en\",\"og:site_name\":\"Malkocompetition for Young Conductors\",\"og:type\":\"Page\",\"og:url\":\"https://malkocompetition.dk/winners/all\",\"og:title\":\"All Winners\",\"twitter:card\":\"summary_large_image\",\"Generator\":\"Drupal 10 (https://www.drupal.org)\",\"MobileOptimized\":\"width\",\"HandheldFriendly\":\"true\",\"viewport\":\"width=device-width, initial-scale=1.0\",\"msapplication-TileColor\":\"#da532c\",\"theme-color\":\"#ffffff\"},\"external\":{\"canonical\":{\"https://malkocompetition.dk/winners/all\":{}},\"icon\":{\"https://malkocompetition.dk/core/misc/favicon.ico\":{\"type\":\"image/vnd.microsoft.icon\"},\"https://malkocompetition.dk/favicon-32x32.png\":{\"type\":\"image/png\",\"sizes\":\"32x32\"},\"https://malkocompetition.dk/favicon-16x16.png\":{\"type\":\"image/png\",\"sizes\":\"16x16\"}},\"apple-touch-icon\":{\"https://malkocompetition.dk/apple-touch-icon.png\":{\"sizes\":\"180x180\"}},\"mask-icon\":{\"https://malkocompetition.dk/safari-pinned-tab.svg\":{\"color\":\"#5bbad5\"}}},\"usage\":{\"tokens\":1451}},\"meta\":{\"usage\":{\"tokens\":1451}}}",
      "total_length": 9420,
      "is_paginated": False,
      "pages": 1,
      "current_page": 1,
      "page_info": "Complete content (9420 characters)",
      "time_ms": 513,
      "timestamp": "2025-09-03T10:00:53.081513"
    },
    "server_id": "default"
  },
  {
    "role": "assistant",
    "content": "\n\n",
    "thought": "\nFirst, I need to find the Malko Competition recipients from the 20th century after 1977. The list from the website shows winners from 1965 onwards, but I need to focus on the 20th century after 1977, which means from 1978 to 2000.\n\nLooking at the list:\n- 1998 Seikyo Kim - Japan\n- 1995 Jan Wagner - Venezuela\n- 1992 Jin Wang - Austria\n- 1989 Fabio Mechetti - Brasil\n- 1986 Kazufumi Yamashita - Japan\n- 1983 Claus Peter Flor - Germany\n- 1980 Maximiano Valdes - Chile\n- 1977 Philip Greenberg - United States\n\nNow, I need to check which of these nationalities are countries that no longer exist. Let me think about each one:\n- Japan: Still exists\n- Venezuela: Still exists\n- Austria: Still exists (though it was part of Austria-Hungary before, but as a country, it exists)\n- Brasil: Still exists (Brazil is a country)\n- Japan: Still exists\n- Germany: Still exists (unified Germany)\n- Chile: Still exists\n- United States: Still exists\n\nBut the question says \"after 1977\", so I need to look at winners from 1978 onwards. The list shows:\n- 1998, 1995, 1992, 1989, 1986, 1983, 1980, 1977\n\n1977 is the year after which we need to consider, so we include 1977 onwards. But let's confirm the exact wording: \"after 1977\" might mean after 1977, so 1978 and later. However, the question says \"after 1977\", so it could be inclusive or exclusive. Let me read the question again: \"from the 20th Century (after 1977)\" – this likely means years after 1977, so 1978 to 2000.\n\nBut the list includes 1977, which is not after 1977. So perhaps I should exclude 1977. Let me see: the winners are from 1965, 1968, 1971, 1974, 1977, etc. The 20th century ends in 2000, so after 1977 would be 1978 to 2000.\n\nFrom the list:\n- 1998 Seikyo Kim - Japan\n- 1995 Jan Wagner - Venezuela\n- 1992 Jin Wang - Austria\n- 1989 Fabio Mechetti - Brasil\n- 1986 Kazufumi Yamashita - Japan\n- 1983 Claus Peter Flor - Germany\n- 1980 Maximiano Valdes - Chile\n\nNow, I need to check which of these nationalities are countries that no longer exist. Let me think about each:\n\n- Japan: Exists\n- Venezuela: Exists\n- Austria: Exists (though it was part of Austria-Hungary, but as a country, it's still there)\n- Brasil: Exists (Brazil is a country, but \"Brasil\" might refer to the state or country. Brazil is a country, so it exists)\n- Japan: Exists\n- Germany: Exists (unified Germany since 1990, but even before, it was West Germany, which exists)\n- Chile: Exists\n\nBut the question says \"country that no longer exists\". So I need to find if any of these nationalities are from countries that have since ceased to exist.\n\nLet me think about historical changes:\n- Brazil was a monarchy until 1889, but it's a country now, and it exists.\n- Venezuela has had some changes, but it's still a country.\n- Austria has been around as a country since 1918, and it exists.\n- Germany: After WWII, there was East and West Germany, but now it's unified, and it exists.\n- Chile: Exists\n- Japan: Exists\n\nBut there might be some subtlety. For example, \"Brasil\" could refer to the state of Brazil in the Netherlands, but that's not likely. Probably, it's Brazil as a country.\n\nI should verify the exact nationality for each winner. The list might have abbreviations or different names.\n\nLet me check the Wikipedia page or more detailed information about each winner to confirm their nationalities.\n\nI'll search for each winner to get more details about their nationalities.\n",
    "tool_calls": [
      {
        "id": "call_bdcf9f41b6f345679726d649",
        "function": {
          "name": "search",
          "arguments": "{\"query\": \"Malko Competition winners 1998 Seikyo Kim nationality\"}"
        },
        "server_id": "search-fusion-mcp"
      }
    ]
  },
  {
    "role": "tool",
    "tool_call_id": "call_bdcf9f41b6f345679726d649",
    "name": "search-fusion-mcp.search",
    "content": {
      "query": "Malko Competition winners 1998 Seikyo Kim nationality",
      "engine": "auto",
      "time_ms": 140,
      "num_results": 3,
      "results": [
        {
          "title": "All Winners | Malko Competition",
          "link": "https://malkocompetition.dk/winners/all",
          "snippet": "Mei-Ann Chen. United States. 1998. Seikyo Kim. Japan. 1995. Jan Wagner. Venezuela. 1992. Jin Wang. Austria. 1989. Fabio Mechetti. Brasil. 1986. Kazufumi ...",
          "source": "google",
          "metadata": {
            "display_link": "malkocompetition.dk"
          }
        },
        {
          "title": "Malko Competition - Wikipedia",
          "link": "https://en.wikipedia.org/wiki/Malko_Competition",
          "snippet": "The Malko Competition is an international competition for young conductors. It is held every three years by the Danish Radio Symphony Orchestra, ...",
          "source": "google",
          "metadata": {
            "display_link": "en.wikipedia.org"
          }
        },
        {
          "title": "Albums - Classical Archives",
          "link": "https://www.classicalarchives.com/albums/b.html",
          "snippet": "... Kim. Beethoven: Piano Sonatas Nos. 21 & 29 Sunwook Kim. Beethoven & Schubert ... 1998 International Violin Competition of Indianapolis various artists.",
          "source": "google",
          "metadata": {
            "display_link": "www.classicalarchives.com"
          }
        }
      ],
      "timestamp": "2025-09-03T10:01:12.227493",
      "success": True
    },
    "server_id": "default"
  },
  {
    "role": "assistant",
    "content": "\n\nBased on the list of Malko Competition winners from the 20th century after 1977, the only recipient whose nationality is a country that no longer exists is Claus Peter Flor, who won in 1983 and was from West Germany (which no longer exists as a separate entity after German reunification in 1990).\n\nanswer>Claus Peteranswer>",
    "thought": "\nFirst, I need to find the Malko Competition recipients from the 20th century after 1977. The list from the website shows winners from 1977 onwards:\n\n- 1977: Philip Greenberg (United States)\n- 1980: Maximiano Valdes (Chile)\n- 1983: Claus Peter Flor (Germany)\n- 1986: Kazufumi Yamashita (Japan)\n- 1989: Fabio Mechetti (Brasil)\n- 1992: Jin Wang (Austria)\n- 1995: Jan Wagner (Venezuela)\n- 1998: Seikyo Kim (Japan)\n- 2005: Mei-Ann Chen (United States)\n- 2009: Joshua Weilerstein (United States)\n- 2012: Rafael Payare (Venezuela)\n- 2015: Tung-Chieh Chuang (Taiwan)\n- 2018: Ryan Bancroft (United States)\n- 2021: Dmitry Matvienko (Belarus)\n- 2024: Samuel Seungwon Lee (South Korea)\n\nNow, I need to find which of these nationalities are countries that no longer exist. Let's check each one:\n\n- United States: Still exists\n- Chile: Still exists\n- Germany: Still exists (though it was East and West Germany before 1990, but since 1998, it's unified Germany)\n- Japan: Still exists\n- Brasil: Still exists (though it was often called Brazil in English, but it's the same country)\n- Austria: Still exists\n- Venezuela: Still exists\n- Japan: Still exists\n- United States: Still exists\n- United States: Still exists\n- Venezuela: Still exists\n- Taiwan: This is a tricky one. Taiwan is not an independent country; it's part of China. However, it's often treated as a separate entity in some contexts, but technically, it's not a sovereign country. But the question says \"country that no longer exists,\" so Taiwan might qualify as it's not recognized as a country by the UN.\n- United States: Still exists\n- Belarus: Still exists\n- South Korea: Still exists\n\nLooking at the list, the only one that might fit is Tung-Chieh Chuang from Taiwan in 2015. Taiwan is not an independent country, so it could be considered as a country that no longer exists in the context of being a sovereign state.\n\nBut let me double-check if there are others. For example, Jin Wang from Austria in 1992 - Austria exists. Fabio Mechetti from Brasil in 1989 - Brasil exists. Kazufumi Yamashita from Japan in 1986 - Japan exists. Claus Peter Flor from Germany in 1983 - Germany exists. Maximiano Valdes from Chile in 1980 - Chile exists. Philip Greenberg from United States in 1977 - United States exists.\n\nSo, the only one that might be problematic is Tung-Chieh Chuang from Taiwan in 2015. But the question says \"from the 20th Century (after 1977)\", and 2015 is in the 21st century, not the 20th. The 20th century ended in 2000, so after 1977 would be up to 2000. Let's see the list:\n\n- 1977: Philip Greenberg (US)\n- 1980: Maximiano Valdes (Chile)\n- 1983: Claus Peter Flor (Germany)\n- 1986: Kazufumi Yamashita (Japan)\n- 1989: Fabio Mechetti (Brasil)\n- 1992: Jin Wang (Austria)\n- 1995: Jan Wagner (Venezuela)\n- 1998: Seikyo Kim (Japan)\n- 2005: Mei-Ann Chen (US)\n- 2009: Joshua Weilerstein (US)\n- 2012: Rafael Payare (Venezuela)\n- 2015: Tung-Chieh Chuang (Taiwan) - but 2015 is 21st century\n- 2018: Ryan Bancroft (US)\n- 2021: Dmitry Matvienko (Belarus)\n- 2024: Samuel Seungwon Lee (South Korea)\n\nSo, for the 20th century after 1977, we have up to 1998. The winners from 1977 to 1998 are:\n\n- 1977: US\n- 1980: Chile\n- 1983: Germany\n- 1986: Japan\n- 1989: Brasil\n- 1992: Austria\n- 1995: Venezuela\n- 1998: Japan\n\nNow, I need to check if any of these countries no longer exist. For example, Brazil (Brasil) was a colony before, but as a country, it exists. Austria has existed since the 19th century. Venezuela exists. Germany has existed since 1990 as a unified country, but before that, it was divided. Since the competition is in the 20th century, and we're looking after 1977, Germany has been unified since 1990, so for 1983, it was West Germany, which no longer exists as a separate entity.\n\nSimilarly, Brazil has existed as a country since 1822, so it's still relevant.\n\nAustria has existed since 1918, so it's still relevant.\n\nVenezuela has existed since 1811, so it's still relevant.\n\nJapan has existed since ancient times, so it's still relevant.\n\nChile has existed since 1810, so it's still relevant.\n\nGermany: West Germany was dissolved in 1990 when it merged with East Germany. So, for the 1983 winner, Claus Peter Flor, he was from West Germany, which no longer exists as a separate country.\n\nAlso, for 1998, Seikyo Kim is from Japan, which still exists.\n\nBut let's check if there are others. For example, if any of the countries have changed names or statuses.\n\nAnother one could be if there's a country that no longer exists, like the Soviet Union, but none of the winners are from the Soviet Union or its republics.\n\nLooking at the list:\n\n- 1989: Fabio Mechetti from Brasil (Brazil exists)\n\n- 1992: Jin Wang from Austria (Austria exists)\n\n- 1995: Jan Wagner from Venezuela (Venezuela exists)\n\n- 1998: Seikyo Kim from Japan (Japan exists)\n\nBut for 1983: Claus Peter Flor from Germany – this was West Germany, which no longer exists.\n\nAlso, for 1977: Philip Greenberg from United States (US exists)\n\nFor 1980: Maximiano Valdes from Chile (Chile exists)\n\nFor 1986: Kazufumi Yamashita from Japan (Japan exists)\n\nFor 1989: Fabio Mechetti from Brasil (Brazil exists)\n\nFor 1992: Jin Wang from Austria (Austria exists)\n\nFor 1995: Jan Wagner from Venezuela (Venezuela exists)\n\nFor 1998: Seikyo Kim from Japan (Japan exists)\n\nSo, the only one that might be a candidate is Claus Peter Flor from Germany in 1983, as West Germany no longer exists.\n\nBut let's confirm if there are others. For example, if any other countries have been dissolved or renamed.\n\nAnother possibility is if there's a winner from a country that no longer exists, like the former Yugoslavia or something, but I don't see any in the list.\n\nAlso, let's check if any of the nationalities are from countries that have changed names. For example, Brazil was often called Brazil in English, but it's the same country.\n\nBut for West Germany, it was officially \"Federal Republic of Germany\" and it no longer exists as a separate entity.\n\nSo, Claus Peter Flor won in 1983 and was from Germany (West Germany), which no longer exists.\n\nBut the question says \"after 1977\", so 1983 is after 1977, so it qualifies.\n\nAlso, the first name is Claus Peter Flor.\n\nBut let's see if there are others. For example, if there's another winner from a country that no longer exists.\n\nAnother one could be if there's a winner from a country that has been dissolved, like the Soviet Union, but none are listed.\n\nAlso, let's check the 1992 winner: Jin Wang from Austria. Austria still exists.\n\n1995: Jan Wagner from Venezuela. Venezuela exists.\n\n1998: Seikyo Kim from Japan. Japan exists.\n\nSo, the only one that seems to fit is Claus Peter Flor from Germany (West Germany).\n\nBut let's double-check if West Germany is considered as no longer existing. Yes, after reunification in 1990, West Germany was dissolved.\n\nAlso, let's see if there's any other winner from a country that no longer exists. For example, if there's a winner from a former state.\n\nAnother one could be if there's a winner from a country that has been renamed, like South Vietnam or something, but I don't see any.\n\nLet me also check the earlier winners:\n\n- 1977: United States (exists)\n\n- 1980: Chile (exists)\n\n- 1983: Germany (West Germany, no longer exists)\n\n- 1986: Japan (exists)\n\n- 1989: Brasil (Brazil, exists)\n\n- 1992: Austria (exists)\n\n- 1995: Venezuela (exists)\n\n- 1998: Japan (exists)\n\nSo, Claus Peter Flor is the only one from a country that no longer exists.\n\nBut let's confirm if there are any other winners after 1977 in the 20th century. The 20th century ended in 2000, so after 1977 would be up to 2000. The winners up to 1998 are in the 20th century. The 2005 winner is in the 21st century.\n\nSo, for the 20th century after 1977, we have winners from 1980, 1983, 1986, 1989, 1992, 1995, 1998.\n\nAmong these, only Claus Peter Flor from Germany (West Germany) is from a country that no longer exists.\n\nTherefore, the first name is Claus Peter.\n\nBut let's make sure there aren't any others. For example, if there's a winner from a country that has been dissolved, like the former Czechoslovakia or something, but none are listed.\n\nAlso, let's check if any of the nationalities are from countries that no longer exist. For example, if there's a winner from a country that has been renamed or dissolved.\n\nAnother possibility is if there's a winner from a country that is now part of another country. For example, if someone was from a country that was absorbed by another.\n\nBut looking at the list, I don't see any such cases.\n\nSo, the only one is Claus Peter Flor.\n\nBut let's double-check the nationality. The list says \"Germany\" for Claus Peter Flor, but at the time, it was West Germany.\n\nAlso, let's confirm if there are any other winners from countries that no longer exist. For example, if there's a winner from a country that was split up or something.\n\nAnother one could be if there's a winner from a country that is now extinct, like the Soviet Union, but none are listed.\n\nAlso, let's check if any of the countries have changed names. For example, Brazil was often called Brazil, but it's the same country.\n\nSo, I think Claus Peter Flor is the correct answer.\n\nBut let's see if there's any other winner that might qualify. For example, if there's a winner from a country that no longer exists, like the former Yugoslavia or something.\n\nBut in the list, I don't see any such nationalities.\n\nTherefore, the first name is Claus Peter.\n\nBut let's make sure about the timing. The question says \"from the 20th Century (after 1977)\". So, 1983 is after 1977 and in the 20th century.\n\nAlso, the nationality on record is a country that no longer exists. West Germany no longer exists.\n\nSo, Claus Peter Flor should be the answer.\n\nBut let's confirm if there are any other winners. For example, if there's a winner from a country that has been dissolved, like the former East Germany or something, but East Germany was dissolved in 1990, and the winners after that would be from unified Germany.\n\nBut Claus Peter Flor is from 1983, which is before 1990, so he was from West Germany.\n\nAnother winner could be from 1992, Jin Wang from Austria. Austria still exists, so that's not it.\n\nSimilarly, Jan Wagner from Venezuela in 1995: Venezuela exists.\n\nSeikyo Kim from Japan in 1998: Japan exists.\n\nSo, only Claus Peter Flor from Germany (West Germany) fits.\n\nTherefore, the first name is Claus Peter.\n\nBut let's make sure about the full name. The list says \"Claus Peter Flor\" – that's the first name Claus Peter.\n\nBut the question asks for the first name, so it should be \"Claus Peter\".\n\nHowever, sometimes in competitions, people might have different names or titles, but I think \"Claus Peter\" is the first name.\n\nAlso, let's check if there's any other winner from a country that no longer exists. For example, if there's a winner from a country that was split, like Czechoslovakia, but none are listed.\n\nSo, I think Claus Peter Flor is the correct answer.\n"
  }
]
    conversations = to_openai_messages(conversations)

    # -------- device & dtype --------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # -------- load model --------
    # ---------- 1) 准备模型/分词器 ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    tools = [{'type': 'function', 'function': {'name': 'mcp-code-executor.execute_code', 'description': 'Execute Python code in the conda environment. For short code snippets only. For longer code, use initialize_code_file and append_to_code_file instead.', 'parameters': {'type': 'object', 'properties': {'code': {'type': 'string', 'description': 'Python code to execute', 'enum': None}, 'filename': {'type': 'string', 'description': 'Optional: Name of the file to save the code (default: generated UUID)', 'enum': None}}, 'required': ['code']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.initialize_code_file', 'description': 'Create a new Python file with initial content. Use this as the first step for longer code that may exceed token limits. Follow with append_to_code_file for additional code.', 'parameters': {'type': 'object', 'properties': {'content': {'type': 'string', 'description': 'Initial content to write to the file', 'enum': None}, 'filename': {'type': 'string', 'description': 'Optional: Name of the file (default: generated UUID)', 'enum': None}}, 'required': ['content']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.append_to_code_file', 'description': 'Append content to an existing Python code file. Use this to add more code to a file created with initialize_code_file, allowing you to build up larger code bases in parts.', 'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string', 'description': 'Full path to the file', 'enum': None}, 'content': {'type': 'string', 'description': 'Content to append to the file', 'enum': None}}, 'required': ['file_path', 'content']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.execute_code_file', 'description': 'Execute an existing Python file. Use this as the final step after building up code with initialize_code_file and append_to_code_file.', 'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string', 'description': 'Full path to the Python file to execute', 'enum': None}}, 'required': ['file_path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.read_code_file', 'description': 'Read the content of an existing Python code file. Use this to verify the current state of a file before appending more content or executing it.', 'parameters': {'type': 'object', 'properties': {'file_path': {'type': 'string', 'description': 'Full path to the file to read', 'enum': None}}, 'required': ['file_path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.install_dependencies', 'description': 'Install Python dependencies in the conda environment', 'parameters': {'type': 'object', 'properties': {'packages': {'type': 'array', 'description': 'List of packages to install', 'enum': None}}, 'required': ['packages']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.check_installed_packages', 'description': 'Check if packages are installed in the conda environment', 'parameters': {'type': 'object', 'properties': {'packages': {'type': 'array', 'description': 'List of packages to check', 'enum': None}}, 'required': ['packages']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.configure_environment', 'description': 'Change the environment configuration settings', 'parameters': {'type': 'object', 'properties': {'type': {'type': 'string', 'description': 'Type of Python environment', 'enum': ['conda', 'venv', 'venv-uv']}, 'conda_name': {'type': 'string', 'description': "Name of the conda environment (required if type is 'conda')", 'enum': None}, 'venv_path': {'type': 'string', 'description': "Path to the virtualenv (required if type is 'venv')", 'enum': None}, 'uv_venv_path': {'type': 'string', 'description': "Path to the UV virtualenv (required if type is 'venv-uv')", 'enum': None}}, 'required': ['type']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-code-executor.get_environment_config', 'description': 'Get the current environment configuration', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'strict': False}}, {'type': 'function', 'function': {'name': 'mcp-server-commands.run_command', 'description': 'Run a command on this linux machine', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'Command with args', 'enum': None}, 'workdir': {'type': 'string', 'description': 'Optional, current working directory', 'enum': None}, 'stdin': {'type': 'string', 'description': "Optional, text to pipe into the command's STDIN. For example, pass a python script to python3. Or, pass text for a new file to the cat command to create it!", 'enum': None}}, 'required': ['command']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.read_file', 'description': 'Read the complete contents of a file as text. DEPRECATED: Use read_text_file instead.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}, 'tail': {'type': 'number', 'description': 'If provided, returns only the last N lines of the file', 'enum': None}, 'head': {'type': 'number', 'description': 'If provided, returns only the first N lines of the file', 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.read_text_file', 'description': "Read the complete contents of a file from the file system as text. Handles various text encodings and provides detailed error messages if the file cannot be read. Use this tool when you need to examine the contents of a single file. Use the 'head' parameter to read only the first N lines of a file, or the 'tail' parameter to read only the last N lines of a file. Operates on the file as text regardless of extension. Only works within allowed directories.", 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}, 'tail': {'type': 'number', 'description': 'If provided, returns only the last N lines of the file', 'enum': None}, 'head': {'type': 'number', 'description': 'If provided, returns only the first N lines of the file', 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.read_media_file', 'description': 'Read an image or audio file. Returns the base64 encoded data and MIME type. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.read_multiple_files', 'description': "Read the contents of multiple files simultaneously. This is more efficient than reading files one by one when you need to analyze or compare multiple files. Each file's content is returned with its path as a reference. Failed reads for individual files won't stop the entire operation. Only works within allowed directories.", 'parameters': {'type': 'object', 'properties': {'paths': {'type': 'array', 'description': None, 'enum': None}}, 'required': ['paths']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.write_file', 'description': 'Create a new file or completely overwrite an existing file with new content. Use with caution as it will overwrite existing files without warning. Handles text content with proper encoding. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}, 'content': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['path', 'content']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.edit_file', 'description': 'Make line-based edits to a text file. Each edit replaces exact line sequences with new content. Returns a git-style diff showing the changes made. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}, 'edits': {'type': 'array', 'description': None, 'enum': None}, 'dryRun': {'type': 'boolean', 'description': 'Preview changes using git-style diff format', 'enum': None}}, 'required': ['path', 'edits']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.create_directory', 'description': 'Create a new directory or ensure a directory exists. Can create multiple nested directories in one operation. If the directory already exists, this operation will succeed silently. Perfect for setting up directory structures for projects or ensuring required paths exist. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.list_directory', 'description': 'Get a detailed listing of all files and directories in a specified path. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is essential for understanding directory structure and finding specific files within a directory. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.list_directory_with_sizes', 'description': 'Get a detailed listing of all files and directories in a specified path, including sizes. Results clearly distinguish between files and directories with [FILE] and [DIR] prefixes. This tool is useful for understanding directory structure and finding specific files within a directory. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}, 'sortBy': {'type': 'string', 'description': 'Sort entries by name or size', 'enum': ['name', 'size']}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.directory_tree', 'description': "Get a recursive tree view of files and directories as a JSON structure. Each entry includes 'name', 'type' (file/directory), and 'children' for directories. Files have no children array, while directories always have a children array (which may be empty). The output is formatted with 2-space indentation for readability. Only works within allowed directories.", 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.move_file', 'description': 'Move or rename files and directories. Can move files between directories and rename them in a single operation. If the destination exists, the operation will fail. Works across different directories and can be used for simple renaming within the same directory. Both source and destination must be within allowed directories.', 'parameters': {'type': 'object', 'properties': {'source': {'type': 'string', 'description': None, 'enum': None}, 'destination': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['source', 'destination']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.search_files', 'description': "Recursively search for files and directories matching a pattern. Searches through all subdirectories from the starting path. The search is case-insensitive and matches partial names. Returns full paths to all matching items. Great for finding files when you don't know their exact location. Only searches within allowed directories.", 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}, 'pattern': {'type': 'string', 'description': None, 'enum': None}, 'excludePatterns': {'type': 'array', 'description': None, 'enum': None}}, 'required': ['path', 'pattern']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.get_file_info', 'description': 'Retrieve detailed metadata about a file or directory. Returns comprehensive information including size, creation time, last modified time, permissions, and type. This tool is perfect for understanding file characteristics without reading the actual content. Only works within allowed directories.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['path']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'filesystem.list_allowed_directories', 'description': 'Returns the list of directories that this server is allowed to access. Subdirectories within these allowed directories are also accessible. Use this to understand which directories and their nested paths are available before trying to access files.', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_create_instance', 'description': 'Create a new browser instance', 'parameters': {'type': 'object', 'properties': {'browserType': {'type': 'string', 'description': 'Browser type', 'enum': ['chromium', 'firefox', 'webkit']}, 'headless': {'type': 'boolean', 'description': 'Whether to run in headless mode', 'enum': None}, 'viewport': {'type': 'object', 'description': 'Viewport size', 'enum': None}, 'userAgent': {'type': 'string', 'description': 'User agent string', 'enum': None}, 'metadata': {'type': 'object', 'description': 'Instance metadata', 'enum': None}}, 'required': []}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_list_instances', 'description': 'List all browser instances', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_close_instance', 'description': 'Close the specified browser instance', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_close_all_instances', 'description': 'Close all browser instances', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_navigate', 'description': 'Navigate to a specified URL', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'url': {'type': 'string', 'description': 'Target URL', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}, 'waitUntil': {'type': 'string', 'description': 'Wait condition', 'enum': ['load', 'domcontentloaded', 'networkidle']}}, 'required': ['instanceId', 'url']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_go_back', 'description': 'Go back to the previous page', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_go_forward', 'description': 'Go forward to the next page', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_refresh', 'description': 'Refresh the current page', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_click', 'description': 'Click on a page element', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'button': {'type': 'string', 'description': 'Mouse button', 'enum': ['left', 'right', 'middle']}, 'clickCount': {'type': 'number', 'description': 'Number of clicks', 'enum': None}, 'delay': {'type': 'number', 'description': 'Click delay in milliseconds', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_type', 'description': 'Type text into an element', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'text': {'type': 'string', 'description': 'Text to input', 'enum': None}, 'delay': {'type': 'number', 'description': 'Input delay in milliseconds', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector', 'text']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_fill', 'description': 'Fill a form field', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'value': {'type': 'string', 'description': 'Value to fill', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector', 'value']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_select_option', 'description': 'Select an option from a dropdown', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'value': {'type': 'string', 'description': 'Value to select', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector', 'value']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_get_page_info', 'description': 'Get detailed page information including full HTML content, page statistics, and metadata', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_get_element_text', 'description': 'Get element text content', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_get_element_attribute', 'description': 'Get element attribute value', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'attribute': {'type': 'string', 'description': 'Attribute name', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector', 'attribute']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_screenshot', 'description': 'Take a screenshot of the page or element', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'fullPage': {'type': 'boolean', 'description': 'Whether to capture the full page', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector (capture specific element)', 'enum': None}, 'type': {'type': 'string', 'description': 'Image format', 'enum': ['png', 'jpeg']}, 'quality': {'type': 'number', 'description': 'Image quality (1-100, JPEG only)', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_wait_for_element', 'description': 'Wait for an element to appear', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'selector': {'type': 'string', 'description': 'Element selector', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId', 'selector']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_wait_for_navigation', 'description': 'Wait for page navigation to complete', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'timeout': {'type': 'number', 'description': 'Timeout in milliseconds', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_evaluate', 'description': 'Execute JavaScript code in the page context', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'script': {'type': 'string', 'description': 'JavaScript code to execute', 'enum': None}}, 'required': ['instanceId', 'script']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'concurrent-browser-mcp.browser_get_markdown', 'description': 'Get page content in Markdown format, optimized for large language models', 'parameters': {'type': 'object', 'properties': {'instanceId': {'type': 'string', 'description': 'Instance ID', 'enum': None}, 'includeLinks': {'type': 'boolean', 'description': 'Whether to include links', 'enum': None}, 'maxLength': {'type': 'number', 'description': 'Maximum content length in characters', 'enum': None}, 'selector': {'type': 'string', 'description': 'Optional CSS selector to extract content from specific element only', 'enum': None}}, 'required': ['instanceId']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'search-fusion-mcp.search', 'description': 'Execute web search and return results\n            \n            Args:\n                query: Search query terms\n                num_results: Number of results to return, default 10\n                engine: Search engine type, options:\n                    - "auto": Automatically select best available search engine (default)\n                    - "google": Prioritize Google search (requires API key)\n                    - "serper": Prioritize Serper search (requires API key)\n                    - "jina": Prioritize Jina AI search\n                    - "duckduckgo": Prioritize DuckDuckGo search\n                    - "exa": Prioritize Exa search (requires API key)\n                    - "bing": Prioritize Bing search (requires API key)\n                    - "baidu": Prioritize Baidu search (requires API key)\n            ', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string', 'description': None, 'enum': None}, 'num_results': {'type': 'integer', 'description': None, 'enum': None}, 'engine': {'type': 'string', 'description': None, 'enum': None}}, 'required': ['query']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'search-fusion-mcp.fetch_url', 'description': 'Fetch web content with intelligent pagination support\n            \n            Args:\n                url: Web URL to fetch\n                use_jina: Whether to prioritize Jina Reader for LLM-optimized content, default True\n                with_image_alt: Whether to generate alt text descriptions for images, default False\n                max_length: Maximum content length per page, auto-paginate if exceeded, default 50000 characters\n                page_number: Specific page to retrieve (starting from 1), default 1\n            ', 'parameters': {'type': 'object', 'properties': {'url': {'type': 'string', 'description': None, 'enum': None}, 'use_jina': {'type': 'boolean', 'description': None, 'enum': None}, 'with_image_alt': {'type': 'boolean', 'description': None, 'enum': None}, 'max_length': {'type': 'integer', 'description': None, 'enum': None}, 'page_number': {'type': 'integer', 'description': None, 'enum': None}}, 'required': ['url']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'search-fusion-mcp.get_available_engines', 'description': 'Get list of currently available search engines and their status', 'parameters': {'type': 'object', 'properties': {}, 'required': []}, 'strict': False}}, {'type': 'function', 'function': {'name': 'search-fusion-mcp.search_wikipedia', 'description': 'Search Wikipedia page content\n            \n            Args:\n                entity: Entity to search for (people, places, concepts, events, etc.)\n                first_sentences: Number of first sentences to return (set to 0 for full content), default 10\n            ', 'parameters': {'type': 'object', 'properties': {'entity': {'type': 'string', 'description': None, 'enum': None}, 'first_sentences': {'type': 'integer', 'description': None, 'enum': None}}, 'required': ['entity']}, 'strict': False}}, {'type': 'function', 'function': {'name': 'search-fusion-mcp.search_archived_webpage', 'description': 'Search archived versions of websites using Wayback Machine\n            \n            Args:\n                url: Website URL to search\n                year: Target year (optional)\n                month: Target month (optional)\n                day: Target day (optional)\n            ', 'parameters': {'type': 'object', 'properties': {'url': {'type': 'string', 'description': None, 'enum': None}, 'year': {'type': 'integer', 'description': None, 'enum': None}, 'month': {'type': 'integer', 'description': None, 'enum': None}, 'day': {'type': 'integer', 'description': None, 'enum': None}}, 'required': ['url']}, 'strict': False}}]



    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype if device.type == "cuda" else torch.float32,
        device_map="cuda:0",
        trust_remote_code=args.trust_remote_code,
        # attn_implementation="eager",  # <<< 关键：强制 eager
    )
    model.eval()

    # ---------- 2) 构造输入 ----------
    prompt = tokenizer.apply_chat_template(conversations, tools=tools, tokenize=False)

    # NOTE: debugging
    # prompt = prompt[:1000]
    #
    enc = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    # ---------- 3) 打补丁并前向 ----------
    turns_list = get_turns(inputs['input_ids'], inputs['attention_mask'], tokenizer)

    cache = QKCache(keep_all_layers=True)  # 或 target_last_layer = model.config.num_hidden_layers - 1
    patch, restore = make_patcher(cache)
    patch()
    with torch.no_grad():
        _ = model(**inputs, use_cache=False, return_dict=True, output_attentions=False)
        # _ = model(**counterfactual_inputs, use_cache=False, return_dict=True, output_attentions=False)
    restore()

    # ---------- 4) 取“最后一层”的 Q/K ----------
    L_last = model.config.num_hidden_layers-1


    q_last = _pick_kth_layer(cache.q, L_last)[:1]  # [1,Hq,T,Dh]
    k_last = _pick_kth_layer(cache.k, L_last)[:1]  # [1,Hkv,T,Dh]

    s_ref, A_ref = accumulate_last_layer_stats_noTT_turns(
        q_last, k_last,
        turns=turns_list,  # 单样本
        g_abs=None,
        alpha=1.0
    )

    s_ref = s_ref / s_ref.sum()
    #
    A_ref = A_ref / (A_ref.sum(0).unsqueeze(0) + 1e-5)
    
    print('=========inputs========')
    print(s_ref)

    print(A_ref)
    
    
    
    
    print('=======counterfactual_inputs========')
    counterfactual_inputs = get_counterfactual_input_ids(inputs['input_ids'], inputs['attention_mask'], turns_list,tokenizer)

    cache = QKCache(keep_all_layers=True)  # 或 target_last_layer = model.config.num_hidden_layers - 1
    patch, restore = make_patcher(cache)
    patch()
    with torch.no_grad():
        _ = model(**counterfactual_inputs, use_cache=False, return_dict=True, output_attentions=False)
    restore()

    # ---------- 4) 取“最后一层”的 Q/K ----------
    L_last = model.config.num_hidden_layers - 1

    q_last = _pick_kth_layer(cache.q, L_last)[:1]  # [1,Hq,T,Dh]
    k_last = _pick_kth_layer(cache.k, L_last)[:1]  # [1,Hkv,T,Dh]

    s_ref_counterfactual, A_ref_counterfactual = accumulate_last_layer_stats_noTT_turns(
        q_last, k_last,
        turns=turns_list,  # 单样本
        g_abs=None,
        alpha=1.0
    )

    s_ref_counterfactual = s_ref_counterfactual / s_ref_counterfactual.sum()
    #
    A_ref_counterfactual = A_ref_counterfactual / (A_ref_counterfactual.sum(0).unsqueeze(0) + 1e-5)
    #
    #
    print(s_ref_counterfactual)

    print(A_ref_counterfactual)

    # # ==========================================================================
    print('=======relative_improvements========')
    print(s_ref / (s_ref_counterfactual + 1e-5))
    print(A_ref / (A_ref_counterfactual + 1e-5) )
    # 以下是利用attention矩阵直接来计算
    # s_list, A_list, Amean = credit_from_attn_matrix_turns(
    #     attn_last,  # [B,H,T,T]
    #     turns_list,  # len=B
    #     g_abs_list=g_abs_list,  # len=B
    #     alpha=1.0
    # )
    # #
    # print(torch.allclose(s_list[0], s_ref, atol=1e-3, rtol=1e-3),
    #       torch.allclose(A_list[0], A_ref, atol=1e-2, rtol=1e-2))




if __name__ == "__main__":
    main()
