# Speculative Decoding 完整实践指南

> 面向工程实现的 .md 版本：涵盖直觉、算法、消息协议、伪代码、超参、日志指标、常见坑与变体。可直接作为实现手册使用。

---

## 0. TL;DR（一句话）
用一个**小草稿模型**（Draft，分布 \(q\)）一次性“猜”出 \(k\) 个 token；再用**大目标模型**（Verifier，分布 \(p\)）做**一次前向（prefill）**并逐位验收：满足 \(\alpha=\min(1, p/q)\) 的位置被接受，遇到首个不通过的位置就由 \(p\) **纠正**一个 token；把“被接受的前缀 + 纠正 token”追加到上下文，循环往复。目标：**在不牺牲分布正确性/质量的前提下，把大模型调用次数大幅减少**。

---

## 1. 何时该用（场景）
- **长输出**（对话/摘要/代码/字幕等）希望提高吞吐或降低成本。
- 有一个与大模型**风格相近**的小模型（蒸馏/量化版），并能共享 tokenizer 与采样策略。
- **端–云协作**：小模型在端侧起草、云侧大模型验证；只上传 **token-ID**，不传原始音频/图像。

---

## 2. 记号与前提
- 上下文：\(x_{1:t}\)
- 草稿序列：\(y_{1:K}\)（一次起草 \(K=k\) 个 token）
- 模型分布：草稿 \(q(\cdot)\)，验证 \(p(\cdot)\)
- 采样参数：**必须一致**（temperature、top-p、repetition penalty、bad-words 等）

---

## 3. 标准算法（Draft → Verify → Align）

### 3.1 步骤概览
1. **Edge/Draft：** 用小模型 \(q\) 从当前上下文自回归采样 \(k\) 个 token：\(y_1,\dots,y_k\)。记录每个被采样 token 的**标量概率** \(q_i = q(y_i\mid x_{1:t},y_{<i})\)。
2. **（可选）门控：** 若块内“难 token 比例”过高（见 §4），或命中规则（数字/日期/命名实体/少见词），则**升级到云**，否则可本地提交（最小闭环阶段可先“总是升云”）。
3. **Cloud/Prefill：** 用大模型 \(p\) 对 \([x_{1:t}, y_{1:k}]\) **一次前向**，得到每个位置下一个 token 的分布；取被草稿选中的 token 概率 \(p_i = p(y_i\mid x_{1:t},y_{<i})\)。
4. **逐位验收：** 对 \(i=1..k\)，计算 \(\alpha_i=\min(1, p_i/q_i)\)。抛硬币决定接受/拒绝；一旦首个位置 \(j\) 被拒：
   - **严格无偏**：从**残差分布** \(r(v)\propto p(v\mid\cdot)-\mathbb{1}[v=y_j]\cdot q_j\) 采样纠正 token \(z\)。
   - **工程捷径**：直接从 \(p(\cdot)\) 采样 \(z\)。
   然后**停止**检查后续位置。
5. **对齐与写回：** 把“被接受前缀 \(y_{1:La}\) +（若有）纠正 \(z\)”追加到两端上下文；同时更新 **KV cache**。进入下一轮。

### 3.2 正确性直觉
接受概率 \(\min(1,p/q)\) 是经典接受–拒绝（accept–reject）校正；首拒位置用残差分布补偿，保证“最终生成分布 = 仅用 \(p\) 自回归采样的分布”。工程捷径虽略破严格无偏，但通常质量几乎不变、实现简单。

---

## 4. 门控（是否升级到云）
> 目的：减少不必要的云验证调用，保留质量。

### 4.1 不确定性信号（逐 token）
- **NLL**：\(\text{nll}_i=-\log q(y_i\mid\cdot)\)
- **熵**：\(H_i=-\sum_v q(v)\log q(v)\)
- **Margin**：top-1 与 top-2 概率差 \(m_i=p_1-p_2\)（越小越不确定）
- **MC-dropout 方差**：在 \(q\) 上重复前向取方差

将其映射为“难 token”布尔掩码：
```
hard_i = (nll_i > T_nll) or (H_i > T_H) or (m_i < T_margin) or (mc_var_i > T_var)
```

### 4.2 块级判定
- **难比例**：`hard_ratio = (#hard_i)/k`
- 满足 `hard_ratio > R_hard` → 升云
- **规则必升**：数字/日期/金额、命名实体（NER）、少见词（基于词频阈值）命中则直接升云

> **建议**：最小闭环先不启用门控（始终升云），跑稳后再加入并调阈值。

---

## 5. 双端消息协议（建议 JSON/Protobuf）

### 5.1 Edge → Cloud（验证请求）
```json
{
  "ctx_id": "<会话ID>",
  "draft_tokens": [y1, ..., yk],
  "q_token_probs": [q1, ..., qk],
  "hard_mask": [0/1, ...],
  "k": k,
  "sampling_cfg": {"temp":1.0, "top_p":0.9, "rep_penalty":1.1}
}
```
> 只传**被采样 token 的标量概率**，不传整分布或原始音频。

### 5.2 Cloud → Edge（验证响应）
```json
{
  "accepted_len": La,
  "correction": z_or_null,
  "metrics": {"alpha_mean": 0.83, "la_over_k": 0.67}
}
```

---

## 6. 伪代码（最小可运行原型）

### 6.1 Edge 侧
```python
k = 6
while not finished:
    y, qprob = draft_k_tokens(q_model, ctx, k, sampling_cfg)
    req = {"ctx_id": ctx_id, "draft_tokens": y, "q_token_probs": qprob,
           "k": k, "sampling_cfg": sampling_cfg}
    resp = rpc_cloud_verify(req)
    La, z = resp["accepted_len"], resp["correction"]

    # 对齐 & 写KV
    ctx.extend(y[:La])
    if z is not None:
        ctx.append(z)
    update_kv_cache_edge(y[:La], z)

    # 动态k（根据最近几轮 la/k 调整）
    ratio = La / k
    if ratio >= 0.8: k = min(k+1, 12)
    elif ratio <= 0.4: k = max(k-1, 3)
```

### 6.2 Cloud 侧
```python
def cloud_verify(req):
    y, k, qprob = req["draft_tokens"], req["k"], req["q_token_probs"]
    logits_list = p_model.prefill(ctx + y)  # 一次前向，长度k
    accepted = 0
    for i in range(k):
        p_i = softmax(logits_list[i])[y[i]]
        # 工程捷径：也可用 p_i >= T_pass 或 p_i/qprob[i] >= T_ratio
        if p_i / max(qprob[i], 1e-9) >= 1.0:
            accepted += 1
        else:
            z = sample_from_p(logits_list[i])  # 或残差分布
            # 写KV：只基于被接受的前缀 + z
            update_kv_cache_cloud(y[:accepted], z)
            return {"accepted_len": accepted, "correction": z}
    update_kv_cache_cloud(y, None)
    return {"accepted_len": k, "correction": None}
```

---

## 7. KV cache 与内存要点
- **Prefill**：一次性把 `ctx + y` 的每步分布算出，并为接受的部分写入 KV。
- **Decode**：每追加一个新 token，只算当步 Query，与缓存的历史 K/V 做注意力。
- **内存**：KV 占用随（层数 × kv_heads × head_dim × 序列长度 × dtype）线性增长；长上下文/并发时是显存瓶颈。必要时启用 **GQA/MQA**、**Paged KV**、**滑动窗口**。

---

## 8. 默认超参（可直接起跑）
- 采样：`temp=1.0, top_p=0.9, repetition_penalty=1.1`（q 与 p 完全一致）
- 起始 `k=6`，范围 `[3,12]`，动态调整阈值：`↑: la/k≥0.8; ↓: la/k≤0.4`
- 门控：起步**关闭**；开启后：`T_nll=val集80分位`，`T_H≈2.5bits`，`T_margin=0.2`，`R_hard=0.3`
- 验证：先用**工程捷径**，跑通后切换到**严格无偏**（\(\min(1,p/q)\)+残差采样）
- 超时兜底：云超时>400ms 时，临时只提交草稿首 token 以保持前进

---

## 9. 在线日志与指标（最小集合）
- `k, accepted_len, la/k, alpha_mean`
- `edge_time_ms（起草k） , cloud_prefill_ms（一次前向）, rtt_ms`
- `A_cum`（累计接受率 = 累计被接受 token / 累计验证 token）
- `bytes_up/down`（只传 token-ID 时通常极小）
- （若启门控）`hard_ratio, triggered?`

> 评测对比（edge_only / cloud_only / speculative）可加入：**TTFT/Total(p50/p95)**、**OTPS**、**CPU Peak RSS / GPU Peak（alloc & reserved）**、**KV 峰值与占比**、**上/下行字节**、**速度提升倍数**、**云调用比**。

---

## 10. 常见坑与排查
- **分布不一致**：q 与 p 的采样温度、top-p、惩罚、bad-words 任一不一致都会显著降低接受率；务必统一。
- **k 过大**：经常中途拒绝会浪费草稿算力；用动态 k。
- **门控阈值偏移**：过严→频繁升云、带宽高；过松→质量/稳定性差。
- **KV 爆显存**：注意并发、上下文长度与精度；启用 GQA/MQA 或滑窗。
- **随机性不可重复**：固定 RNG 种子与 coin-flip 顺序，便于对齐与回溯。

---

## 11. 变体速览
- **Self-Speculative（同模自证）**：不引入第二个模型，由单模型产生候选再自证；部署简单，提速略逊。
- **Tree/Medusa 风格**：一次生成多分支候选树，用主干验证；更高并行度，实现复杂。
- **EAGLE/Lookahead**：训练一个预测未来 logits 的辅助头当草稿器。

---

## 12. 简要正确性小结（严格无偏版）
- 验收因子 \(\min(1,p/q)\) 确保被接受 token 的边际分布与 \(p\) 一致；
- 首拒位置用残差分布采样，补偿“被拒 token 的 q 质量”，整体序列分布与仅用 \(p\) 逐 token 采样**等价**。

---

## 13. 清单（Checklist）
- [ ] q 与 p 共享 tokenizer 与采样参数
- [ ] 起草器能输出 \([y_{1:k}], [q_i]\)
- [ ] 云端能做一次前向并逐位取 \(p_i\)
- [ ] 验收与纠正（工程或无偏）
- [ ] 双端 KV 与上下文对齐
- [ ] 动态 k 与（可选）门控
- [ ] 日志：`k, La, la/k, A_cum, prefill_ms, rtt_ms`
- [ ] 评测：质量、时延、内存、带宽、协作指标齐全

---

> 备注：本指南面向工程快速落地。若撰写论文/报告，建议补充与“纯 \(p\) 自回归采样”**分布一致性的推导**与**复杂度分析**（大模型前向调用次数、prefill 并行度、端–云通信量上界）。

