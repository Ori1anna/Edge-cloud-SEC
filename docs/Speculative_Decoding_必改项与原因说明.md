# Speculative Decoding（简化版）必改项与原因说明

> 目标：让当前实现严格满足你的三条约束  
> 1) **仅用一种 uncertainty metric（熵）**做门控，便于调试；  
> 2) **Cloud 验证**采用工程捷径：一次前向（prefill）→ 概率阈值判定 → **首错就地以 p(·) 采样纠正**；  
> 3) **不传音频出端**（可先“不只传 token id”，但**绝不上传音频特征**）。

---

## 改动总览（一眼看懂）

1. **禁止云端携带音频特征**（必须）  
   - 方案 A（推荐）：在调用侧**只传 `input_ids/attention_mask`** 给云端验证；  
   - 方案 B：在云端 `verify_tokens` 内部**禁用**接收 `input_features/*`。  
   - **原因**：当前实现会把 `input_features` 一并传上云（与“音频不出端”冲突）。

2. **统一云端 prefill 的实现为 `forward(logits)` 版本**（建议）  
   - 保留“**一次前向 → 阈值判定 → 首错采样纠正**”的逻辑；  
   - 移除/停用 `generate(..., max_new_tokens=0, output_scores=True)` 的取分布路径，避免歧义与不必要的开销。

3. **防止把 `None` 追加进上下文**（已修，确认保留）  
   - 写回时仅在 `correction_token is not None` 时追加。

4. **返回值统一为 `(text, metrics)`**（避免测试崩）  
   - 工程里存在两个结尾版本：一个只返回文本、一个返回二元组；请**统一为二元组**以匹配测试用例。

5. **阈值建议与校准**（强烈推荐）  
   - 将 `prob_threshold` 初值调低至 **0.20–0.30**（建议 0.25），在 dev 集做一次分位数扫以命中**目标接受率 0.6–0.8**；  
   - 说明：测试里用 `prob_threshold=0.8` 更偏“探索/压测”配置，评测期会导致频繁首位拒绝。

6. **设备与张量对齐**（已覆盖，继续遵循）  
   - 在云端 `verify_tokens` 内对 `context` 张量统一 `.to(self.device)`，避免 device mismatch。

7. **门控与采样一致性**（保持现状）  
   - 门控仅用**熵**，逻辑清晰；  
   - 确保 **edge 的 q** 与 **cloud 的 p** 采样参数（`temperature/top_p/repetition_penalty`）**一致**，避免接受率异常。

---

## 逐项修改与示例补丁

### 1) 禁止云端携带音频特征（必须）

**改法 A（调用侧过滤，推荐）** — 在调用 `verify_tokens` 前，构造**瘦身上下文**：
```python
# speculative_decoding.py 里，调用云端前：
ctx4cloud = {
    "input_ids": current_context["input_ids"],
    "attention_mask": current_context["attention_mask"],
}
accepted_tokens, correction_token, needs_correction =     self.cloud_model.verify_tokens(ctx4cloud, draft_tokens, self.prob_threshold)
```
> 这样即使 `current_context` 内含有 `input_features/*`，也不会被上传。

**改法 B（云端内禁用，也可叠加）** — 删/注如下代码：
```python
# cloud_model.py 的 verify_tokens 内
# extra = {}
# if 'input_features' in context:
#     extra['input_features'] = context['input_features'].to(self.device)
# if 'feature_attention_mask' in context:
#     extra['feature_attention_mask'] = context['feature_attention_mask'].to(self.device)
```
> 无论上层传不传，云端都**不会**把音频特征带入前向。

**原因**：满足“音频不出端”的硬约束；同时也让验证逻辑**只基于文本 token**，与“只传 token（或必要 tokens）”一致。

---

### 2) 统一云端 prefill 为 `forward` 版本（建议）

**保留的核心逻辑**（已实现）：  
- `logits = model(input_ids=ctx+y, ...).logits`；  
- 令 `m = len(ctx)`、`k = len(y)`，第 `i` 个草稿 `y_i` 的概率在 `logits[m-1+i]`；  
- 若 `p_i ≥ T_pass` → 接受；否则在该位 `softmax(logits[m-1+i])` 上**采样纠正**并**结束本轮**。

**建议移除/停用**：通过 `generate(..., max_new_tokens=0, output_scores=True)` 取 `scores` 的等价路径，避免两套实现并存。

**原因**：  
- `forward` 更直接、可控、少胶水；  
- 避免未来误将 `max_new_tokens=0` 的 `scores` 当成“草稿位分布”（历史上常见踩坑点）；  
- 统一实现便于计时与报表（`cloud_prefill_ms`）。

---

### 3) 防止把 `None` 追加进上下文（已修，保留）

在“首错纠正”分支里请确保只在 `correction_token is not None` 时才追加：
```python
append_list = accepted_tokens + ([correction_token] if correction_token is not None else [])
current_context = self._update_context(current_context, append_list)
```

**原因**：极端情况下（全通过或异常），避免把 `None` 作为 token id 拼到 `input_ids` 里导致崩溃。

---

### 4) 统一 `generate` 返回值为 `(text, metrics)`（必须与测试匹配）

工程里可见两个版本：
- 只返回文本：`return generated_text`（会让测试 `spec_text, spec_metrics = ...` 报错）。  
- 返回二元组：`return generated_text, metrics`（**应保留此版**）。

**原因**：测试脚本显式解包两项，且指标对后续分析（接受率、纠正率、云调用次数等）很关键。

---

### 5) `prob_threshold` 的默认值与校准（强烈推荐）

**建议**：初值设为 **0.25**；在 dev 集做一次小网格扫：
```python
for T in [0.15, 0.20, 0.25, 0.30, 0.35]:
    acc_rate, total_time = eval_on_dev(prob_threshold=T)
    # 以 0.6~0.8 的目标接受率为主，时延为辅选 T
```
**原因**：  
- 过高阈值会导致**几乎每轮首位拒绝**，使吞吐恶化且纠正率畸高；  
- 通过“目标接受率”对齐阈值，有利于不同数据/模型下的**稳定可比**。

---

### 6) 设备与张量对齐（延续保持）

云端内将 `context` 张量 `.to(self.device)`，确保与云端模型设备一致；同时边端/云端的 `temperature/top_p/repetition_penalty` 等采样配置要**完全一致**。

**原因**：跨设备拼接 `input_ids`/mask 会直接抛 `RuntimeError`；采样配置不一致会让接受率异常偏低。

---

### 7) 门控与采样一致性（检查项）

- **门控**：当前仅用**熵**，实现清晰，便于调试与阈值调整（保持）。  
- **采样一致性**：务必确保边端与云端 **temperature/top_p/repetition_penalty** 等参数一致（避免因分布错位导致接受率骤降）。

---

## 变更后，机制与约束的对应关系

- **工程捷径验证**：一次前向 → 阈值验收 → **首错采样纠正** ✅（云端 `forward` 版）。  
- **只传 tokens，不传音频**：调用侧过滤或云端禁用 ✅。  
- **单一不确定性指标**：熵门控 ✅。  
- **Qwen-2.5-Omni 3B/7B 适配**：两侧均用 Omni 模型与处理器，tokenizer 对齐 ✅。

---

## 最终 Checklist（提交前自检）

- [ ] 云端验证**不再**接收任何 `input_features/*`（调用侧或云端侧屏蔽）  
- [ ] 云端验证统一走 `forward(logits)` 路径；位移索引使用 `m-1+i`  
- [ ] 只在 `correction_token is not None` 时写回该 token  
- [ ] `generate(...)` 统一返回 `(text, metrics)`  
- [ ] `prob_threshold` 初值 0.25，并在 dev 集完成阈值校准  
- [ ] 保证 edge/cloud 采样参数一致；门控仅用熵  
- [ ] （可选）记录 `k, accepted_len, la/k, cloud_prefill_ms, rtt_ms` 便于后续调参
