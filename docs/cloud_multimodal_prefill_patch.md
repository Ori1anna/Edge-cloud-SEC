
# Cloud 端 *multimodal prefill* 补丁方案（Qwen‑2.5‑Omni｜结合你当前实现）

> 目标：当 **uncertainty 超阈值**触发“升云”时，让 **Cloud（7B）** 在 *prefill/verification* 阶段**真正使用多模态输入**（文本 + 音频特征），并用**显式前向**稳健地取得草稿位置的 logits 做“阈值验收 + 首错纠正”。

---

## 1) 现在的问题是什么

1. **Cloud 验证未消费音频特征**  
   你在 `verify_tokens(context, draft_tokens, threshold)` 里只把 `input_ids/attention_mask` 传进了 Cloud 的 `generate(...)`，**没有**传 `context["input_features"] / context["feature_attention_mask"]`。因此 Cloud 在验证时**“听不到音频”**，与 Edge（3B）起草时的条件不一致，容易导致接受率偏低。  
   参考：Qwen‑2.5‑Omni 是端到端多模态模型，音频等模态需随文本一并提供（由 `Qwen2_5OmniProcessor` 产出 `input_features` / `feature_attention_mask`）。citeturn0search0turn0search1

2. **`generate(max_new_tokens=0)` 不是通用的 *prefill logits* 入口**  
   Transformers 的 `generate(..., output_scores=True)` 的 `scores` 指“**每一步生成**”的 logits；当 `max_new_tokens=0` 时，`scores` 可能为空或与预期不符。要在“上下文 + 草稿”上拿到“**预测草稿每步的分布**”，应改为**显式 `forward`** 并从 `ModelOutput.logits` 中按位置对齐读取。citeturn0search2

3. **Omni 顶层 vs Thinker 子模型**  
   官方文档建议：只做**文本生成/验证**可使用 **`Qwen2_5OmniThinkerForConditionalGeneration`**（更省显存）；若需要在云端**利用音频特征条件化**，可继续用 **`Qwen2_5OmniForConditionalGeneration`** 并传入多模态特征。citeturn0search0

---

## 2) 修改方案

### 方案 A｜继续使用 **Omni 顶层**，改为 **显式 `forward` 多模态**（推荐）

**思路**  
- 在 Cloud 端把 `context` 中的 `input_ids/attention_mask` 与 `input_features/feature_attention_mask` **一起**喂给模型，**不走 `generate(max_new_tokens=0)`**，而是 `model(**inputs, return_dict=True, use_cache=False)` 直接取 `logits`。  
- 这样能在与 Edge 一致的多模态条件下，对“草稿块（k 个 token）”进行**阈值验收 + 首错纠正**。

**关键代码（`cloud_model.py::verify_tokens` 替换核心）：**
```python
@torch.no_grad()
def verify_tokens(self, context, draft_tokens, threshold: float):
    if not draft_tokens:
        return [], None, False

    device = next(self.model.parameters()).device
    # 1) 拼接“上下文 + 草稿”
    ctx_ids  = context["input_ids"].to(device)
    ctx_mask = context["attention_mask"].to(device)
    y = torch.tensor([draft_tokens], device=device, dtype=ctx_ids.dtype)

    full_ids  = torch.cat([ctx_ids,  y], dim=1)
    full_mask = torch.cat([ctx_mask, torch.ones_like(y)], dim=1)

    # 2) 组装多模态输入（文本 + 音频特征）
    mm_inputs = {"input_ids": full_ids, "attention_mask": full_mask}
    if "input_features" in context:
        mm_inputs["input_features"] = context["input_features"].to(device)
    if "feature_attention_mask" in context:
        mm_inputs["feature_attention_mask"] = context["feature_attention_mask"].to(device)

    # 3) 显式前向，稳健取得 logits
    out = self.model(**mm_inputs, return_dict=True, use_cache=False)
    logits = out.logits                 # [B, M+K, V]
    M = ctx_ids.shape[1]; K = len(draft_tokens)

    # 4) 逐位阈值验收 + 首错纠正（工程捷径）
    accepted = 0
    for i in range(K):
        pos   = M - 1 + i
        probs = torch.softmax(logits[0, pos], dim=-1)
        p_i   = probs[draft_tokens[i]].item()
        if p_i >= threshold:
            accepted += 1
        else:
            corr = torch.multinomial(probs, 1).item()
            return draft_tokens[:accepted], corr, True

    return draft_tokens, None, False
```

**为什么可行**  
- 完全遵循 Qwen‑2.5‑Omni 的多模态输入设计，Cloud 验证与 Edge 起草在**条件上保持一致**（都“听到”音频），接受率和稳定性更好。citeturn0search0  
- 规避 `generate(..., output_scores=True)` 在 `max_new_tokens=0` 下的“不确定行为”，前向语义清晰可控（`logits[:, M-1+i, :]` 对应预测 `y_i` 的分布）。citeturn0search2



---

## 3) 与你现有 Spec 流程如何对齐

1. **触发**：边端仍用熵门控（token‑level Shannon 熵）决定是否升云；  
2. **拼接**：把草稿 `y_{1:K}` 拼到上下文后做 Cloud 前向；  
3. **验证**：逐位比较 `p(y_i | context + y_{<i}) >= T_pass`，**首错即纠正**（从该位分布采样 1 个 token）；  
4. **KV 更新**：写回“已接受前缀 + 纠正 token”，继续下一轮；  
5. **音频**：方案 A 直接随 `mm_inputs` 上云；方案 B 改为在 prompt 中注入“音频线索文本”。

---

## 4) 常见坑与自检清单

- **位置对齐**：第 `i` 个草稿（从 1 开始）用 `pos = M - 1 + (i-1)`；  
- **阈值类型**：边端的熵阈值是 **nats**；Cloud 的判定阈值是 **概率**，不要混淆；  
- **处理器一致性**：Cloud 侧 `processor/tokenizer` 应与 7B checkpoint 匹配；  
- **是否需要语音输出**：如不需要，`return_audio=False`、或加载时关闭 talker 分支以省显存；  
- **`generate` 的 `scores` 语义**：仅对应“生成步”的 logits，非 prefill；prefill 建议显式 `forward`。citeturn0search2

---

## 5) 参考

- **Hugging Face 文档：Qwen‑2.5‑Omni（多模态输入、Thinker‑Talker 架构、处理器）**。citeturn0search0  
- **Qwen2.5‑Omni 官方仓库（模型说明、量化与部署提示）**。citeturn0search3  
- **Transformers `generate(..., output_scores=True)` 的 `scores` 语义（“生成步” logits）**。citeturn0search2
