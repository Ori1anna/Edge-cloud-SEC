# Speculative Decoding 简单版：代码审查与修复指南（.md）

> 目标：满足以下三点并让最小闭环稳定运行。
> 1) 仅使用 **一种 uncertainty metric**（便于调试）。  
> 2) **Cloud 验证**采用工程捷径：一次前向（prefill）→ 概率阈值判定 → 找到**首个未通过**的位置并按 \(p(\cdot)\) 直接采样**纠正**。  
> 3) 暂不确认 Qwen-Omni 是否仅传 token-id 可行，**先传必要 tokens**，**不传音频**。

---

## 一、总体结论
- 你的实现已经基本搭好“**熵门控 + 阈值验收 + 首错纠正**”的骨架。  
- 关键问题在 **Cloud 端 prefill 的 logits 取法**：当前用 `generate(..., max_new_tokens=0)` 取不到草稿位分布，导致退化为“全拒绝 + 近似随机纠正”。  
- 此外需要：
  1) **防止把 `None` 追加**进上下文；
  2) **把阈值下调**并做一次 **dev 集校准**（建议 0.20–0.30）。

---

## 二、必须修改 #1：Cloud 端 prefill 的 logits 取法
**问题**  
`generate(..., max_new_tokens=0)` 的 `outputs.scores` 只对应“新生成的 token”。当 `max_new_tokens=0` 时基本为空，你的代码会落到 dummy logits，`softmax(0)` 变均匀分布，阈值稍高就“首位拒绝”。

**正确做法**  
直接 **forward** 得到 `logits`，再做**位移对齐**：
- 上下文长度 `m = len(ctx)`；草稿长度 `k`。  
- 把 `ctx + y1..yk` 喂给模型，得到 `[1, m+k, V]` 的 `logits`。  
- `p(y1)` 用 `logits[0, m-1]`；`p(y2)` 用 `logits[0, m]`；一般地 `p(yi)` 用 `logits[0, m+i-2]`。
- 逐位与 `threshold` 比较，遇到首个未通过的位置 `j`，在 **同一位置** 的分布上 `sample_from_p` 得到纠正 `z`。  
- 只把“被接受前缀 + z”写回。

**可直接替换的实现**（`cloud_model.py`）：
```python
# cloud_model.py

def verify_tokens(self, context: dict, draft_tokens: list, threshold: float = 0.25):
    """
    Prefill 一次（forward），按阈值逐位验收；首错用 p(·) 直接采样纠正
    返回: (accepted_tokens, correction_token, needs_correction)
    """
    import torch
    if not draft_tokens:
        return [], None, False

    # 1) 设备与拼接（把 edge 张量搬到 cloud 设备）
    ctx_ids  = context['input_ids'].to(self.device)
    ctx_mask = context['attention_mask'].to(self.device)
    y = torch.tensor([draft_tokens], device=self.device)
    full_ids  = torch.cat([ctx_ids,  y], dim=1)
    full_mask = torch.cat([ctx_mask, torch.ones_like(y)], dim=1)

    # 可选：若上下文里带有音频特征键，可原样带上；若你坚持“不传音频”，可以注释
    extra = {}
    if 'input_features' in context:
        extra['input_features'] = context['input_features'].to(self.device)
    if 'feature_attention_mask' in context:
        extra['feature_attention_mask'] = context['feature_attention_mask'].to(self.device)

    # 2) 一次前向，取 logits
    with torch.no_grad():
        out = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            return_dict=True,
            **extra
        )
    logits = out.logits[0]  # [m+k, V]
    m = ctx_ids.shape[1]
    k = len(draft_tokens)

    # 3) 逐位计算 p(yi) 并找首个未通过
    accepted = 0
    for i in range(k):
        pos = m - 1 + i            # 对应 yi 的预测位置
        probs = torch.softmax(logits[pos], dim=-1)
        p_i = probs[draft_tokens[i]].item()
        if p_i >= threshold:
            accepted += 1
        else:
            # 4) 首错纠正：直接按 p(·) 采样一个 token
            corr = torch.multinomial(probs, 1).item()
            return draft_tokens[:accepted], corr, True

    # 全部通过
    return draft_tokens, None, False
```

---

## 三、必须修改 #2：避免将 `None` 追加进上下文
在“首错纠正”分支里请确保只在 `correction_token is not None` 时才追加。

**补丁**（`speculative_decoding.py`）：
```python
# 原：current_context = self._update_context(current_context, accepted_tokens + [correction_token])
append_list = accepted_tokens + ([correction_token] if correction_token is not None else [])
current_context = self._update_context(current_context, append_list)
```

---

## 四、强烈建议：阈值下调 + 一次校准
- 将默认 `prob_threshold` 设为 **0.25**（常见可用范围 0.15–0.35）。
- 在一小段 dev 样本上网格搜索阈值：`T ∈ {0.15,0.20,0.25,0.30,0.35}`，统计 **接受率**（被接受 token / 验证 token）与**端到端时延**，选到目标接受率（0.6–0.8）。

**示意代码**：
```python
candidates = [0.15, 0.20, 0.25, 0.30, 0.35]
best = None
for T in candidates:
    acc_rate, total_time = run_dev_eval(prob_threshold=T)
    # 例如：优先 acc_rate≈0.7，其次 total_time 最小
    score = (abs(acc_rate-0.7), total_time)
    best = min([best, (score, T)], key=lambda x: x[0] if x else (1e9, None))
print('chosen T_pass =', best[1])
```

---

## 五、设备与 KV 提示
- **设备对齐**：在 Cloud 端将 `context` 中的张量显式 `.to(self.device)`（上面补丁已包含）。
- **KV 缓存（后续提速）**：当前最小版不强依赖 KV；要进一步提速，建议：
  - Cloud 端只把“**被接受前缀 + 纠正 token**”写入 KV；
  - Edge 端同样写入，下一轮从新位置继续。

---

## 六、不传音频与一致性
- 本实现满足“**不传音频**”的要求：验证阶段仅用文本 token。  
- 如果后续发现质量偏低，可考虑在 Edge 侧先把音频**摘要为少量文本提示**再随 token 一并上传（仍不传原始音频）。

---

## 七、流程复核（修复后的运行链路）
1) **Edge 起草**：生成 \(k\) 个 token（可保留每步熵用于门控调试）。  
2) **发送**：把草稿 tokens 与采样配置发给 Cloud。  
3) **Cloud 验证**：`forward(ctx+y)` → 取每步 `p_i` → 与 `T_pass` 比较 → **首错**按 \(p(\cdot)\) 采样纠正 `z`。  
4) **写回**：两端追加“被接受前缀 + z”；进入下一轮。  
5) **阈值与 k**：阈值经 dev 集校准；`k` 可先固定或用 `la/k` 的简单自适应规则。

---

## 八、待办清单（最少改动）
- [ ] 用上面的 `verify_tokens` 替换 Cloud 端实现（前向、位移对齐、阈值与首错纠正）。
- [ ] 写回上下文时加 `None` 防护。
- [ ] 默认 `prob_threshold=0.25` 并跑一次 dev 校准（目标接受率 0.6–0.8）。
- [ ] （可选）记录 `k, accepted_len, la/k, prefill_ms, rtt_ms` 以便后续调参。

完成以上修改，你的“**简单版 Speculative Decoding**（熵门控 + 阈值验收 + 首错纠正）”即可稳定工作，后续可在此基础上加入动态 `k` 与更丰富的门控信号。

