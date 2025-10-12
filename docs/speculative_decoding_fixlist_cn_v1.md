# Speculative Decoding 修复清单（改什么 & 为什么）

> 适用对象：你在 `SimpleSpeculativeDecoding` 中的增量起草（edge）+ 升云校对（cloud）实现。
>
> 目标：消除“字字逗”的标点泛滥、短环复读与提前截断；让中文一句话输出自然、连贯、可控。

---

## TL;DR 一次性要改的点

1. **温度只用一次**：去掉“先 `/0.7` 再 `/temperature`”的重复加热。
2. **步进别传 1×1 `attention_mask`**：它会让位置错位；必要时传 `position_ids`。
3. **中文禁“紧邻同字”+ 去标点 2/3-gram 约束**：拦截“真想 真想 …”“员 在 员 在 …”等短环。
4. **提高逗号门槛 & 用字符级密度检测**：至少 4 个中文内容字后才放逗号；标点泛滥用**字符**统计。
5. **重复惩罚只打在中文内容字**：对标点不施压，避免和门控博弈。
6. **停止准则匹配“一句中文短句”**：`n_sentences=1, min_new_tokens≈16`，句读“该来就来”。
7. **调参降低切碎**：两档推荐（流畅优先 vs 质量优先）：`(thr≈5.5~6.0,k=4)` 或 `(thr≈3.0~3.5,k=6)`。

---

## A. 只保留一次温度缩放（避免“过尖”导致标点胜出）

**问题**：你在增量步进里先 `logits/0.7`，末尾又 `/temperature(=0.7)`，等价 **除以 0.49**，把分布压得过尖，硬门控一旦放行，标点更易压制内容 token。

**改法**：在应用完所有惩罚/门控后 **只做一次** `/temperature`。

```python
# 现状（应删除第一次 /0.7）
logits = outputs.logits[0, -1, :].float()
logits_temp = logits.clone() / 0.7      # ← 删除
# ... 对 logits_temp 施加各类惩罚/门控 ...
temperature = 0.7
logits_scaled = logits_temp / temperature
next_token = torch.argmax(logits_scaled, dim=-1).item()
```

---

## B. 去掉 1×1 attention_mask；必要时显式 position_ids（防位置错位）

**问题**：增量步进时传 `attention_mask=torch.ones(1,1)` 会误导部分实现把当前位置“重置”，典型表现就是**在字间夹入标点或口头语**尝试自修复。

**改法**：步进时**不要传** 1×1 `attention_mask`。如需稳健，可在 prefill 后记录“最后位置”，步进时显式给 `position_ids`。

```python
# 删除：
# if 'attention_mask' in context:
#     step_inputs['attention_mask'] = torch.ones(1,1, device=...)

# 可选：prefill 结束
enhanced_context['position_id_last'] = context['input_ids'].shape[1] - 1

# 可选：步进中（第 step 步）
if 'position_id_last' in context:
    pid = torch.tensor([[context['position_id_last'] + 1 + step]],
                       device=self.edge_model.device, dtype=torch.long)
    step_inputs['position_ids'] = pid
```

---

## C. 中文轻量“不重复”约束（不伤自然）

**问题**：为保中文流畅你把 n-gram ban 全关，但这会放任**紧邻 unigram 复读**与**短环**。

**改法**：
- 禁止**紧邻同字**（仅中文内容字）；
- 在“去标点历史”上做温和的 2/3-gram ban（只对中文内容字生效）。

```python
def _is_cjk_id(tid:int)->bool:
    try:
        s = self.edge_model.processor.tokenizer.decode([tid], skip_special_tokens=True)
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)
    except:
        return False

# 1) 紧邻同字禁止（中文内容字）
if draft_tokens:
    last = draft_tokens[-1]
    if _is_cjk_id(last):
        logits_temp[last] = -float('inf')

# 2) 去标点的内容历史上做 n-gram 限制（CJK）
PUNCT_IDS = {...}  # 见 D 节，或按你现有映射
content_hist = [t for t in (context['input_ids'][0].tolist() + draft_tokens) if t not in PUNCT_IDS]
if len(content_hist) >= 2 and all(_is_cjk_id(t) for t in content_hist[-6:]):
    # 以 trigram 为例（也可 bigram）
    trigram_map = {}
    for x, y, z in zip(content_hist[:-2], content_hist[1:-1], content_hist[2:]): 
        trigram_map.setdefault((x, y), set()).add(z)
    if len(content_hist) >= 2:
        a, b = content_hist[-2], content_hist[-1]
        banned = list(trigram_map.get((a, b), []))
        if banned:
            logits_temp[banned] = -float('inf')
```

---

## D. 提高逗号门槛 + 字符级“标点泛滥”检测

**问题**：中文多为单字 token，`since_punct >= 2` 太松，容易形成“**字字逗**”。基于 token id 的密度统计会被词表/全半角变体干扰。

**改法**：至少 4 个中文内容字后才放逗号/顿号/冒号；句号至少 8 个内容字才鼓励。同时用**字符级**密度检测。

```python
def _ids_for(chars):
    ids = []
    for ch in chars:
        enc = self.edge_model.processor.tokenizer.encode(ch, add_special_tokens=False)
        if enc: ids.append(enc[0])
    return set(ids)

PUNCT_IDS   = _ids_for(['，','。','、','：',':','；','！','？'])
COMMA_LIKE  = _ids_for(['，','、','：',':'])
PERIOD_LIKE = _ids_for(['。'])

# 统计自上一个标点起的 CJK 内容字数
hist = context['input_ids'][0].tolist() + draft_tokens
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS: break
    try:
        s = self.edge_model.processor.tokenizer.decode([t], skip_special_tokens=True)
        if any('\u4e00' <= ch <= '\u9fff' for ch in s): since_punct += 1
    except: pass

# 门槛
if since_punct < 4:        # ← 逗号/顿号/冒号
    for pid in COMMA_LIKE: logits_temp[pid] = -float('inf')
if since_punct < 8:        # ← 句号：未到时强抑制
    for pid in PERIOD_LIKE:
        if not torch.isinf(logits_temp[pid]): logits_temp[pid] -= 8.0

# 字符级泛滥检测（替代 token id 计数）
draft_text = self.edge_model.processor.tokenizer.decode(draft_tokens, skip_special_tokens=True)
punct_chars = set('，。：:；！？、')
punct_density = sum(c in punct_chars for c in draft_text) / max(1, len(draft_text))
if punct_density > 0.4:
    needs_cloud_verification = True
```

---

## E. 重复惩罚仅作用于“中文内容字”

**问题**：对**全部** token 做重复惩罚会和标点门控/兜底互相博弈，模型转而“找替代标点”。

**改法**：只对 CJK 内容字施加乘性惩罚（1.2~1.25），**不动标点**。

```python
repetition_penalty = 1.22
if recent_tokens:
    for token_id in set(recent_tokens):
        if _is_cjk_id(token_id):
            if logits_temp[token_id] > 0: logits_temp[token_id] /= repetition_penalty
            else:                         logits_temp[token_id] *= repetition_penalty
```

---

## F. 停止准则与任务契合（中文一行短句）

**问题**：`n_sentences=2, min_new_tokens=32` 不匹配“一句中文短句”，要么提早停，要么晚停靠 `max_new_tokens` 断。

**改法**：

```python
stopping_criteria = create_stopping_criteria(
    self.edge_model.processor.tokenizer,
    n_sentences=1,                  # 一句
    sentence_end_chars=("。","."),  # 句号即停
    min_new_tokens=16,              # 至少生成 16 token 再允许停
    prompt_type="concise"           # 或按你的任务选择
)
```

> 与 D 节“句号门控”配套后，句读“该来就来”，不会被强压或提前误停。

---

## G. 降低“频繁升云”对节律的切碎效应（调参）

结合你近几轮统计（接受率 ~0.35、几乎每块升云），建议二选一：

- **流畅优先**：`entropy_threshold = 5.5 ~ 6.0`，`k = 4`  
  → 云调用更少，连续片段更长，语势更顺。

- **质量优先**：`entropy_threshold = 3.0 ~ 3.5`，`k = 6`  
  → 云端更勤，但单次校正覆盖更长草稿，减少“块块截断”。

> 当前（4.0, 5）落在“高不确定位点较多但每段偏短”的尴尬区间，容易**生成-打断-回滚**频发，从而滋生“填充性标点”。

---

## 依赖/互斥关系提示

- A 与 D/E/F 是**强相关**：不过尖 + 合理门控 + 合适停条件 → 句读自然。  
- B 是**必要条件**：位置一旦错，语言模型会用标点“缝合”，所有上层策略都会被噪声放大。  
- C 与 E **互补**：前者切断结构性短环，后者抑制短期复读倾向。  
- G 属于**策略层**，优先选“流畅优先”作为默认。

---

## 回归测试建议（3 个核心指标）

1. **Punctuation density（字符级）**：目标 10%–25%（中文短句），若 >40% 视为异常。  
2. **Repeat rate（内容字 2/3-gram）**：短环比例显著下降；“紧邻同字”近零。  
3. **Cloud call rate / Acceptance rate**：升云次数显著下降；接受率 >0.6 更理想。

> 建议在 20–50 条验证集上做前后 A/B，对比三项指标与人工可读性。

---

## 最后说明

以上改动都可**仅在 `_generate_draft_tokens_incremental(...)` 与创建 `stopping_criteria` 的位置**完成；其余 KV 同步/纠错回滚逻辑你已修好，无需再动。若需要，我可以基于你当前文件给出**最小可合并 diff**。

祝顺利！
