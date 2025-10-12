# Make the Model Output 2–3 Sentences (Not Just One)
**Why you're seeing only one sentence** and **how to fix it** with precise code changes.

---

## TL;DR (What to change)

1. **Prompt**: Your “detailed” Chinese template explicitly forces *one* sentence (“不得再写第二句”). Change it to allow **2–3 sentences**.
2. **Stopping criteria**: You hard‑set `n_sentences=1` and also stop after sentences in another place. Make this **configurable** (e.g., 2 or 3) and remove duplicated stop logic.
3. **Punctuation gate**: You heavily suppress `。` until 8 CJK tokens. Lower this to **5–6** so sentences can close naturally.
4. **EOS rules**: You stop on `\n` (newline). Remove this so a newline **doesn’t prematurely end** generation.
5. **Budget**: Increase `max_new_tokens` to **96–128** for multi‑sentence output. Tune `k` and `entropy_threshold` for quality.

---

## Root Causes (in your current code)

- **Prompt-level constraint**  
  In your runner’s “detailed” Chinese prompt, you require: “**只输出‘一句中文长句’…不得再写第二句**”。这会强制模型只写一句。

- **Stopping criteria fixed to one sentence**  
  You call `create_stopping_criteria(..., n_sentences=1, sentence_end_chars=("。","."), min_new_tokens=16, min_chars=50, prompt_type="concise")` — this stops after **one sentence** and also uses the **concise** mode.

- **Extra sentence-stop guard**  
  Later you also count sentence endings and `break` after 2 when `len(generated_tokens) > 32` — this conflicts with the earlier criteria and is **hard‑coded**, not tied to a config.

- **Period (“。”) gate is too strict**  
  You suppress `。` unless `since_punct >= 8`, biasing the model **away from closing** a sentence.

- **Newline treated as EOS**  
  `_is_eos_token` returns True for tokens containing `'\n'`, which can **cut off multi‑sentence** paragraphs.

---

## Minimal, Safe Patch (diff-style)

> Below are **surgical edits** to make your system produce naturally **2–3 sentences**.

### 1) Make sentence target configurable

Add these parameters to `SimpleSpeculativeDecoding.__init__`:
```python
def __init__(..., entropy_threshold: float = 1.5, k: int = 5,
             target_sentences: int = 2,  # NEW
             min_chars: int = 90,        # NEW (for detailed)
             min_new_tokens_sc: int = 48 # NEW (stopping-criteria budget)
             ):
    ...
    self.target_sentences = target_sentences
    self.min_chars = min_chars
    self.min_new_tokens_sc = min_new_tokens_sc
```

### 2) Use “detailed” stopping profile and N sentences

Replace your current `create_stopping_criteria` call:
```python
stopping_criteria = create_stopping_criteria(
    self.edge_model.processor.tokenizer,
    n_sentences=self.target_sentences,             # CHANGED (was 1)
    sentence_end_chars=("。", "."),                # keep stable
    min_new_tokens=self.min_new_tokens_sc,        # CHANGED (was 16)
    min_chars=self.min_chars,                     # CHANGED (was 50)
    prompt_type="detailed"                        # CHANGED (was "concise")
)
```

### 3) Remove hard-coded “stop after 2 sentences” block

Delete this block (it duplicates logic):
```python
if len(generated_tokens) > 32:
    sentence_count = 0
    for token in generated_tokens:
        if self._is_sentence_end_token(token):
            sentence_count += 1
            if sentence_count >= 2:
                should_stop = True
                break
```

> Rely on `stopping_criteria` exclusively to avoid conflicting exits.

### 4) Allow earlier full stops (。)

In `_generate_draft_tokens_incremental`, adjust the period gate:
```python
# Period: require at least 5–6 CJK tokens (was 8)
period_min = 5  # or 6
if since_punct < period_min:
    for punct_id in PERIOD_LIKE:
        if not torch.isinf(logits_temp[punct_id]):
            logits_temp[punct_id] -= 3.5    # was -8.0, too strong
```
Keep comma gate at 4; it’s fine.

### 5) Don’t treat newline as EOS

In `_is_eos_token`, **remove**:
```python
if '\n' in token_text:
    return True
```
And also **remove** `token_text in ['\n', ...]` from the early list.

### 6) Increase token budget

- In your runner, bump `max_new_tokens` when calling `spec_decoding.generate(...)` to **96–128** for multi‑sentence:
```python
generated_text, latency_metrics = spec_decoding.generate(
    audio_features=audio_waveform,
    prompt=prompt_template,
    max_new_tokens=128  # was 64
)
```

- Optionally expose CLI flags: `--target_sentences`, `--min_chars`, `--max_new_tokens` and plumb them to the class.

### 7) Fix the “detailed” prompt itself

Change the Chinese “detailed” template to **allow 2–3 sentences** and remove “不得再写第二句”. Suggested replacement:

```text
任务：请生成一段“情感说明文字”，按以下顺序组织并保持自然流畅：
(1) 先用2–3个声学/韵律线索描述说话方式（语速、音调起伏、音量强弱、停顿连贯度、音色/紧张度等）；
(2) 据此给出最可能的单一情绪；
(3) 如语义暗示缘由，可用极简短语点到为止（可用“可能/似乎/大概”表不确定）。

输出要求：
- 输出**2–3句中文**，总计约90–140字；以自然的句号“。”结束；
- 使用第三人称或“说话人”指代；不要出现第一/第二人称；不设问/寒暄；
- 不编造具体人物/时间/地点；不含表情符号、英文、Markdown/代码。
```

> 修改位置在你的实验 runner 的 `get_prompt_template(prompt_type="detailed", language="chinese")`。

### 8) Hyperparameter tips for longer, stable outputs

- `k`: **4–5** (短一点更稳)  
- `entropy_threshold`: **3.0–3.5** for quality（1.5 过低会频繁触发云校验、卡顿）  
- `temperature`: keep **0.7**  
- `max_new_tokens`: **128**  
- `min_chars`: **90–140** (搭配 2–3 句)  

---

## Quick Sanity Checklist (10‑sample smoke test)

- [ ] 90% 样本输出有 **2–3 句**，总长 90–140 字；  
- [ ] 第一/第二人称 **未出现**；  
- [ ] 情绪词 + 至少两条声学线索 **同时出现**；  
- [ ] 无模板泄漏（Human/User/Assistant/System）；  
- [ ] 标点分布自然，句号出现的**最短间距 ≥ 5 个 CJK 内容 token**；  
- [ ] `acceptance_rate` ≥ 0.6，`cloud_calls` 合理（< 1.2/样本）。

> 如仍过短：增大 `min_chars` 与 `min_new_tokens_sc`；如过长：把 `target_sentences` 改为 2，并把 `max_new_tokens` 收窄到 96。

---

## Optional (nice-to-have)

- **Expose `target_sentences` to CLI** for quick A/B runs without code edits.  
- **Sentence-end set**：如希望允许“！”“？”也计数，可将 `_is_sentence_end_token` 中的集合扩展到 `['。','！','？','.']`，同时 **不要**把它们加入 EOS。

---

## Why these changes work

- **Prompt** 是第一约束：明确 2–3 句 + 长度范围，模型才会朝多句段落采样。  
- **Stopping criteria** 是第二约束：以 `n_sentences` + `min_chars` 控制停止点，避免早停。  
- **Punctuation gate** 放松句号抑制，保证句子能自然闭合，再进入下一句。  
- **去掉换行停止** 防止非内容性 token 触发早停。  
- **更大的 token 预算** 给足多句段落所需空间。

---

**Ready to run**: after applying the above, rerun your 10‑sample test；若还有异常，我会再根据日志中 `acceptance_rate / cloud_calls / punctuation density` 定位进一步的解码层问题。