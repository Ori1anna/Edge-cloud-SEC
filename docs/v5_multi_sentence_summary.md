# V5 Multi-Sentence Fix - 快速参考

## 问题

V4成功解决标点和重复，但输出只有**1句话** (30-52 tokens)。

## 解决

8项修改，实现**2-3句话**输出 (90-140字)。

---

## 修改清单

| ✅ | 修改 | 文件 | 行号 | 改动 |
|----|------|------|------|------|
| ✅ | 可配置参数 | `speculative_decoding.py` | 24-26 | +`target_sentences, min_chars, min_new_tokens_sc` |
| ✅ | Stopping criteria | `speculative_decoding.py` | 327-331 | `n=2, min=48, chars=90, detailed` |
| ✅ | 删除硬编码 | `speculative_decoding.py` | 612-614 | 删除2句硬编码逻辑 |
| ✅ | 句号闸门 | `speculative_decoding.py` | 1004-1009 | `8→5`, `-8.0→-3.5` |
| ✅ | Newline EOS | `speculative_decoding.py` | 129-134 | 删除`'\n'` |
| ✅ | max_new_tokens | 两个文件 | 174, 477 | `32/64→128` |
| ✅ | Prompt | run脚本 | - | "两到三句"（用户完成）|
| ✅ | 超参数 | run脚本 | 370, 709 | 默认3.0 |

---

## 测试命令

```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10
```

默认使用`entropy_threshold=3.0`（质量优先）

---

## 预期效果

| 指标 | V4 (1句) | V5 (预期2-3句) |
|------|----------|----------------|
| **句子数** | 1 | 2-3 |
| **Tokens** | 30-52 | 80-120 |
| **字符数** | 50-80 | 90-140 |
| **BERTScore F1** | 0.149 | >0.20 |

---

## 关键变化

### 1. 停止条件

**之前**：1句 + 50字 + 16 tokens
**现在**：2句 + 90字 + 48 tokens

### 2. 句号闸门

**之前**：8字 + -8.0惩罚（难以结束句子）
**现在**：5字 + -3.5惩罚（自然结束句子）

### 3. Token预算

**之前**：64 tokens（空间不足）
**现在**：128 tokens（充足空间）

---

## 成功标志

- [ ] 输出样例有2-3个句号
- [ ] 平均长度翻倍（80-120 tokens）
- [ ] 连贯性保持（无重复、无标点泛滥）
- [ ] Acceptance rate >0.70


