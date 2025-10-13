# Edge Baseline对齐 - 快速开始指南

## 🚀 运行对齐后的Edge Baseline

### 完整测试命令（10个样本）

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0 \
    --output_name edge_cpu_limited_mer_aligned
```

### 完整测试命令（100个样本）

```bash
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0 \
    --output_name edge_cpu_limited_mer_aligned_100
```

---

## 📋 参数说明

### 必需参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset_path` | `data/processed/mer2024/manifest_audio_only_final.json` | 数据集路径 |

### 任务配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset_type` | `unified` | 数据集类型 |
| `--caption_type` | `audio_only` | 使用audio-only标注 |
| `--language` | `chinese` | 生成中文 |
| `--prompt_type` | `detailed` | 使用详细prompt（2-3句话） |
| `--max_samples` | `10` 或 `100` | 处理的样本数 |
| `--output_name` | `edge_cpu_limited_mer_aligned` | 输出文件名 |

### **CPU限制参数**（重要！）

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--max_cpu_cores` | 2 | **2** | 限制CPU核心数<br>模拟iPhone 15 Plus的2个性能核心 |
| `--max_memory_gb` | 16.0 | **16.0** | 限制内存（GB）<br>Qwen2.5-Omni-3B需要约12-14GB |

**注意**：
- ✅ 虽然有默认值，但**建议显式指定**以确保一致性
- ✅ 这些参数模拟移动设备的硬件限制
- ✅ 如果不指定，会使用默认值（2核心，16GB）

---

## ✅ 验证清单

### 1. 检查输出格式

```bash
# 查看前3个样本的生成文本
grep "generated_text" experiments/results/edge_cpu_limited_mer_aligned.json | head -3
```

**预期**：
- ✅ 无对话式内容（"你要是还有啥想法..."）
- ✅ 客观描述
- ✅ 2-3句话
- ✅ 约90-140字

**示例好的输出**：
```
"说话人的声音低沉，音高变化不大，语气平缓，没有明显停顿。他可能在表达一种无奈的情绪。"
```

**示例坏的输出**（对齐前）：
```
"说话人...你要是还有啥想法或者想补充的，随时跟我说哈。"  ❌
```

### 2. 检查Cloud调用次数

```bash
# 应该为0（Edge-only模式）
grep "total_cloud_calls" experiments/results/edge_cpu_limited_mer_aligned.json | head -1
```

**预期**：`"total_cloud_calls": 0`

### 3. 检查指标

```bash
# 查看总体指标
grep -A 10 '"metrics"' experiments/results/edge_cpu_limited_mer_aligned.json | head -15
```

**预期**：
- BLEU: ~0.020（可能比对齐前的0.0305低，正常）
- CIDEr: ~0.45（可能比对齐前的0.5097低，正常）
- BERTScore F1: ~0.16（语义相似度，更可靠）

### 4. 检查无病态模式

```bash
# 检查是否有标点泛滥
grep "generated_text.*，.*，.*，.*，.*，" experiments/results/edge_cpu_limited_mer_aligned.json
```

**预期**：应该没有或极少（标点泛滥已被修复）

---

## 📊 对比测试

### 运行全套baseline对比

```bash
# 1. Edge Baseline (对齐后)
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0 \
    --output_name edge_cpu_limited_mer_aligned_100

# 2. Speculative Decoding
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --output_name speculative_decoding_mer_aligned_100

# 3. Cloud Baseline（GPU）
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100 \
    --output_name cloud_mer_aligned_100
```

### 预期排序

| 模型 | 预期BLEU | 预期CIDEr | 预期BERTScore F1 | 输出质量 |
|------|----------|-----------|------------------|----------|
| **Cloud Baseline** | ~0.024 | ~0.50 | **~0.19** | ✅ 最高 |
| **Spec Decoding** | ~0.025 | ~0.50 | ~0.19 | ✅ 接近Cloud |
| **Edge Baseline** | ~0.020 | ~0.45 | ~0.16 | ✅ 基准 |

**关键点**：
- ✅ Cloud ≥ Spec Decoding > Edge（排序符合直觉）
- ✅ Edge的BLEU下降是正常的（删除了错误内容）
- ✅ BERTScore是更可靠的语义指标
- ✅ 现在可以准确评估Cloud的纠正效果

---

## 🔍 故障排查

### 问题1: ImportError或ModuleNotFoundError

**原因**：依赖未安装或环境未激活

**解决**：
```bash
conda activate sec-gpu  # 或你的环境名
pip install -r requirements.txt
```

### 问题2: OOM (Out of Memory)

**症状**：`RuntimeError: [enforce fail at alloc_cpu.cpp:...]`

**原因**：内存不足

**解决**：
```bash
# 增加内存限制
--max_memory_gb 20.0

# 或减少batch size/样本数
--max_samples 10
```

### 问题3: "entropy_threshold=999.0" 但仍有Cloud调用

**症状**：`total_cloud_calls > 0`

**检查**：
```bash
# 查看日志确认entropy_threshold
grep "entropy_threshold" <log_file>
```

**原因**：可能spec_decoder初始化失败，回退到了其他逻辑

**解决**：检查日志中的错误信息

### 问题4: 生成速度很慢

**预期速度**：
- Edge Baseline (对齐前，HF generate): ~1-2分钟/10样本
- Edge Baseline (对齐后，Spec逻辑): ~2-3分钟/10样本

**原因**：自定义逐token循环比HF generate稍慢（正常）

**优化**：无需优化，速度差异可接受

### 问题5: 输出仍有对话式内容

**检查**：
```bash
grep "你要是\|随时跟我说\|有啥想法" experiments/results/edge_cpu_limited_mer_aligned.json
```

**如果仍有**：
1. 确认使用了新方法 `generate_draft_with_spec_logic()`
2. 检查日志确认 "Using Speculative Decoding Edge logic"
3. 确认entropy_threshold=999.0

---

## 📝 关键配置总结

### Edge Baseline对齐配置

```python
# src/models/edge_model.py
def generate_draft_with_spec_logic(
    self,
    audio_features,
    prompt,
    max_new_tokens=128,        # ✅ 与Spec Decoding一致
    target_sentences=2,        # ✅ 2句话
    min_chars=90,              # ✅ 最少90字
    min_new_tokens_sc=48,      # ✅ 最少48 tokens
    prompt_type="detailed"     # ✅ 详细prompt
)
```

### 关键差异

| 特性 | 对齐前 | 对齐后 |
|------|--------|--------|
| **生成方式** | `model.generate()` | `model.thinker()` 循环 |
| **max_new_tokens** | 64 | 128 |
| **重复惩罚** | 1.05，所有 | 1.22，仅CJK |
| **N-gram** | 2-gram（含标点） | 3-gram（仅内容） |
| **标点控制** | ❌ 无 | ✅ 硬闸门 |
| **Stopping criteria** | ❌ 无 | ✅ 2句+90字 |

---

## 🎯 验证成功标准

运行测试后，检查以下各项：

- [ ] Edge Baseline能正常运行完成
- [ ] 生成的文本无对话式内容
- [ ] 输出长度符合预期（2-3句话，90-140字）
- [ ] Cloud调用次数为0
- [ ] 无标点泛滥（"你，明，知，道"）
- [ ] 无语气词重复（"呢？呢！呢？"）
- [ ] Edge BLEU < Spec Decoding < Cloud（正确排序）
- [ ] BERTScore: Cloud最高

**全部通过 = 对齐成功！** ✅

---

## 📖 相关文档

- 完整实施记录：`docs/edge_baseline_alignment_implementation.md`
- 生成逻辑对比：`docs/edge_generation_logic_comparison.md`
- 对齐计划：`docs/edge_baseline_alignment_plan.md`
- Baseline对比分析：`docs/baseline_comparison_analysis.md`

---

**准备好了吗？运行测试命令开始验证！** 🚀

