# Edge Baseline对齐实施记录

## 实施时间
2025年（根据用户请求）

## 实施目标
让Edge Baseline使用与Speculative Decoding完全相同的Edge生成逻辑，确保两者可以公平对比。

---

## 修改清单

### 修改1: `src/models/edge_model.py`

**位置**: 第930-1047行

**添加内容**: 新方法 `generate_draft_with_spec_logic()`

**功能**:
```python
def generate_draft_with_spec_logic(self, 
                                   audio_features: torch.Tensor,
                                   prompt: str,
                                   max_new_tokens: int = 128,
                                   target_sentences: int = 2,
                                   min_chars: int = 90,
                                   min_new_tokens_sc: int = 48,
                                   prompt_type: str = "detailed") -> tuple[str, dict]
```

**实现原理**:
1. 创建 `SimpleSpeculativeDecoding` 实例
2. 设置 `entropy_threshold=999.0` 强制Edge-only模式（Cloud永远不会被调用）
3. 调用 `spec_decoder.generate()` 使用完全相同的Edge生成逻辑
4. 返回生成文本和指标

**关键特性**:
- ✅ 使用自定义逐token生成循环（不是HF `generate()`）
- ✅ CJK-aware重复惩罚（1.22，仅内容）
- ✅ Content-only 3-gram ban（去除标点）
- ✅ Hard标点闸门（4字逗号，5字句号）
- ✅ Same-character blocking（阻止CJK重复）
- ✅ Fallback机制（top-k非标点）
- ✅ Stopping criteria（2句话+90字+48 tokens）

---

### 修改2: `experiments/runs/run_edge_baseline_cpu_limited.py`

**位置**: 第447-466行

**修改前**:
```python
generated_text, detailed_latency = edge_model.generate_draft(
    audio_waveform, prompt_template, max_new_tokens=64
)
```

**修改后**:
```python
generated_text, detailed_latency = edge_model.generate_draft_with_spec_logic(
    audio_waveform, 
    prompt_template, 
    max_new_tokens=128,         # 64 → 128
    target_sentences=2,         # 新增
    min_chars=90,               # 新增
    min_new_tokens_sc=48,       # 新增
    prompt_type=prompt_type     # 新增
)
```

**参数变化**:
| 参数 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| `max_new_tokens` | 64 | 128 | 与Spec Decoding一致 |
| `target_sentences` | N/A | 2 | 新增：目标句子数 |
| `min_chars` | N/A | 90 | 新增：最少字符数 |
| `min_new_tokens_sc` | N/A | 48 | 新增：最少token数 |
| `prompt_type` | N/A | prompt_type | 新增：传递prompt类型 |

---

## 对齐验证清单

### 生成逻辑对齐

| 特性 | Edge Baseline（修改前） | Edge Baseline（修改后） | Spec Decoding Edge |
|------|------------------------|------------------------|-------------------|
| **生成API** | `model.generate()` | `model.thinker()` 循环 | `model.thinker()` 循环 |
| **重复惩罚** | 1.05，所有token | 1.22，仅CJK | 1.22，仅CJK |
| **N-gram ban** | 2-gram（含标点） | 3-gram（仅内容） | 3-gram（仅内容） |
| **标点闸门** | ❌ 无 | ✅ 4/5字 | ✅ 4/5字 |
| **Same-char** | ❌ 无 | ✅ CJK阻止 | ✅ CJK阻止 |
| **Fallback** | ❌ 无 | ✅ Top-k非标点 | ✅ Top-k非标点 |
| **Stopping criteria** | ❌ 无 | ✅ 2句+90字+48t | ✅ 2句+90字+48t |

### 参数对齐

| 参数 | Edge Baseline（修改前） | Edge Baseline（修改后） | Spec Decoding |
|------|------------------------|------------------------|---------------|
| `max_new_tokens` | 64 | 128 | 128 |
| `target_sentences` | N/A | 2 | 2 |
| `min_chars` | N/A | 90 | 90 |
| `min_new_tokens_sc` | N/A | 48 | 48 |
| `entropy_threshold` | N/A | 999.0（Edge-only） | 3.0-5.5（正常） |

---

## 预期效果

### 输出质量变化

**修改前** (HF `generate()`):
```
"说话人的声音有些低沉，音高变化不大，语气平缓，没有明显的停顿时断，整体给人一种平静的感觉。他可能是在表达一种无奈的情绪。你要是还有啥想法或者想补充的，随时跟我说哈。"
```
- ❌ 包含对话式结尾
- ❌ 不符合任务要求
- ❌ BLEU虚高（0.0305）

**修改后** (Spec Decoding逻辑):
```
"说话人的声音低沉，音高变化不大，语气平缓，没有明显停顿。他可能在表达一种无奈的情绪。"
```
- ✅ 纯客观描述
- ✅ 符合任务要求
- ✅ BLEU真实（预期~0.020）

### 指标对比

| 模型 | BLEU (修改前) | BLEU (预期) | 变化 | 输出质量 |
|------|--------------|------------|------|----------|
| **Edge Baseline** | 0.0305 | ~0.020 | -33% | ❌ → ✅ |
| **Spec Decoding** | 0.0250 | ~0.025 | 不变 | ✅ |
| **Cloud Baseline** | 0.0239 | ~0.024 | 不变 | ✅ |

**关键点**:
- Edge Baseline的BLEU会**下降**（正常！删除了错误内容）
- 但输出质量**提升**（符合任务要求）
- Spec Decoding > Edge Baseline（正确排序）
- 可以准确评估Cloud的纠正效果

### BERTScore对比（语义级别，更可靠）

| 模型 | BERTScore F1 (修改前) | BERTScore F1 (预期) | 变化 |
|------|----------------------|-------------------|------|
| **Edge Baseline** | 0.1655 | ~0.16 | 小幅下降或不变 |
| **Spec Decoding** | 0.1900 | ~0.19 | 不变 |
| **Cloud Baseline** | 0.1938 | ~0.19 | 不变 |

**预期排序**: Cloud > Spec Decoding > Edge（符合直觉）

---

## 测试命令

### 运行对齐后的Edge Baseline

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

**参数说明**：
- `--max_cpu_cores 2`: 限制为2个CPU核心（模拟iPhone 15 Plus的2个性能核心）
- `--max_memory_gb 16.0`: 限制内存为16GB（Qwen2.5-Omni-3B模型需要足够内存）

### 验证对齐效果

**检查1: 输出格式**
```bash
# 查看生成的文本
grep "generated_text" experiments/results/edge_cpu_limited_mer_aligned.json | head -5
```

**预期**:
- ✅ 无对话式内容（"你要是还有啥想法..."）
- ✅ 客观描述
- ✅ 2-3句话
- ✅ 约90-140字

**检查2: Cloud调用次数**
```bash
# 应该为0（Edge-only模式）
grep "total_cloud_calls" experiments/results/edge_cpu_limited_mer_aligned.json
```

**预期**: `"total_cloud_calls": 0`

**检查3: 指标对比**
```bash
# 对比BLEU和BERTScore
# Edge Baseline的BLEU应该下降，但BERTScore接近或略降
```

---

## 技术细节

### 为什么使用 `entropy_threshold=999.0`？

**原理**:
```python
# 在Speculative Decoding中
if uncertainty < entropy_threshold:
    # 接受所有Edge tokens，不调用Cloud
    accept_all_edge_tokens()
else:
    # 调用Cloud验证
    call_cloud_for_verification()
```

**设置999.0的效果**:
- 任何uncertainty值都 < 999.0
- 永远不会触发Cloud验证
- 只使用Edge生成逻辑
- 相当于"Edge-only mode"

### 为什么不直接复制代码？

**方案对比**:

| 方案 | 优点 | 缺点 |
|------|------|------|
| **A: 调用Spec Decoding** | ✅ 100%一致<br>✅ 代码复用<br>✅ 自动同步 | ⚠️ 依赖Spec Decoding |
| **B: 复制代码** | ✅ 独立 | ❌ 代码重复<br>❌ 维护困难<br>❌ 容易不一致 |
| **C: 提取共享模块** | ✅ 清晰架构 | ⚠️ 需要重构 |

**选择方案A的理由**:
- 保证100%一致（使用相同代码）
- 最小修改量（只添加一个wrapper方法）
- 未来修改Spec Decoding，Edge Baseline自动更新

### 性能影响

**额外开销**:
1. 创建dummy CloudModel实例（但不加载权重）: ~0.1s
2. 创建SimpleSpeculativeDecoding实例: ~0.1s
3. 封装层开销: 忽略不计

**总开销**: < 0.2s per sample（可接受）

**优化建议**:
- 可以缓存spec_decoder实例，避免每次重新创建
- 但为了代码简洁，当前实现可以接受

---

## 回退方案

如果对齐后出现问题，可以轻松回退：

### 回退代码

在 `run_edge_baseline_cpu_limited.py` 第458行：

```python
# 回退到原来的方法
generated_text, detailed_latency = edge_model.generate_draft(
    audio_waveform, prompt_template, max_new_tokens=64
)
```

### 回退条件

如果出现以下情况，考虑回退：
1. ❌ 运行时错误（无法生成）
2. ❌ 内存不足（OOM）
3. ❌ 速度太慢（> 5x原来）

**但注意**: BLEU下降不是回退理由！这是预期的，因为删除了错误内容。

---

## 验证清单

对齐实施后，验证以下各项：

### 功能验证
- [ ] Edge Baseline能正常运行
- [ ] 生成的文本无对话式内容
- [ ] 输出长度符合预期（2-3句话，90-140字）
- [ ] Cloud调用次数为0
- [ ] 无OOM或崩溃

### 对齐验证
- [ ] Edge Baseline输出与Spec Decoding Edge输出格式一致
- [ ] 无标点泛滥（"你，明，知，道"）
- [ ] 无语气词重复（"呢？呢！呢？"）
- [ ] 无单字+标点（"我：话：哎："）

### 指标验证
- [ ] Edge Baseline BLEU下降（正常）
- [ ] Cloud > Spec Decoding > Edge（排序正确）
- [ ] BERTScore: Cloud最高（语义质量）

---

## 后续工作

### 可选优化

1. **缓存spec_decoder实例**
   ```python
   # 在EdgeModel.__init__中创建一次
   self.spec_decoder = SimpleSpeculativeDecoding(...)
   
   # 在generate_draft_with_spec_logic中复用
   return self.spec_decoder.generate(...)
   ```

2. **提取共享模块**（长期）
   - 创建 `src/generation/edge_generation_logic.py`
   - 被Edge Baseline和Spec Decoding共同使用
   - 更清晰的架构

3. **性能profiling**
   - 对比对齐前后的速度
   - 识别瓶颈

### 文档更新

- [x] 创建对齐实施记录（本文档）
- [ ] 更新README说明Edge Baseline已对齐
- [ ] 更新实验结果解释文档

---

## 总结

### 修改内容
1. ✅ 添加 `edge_model.generate_draft_with_spec_logic()` 方法
2. ✅ 修改 `run_edge_baseline_cpu_limited.py` 调用新方法
3. ✅ 参数对齐（128 tokens, 2句话, 90字, 48 min tokens）

### 核心原理
- 使用Speculative Decoding的生成逻辑，但强制Edge-only模式
- 设置 `entropy_threshold=999.0` 确保Cloud永远不被调用
- 100%代码复用，保证完全一致

### 预期效果
- Edge Baseline的BLEU会下降（删除错误内容）
- 但输出质量提升（符合任务要求）
- 可以准确评估Cloud的纠正效果
- 排序符合直觉：Cloud > Spec Decoding > Edge

**对齐完成！可以开始测试了。** 🚀

