# Edge Baseline对齐计划 - 与Speculative Decoding保持一致

## 问题描述

用户提出的关键洞察：
> "我觉得应该让edge baseline也使用我设计speculative decoding时候的那一套edge生成逻辑。因为speculative decoding实际上是让cloud来纠正edge的错误输出。如果edge baseline和speculative decoding的edge生成逻辑都不一样，比较和纠正的目的就无法达到了。"

**这个观点完全正确！**

---

## 核心问题

### Speculative Decoding的本质

```
Edge生成draft tokens → Cloud验证/纠正 → 最终输出
```

**如果Edge baseline和Speculative Decoding中的Edge逻辑不同**：
- Edge baseline评估的是"逻辑A"的性能
- Speculative Decoding中Cloud纠正的是"逻辑B"的输出
- **无法准确评估Cloud的纠正效果**

### 当前的不一致

#### Edge Baseline (当前)

**代码位置**: `src/models/edge_model.py` 第202-210行

```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    do_sample=False,  # 贪心解码
    no_repeat_ngram_size=2,  # 2-gram禁止
    repetition_penalty=1.05,  # 轻度重复惩罚
    pad_token_id=self.processor.tokenizer.eos_token_id,
    # ❌ 没有stopping_criteria
    # ❌ 没有advanced anti-repetition（CJK感知）
    # ❌ 没有punctuation gate（标点闸门）
)
```

**特点**：
- ❌ 简单的generation
- ❌ 只有基础的`no_repeat_ngram_size=2`和`repetition_penalty=1.05`
- ❌ 没有停止条件控制
- ❌ 没有语言感知的约束

#### Speculative Decoding中的Edge (当前)

**代码位置**: `src/speculative_decoding.py` 第899-1044行

```python
# 1. 重复惩罚（仅针对CJK内容token）
repetition_penalty = 1.22  # 仅对CJK
for token_id in unique_recent:
    if _is_cjk(token_id):  # ✅ 语言感知
        if logits[token_id] > 0:
            logits[token_id] /= repetition_penalty

# 2. CJK特殊anti-repetition
# 2.1) 阻止immediate same-char repetition
if draft_tokens and _is_cjk(last_token):
    logits[last_token] = -inf

# 2.2) Content-only trigram ban (去除标点后)
content_hist = [t for t in full_history if t not in PUNCT_IDS]
# 3-gram ban on content tokens

# 3. Hard punctuation gate（标点闸门）
# 逗号：至少4个CJK字符
if since_punct < 4:
    logits[comma_like] = -inf

# 句号：至少5个CJK字符，轻度抑制
if since_punct < 5:
    logits[period] -= 3.5

# 4. 温和解码
temperature = 0.7
next_token = argmax(logits / temperature)

# 5. Fallback：如果选中标点，从top-k选非标点
if next_token in PUNCT_IDS and since_punct < 4:
    next_token = first_non_punct_from_topk()
```

**特点**：
- ✅ **语言感知**：CJK特殊处理
- ✅ **Content-only n-gram**：去除标点后的trigram ban
- ✅ **标点闸门**：基于中文字符数的硬约束
- ✅ **Fallback机制**：避免违规标点
- ✅ **温度缩放**：0.7
- ✅ **重复惩罚仅针对内容**：避免标点相对优势

---

## 📊 关键差异对比

| 特性 | Edge Baseline | Spec Decoding Edge | 影响 |
|------|---------------|-------------------|------|
| **生成方式** | `model.generate()` | 逐token incremental | 控制粒度 |
| **重复惩罚** | 1.05，所有token | 1.22，仅CJK内容 | 标点行为 |
| **N-gram ban** | 2-gram（含标点） | 3-gram（仅内容） | 中文流畅度 |
| **标点控制** | ❌ 无 | ✅ 硬闸门（4/5字） | 标点泛滥 |
| **Same-char block** | ❌ 无 | ✅ CJK immediate ban | 重复字符 |
| **Temperature** | 参数传递 | 0.7固定 | 随机性 |
| **Fallback** | ❌ 无 | ✅ Top-k非标点 | 鲁棒性 |
| **Stopping criteria** | ❌ 无 | ✅ 2句话+90字+48 tokens | 输出长度 |

---

## 💡 为什么需要对齐？

### 场景1: 评估Cloud的纠正能力

**如果不对齐**：
```
Edge Baseline (简单逻辑A):
  输出: "说话人...你要是还有啥想法随时跟我说哈。"
  BLEU: 0.03, CIDEr: 0.51

Speculative Decoding:
  Edge (复杂逻辑B): "说话人...（无对话式）"
  Cloud纠正后: "说话人...情绪平静。"
  BLEU: 0.025, CIDEr: 0.48

错误结论: Cloud纠正后反而变差了！❌
```

**如果对齐**：
```
Edge Baseline (逻辑B):
  输出: "说话人...（无对话式）"
  BLEU: 0.020, CIDEr: 0.45

Speculative Decoding:
  Edge (逻辑B): "说话人...（无对话式）"
  Cloud纠正后: "说话人...情绪平静。"
  BLEU: 0.025, CIDEr: 0.48

正确结论: Cloud纠正后提升了！✅
```

### 场景2: 评估Speculative Decoding的加速效果

**目标**：在保持质量的前提下加速

```
Edge Baseline质量: Q_edge
Cloud Baseline质量: Q_cloud
Speculative Decoding质量: Q_spec

期望: Q_spec ≈ Q_cloud （质量接近Cloud）
延迟: T_spec < T_cloud （速度提升）

但前提是: Edge Baseline使用相同逻辑，Q_edge才是有意义的基准
```

---

## 🔧 对齐方案

### 方案A: 让Edge Baseline调用Speculative Decoding的生成逻辑（推荐）

#### 优点：
- ✅ 完全一致（100%对齐）
- ✅ 代码复用，减少维护成本
- ✅ 任何改进自动同步

#### 缺点：
- 需要重构Edge Baseline代码
- 依赖于Speculative Decoding的实现

#### 实现：

**修改 `src/models/edge_model.py`**，添加新方法：

```python
def generate_draft_with_spec_logic(self, 
                                   audio_features: torch.Tensor,
                                   prompt: str,
                                   max_new_tokens: int = 128,
                                   target_sentences: int = 2,
                                   min_chars: int = 90,
                                   min_new_tokens_sc: int = 48,
                                   prompt_type: str = "detailed") -> tuple[str, dict]:
    """
    Generate using the SAME logic as Speculative Decoding's edge generation
    This ensures Edge Baseline is directly comparable to Speculative Decoding
    
    Args:
        audio_features: Audio waveform tensor
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        target_sentences: Target number of sentences (for stopping criteria)
        min_chars: Minimum characters (for stopping criteria)
        min_new_tokens_sc: Minimum new tokens before stopping
        prompt_type: Type of prompt
        
    Returns:
        Tuple of (generated_text, metrics)
    """
    # Import speculative decoding logic
    from ..speculative_decoding import SimpleSpeculativeDecoding
    
    # Create a dummy cloud model (won't be used)
    # We only need the Edge generation logic
    from .cloud_model import CloudModel
    dummy_cloud = CloudModel(model_name=self.model_name, device=self.device)
    
    # Create spec decoder
    spec_decoder = SimpleSpeculativeDecoding(
        edge_model=self,
        cloud_model=dummy_cloud,
        k=5,  # Draft block size
        entropy_threshold=999.0,  # Never call cloud (Edge only mode)
        target_sentences=target_sentences,
        min_chars=min_chars,
        min_new_tokens_sc=min_new_tokens_sc
    )
    
    # Use spec decoder's generation logic but force Edge-only
    generated_text, metrics = spec_decoder.generate(
        audio_waveform=audio_features,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        prompt_type=prompt_type
    )
    
    return generated_text, metrics
```

**修改 `experiments/runs/run_edge_baseline_cpu_limited.py`**：

```python
# 第449行，改为使用新方法
generated_text, detailed_latency = edge_model.generate_draft_with_spec_logic(
    audio_waveform, 
    prompt_template,
    max_new_tokens=128,
    target_sentences=2,
    min_chars=90,
    min_new_tokens_sc=48,
    prompt_type=prompt_type
)
```

---

### 方案B: 提取Speculative Decoding的生成逻辑为共享模块

#### 优点：
- ✅ 更清晰的代码结构
- ✅ 易于测试和维护
- ✅ 可以独立优化

#### 缺点：
- 需要较大的重构工作
- 需要仔细设计接口

#### 实现：

**创建 `src/generation/edge_generation.py`**：

```python
class EdgeGenerationLogic:
    """
    Shared Edge generation logic for both:
    1. Edge Baseline
    2. Speculative Decoding's draft generation
    """
    
    def __init__(self, edge_model, tokenizer):
        self.edge_model = edge_model
        self.tokenizer = tokenizer
        
    def generate_tokens_incremental(self, 
                                    context: dict, 
                                    k: int,
                                    temperature: float = 0.7) -> list:
        """
        Generate k tokens using the advanced logic with:
        - CJK-aware repetition penalty
        - Content-only n-gram ban
        - Hard punctuation gate
        - Fallback mechanism
        """
        # ... 从 speculative_decoding.py 中提取核心逻辑 ...
        
    def _apply_repetition_penalty(self, logits, recent_tokens):
        """Apply CJK-aware repetition penalty"""
        ...
        
    def _apply_ngram_ban(self, logits, history):
        """Apply content-only trigram ban"""
        ...
        
    def _apply_punctuation_gate(self, logits, history):
        """Apply hard punctuation gate"""
        ...
```

然后在Edge Baseline和Speculative Decoding中都使用这个模块。

---

### 方案C: 复制核心逻辑到Edge Baseline（不推荐）

#### 优点：
- ✅ 简单直接

#### 缺点：
- ❌ 代码重复
- ❌ 维护成本高（需要同步修改）
- ❌ 容易出现不一致

---

## 📝 推荐实施步骤

### Step 1: 采用方案A（最快）

1. **修改 `src/models/edge_model.py`**：
   - 添加 `generate_draft_with_spec_logic()` 方法
   - 设置 `entropy_threshold=999.0` 强制Edge-only模式

2. **修改 `experiments/runs/run_edge_baseline_cpu_limited.py`**：
   - 调用新方法而不是 `generate_draft()`
   - 传递相同的stopping criteria参数

3. **验证对齐**：
   - 运行Edge Baseline，检查输出格式
   - 确认没有对话式内容
   - 确认stopping criteria生效（2-3句话）

### Step 2: 重新运行实验

```bash
# Edge Baseline (对齐后)
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100

# Speculative Decoding
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100

# Cloud Baseline（参考）
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 100
```

### Step 3: 验证对齐效果

**检查点**：
- ✅ Edge Baseline输出没有对话式内容
- ✅ Edge Baseline输出长度与Spec Decoding Edge相似（2-3句话）
- ✅ Edge Baseline的BLEU/CIDEr可能下降（正常，因为删除了错误内容）
- ✅ Cloud > Speculative Decoding ≥ Edge（质量排序符合直觉）

### Step 4: 长期重构（可选）

采用方案B，提取共享模块，提升代码质量。

---

## 📊 预期结果对比

### 对齐前（当前）

| 模型 | BLEU | CIDEr | BERTScore F1 | 输出特点 |
|------|------|-------|--------------|----------|
| **Edge Baseline** | 0.0305 | 0.5097 | 0.1655 | ❌ 对话式，不可比 |
| **Spec Decoding** | 0.0250 | 0.4800 | 0.1900 | ✅ 客观描述 |
| **Cloud Baseline** | 0.0239 | 0.4967 | 0.1938 | ✅ 客观描述 |

**问题**：
- Edge Baseline的高分是"虚假的"（包含错误内容）
- 无法准确评估Cloud的纠正效果

### 对齐后（预期）

| 模型 | BLEU | CIDEr | BERTScore F1 | 输出特点 |
|------|------|-------|--------------|----------|
| **Edge Baseline** | ~0.020 | ~0.45 | ~0.16 | ✅ 客观描述，与Spec Edge一致 |
| **Spec Decoding** | ~0.025 | ~0.50 | ~0.19 | ✅ Cloud纠正后提升 |
| **Cloud Baseline** | ~0.024 | ~0.50 | ~0.19 | ✅ 最高质量 |

**优势**：
- ✅ Edge Baseline真实反映Edge能力
- ✅ Spec Decoding显示Cloud纠正效果（相对Edge Baseline提升）
- ✅ Cloud Baseline最高（符合直觉）
- ✅ 所有对比都有意义

---

## 🎯 核心结论

用户的洞察完全正确：

> **Edge Baseline必须使用与Speculative Decoding相同的Edge生成逻辑，否则比较没有意义。**

### 为什么？

1. **Speculative Decoding的本质**：Cloud纠正Edge的错误
2. **如果Edge逻辑不同**：我们评估的不是同一个系统
3. **无法准确衡量Cloud的贡献**：基准不一致

### 怎么做？

**推荐方案A**：让Edge Baseline直接调用Speculative Decoding的生成逻辑（设置`entropy_threshold=999.0`强制Edge-only）

### 预期效果？

- Edge Baseline的BLEU/CIDEr会下降（正常，删除错误内容）
- 但输出质量提升（符合任务要求）
- Cloud > Spec Decoding > Edge（排序符合直觉）
- **最重要**：可以准确评估Cloud的纠正效果和加速收益

---

## 下一步

**需要我帮您实施方案A吗？**

我将：
1. 修改 `src/models/edge_model.py` 添加新方法
2. 修改 `experiments/runs/run_edge_baseline_cpu_limited.py` 使用新方法
3. 重新运行实验验证对齐效果

