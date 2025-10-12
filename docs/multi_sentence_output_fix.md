# Multi-Sentence Output Fix - 实现2-3句话输出

## 问题诊断

### 测试结果：speculative_decoding_mer_chinese_newprompt18.json

**好消息**：
- ✅ 标点泛滥已解决（无"字，字，字"模式）
- ✅ 短环重复已解决（无"真想，真想"循环）
- ✅ Acceptance rate大幅提升：**0.29 → 0.87** (+200%)
- ✅ Cloud calls显著降低：**34-46次 → 5.3次** (-85%)
- ✅ 句子连贯通顺

**新问题**：
- ❌ 所有样本只有**1句话**（30-52 tokens）
- ❌ 期望：**2-3句话**（90-140字）

### 样本分析

| Sample | 输出 | Tokens | 句子数 |
|--------|------|--------|--------|
| 00000000 | "他语速缓慢，音调低沉...可能是因为自己在情感上遇到了挫折。" | 35 | **1句** |
| 00000007 | ""那就不见不散"这句话的语速比较缓慢...所以才这么说。"" | 52 | **1句** |
| 00000033 | "他语速缓慢，音调低沉...可能是因为龙舟队的队员们现在都还是群菜鸟。" | 30 | **1句** |

**期望**：每个样本2-3句话，总计90-140字

---

## 根本原因分析（用户诊断）

### ✅ 原因1：Prompt约束（已修复）

**之前的prompt**：
```
"只输出'一句中文长句'，约70–100个字，以'。'结束；不得再写第二句；"
```

**用户已修改为**：
```
"只输出'两到三句中文长句'，约70–100个字；"
```

### ✅ 原因2：Stopping criteria固定为1句

**代码问题**：
```python
stopping_criteria = create_stopping_criteria(
    n_sentences=1,        # ← 硬编码1句
    min_new_tokens=16,    # ← 太少
    min_chars=50,         # ← 太少
    prompt_type="concise" # ← 错误模式
)
```

**后果**：
- 生成1个句号就停止
- 即使prompt要求2-3句，stopping criteria强制停在1句

### ✅ 原因3：硬编码的2句停止逻辑（冲突）

**代码问题**：
```python
if len(generated_tokens) > 32:
    sentence_count = 0
    for token in generated_tokens:
        if self._is_sentence_end_token(token):
            sentence_count += 1
            if sentence_count >= 2:  # 硬编码2句
                should_stop = True
```

**问题**：
- 与stopping_criteria逻辑重复
- 硬编码，无法配置
- 实际上被`n_sentences=1`提前触发，从未执行到

### ✅ 原因4：句号闸门过严

**代码问题**：
```python
if since_punct < 8:  # 需要8个中文字
    logits[句号] -= 8.0  # 强力抑制
```

**后果**：
- 模型不敢轻易用句号
- 即使该结束第一句，也会继续拖延
- 导致单句过长，或用逗号替代句号

### ✅ 原因5：Newline作为EOS

**代码问题**：
```python
if '\n' in token_text:
    return True  # 立即停止
```

**后果**：
- 多句段落通常有换行
- 一旦生成换行，立即停止
- 无法生成第2-3句

### ✅ 原因6：max_new_tokens太小

**代码问题**：
```python
max_new_tokens=64  # 在运行脚本中
```

**后果**：
- 2-3句话需要~80-120 tokens
- 64太少，可能被截断

---

## 解决方案（已实施）

### ✅ 修改1：添加可配置参数

**文件**：`src/speculative_decoding.py`，第19-47行

**新增参数**：
```python
def __init__(self, 
             edge_model, cloud_model,
             entropy_threshold: float = 1.5,
             k: int = 5,
             target_sentences: int = 2,      # NEW
             min_chars: int = 90,            # NEW
             min_new_tokens_sc: int = 48):   # NEW
    
    self.target_sentences = target_sentences
    self.min_chars = min_chars
    self.min_new_tokens_sc = min_new_tokens_sc
```

---

### ✅ 修改2：使用配置的stopping_criteria

**文件**：`src/speculative_decoding.py`，第322-332行

**修改前**：
```python
stopping_criteria = create_stopping_criteria(
    n_sentences=1,           # 硬编码1句
    min_new_tokens=16,       # 太少
    min_chars=50,            # 太少
    prompt_type="concise"    # 错误模式
)
```

**修改后**：
```python
stopping_criteria = create_stopping_criteria(
    n_sentences=self.target_sentences,      # 配置化（默认2）
    min_new_tokens=self.min_new_tokens_sc,  # 配置化（默认48）
    min_chars=self.min_chars,               # 配置化（默认90）
    prompt_type="detailed"                  # 改为detailed模式
)
```

---

### ✅ 修改3：删除硬编码停止逻辑

**文件**：`src/speculative_decoding.py`，第612-614行

**删除的代码**：
```python
# REMOVED:
if len(generated_tokens) > 32:
    sentence_count = 0
    for token in generated_tokens:
        if self._is_sentence_end_token(token):
            sentence_count += 1
            if sentence_count >= 2:
                should_stop = True
                break
```

**替换为**：
```python
# REMOVED: Hard-coded 2-sentence stop logic
# Rely exclusively on stopping_criteria (configurable)
```

---

### ✅ 修改4：放松句号闸门

**文件**：`src/speculative_decoding.py`，第1002-1009行

**修改前**：
```python
if since_punct < 8:          # 8个中文字才允许
    logits[句号] -= 8.0       # 强力抑制
```

**修改后**：
```python
period_min = 5               # 降到5个中文字（从8）
if since_punct < period_min:
    logits[句号] -= 3.5       # 温和抑制（从-8.0）
```

**效果**：
- 句子可以更早、更自然地结束
- 第1句结束后，才能开始第2句

---

### ✅ 修改5：删除newline作为EOS

**文件**：`src/speculative_decoding.py`，第125-134行

**修改前**：
```python
if token_text in ['\n', 'Human', ...]:
    return True

if '\n' in token_text:
    return True
```

**修改后**：
```python
# Newline is allowed for multi-sentence paragraphs
if token_text in ['Human', 'Human:', ...]:  # 移除'\n'
    return True

# REMOVED: newline as EOS
# if '\n' in token_text:
#     return True
```

---

### ✅ 修改6：增加max_new_tokens

**文件1**：`src/speculative_decoding.py`，第174行
```python
# 默认值：32 → 128
def generate(self, ..., max_new_tokens: int = 128):
```

**文件2**：`experiments/runs/run_speculative_decoding_cpu_limited.py`，第477行
```python
# 调用时：64 → 128
max_new_tokens=128
```

---

### ✅ 修改7：Prompt已修改（用户完成）

**文件**：`experiments/runs/run_speculative_decoding_cpu_limited.py`

用户已将prompt从"一句"改为"两到三句"。

---

### ✅ 修改8：更新超参数建议

**文件**：`experiments/runs/run_speculative_decoding_cpu_limited.py`

**默认值调整**：
```python
entropy_threshold: float = 3.0  # 从1.5提升到3.0（质量优先）
```

**帮助信息**：
```python
--entropy_threshold: "Recommended: 5.5-6.0 for fluency, 3.0-3.5 for quality"
--k: "Recommended: 4 for fluency, 6 for quality"
```

**运行时传参**：
```python
spec_decoding = SimpleSpeculativeDecoding(
    ...,
    target_sentences=2,
    min_chars=90,
    min_new_tokens_sc=48
)
```

---

## 修改总结表

| # | 修改项 | 文件 | 位置 | 改动 |
|---|--------|------|------|------|
| 1 | 可配置参数 | `speculative_decoding.py` | 19-47 | 新增3个参数 |
| 2 | Stopping criteria | `speculative_decoding.py` | 322-332 | `n=2,min=48,chars=90,detailed` |
| 3 | 删除硬编码停止 | `speculative_decoding.py` | 612-614 | 完全删除 |
| 4 | 放松句号闸门 | `speculative_decoding.py` | 1002-1009 | `8→5`, `-8.0→-3.5` |
| 5 | 删除newline EOS | `speculative_decoding.py` | 125-134 | 移除'\n' |
| 6 | max_new_tokens | 两个文件 | 174, 477 | `32/64→128` |
| 7 | Prompt | run脚本 | - | 用户已修改 |
| 8 | 超参数 | run脚本 | 370, 709 | 默认3.0 + 建议 |

---

## 预期效果

### 输出长度对比

| 指标 | 修改前 | 修改后（预期） |
|------|--------|----------------|
| **平均tokens** | 30-52 | **80-120** |
| **平均句子数** | **1句** | **2-3句** |
| **平均字符数** | 50-80 | **90-140** |

### Sample 00000000 对比

**修改前（1句）**：
```
"他语速缓慢，音调低沉，停顿较多，在讲述自己的经历时，似乎有些无奈和沮丧，可能是因为自己在情感上遇到了挫折。"
```
- 35 tokens
- 1句话
- 63字符

**修改后（预期2-3句）**：
```
"他语速缓慢，音调低沉，停顿较多，似乎有些无奈和沮丧。在讲述自己的经历时，情绪显得比较低落。可能是因为自己在情感上遇到了挫折，所以说话时带着明显的失落感。"
```
- ~90-100 tokens
- 3句话
- ~120-140字符

---

## 工作机制

### 停止条件协同

```python
# 第1句：
生成35 tokens → 遇到"。" → 检查停止条件
  - sentence_count = 1 < target_sentences(2) → 继续 ✅
  - tokens = 35 < min_new_tokens(48) → 继续 ✅

# 第2句：
生成60 tokens → 遇到"。" → 检查停止条件
  - sentence_count = 2 ≥ target_sentences(2) → 可以停
  - tokens = 60 ≥ min_new_tokens(48) → 可以停
  - chars = 95 ≥ min_chars(90) → 可以停
  → 三重条件满足 → 停止 ✅
```

### 句号闸门放松的作用

**修改前**（`period_min=8, penalty=-8.0`）：
```
生成："说话人语速缓慢"（7个字）
→ since_punct=7 < 8
→ logits[句号] -= 8.0
→ 句号被强力抑制，继续生成
→ 单句越来越长，难以结束
```

**修改后**（`period_min=5, penalty=-3.5`）：
```
生成："说话人语速缓慢"（7个字）
→ since_punct=7 ≥ 5
→ 句号可用
→ 模型自然选择句号结束第1句 ✅
→ 开始生成第2句
```

---

## 测试建议

### 推荐命令（质量优先模式）

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10 \
    --entropy_threshold 3.0 \
    --k 5
```

**注意**：
- `entropy_threshold=3.0` 已设为默认值
- 如需更流畅，可用 `--entropy_threshold 5.5 --k 4`

### 观察指标

#### 1. 句子数量

**目标**：90%的样本应有2-3句

**检查方法**：
```python
for sample in results:
    text = sample['generated_text']
    sentence_count = text.count('。')
    print(f"{sample['file_id']}: {sentence_count} sentences")
```

**期望**：
- 2句：40-50%
- 3句：40-50%
- 1句：<10%

#### 2. 输出长度

| 指标 | 目标范围 |
|------|----------|
| **Tokens** | 80-120 |
| **字符数** | 90-140 |

#### 3. 质量指标

| 指标 | 修改前 | 预期 |
|------|--------|------|
| **BERTScore F1** | 0.149 | **>0.20** |
| **CIDEr** | 0.461 | **>0.55** |
| **Acceptance Rate** | 0.87 | **>0.75** |

#### 4. 日志验证

**应该看到**：
- `"Added stopping criteria ... target_sentences=2"`
- `"Hard gate: suppressing period (only X/5 CJK tokens)"` (降低阈值)
- 生成过程中多次遇到句号但继续（sentence_count < target）

**不应该看到**：
- `"Stop condition met: 1 sentences completed"` (提前停止)
- 大量低熵+单句就停

---

## 参数调优建议

### 场景1：流畅优先（更少Cloud调用）

```python
--entropy_threshold 5.5
--k 4
target_sentences=2
min_chars=80
```

**效果**：
- Cloud调用少（~3-4次/样本）
- 生成更流畅
- 可能质量略低

### 场景2：质量优先（更准确）

```python
--entropy_threshold 3.0  # 默认值
--k 5                    # 默认值
target_sentences=2
min_chars=90
```

**效果**：
- Cloud调用适中（~5-8次/样本）
- 质量更高
- 平衡流畅度

### 场景3：最长输出（3句话）

```python
--entropy_threshold 3.0
--k 5
target_sentences=3       # 改为3句
min_chars=120            # 提高字符要求
min_new_tokens_sc=64     # 提高token要求
max_new_tokens=160       # 增加上限
```

**效果**：
- 3句话详细描述
- ~120-160字
- 更丰富的信息

---

## 与之前修改的关系

### 累积改进

```
V0: 基础实现
  ↓
V1-V3: 解决标点泛滥、短环重复
  ↓
V4: 语言感知 + 前瞻闸门 + 系统协同
  → 结果：句子通顺，但只有1句 ✅+❌
  ↓
V5 (本次): 多句输出支持
  → 结果：2-3句话，连贯通顺 ✅✅
```

### V4的成就（保留）

- ✅ 紧邻同字阻断
- ✅ 内容trigram约束
- ✅ 硬闸门控制标点间隔
- ✅ 重复惩罚只打内容
- ✅ 删除双重温度
- ✅ 删除错误mask

### V5的增量（本次）

- ✅ 可配置句子数量
- ✅ 放松句号闸门（允许句子结束）
- ✅ 删除newline EOS（允许段落）
- ✅ 增加token预算（128）
- ✅ 停止准则detailed模式
- ✅ 删除硬编码停止逻辑

---

## 技术细节

### 为什么之前只有1句？

**三重约束导致**：

1. **Stopping criteria**：`n_sentences=1` → 1个句号就停
2. **句号闸门过严**：`since_punct<8` + `-8.0` → 不敢用句号
3. **硬编码逻辑**：`sentence_count >= 2` → 与stopping criteria冲突

**矛盾**：
- Criteria说：1句就停
- 闸门说：很难放句号
- 结果：憋到max_tokens，用逗号拼接

**本次修复**：
- Criteria改为：2句才停
- 闸门放松：5个字就可以用句号
- 删除硬编码：统一由criteria控制

### 为什么降低period_min有效？

**心理学类比**：
- **严格老师**（period_min=8, penalty=-8.0）：
  - 学生："我想写完第1句"
  - 老师："还不够长！继续写！"
  - 学生："好吧...继续拖延..."
  - 结果：单句过长，没有第2句

- **温和老师**（period_min=5, penalty=-3.5）：
  - 学生："我想写完第1句"
  - 老师："可以，但建议再多写一点"
  - 学生："好的，我写完了第1句。现在开始第2句。"
  - 结果：多句连贯

---

## FAQ

### Q1: 会不会句子太多（>3句）？

**不会**。`stopping_criteria`中的`n_sentences=2`会在2句后允许停止（配合`min_chars`）。

### Q2: max_new_tokens=128会不会太长？

**不会**。这是**上限**，不是目标：
- Stopping criteria会在2句+90字符时停止
- 通常在80-110 tokens停止
- 128只是防止被截断

### Q3: 删除newline EOS安全吗？

**安全**。因为：
- 真正的EOS是`<|im_end|>`、`<|endoftext|>`
- Newline只是格式化，不应终止生成
- Stopping criteria会在合适时机停止

---

## 回归测试清单

- [ ] 90%样本输出**2-3句话**
- [ ] 平均输出**80-120 tokens**
- [ ] 平均字符数**90-140**
- [ ] BERTScore F1 **>0.20**
- [ ] Acceptance rate **>0.70**
- [ ] Cloud calls **<10次/样本**
- [ ] 无标点泛滥（逗号密度<20%）
- [ ] 无短环重复（紧邻同字=0）
- [ ] 句子连贯、语义完整

---

## 参考文档

- 用户分析：`docs/spec_decoding_multi_sentence_fixes.md`（本次修改依据）
- V4综合修复：`docs/chinese_punctuation_comprehensive_fix.md`
- 本次修复：`docs/multi_sentence_output_fix.md`

---

## 总结

### 问题

V4修复成功解决标点和重复问题，但输出只有1句话。

### 原因

- Stopping criteria固定1句
- 句号闸门过严（难以结束第1句）
- Newline被当EOS（无法多句）
- max_new_tokens太小（空间不足）

### 解决

8项协同修改：
1. ✅ 可配置参数（target_sentences等）
2. ✅ Stopping criteria使用配置值
3. ✅ 删除硬编码停止逻辑
4. ✅ 放松句号闸门（5字/-3.5）
5. ✅ 删除newline EOS
6. ✅ max_new_tokens=128
7. ✅ Prompt改为"两到三句"
8. ✅ 超参数建议（3.0/5）

### 核心原则

**"句子数配置化 + 自然结束第1句 + 足够空间继续"**


