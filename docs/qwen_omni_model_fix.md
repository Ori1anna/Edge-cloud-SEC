# Qwen2.5-Omni Model Loading Fix

## ⚠️ 错误信息

```
You are using a model of type qwen2_5_omni to instantiate a model of type qwen2_audio. 
This is not supported for all configurations of models and can yield errors.
```

---

## 🔍 问题分析

### 原代码（错误）

```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if 'cuda' in device else torch.float32,
    device_map=device if 'cuda' in device else None
)
```

**问题**:
- ❌ 使用了 **`Qwen2AudioForConditionalGeneration`** (旧版本的类)
- ❌ Qwen2.5-Omni 模型应该使用专门的 **`Qwen2_5OmniForConditionalGeneration`** 类
- ❌ 类型不匹配导致警告和潜在错误

---

## ✅ 官方正确用法

### 来源：`Qwen2.5-Omni-README.md` Line 744-760

```python
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# default: Load the model on the available device(s)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype="auto",      # ← 使用 "auto" 而不是手动指定
    device_map="auto"        # ← 使用 "auto" 自动分配
)

processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
```

---

## 🔧 修复方案

### 修改1: 导入正确的类 (Line 26)

```python
# 错误 ❌
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# 正确 ✅
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
```

---

### 修改2: 使用官方推荐的加载方式 (Line 46-75)

```python
def load_qwen_model(model_name: str, device: str):
    """
    Load Qwen-2.5-Omni model and processor (text-only mode)
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    
    # Load model (use torch_dtype="auto" as recommended by official docs)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",    # ✅ 官方推荐使用 "auto"
        device_map="auto"       # ✅ 自动分配设备
    )
    
    # Disable audio output to save memory (we only need text output)
    model.disable_talker()      # ✅ 节省约2GB显存
    
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model, processor
```

---

### 修改3: 添加 `return_audio=False` (Line 137-148)

```python
# Generate (text-only, no audio output)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        top_p=0.9,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        return_audio=False  # ✅ 只返回文本输出
    )
```

---

## 📚 官方文档参考

### 1. Text-only Generation Mode

**来源**: `Qwen2.5-Omni-README.md` Line 1001-1023

```python
# Method 1: Disable talker after loading
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="auto"
)
model.disable_talker()  # Save about 2GB of GPU memory

# Method 2: Set return_audio=False during generation
text_ids = model.generate(**inputs, return_audio=False)
```

**说明**:
- `model.disable_talker()`: 禁用音频生成模块，节省约2GB显存
- `return_audio=False`: 生成时只返回文本，不生成音频

---

### 2. 推荐的加载方式

**来源**: `Qwen2.5-Omni-README.md` Line 744-760

```python
# Recommended: Use torch_dtype="auto" and device_map="auto"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B", 
    torch_dtype="auto",      # Auto-select dtype based on device
    device_map="auto"        # Auto-distribute layers across devices
)
```

**优势**:
- ✅ 自动选择最佳数据类型（FP16/BF16/FP32）
- ✅ 自动分配模型层到可用设备
- ✅ 更好的内存管理

---

### 3. FlashAttention-2 加速（可选）

**来源**: `Qwen2.5-Omni-README.md` Line 1043-1064

```python
# For better performance, enable FlashAttention-2
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Enable Flash Attention
)
```

**需要先安装**:
```bash
pip install -U flash-attn --no-build-isolation
```

---

## 🎯 修改前后对比

### Before (错误)

| 项目 | 原版本 | 问题 |
|------|--------|------|
| Import | `Qwen2AudioForConditionalGeneration` | ❌ 类型不匹配 |
| Processor | `AutoProcessor` | ❌ 不是专用处理器 |
| torch_dtype | `torch.float16` | ⚠️ 手动指定 |
| device_map | `device if 'cuda' in device else None` | ⚠️ 手动管理 |
| Talker | 未禁用 | ⚠️ 浪费2GB显存 |
| return_audio | 未设置 | ⚠️ 可能返回音频 |

### After (正确)

| 项目 | 修复版本 | 优势 |
|------|---------|------|
| Import | `Qwen2_5OmniForConditionalGeneration` | ✅ 正确类型 |
| Processor | `Qwen2_5OmniProcessor` | ✅ 专用处理器 |
| torch_dtype | `"auto"` | ✅ 自动优化 |
| device_map | `"auto"` | ✅ 自动分配 |
| Talker | `model.disable_talker()` | ✅ 节省2GB |
| return_audio | `False` | ✅ 明确只要文本 |

---

## 🚀 预期效果

### Before (原版本)

```
WARNING: You are using a model of type qwen2_5_omni to instantiate 
a model of type qwen2_audio. This is not supported...
```

### After (修复后)

```
INFO: Loading model: Qwen/Qwen2.5-Omni-7B
INFO: Model loaded successfully on cuda:0
```

**无警告，正常加载** ✅

---

## 📊 性能优化总结

| 优化项 | 效果 | 说明 |
|--------|------|------|
| 使用正确的类 | 消除警告 | 避免类型不匹配 |
| `torch_dtype="auto"` | 自动优化 | 根据设备选择最佳类型 |
| `device_map="auto"` | 自动分配 | 多GPU自动平衡 |
| `disable_talker()` | 节省2GB显存 | 只需文本输出 |
| `return_audio=False` | 加快生成 | 跳过音频生成 |

---

## ✅ 测试命令

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

python tools/extract_emotion_labels.py \
    --input_json experiments/results/cloud_mer_en_test1.json \
    --output_csv MERTools/MER2024/ov_store/predict-openset-qwen-fixed.csv \
    --model_name Qwen/Qwen2.5-Omni-7B \
    --device cuda:0
```

**预期**: 
- ✅ 无警告信息
- ✅ 正常加载模型
- ✅ 成功生成标签

---

## 📝 总结

### 核心修改

1. ✅ **Import**: `Qwen2_5OmniForConditionalGeneration` + `Qwen2_5OmniProcessor`
2. ✅ **Loading**: `torch_dtype="auto"` + `device_map="auto"`
3. ✅ **Optimization**: `model.disable_talker()` + `return_audio=False`

### 参考文档

- `Qwen2.5-Omni-README.md` Line 744-760 (推荐加载方式)
- `Qwen2.5-Omni-README.md` Line 1001-1023 (Text-only mode)
- `qwen2_5_omni.md` Line 106-158 (Text-only generation)

**所有修改都基于官方文档推荐！** 📚✅

