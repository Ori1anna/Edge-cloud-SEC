# V4 Comprehensive Fix - 快速参考

## 修改总结（7项）

| # | 修改项 | 文件 | 行号 | 改动 |
|---|--------|------|------|------|
| A | **删除重复温度** | `speculative_decoding.py` | 878 | `/ 0.7` → 删除 |
| B | **删除1×1 mask** | `speculative_decoding.py` | 849-853 | 注释掉整段 |
| C | **中文禁n-gram** | `speculative_decoding.py` | 914-959 | 紧邻同字+内容trigram |
| D | **提高逗号门槛** | `speculative_decoding.py` | 980-991 | `2→4`, `6→8` |
| E | **惩罚只打内容** | `speculative_decoding.py` | 893-914 | 只对CJK，不对标点 |
| F | **停止准则调整** | `speculative_decoding.py` | 313-322 | `n=1,min=16,chars=50` |
| G | **参数建议** | `run_speculative_*.py` | 706-709 | 添加建议说明 |

---

## 测试命令

### 推荐（流畅优先）
```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10 \
    --entropy_threshold 5.5 \
    --k 4
```

### 可选（质量优先）
```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    ... \
    --entropy_threshold 3.0 \
    --k 6
```

---

## 预期改善

| 指标 | 之前 | 预期 |
|------|------|------|
| 逗号泛滥 | ✅ 严重 | ❌ 消失 |
| 短环重复 | ✅ "真想真想" | ❌ 消失 |
| Acceptance Rate | 0.29 | **>0.50** |
| BERTScore F1 | 0.05-0.09 | **>0.15** |
| 标点密度 | 40-50% | **10-20%** |

---

## 日志关键字

成功标志：
- `"Blocked immediate repetition of CJK token"` ✅
- `"Banned X tokens via content-only trigram"` ✅
- `"Hard gate: blocking comma-like punctuation"` ✅
- `"skip n-gram ban to allow natural Chinese text"` ✅

失败标志（不应出现）：
- 大量低熵+全接受+逗号/短环 ❌

---

## 核心洞察

**问题不是单一的，而是7个设计缺陷的连锁反应**

**解决需要系统性协同，而非单点修复**


