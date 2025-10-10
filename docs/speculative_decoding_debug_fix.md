# Speculative Decoding — Debug & Fix Plan (MER2024 · Chinese · `prompt_type=detailed`)

> This note consolidates the concrete issues we observed (premature truncation, incoherent endings), pinpoints likely root causes in the speculative-decoding path, and provides **minimal patches** you can apply immediately. It also includes a practical tuning playbook and a sanity‑check checklist.

---

## 1) Symptoms (from your `speculative_decoding_mer_chinese_newprompt1.json`)

- **Premature truncation**: outputs end mid‑sentence or right after a period-like symbol; sometimes leak templating artifacts like `Human`.
- **Incoherent endings**: fragmented half‑words/clauses; second sentence drifts into chit‑chat or question forms.

These map to stopping logic and verification rhythm problems (see below).

---

## 2) Root causes (what’s going wrong)

### A. Over‑aggressive stopping logic (string‑level)
- The speculative path treats **common characters** (e.g., Chinese period `。`, newline, words like `Please`) as “EOS-like” and stops **as soon as one appears**. This **terminates valid sentences early**.
- Stopping is implemented **after decoding to text**, which is brittle with BPE/WordPiece (you often get *partial* characters or half‑tokens).

**Correct approach**: drive stopping by **token ids**, using the model’s `eos_token_id` (and model-specific stop sequences if any). If you want “one sentence only”, do it in **post‑processing** (count sentence‑final punctuation) rather than hijacking token‑level EOS.

> *Speculative decoding is designed to keep the target model’s distribution unchanged; ad‑hoc string stops alter the distribution.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192), [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)  
> *Use Hugging Face `StoppingCriteria` at the id level, not string contains.* [HF StoppingCriteria](https://huggingface.co/docs/transformers.js/main/en/api/generation/stopping_criteria)

### B. Handwritten sampling/penalties on specific Chinese tokens
- The draft path applies manual penalties to frequent Chinese tokens (e.g., characters like “声”, “，” etc.) and implements a partial **custom top‑p** filter. This distorts the distribution and causes **ungrammatical fragments**.
- In speculative decoding, the **draft** should be **deterministic or low‑temperature**, and **verification** should be the only gate that preserves the target distribution.

> *Prefer using the library’s built‑in generation controls; avoid custom top‑p/penalties unless you fully control the math.* [HF Generation args](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation)

### C. Verification is too permissive (rank threshold too large)
- Accepting a draft token whenever the target’s rank is within **top‑20** often lets **mediocre** candidates slip through repeatedly, compounding minor errors and yielding incoherent tails.
- Tighten to **top‑3 ~ top‑5** to keep accepted runs faithful and coherent.

> *Speculative decoding (and Medusa‑style multi‑token guessing) both rely on high‑quality prefixes; long tails require stricter acceptance to avoid drift.* [Speculative Decoding](https://arxiv.org/abs/2211.17192), [Medusa](https://arxiv.org/abs/2401.10774)

### D. Context sync after corrections
- After any correction by the target model, you **must** rebuild the **new prefix** and re‑generate the *next* draft from this corrected state (including attention mask / KV cache if used). Any desync leads to “fractured” endings.

### E. Triggering verification by per‑token entropy
- With `entropy_threshold` too high and checked at every step, you either **rarely** verify (letting the 3B draft run too long) or you verify **too frequently**, fragmenting the rhythm.
- Use a **lower threshold** and **evaluate every k tokens** instead of every step to stabilize cadence.

---

## 3) Minimal code fixes (drop‑in changes)

> Keep your overall structure. Change only what is necessary to fix truncation and fragmentation.

### 3.1 Token‑id based early stop (first full stop → stop)
Add a proper `StoppingCriteria` and pass it to **both** streaming and non‑streaming `generate(...)` calls:

```python
from transformers import StoppingCriteria, StoppingCriteriaList

class StopAtFirstSentence(StoppingCriteria):
    def __init__(self, tokenizer, stop_chars=("。", "."), min_new_tokens=24):
        self.stop_ids = [tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in stop_chars]
        self.min_new_tokens = min_new_tokens
        self.generated = 0

    def __call__(self, input_ids, scores, **kwargs):
        # count only newly generated tokens
        self.generated += 1
        if self.generated < self.min_new_tokens:
            return False
        return input_ids[0, -1].item() in self.stop_ids

stop_criteria = StoppingCriteriaList([StopAtFirstSentence(tokenizer)])
```

- Pass `stopping_criteria=stop_criteria` and **do not** string‑search for `<|im_end|>` in streamer text (you used `skip_special_tokens=True`, so you won’t see it anyway).  
- Provide **multiple EOS ids** if applicable: e.g., `[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]`.

### 3.2 Remove handwritten token penalties & custom top‑p
- Delete any ad‑hoc down‑weighting of specific Chinese tokens and hand‑rolled nucleus sampling.  
- Keep the **draft** deterministic: `do_sample=False`. This alone improves acceptance & coherence.

### 3.3 Tighten verify acceptance
- Shrink `rank_threshold` from **20 → 3~5**.  
- Log and monitor: *avg accepted tokens per verification*, *avg corrected tokens*, *acceptance rate*. Target: relatively **long accepted runs** with **few corrections**.

### 3.4 Strict context sync after correction
- After a correction, **rollback to the pre‑draft length**, append the **correct token**, rebuild `input_ids/attention_mask` (and KV cache if used), then continue drafting from this corrected prefix. Ensure `should_stop` is explicitly set in all branches.

### 3.5 Verification cadence & threshold
- Set `entropy_threshold ≈ 1.5` (nats) as a starting point.  
- Only *evaluate* entropy **every k tokens** (e.g., k=5), not at every step, to avoid jitter.  
- Increase `max_new_tokens` for detailed Chinese (e.g., **96–128**) and rely on the **stop‑at‑first‑period** criterion to end at the first complete sentence.

---

## 4) Prompt & generation pairing (content first, no chit‑chat)

Keep the official **system** (for Omni audio path). Put constraints in **user** only. Use the “information‑first” prompts (声学线索 → 单一情绪 → 极简因由，单句长句 30–70 汉字). Then pair with:

- `do_sample=False, no_repeat_ngram_size=2, repetition_penalty≈1.05`
- `eos_token_id=[model_eos, im_end_if_any]`
- `stopping_criteria=StopAtFirstSentence(...)`

See also: [HF generation args](https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation), [HF stopping criteria](https://huggingface.co/docs/transformers.js/main/en/api/generation/stopping_criteria), [HF chat templates](https://huggingface.co/docs/transformers/main/chat_templating).

---

## 5) Sanity‑check checklist (run on 20–50 samples)

1. **No premature stops**: each output ends exactly at the **first `。`** (or `.`) after *≥24* new tokens.  
2. **No template leakage**: strings like `Human/User` never appear (stop by ids, not by strings).  
3. **Acceptance rhythm**: median **accepted span ≥3 tokens**; few corrections; stable cadence.  
4. **No token penalties**: distribution shaped only by the model and official logits processors.  
5. **Length match**: detailed Chinese averages **30–70 汉字**; concise **8–16 汉字**.  
6. **Evaluation sanity**: BERTScore (zh) with `rescale_with_baseline=True`; BLEU/chrF with Chinese‑friendly configs.

---

## 6) References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding* (arXiv:2211.17192).  
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling* (arXiv:2302.01318).  
- Hugging Face *StoppingCriteria* (Transformers docs).  
- Hugging Face *Text generation* (generation args like `bad_words_ids`, `eos_token_id`).  
- Hugging Face *Chat templates* & `apply_chat_template`.  
- *Medusa*: multi‑token prediction with multiple heads (context quality still crucial).
