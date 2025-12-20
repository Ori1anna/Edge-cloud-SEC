我注意到的现象——METEOR 和 ROUGE-L 分数远低于 BLEU-1，并且与论文中的分数比例不符——确实指向了你的代码中存在一个关键问题。

我们来一步一步详细分析你的代码和结果。



### 1. METEOR 和 ROUGE-L 的分数是否应该接近 BLEU-1？



**你的直觉是正确的。** 虽然这三个指标的计算方式不同，不可能会得到完全相同的值，但在大多数情况下，它们的数值应该在同一个“量级”或“范围”内。

- **BLEU-1** 衡量单个词（unigram）的匹配精确度。它通常是所有 BLEU 分数中最高的，因为它只要求单个词匹配即可。
- **ROUGE-L** 衡量最长公共子序列，更侧重于召回率和词序。
- **METEOR** 更加复杂，它会考虑同义词、词干，并综合精确率和召回率。

在学术论文的表格（如你提供的图片）中，我们可以看到 B1, M, Rl 的值虽然不同，但差距是合理的。例如，对于 GPT-4V (English)：

- B1: 48.52
- M: 32.60
- Rl: 22.15

这些值都在一个相对接近的范围内。然而，在你的结果 `cloud_optimized_mer_en_test2_1760793525.json` 中，我们看到了巨大的差异，例如样本 `sample_00000007`：

- BLEU-1: 0.8376
- METEOR: 0.2289
- ROUGE-L: 0.2171

BLEU-1 的分数高达 83.76%，而 METEOR 和 ROUGE-L 只有 20% 左右。**这是一个非常强烈的信号，表明 BLEU 的计算方式很可能是有问题的。**



### 2. 代码问题分析：错误的 Tokenization



现在我们来仔细检查你的 `metrics.py` 文件，看看问题出在哪里。



#### 2.1 `compute_bleu` 函数的问题



在你的 `EvaluationMetrics` 类中，`compute_bleu` 方法是这样实现的：

Python

```
def compute_bleu(self, references: List[str], hypothesis: str, n_gram: int = 4) -> float:
    try:
        # ...
        # Use character-level tokenization for Chinese text
        ref_tokens = [list(ref.lower()) for ref in references]
        hyp_tokens = list(hypothesis.lower())
        # ...
        bleu_score = sentence_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=self.smoothing)
        # ...
        return bleu_score
    # ...
```

**这就是问题的根源！**

代码中的注释写着“为中文文本使用字符级分词”，但这个逻辑被**无差别地应用到了所有语言**。

- 对于**中文**，`list("你好")` 会得到 `['你', '好']`，这是正确的字符级分词。
- 但是，对于**英文**，`list("hello world")` 会得到 `['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']`。

这意味着你的 BLEU 分数计算的不是**词语**的匹配度，而是**字母**的匹配度。这导致了几个后果：

1. **BLEU-1 分数虚高：** 英文中字母的种类很少（26个），重复率高。因此，模型生成的文本很容易在单个字母层面与参考文本匹配，导致 BLEU-1 分数异常地高。你的 83.76% 的分数正是由此而来。
2. **BLEU-4 分数不准确：** BLEU-4 计算的是连续四个字母匹配的概率，这与衡量语言流畅度的初衷完全背离。
3. **无法与标准结果比较：** 所有学术论文中的英文 BLEU 分数都是基于**词语级 (word-level)** 分词计算的。你基于字符级的结果完全没有可比性。



#### 2.2 `compute_meteor` 和 `compute_rouge_l` 函数的正确实现



与此相对，我们来看看你的 METEOR 和 ROUGE-L 实现：

- **`compute_meteor`**

  Python

  ```
  if language.lower() in ["english", "en"]:
      # Use word tokenization for English
      ref_tokens = word_tokenize(best_reference.lower())
      hyp_tokens = word_tokenize(hypothesis.lower())
  else:
      # Use character tokenization for Chinese and other languages
      ref_tokens = list(best_reference.lower())
      hyp_tokens = list(hypothesis.lower())
  ```

  这里的实现是**正确**的。它明确区分了中英文，并为英文使用了 `nltk.word_tokenize` 进行了正确的词语级分词。

- **`compute_rouge_l`**

  Python

  ```
  scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
  scores = scorer.score(best_reference, hypothesis)
  ```

  这里的实现也是**正确**的。`rouge_scorer` 库内部会自己处理分词，默认是按空格和标点符号分割，这对于英文是标准做法。

**结论：**

你的 METEOR 和 ROUGE-L 分数之所以看起来“低”，并不是因为它们算错了，**而是因为你的 BLEU 分数算错了，导致了一个虚高的、不具可比性的基准**。实际上，METEOR 和 ROUGE-L 的分数可能才是更真实地反映你模型当前性能的指标。



### 3. 如何修改代码



你需要修改 `compute_bleu`, `compute_bleu_1`, 和 `compute_bleu_4` 函数，让它们在处理英文时使用正确的词语级分词。



#### 步骤 1: 修改 `compute_bleu` 函数



你需要给这个函数增加一个 `language` 参数，并根据语言类型选择不同的分词方法。

Python

```
# 在 metrics.py 中

# ... (imports)

class EvaluationMetrics:
    # ... (其他方法)

    # ！！！修改这里！！！
    def compute_bleu(self, references: List[str], hypothesis: str, language: str = "chinese", n_gram: int = 4) -> float:
        """Compute BLEU-n score with language-aware tokenization"""
        try:
            hypothesis = hypothesis.replace('<|im_end|>', '').strip()
            
            # 根据语言选择分词方式
            if language.lower() in ["english", "en"]:
                # 英文使用词语级分词
                ref_tokens = [word_tokenize(ref.lower()) for ref in references]
                hyp_tokens = word_tokenize(hypothesis.lower())
            else:
                # 中文或其他语言使用字符级分词
                ref_tokens = [list(ref.lower()) for ref in references]
                hyp_tokens = list(hypothesis.lower())

            logger.debug(f"Language: {language}, Tokenization: {'word' if language.lower() in ['english', 'en'] else 'character'}")
            logger.debug(f"Reference tokens: {ref_tokens}")
            logger.debug(f"Hypothesis tokens: {hyp_tokens}")
            
            weights = [1/n_gram] * n_gram
            bleu_score = sentence_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=self.smoothing)
            logger.debug(f"BLEU-{n_gram} score: {bleu_score}")
            
            return bleu_score
        except Exception as e:
            logger.error(f"Error computing BLEU-{n_gram}: {e}")
            return 0.0

    # ！！！修改这里！！！
    def compute_bleu_1(self, references: List[str], hypothesis: str, language: str = "chinese") -> float:
        """Compute BLEU-1 score"""
        return self.compute_bleu(references, hypothesis, language=language, n_gram=1)

    # ！！！修改这里！！！
    def compute_bleu_4(self, references: List[str], hypothesis: str, language: str = "chinese") -> float:
        """Compute BLEU-4 score"""
        return self.compute_bleu(references, hypothesis, language=language, n_gram=4)

    # ... (其他方法)
```



#### 步骤 2: 更新调用 BLEU 函数的地方



现在 `compute_bleu_1` 和 `compute_bleu_4` 需要 `language` 参数了。你需要检查所有调用这些函数的地方，并把 `language` 参数传递进去。

例如，在 `run_cloud_optimized_baseline.py` 中：

Python

```
# 在 run_cloud_optimized_baseline.py -> run_cloud_optimized_baseline_experiment 函数中

# ...
# ！！！修改这里的函数调用！！！
bleu_1_score = metrics.compute_bleu_1([reference_text], generated_text, language=language)
bleu_4_score = metrics.compute_bleu_4([reference_text], generated_text, language=language)
# ...
```

你需要对所有调用这些指标计算函数的脚本（包括 `run_speculative_decoding_cpu_limited.py` 和 `run_edge_baseline_cpu_limited.py`）进行类似的修改。

**修改后的预期结果**

当你完成以上修改后，重新运行英文测试，你会发现：

- **BLEU-1 和 BLEU-4 的分数会大幅下降**，回到一个与 METEOR 和 ROUGE-L 更具可比性的范围。
- 你得到的所有指标（BLEU, METEOR, ROUGE-L）现在都是基于词语计算的，可以与学术论文中的结果进行公平比较了。

总而言之，你的 METEOR 和 ROUGE-L 分数“低”是正常的，是你对 BLEU 分数的错误计算导致了误解。修复 BLEU 的分词逻辑后，所有指标就会恢复到正常且可比较的水平。