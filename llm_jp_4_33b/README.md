# LLM-jp-4 33B Base / Thinking on Google Colab

2026年8月18日に公開された **LLM-jp-4 33B** の、

- `llm-jp/llm-jp-4-33b-base`
- `llm-jp/llm-jp-4-33b-thinking`

を、Google Colab上で動かすための実験です。

今回のリポジトリでは、BaseとThinkingを別Notebookに分けています。

```text
LLM_JP_4_33B_Base.ipynb
LLM_JP_4_33B_Thinking.ipynb
```

どちらも33B denseモデルなので、FP16/BF16のままでは非常に大きく、Notebookでは **bitsandbytes NF4 4bit量子化**を利用しています。

著者環境では、**Google Colab無料版のNVIDIA T4ではBase版を4bitでもロードできませんでした**。

一方、Google Colab Proで割り当てられた

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
GPU VRAM: 94.97 GB
```

では、Base版について

```text
4bit model load
Japanese completion
Gradio UI
```

まで正常に動作することを確認しました。

Thinking版も同じ33B denseクラスなので、無料版T4では同様に厳しく、Colab Proの大容量GPUを第一ターゲットとしています。

---

## Models

### Base

```text
llm-jp/llm-jp-4-33b-base
```

https://huggingface.co/llm-jp/llm-jp-4-33b-base

Baseモデルは、pre-training / mid-trainingまでを行った基盤モデルです。

そのため、主な使い方は、

```text
Prompt
  ↓
Base Model
  ↓
文章の続きを生成
```

というcompletionです。

---

### Thinking

```text
llm-jp/llm-jp-4-33b-thinking
```

https://huggingface.co/llm-jp/llm-jp-4-33b-thinking

Thinkingモデルは、Baseモデルに対してpost-trainingされた推論モデルです。

LLM-jp-4のThinking系列では、SFT / DPOを用いたalignmentと、Harmony Response Formatを利用した推論形式が採用されています。

Notebookでは、

```text
reasoning_effort
=
low / medium / high
```

を切り替えて試せるようにしています。

---

## Why 4bit?

33B denseモデルをFP16/BF16で保持すると、単純計算でも、

```text
33B × 2 bytes
≈ 66 GB
```

程度になります。

そこでNotebookでは、bitsandbytesのNF4 4bit量子化を利用します。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

4bit化した場合でも、

```text
33B × 0.5 byte
≈ 16.5 GB
```

程度のweight容量が必要です。

さらに実際には、

- quantization metadata
- CUDA context
- activation
- KV cache
- temporary buffer

などが必要です。

---

## Google Colab Free / T4

著者環境では、無料版ColabのNVIDIA T4でBase版を試しました。

結果は、

```text
Google Colab Free
NVIDIA T4
VRAM 約14〜15 GB

LLM-jp-4 33B Base
NF4 4bit
→ model load ×
```

でした。

33B denseの4bit weightだけでも概算16.5GBあり、T4のVRAM容量を超えるため、この結果は不自然ではありません。

したがって、通常の

```text
Transformers
+
bitsandbytes NF4 4bit
```

という構成では、無料版T4は現実的ではありません。

Thinking版も同じ33B dense backboneなので、同様に厳しいと考えられます。

---

## Google Colab Pro

Base版では、Google Colab Proで、

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition
GPU VRAM: 94.97 GB
```

が割り当てられました。

実際に4bitでロードした結果は、

```text
model loaded: llm-jp/llm-jp-4-33b-base
input device: cuda:0
4bit: True

GPU allocated: 18.76 GB
GPU reserved : 19.15 GB
```

でした。

つまり、実際のロードでは約19GB前後のVRAMを使用しました。

この結果から、

> **33B denseのNF4 4bit実行では、24GB級以上のGPUが現実的な目安**

と考えられます。

---

## Final GPU Memory

Base版でGradioまで実行した後のGPUメモリは、

```text
GPU allocated: 18.77 GB
GPU reserved : 19.14 GB
GPU free      : 75.17 GB
GPU total     : 94.97 GB
```

でした。

モデルロード直後と大きく変わらず、推論・Gradioまで安定して動作しました。

---

# Base Notebook

Notebook:

```text
LLM_JP_4_33B_Base.ipynb
```

構成は、

```text
Google Colab GPU確認
        ↓
Google Drive / Hugging Face cache
        ↓
LLM-jp-4 33B BaseをNF4 4bitでロード
        ↓
日本語completion
        ↓
Gradio Completion UI
        ↓
GPUメモリ確認
```

です。

---

## Base Model Load

```python
MODEL_ID = "llm-jp/llm-jp-4-33b-base"
```

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)
```

---

## Japanese Completion

Baseモデルでは、質問応答ではなく文章の続きを生成させます。

Prompt:

```text
人工知能の歴史を振り返ると、近年の大規模言語モデルの発展には
```

実際の生成では、

```text
驚嘆するばかりです。GPT-3 のようなモデルは、テキスト生成や翻訳、質問応答など、多くのタスクで驚異的なパフォーマンスを発揮しています。
```

と続き、その後も人工知能史やニューラルネットワークについて日本語で文章が生成されました。

Baseモデルとして自然なcompletionが得られています。

---

## Base Gradio UI

Base版では、Chat UIではなくCompletion UIにしています。

```text
Prompt
+
max_new_tokens
+
temperature
        ↓
LLM-jp-4 33B Base
        ↓
Continuation
```

著者環境では、

```text
Gradio: 6.20.0
```

でpublic URLが生成され、正常に動作しました。

---

# Thinking Notebook

Notebook:

```text
LLM_JP_4_33B_Thinking.ipynb
```

構成は、

```text
Google Colab GPU確認
        ↓
Google Drive / Hugging Face cache
        ↓
LLM-jp-4 33B ThinkingをNF4 4bitでロード
        ↓
Harmony chat template
        ↓
reasoning_effort
        ↓
日本語Thinking
        ↓
low / medium / high比較
        ↓
Gradio UI
        ↓
GPUメモリ確認
```

です。

---

## Harmony Response Format

Thinking版では、通常のcompletionではなく、chat templateを利用します。

```python
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort=reasoning_effort,
)
```

Notebookでは、

```text
clean output
raw Harmony output
```

の両方を確認できるようにしています。

---

## reasoning_effort

Thinking版では、

```text
low
medium
high
```

を切り替えて試します。

例:

```python
clean, raw = thinking_generate(
    question,
    reasoning_effort="medium",
    max_new_tokens=512,
)
```

Thinkingモデルは通常回答より長いreasoningを生成する可能性があるため、Colabでは最初から非常に大きな`max_new_tokens`を使わず、

```text
low     : 256〜384
medium  : 384〜512
high    : 512〜1024
```

程度から試す構成にしています。

---

## Japanese Thinking Test

Notebookでは、

```text
日本語で答えてください。
9.11と9.8では、どちらの数が大きいですか？
理由も説明してください。
```

という簡単な推論問題を用意しています。

さらに、

```text
生成AIのハルシネーションが起きる理由を簡潔に説明してください。
```

という同一promptに対して、

```text
low
medium
high
```

を切り替え、reasoningの違いを比較できます。

---

## Thinking Gradio UI

Gradioでは、

```text
Prompt
+
reasoning_effort
+
max_new_tokens
        ↓
LLM-jp-4 33B Thinking
        ↓
Decoded output
+
Raw Harmony output
```

を確認できます。

Thinkingモデルでは長い生成が起こり得るため、同時実行数は1に制限しています。

---

# Base vs Thinking

今回の2モデルの違いは、単にモデル名だけではありません。

```text
LLM-jp-4 33B Base
→ pre-training / mid-training
→ completion model
→ 文章の続きを生成

LLM-jp-4 33B Thinking
→ post-trained reasoning model
→ Harmony Response Format
→ reasoning_effort
→ 推論を伴う回答
```

という違いがあります。

したがって、このリポジトリでも、BaseとThinkingを無理に同じUIへまとめず、それぞれの用途に合わせてNotebookを分けています。

---

# Colabで見えた実行境界

今回の実験では、かなり分かりやすい境界が見えました。

```text
8B級
→ 無料版T4でも比較的現実的

27B級
→ 4bitでも無料版T4では厳しい場合がある

33B dense
→ 無料版T4では困難
→ 約19GB VRAMを実測
→ 24GB級以上のGPUが現実的
```

これはモデル名や「4bit」という言葉だけでは分からない、実際のColab上の境界です。

---

# What Worked

## Base

```text
Google Colab Free / T4
Model Load          ×

Google Colab Pro / RTX PRO 6000
Model Load          ○
NF4 4bit            ○
Japanese Completion ○
Gradio              ○
```

## Thinking

Thinking版は同じ33B denseクラスなので、

```text
Google Colab Free / T4
→ 厳しい

Google Colab Pro / 大容量GPU
→ 4bitで実行対象
```

という位置づけです。

Thinking版の実測値は、実行後にREADMEへ追記してください。

---

# Repository Structure

例:

```text
LLM_JP_4_33B/
├── README.md
├── LLM_JP_4_33B_Base.ipynb
└── LLM_JP_4_33B_Thinking.ipynb
```

Base / Thinkingを同じディレクトリで管理する場合は、このREADME.mdをトップに置く構成を想定しています。

---

# Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

今回のNotebookは、上記書籍の公式サンプルそのものではありません。

書籍刊行後に公開された新しいopen-weightモデルを、書籍で扱ったGoogle Colabでの実装方法をベースに追加検証したものです。

本Notebookでより興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

をぜひお手にとってください。

---

# Links

- LLM-jp: https://llm-jp.nii.ac.jp/
- LLM-jp Hugging Face: https://huggingface.co/llm-jp
- LLM-jp-4 33B Base: https://huggingface.co/llm-jp/llm-jp-4-33b-base
- LLM-jp-4 33B Thinking: https://huggingface.co/llm-jp/llm-jp-4-33b-thinking
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEは2026年8月18〜19日時点のNotebookおよび実験結果に基づきます。Google ColabのGPU割り当て、VRAM、CUDA、PyTorch、Transformers、bitsandbytes、LLM-jp側の実装は今後変更される可能性があります。モデル利用時は配布元の最新README、ライセンス、利用条件を確認してください。
