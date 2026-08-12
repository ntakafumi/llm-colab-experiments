# LLM-jp-4 8B on Google Colab

![LLM-jp](https://img.shields.io/badge/LLM--jp-4-0066CC)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.2.0-FFD21E)
![bitsandbytes](https://img.shields.io/badge/bitsandbytes-4bit%20NF4-4B8BBE)

LLM-jpが公開している **LLM-jp-4 8B** 系列の

- Base
- Instruct
- Thinking

の3モデルを、Google Colab上でbitsandbytesによる4bit量子化を用いて試す実験です。

同じ8Bクラスのモデルでも、**Base / Instruct / Thinkingでは目的と使い方が異なります。**  
このディレクトリでは、その違いがNotebookのコードから分かるように、3つを独立したNotebookとして用意しています。

> **確認結果（2026-08-13）**  
> `llm-jp-4-8b-instruct` はGoogle Colab Proで動作を確認しました。  
> 一方、筆者が試した無料版Google Colab環境では、4bit量子化を利用してもモデルロード中にGPUメモリ不足となり、動作させることができませんでした。

---

## Models

| Model | Role | Hugging Face |
|---|---|---|
| LLM-jp-4 8B Base | 文章の続きを生成する基盤モデル | https://huggingface.co/llm-jp/llm-jp-4-8b-base |
| LLM-jp-4 8B Instruct | 指示追従・通常チャット向けモデル | https://huggingface.co/llm-jp/llm-jp-4-8b-instruct |
| LLM-jp-4 8B Thinking | 推論を行うチャット向けモデル | https://huggingface.co/llm-jp/llm-jp-4-8b-thinking |

今回のNotebookでは、モデルをそのままGPUへ載せるのではなく、**bitsandbytes / NF4による4bit量子化**を利用します。

---

## Notebooks

### 1. LLM-jp-4 8B Base

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/llm-colab-experiments/blob/main/llm-jp-4/LLMJP4_8B_Base_Colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Base-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/llm-jp/llm-jp-4-8b-base)

```text
LLMJP4_8B_Base_Colab.ipynb
```

Baseモデルは、Instruct / Thinkingモデルのようなチャットモデルとしては扱いません。

たとえば、

```text
人工知能とは、
```

という文章の書き出しを与え、その**続きを生成**させます。

このNotebookでは、

- `apply_chat_template()`を使わない
- system / user形式のmessagesを使わない
- Harmony用のresponse parserを使わない
- 文字列promptを直接tokenizeする
- GradioもチャットではなくシンプルなText Completion UIにする

という構成にしています。

---

### 2. LLM-jp-4 8B Instruct

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/llm-colab-experiments/blob/main/llm-jp-4/LLMJP4_8B_Instruct_Colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Instruct-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/llm-jp/llm-jp-4-8b-instruct)

```text
LLMJP4_8B_Instruct_Colab.ipynb
```

Instructモデルは通常のチャット用途向けです。

LLM-jp-4のInstructモデルではHarmony形式の応答を扱うため、Notebookでは、

```python
trust_remote_code=True
```

を指定し、モデルリポジトリに含まれるTokenizer / Parserを利用します。

入力は、

```python
tokenizer.apply_chat_template(...)
```

で構築し、生成後は、

```python
parsed = tokenizer.parse_response(response)
content = parsed.get("content")
```

として最終回答部分を取り出します。

これにより、Harmony形式のspecial token等をそのまま表示せず、通常のチャット回答として扱います。

---

### 3. LLM-jp-4 8B Thinking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/llm-colab-experiments/blob/main/llm-jp-4/LLMJP4_8B_Thinking_Colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Thinking-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/llm-jp/llm-jp-4-8b-thinking)

```text
LLMJP4_8B_Thinking_Colab.ipynb
```

Thinkingモデルでは、Instructモデルと同様にHarmony形式を利用します。

さらにchat templateに、

```python
reasoning_effort="medium"
```

を指定します。

今回のNotebookではUIをシンプルにするため、`reasoning_effort` は **medium固定**です。

生成された応答は専用parserで処理し、Gradioには最終回答だけを表示します。

---

## Base / Instruct / Thinkingの違い

| Model | 入力 | 主な出力 | UI |
|---|---|---|---|
| Base | 文章の書き出し | 文章の続き | Text Completion |
| Instruct | 指示・質問 | 通常の回答 | Chat |
| Thinking | 質問 | 推論を経た回答 | Chat |

たとえば、

```text
人工知能とは、
```

の続きを生成したい場合はBaseモデル、

```text
人工知能とは何ですか？
```

と質問して回答を得たい場合はInstructモデル、

より推論を必要とする質問を扱いたい場合はThinkingモデル、

というように使い分けます。

---

## 4bit quantization

3つのNotebookでは、共通してbitsandbytesによるNF4量子化を利用しています。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)
```

GPUがBF16に対応している場合は`torch.bfloat16`、対応していない場合は`torch.float16`を利用します。

```python
COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)
```

---

## Environment

Notebookでは主に次のライブラリを利用します。

```text
torch
transformers==5.2.0
accelerate>=1.13.0
bitsandbytes
sentencepiece
gradio
```

モデルキャッシュは、必要に応じてGoogle Driveへ保存します。

---

## Tested environment

実験日: **2026-08-13**

### Google Colab Pro

`llm-jp-4-8b-instruct`について、Google Colab Pro環境でモデルロードからGradioチャットまで動作を確認しました。

Base / Thinkingについても、同じLLM-jp-4 8B系列を同じ4bit構成で試すNotebookとして用意しています。

> Base / Thinkingの実測結果については、確認後に必要に応じて本READMEを更新してください。

### Google Colab Free

筆者が2026-08-13に試した無料版Google Colab環境では、`llm-jp-4-8b-instruct` のロード中にCUDA Out of Memoryとなりました。

モデルロードの約95%まで進んだ時点で、

```text
GPU total capacity : 14.56 GiB
GPU memory in use  : 14.53 GiB
GPU free            : 31.81 MiB
```

となり、追加の112MiBを確保できずロードに失敗しました。

実際のエラーは次のようなものでした。

```text
OutOfMemoryError: CUDA out of memory.
Tried to allocate 112.00 MiB.
GPU 0 has a total capacity of 14.56 GiB
of which 31.81 MiB is free.
```

そのため、**今回のTransformers + bitsandbytes 4bit構成では、筆者の無料版Colab環境でLLM-jp-4 8B Instructを動作させることはできませんでした。**

Base / Thinkingも同じ8B系列であるため、今回の構成では無料版Colabでの利用は推奨していません。

ただし、これは

> 「LLM-jp-4 8Bは無料版Colabでは絶対に動かない」

という意味ではありません。

Google Colabでは、割り当てられるGPUやメモリなどが時期や利用状況によって変化する可能性があります。

また、

- CPU offload
- 別の量子化方式
- 事前量子化済みモデル
- GGUFなど別形式のモデル

を利用すれば、異なる結果になる可能性があります。

ここでは、**今回作成したTransformers + bitsandbytes 4bit Notebookを、実験時点の無料版Colab環境で実行した結果**を記録しています。

---

## Why can an 8B 4bit model still run out of VRAM?

単純計算だけを見ると、

```text
8.6B parameters × 4bit ≒ 4.3 GB
```

程度に見えます。

しかし、実際にはモデル重み以外にも、

- 量子化用のscale等
- 4bit計算用の内部表現
- CUDA / PyTorchが利用するメモリ
- モデルロード時の一時バッファ
- 各種buffer
- KV Cache
- 生成時の中間テンソル

などが必要です。

したがって、

> **8Bモデルを4bit量子化すれば4～5GBのVRAMだけで動く**

という単純な計算にはなりません。

今回、無料版Colabで実際に約14.5GiBまでGPUメモリを使用したことは、量子化モデルの実メモリ使用量を考える上でも興味深い結果でした。

---

## `trust_remote_code=True`について

Instruct / Thinkingモデルでは、

```python
trust_remote_code=True
```

を指定しています。

これにより、Hugging Face Hub上のモデルリポジトリに含まれるPythonコードを読み込んで実行します。

実行時にはHugging Faceから、

```text
Make sure to double-check they do not contain any added malicious code.
To avoid downloading new versions of the code file, you can pin a revision.
```

という警告が表示される場合があります。

本NotebookではLLM-jp公式モデルリポジトリを利用していますが、再現性やセキュリティをより厳密に管理する場合には、利用するrevisionを固定することも検討してください。

---

## Files

```text
llm-jp-4/
├── README.md
├── LLMJP4_8B_Base_Colab.ipynb
├── LLMJP4_8B_Instruct_Colab.ipynb
└── LLMJP4_8B_Thinking_Colab.ipynb
```

---

## References

- LLM-jp-4 8B Base  
  https://huggingface.co/llm-jp/llm-jp-4-8b-base

- LLM-jp-4 8B Instruct  
  https://huggingface.co/llm-jp/llm-jp-4-8b-instruct

- LLM-jp-4 8B Thinking  
  https://huggingface.co/llm-jp/llm-jp-4-8b-thinking

- LLM-jp-4 Cookbook  
  https://github.com/llm-jp/llm-jp-4-cookbook

- LLM-jp  
  https://llm-jp.nii.ac.jp/

---

## Related repository

この実験は、個人的なGoogle Colab実験リポジトリ

```text
ntakafumi/llm-colab-experiments
```

の一部です。

https://github.com/ntakafumi/llm-colab-experiments/

このリポジトリは、書籍『ブラウザで動かすLLM実装入門』の公式サンプルとは独立した、著者個人の追加実験用リポジトリです。

---

## Notes

- Google ColabのGPU、メモリ、CUDA、Pythonパッケージは将来変更される可能性があります。
- 同じNotebookでも、実行時期によって結果が異なる場合があります。
- 各モデルの最新ライセンス・利用条件は配布元を確認してください。
- Instruct / Thinkingモデルではモデルリポジトリのremote codeを利用しています。
- 本READMEの動作結果は、2026-08-13時点の著者環境に基づいています。
