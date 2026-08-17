# Nanbeige4.2-3B on Google Colab Free

Nanbeige LLM Labの軽量Agentモデル **Nanbeige4.2-3B** を、Google Colab無料版のNVIDIA T4 GPU上で動かす実験です。

今回のNotebookでは、

```text
Nanbeige/Nanbeige4.2-3B
```

をFP16で読み込み、

- 日本語チャット
- Thinking ON / OFF
- 簡単な推論
- Tool Calling
- Gradio UI
- GPU / RAM使用量の確認

までを試します。

著者環境では、**Google Colab無料版 / NVIDIA T4 GPUで、FP16のままモデルロードから日本語生成、Thinking、Tool Calling、Gradio UIまで動作することを確認しました。**

---

## Model

- Model: `Nanbeige/Nanbeige4.2-3B`
- Developer: Nanbeige LLM Lab
- Task: Text Generation / Agentic LLM
- Model size: 4B total class / 3B non-embedding class
- Official weights: approximately 8.36 GB
- Precision in this Notebook: FP16
- Runtime: Transformers
- Environment: Google Colab Free
- GPU: NVIDIA T4
- Interface: Notebook + Gradio
- License: Apache-2.0

Nanbeige4.2-3Bは、Nanbeige4.2-3B-Baseをベースにしたcompact agentic modelで、通常のチャットだけでなく、

```text
Reasoning
Tool Use
Office Agent
Code Agent
Agentic Workflow
```

などを意識したモデルです。

公式モデルカード:

https://huggingface.co/Nanbeige/Nanbeige4.2-3B

---

## Notebook

`Nanbeige4_2_3B.ipynb`

Notebookの構成は、拙著で利用しているGoogle Colab Notebookとできるだけ同じ流れにしています。

```text
Google ColabのGPU確認
        ↓
Google Drive / Hugging Face cache設定
        ↓
Nanbeige4.2-3BをFP16で読み込む
        ↓
Transformers互換性修正
        ↓
日本語チャット
        ↓
Thinking OFF / ON
        ↓
Tool Calling
        ↓
RAM / VRAM cleanup
        ↓
Gradio UI
        ↓
GPUメモリ確認
```

---

## Environment

著者環境では、

```text
Python:       3.12.13
PyTorch:      2.11.0+cu128
Transformers: 4.57.6
GPU:          NVIDIA Tesla T4
GPU VRAM:     14.56 GB
```

で動作確認しました。

依存関係は、

```python
%pip -q install -U "transformers>=4.57,<5" accelerate sentencepiece
```

としています。

最新モデルではcustom model codeとTransformers側のAPI変更が衝突する場合があります。

そのため、このNotebookではTransformers 5系ではなく、4.57系を利用しています。

---

## Google Drive Cache

Hugging FaceのcacheはGoogle Driveへ保存します。

```python
PROJECT_DIR = Path(
    "/content/drive/MyDrive/Colab Notebooks/LocalLLM"
)

CACHE_DIR = (
    PROJECT_DIR
    / "Program"
    / "hf_cache_nanbeige4_2_3b"
)
```

Nanbeige4.2-3Bのweightsは約8.36GBあるため、一度ダウンロードしたモデルを再利用できるようにしています。

---

## FP16でモデルをロード

今回の本命は4bit量子化ではなく、**FP16です**。

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)
```

著者環境では、Google Colab無料版T4上で正常にロードできました。

実測値は、

```text
model loaded: Nanbeige/Nanbeige4.2-3B
input device: cuda:0
model dtype: torch.float16
memory footprint: 7.77 GB
GPU allocated: 7.78 GB
GPU reserved : 8.46 GB
```

でした。

つまり、

```text
Nanbeige4.2-3B
+
FP16
+
T4 14.56 GB
```

の組み合わせでも、短いcontextの通常推論であれば十分に実行可能でした。

---

## Why FP16?

公式weightsは約8.36GBです。

Google Colab無料版T4のVRAMは著者環境で14.56GBだったため、

```text
weights
+
activation
+
CUDA context
+
generation buffer
```

を考慮しても、短い生成であればFP16のまま動かせる可能性があります。

今回実際に、

```text
GPU allocated ≈ 7.8 GB
GPU total     ≈ 14.56 GB
```

でロードできました。

そのため、最初から4bitへ落とさず、

> **まずFP16でモデル本来の挙動を確認する**

方針にしています。

Notebook末尾には、FP16でOOMした環境向けにNF4 4bit fallbackセルも残しています。

---

## Important: Transformers Compatibility

今回、一番大きかったのはGPUメモリではなく、**Nanbeige4.2-3Bのcustom codeと現在のTransformersの互換性**でした。

最初の生成では、

```text
AttributeError:
'DynamicCache' object has no attribute 'get_max_length'
```

が発生しました。

Nanbeige側のcustom model codeが旧KV cache APIである、

```python
get_max_length()
```

を利用している一方、現在のTransformersではAPIが変更されています。

そこでNotebookでは互換aliasを追加しています。

```python
from transformers.cache_utils import Cache

if not hasattr(Cache, "get_max_length"):

    def get_max_length_compat(self):
        return self.get_max_cache_shape()

    Cache.get_max_length = get_max_length_compat
```

これにより最初のAPIエラーを回避します。

---

## KV Cacheを無効化

次に、

```text
RuntimeError:
The size of tensor a (...) must match
the size of tensor b (...)
```

というRoPE / position系のエラーが発生しました。

これは生成時のKV cache処理でtoken長がずれる互換性問題と考えられます。

そのため、このNotebookの通常生成では、

```python
use_cache=False
```

を指定します。

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=max_new_tokens,
    use_cache=False,
    ...
)
```

これによって、著者環境では通常生成、Thinking、Tool Callingまで実行できました。

ただし、

```text
use_cache=False
```

では過去のKey / Valueを再利用しないため、長い生成ほど速度面で不利になります。

したがって、このNotebookでは、

```text
Normal   : 128～256 tokens程度
Thinking : 256～512 tokens程度
```

の短めの生成から試すことを推奨します。

---

## Japanese Chat

公式モデルカード上の言語は英語・中国語ですが、日本語でも試しました。

質問例:

```text
生成AIにおけるハルシネーションとは何ですか？
大学1年生にも分かるように簡潔に説明してください。
```

著者環境では、日本語で、

```text
ハルシネーションとは、
生成AIが存在しない事実や情報を
真実らしく生成してしまう現象
```

という内容を含む自然な説明が得られました。

つまり、公式に日本語特化モデルとして案内されているわけではありませんが、今回の簡単な日本語テストでは十分に応答しました。

---

## Thinking OFF / ON

Nanbeige4.2-3Bのchat templateでは、

```python
enable_thinking=False
```

または、

```python
enable_thinking=True
```

を切り替えられます。

比較用に、

```text
9.11と9.8では、どちらの数が大きいですか？
```

という問題を与えました。

Thinking OFFでは、日本語で、

```text
9.8の方が大きい
```

と正しく回答しました。

Thinking ONでは、内部的な推論内容が英語で長く出力されることも確認できました。

したがって、

```text
Normal
→ 最終回答を簡潔に得る

Thinking
→ 推論過程を含む長い出力
```

という違いがあります。

なお、Thinkingを有効にすると生成token数が増えやすいため、無料版T4では`max_new_tokens`を小さく設定しています。

---

## Tool Calling

Nanbeige4.2-3BはTool Callingにも対応しています。

Notebookでは、

```text
multiply(a, b)
```

という単純な掛け算ツールをモデルへ提示し、

```text
123.4 × 56.7を計算してください。
必要なら利用可能なツールを使ってください。
```

と質問しました。

著者環境では、

```xml
<tool_call>
<function=multiply>
<parameter=a>
123.4
</parameter>
<parameter=b>
56.7
</parameter>
</function>
</tool_call>
```

というTool Calling出力が得られました。

公式では、

```python
tool_call_format="xml"
```

が推奨されています。

このNotebookでもXML形式を利用しています。

---

## RAM / VRAM Cleanup

Tool Callingや複数回の推論後、Gradioを起動する前にRAM / VRAM cleanupを入れています。

```python
gc.collect()
torch.cuda.empty_cache()
```

に加えて、不要な一時変数やIPythonのoutput cache、過去のtracebackへの参照も可能な範囲で解放します。

今回の実測ではcleanup前後で、

```text
Python RAM : 2.14 GB

GPU allocated: 7.82 GB
GPU reserved : 8.46 GB
GPU free     : 5.95 GB
```

と大きな数値変化はありませんでした。

したがって、

> cleanupによって2GB以上のRAMが回収された

という結果ではありません。

一方で、複数回の例外や推論を行ったColabセッションでGradioを起動する前の予防的なcleanupとして、この処理を残しています。

---

## Gradio UI

Notebookの後半ではGradio UIも利用できます。

```text
Prompt
+
Normal / Thinking
        ↓
Nanbeige4.2-3B
        ↓
Answer
```

というシンプルな構成です。

著者環境では、

```text
Gradio: 6.20.0
```

で、Google Colab無料版T4からGradio public URLを起動できました。

通常モードの日本語チャットまで正常に動作しました。

---

## Google Colab Free / T4

今回の最も重要な結果です。

著者環境では、

```text
Google Colab Free
NVIDIA Tesla T4
GPU VRAM: 14.56 GB
```

で、

```text
FP16 Model Load  ⚪︎
Japanese Chat    ⚪︎
Thinking OFF     ⚪︎
Thinking ON      ⚪︎
Tool Calling     ⚪︎
Gradio           ⚪︎
```

まで確認しました。

最終的なGPUメモリは、

```text
GPU allocated: 7.82 GB
GPU reserved : 8.46 GB
GPU free      : 5.95 GB
GPU total     : 14.56 GB
```

でした。

つまり、

> **Nanbeige4.2-3Bは、Google Colab無料版T4でもFP16のまま一連のAgent系機能を試すことができました。**

---

## What Worked

今回の結果をまとめると、

```text
Nanbeige4.2-3B
+
FP16
+
Google Colab Free / T4
```

で、

```text
Model Load        ○
Japanese Chat     ○
Thinking OFF      ○
Thinking ON       ○
Tool Calling      ○
Gradio            ○
```

でした。

一方で、重要な注意点として、

```text
default KV cache
→ 現行Transformersとの互換性問題あり

use_cache=False
→ 正常生成
```

という結果でした。

したがって、

> **モデルそのものがT4で動かないのではなく、現行TransformersとのKV cache互換性に注意が必要**

というのが今回の大きなポイントです。

---

## Why This Is Interesting

Nanbeige4.2-3Bは、大型LLMではありません。

それでも、

```text
Japanese Chat
Reasoning
Thinking
Tool Calling
Gradio
```

まで、無料版T4で試すことができました。

最近のopen-weightモデルは大型化しており、

```text
4bitでもFree T4では難しい
```

ケースも増えています。

その中で、

> **3B級のAgentモデルをFP16のまま無料版Colabで動かせる**

というのは、教育・実験用途ではかなり扱いやすい結果です。

---

## 4bit Fallback

Notebook末尾には、

```text
FP16でOOMした場合
```

に備えてNF4 4bit版のロードセルも入れています。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

ただし、著者環境の無料版T4ではFP16で動作したため、**4bit fallbackは使用していません。**

そのため、

```text
Nanbeige4.2-3Bの4bit版でも
すべて同じ機能が動作した
```

という意味ではありません。

---

## Important Notes

このNotebookは、著者が実際にGoogle Colab無料版T4 GPU上で試した実験記録です。

Google ColabのGPU、GPUメモリ、CUDA、PyTorch、Transformers、Gradioなどの環境は今後変更される可能性があります。

特に今回のNanbeige4.2-3Bは、

```text
trust_remote_code=True
```

でモデル独自コードを利用します。

Hugging Face上のcustom model codeが更新されると、将来的に今回のcompatibility patchが不要になったり、逆に別の修正が必要になったりする可能性があります。

また、`use_cache=False`は互換性回避のための設定であり、最適な高速推論構成とは限りません。

モデル出力は確率的であり、日本語品質やTool Calling結果はpromptやsampling parameterによって変化します。

各モデルを利用する際には、配布元の最新README、ライセンス、利用条件を確認してください。

---

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルそのものではありません。

書籍著者が、刊行後に公開された新しいopen-weightモデルを個人的に試した追加実験です。

本Notebookでより興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

をぜひお手にとってください。

---

## Links

- Nanbeige4.2-3B: https://huggingface.co/Nanbeige/Nanbeige4.2-3B
- Notebook: `Nanbeige4_2_3B.ipynb`
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月17日時点のものです。Google Colabのランタイム、GPU、GPUメモリ、CUDA、PyTorch、Transformers、Gradio、Nanbeige側のcustom code等は今後変更される可能性があります。
