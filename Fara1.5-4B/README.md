# Microsoft Fara1.5-4B on Google Colab

Microsoftの **Fara1.5-4B** を、Google Colab上で4bit量子化して動かす実験です。

今回のNotebookでは、

```text
microsoft/Fara1.5-4B
```

を利用し、ブラウザ画面のスクリーンショットとタスクを入力して、

- 現在の画面が何のための画面かを理解する
- 画面内のUI要素を認識する
- ユーザーの目的に対して次に行うべき操作を推定する
- `left_click` や `terminate` などのComputer Use actionを出力する

といった動作を試します。

外部APIは使用せず、Hugging Faceからモデルをダウンロードし、Google Colab上でローカル推論します。

---

## Model

- Model: `microsoft/Fara1.5-4B`
- Developer: Microsoft
- Model type: Multimodal Computer Use Agent
- Size: 4B class
- Input: Browser screenshot + task
- Output: Computer Use action / screen understanding response
- Inference: Local inference on Google Colab
- Quantization: bitsandbytes NF4 4bit

Fara1.5-4Bは、一般的な画像説明用Vision-Language Modelとは少し異なります。

例えば通常のVLMであれば、

```text
Screenshot
    ↓
「これはYouTubeのホーム画面です」
```

のような画像理解が中心になります。

Fara1.5-4Bでは、それに加えて、

```text
Screenshot
+
User Goal
    ↓
Next Action
```

というComputer Use Agentとしての動作を行います。

---

## Notebook

`Fara1_5_4B_Colab.ipynb`

Notebookの構成は、拙著で利用しているPhi-4 Notebookにできるだけ近づけています。

```text
Google ColabのGPU確認
        ↓
Google DriveへのHugging Face cache保存
        ↓
Fara1.5-4Bを4bitで読み込む
        ↓
応答生成関数
        ↓
動作確認用ブラウザ画面を生成
        ↓
次のブラウザ操作を予測
        ↓
Gradio UI
```

TorchとGradioについては、可能な限りGoogle Colabの既定環境を利用します。

Transformers、Accelerate、bitsandbytesのみ追加しています。

```python
%pip -q install -U \
    "transformers>=5.2,<6" \
    accelerate \
    bitsandbytes
```

---

## 4bit Quantization

書籍のPhi-4 Notebookと同様に、bitsandbytesによるNF4 4bit量子化を利用します。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)
```

モデルは次のように読み込みます。

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)
```

今回実行した有料版Colab環境では、モデル読み込み直後のGPU allocatedは約3.08GBでした。

ただし、これはモデルロード直後の値であり、スクリーンショット処理や生成時には追加のGPUメモリが必要になります。

---

## How Fara1.5 Works

このNotebookでは、スクリーンショットとタスクをFara1.5-4Bへ入力します。

例えば、

```text
Search YouTube for "explainable AI".
```

というタスクとYouTubeのホーム画面を入力すると、

```text
<tool_call>
{"name":"computer_use","arguments":{
    "action":"left_click",
    "coordinate":[469,54]
}}
</tool_call>
```

のように、次に行うべき1アクションを返します。

重要なのは、Fara1.5は原則として、

```text
Complete Action Sequence
```

を一度に生成するモデルではなく、

```text
Current Screenshot
        ↓
Next Action
        ↓
Updated Screenshot
        ↓
Next Action
        ↓
...
```

という逐次的なComputer Use Agentとして利用するモデルである点です。

---

## Screen Understanding

Fara1.5-4Bは、操作を出力するだけではありません。

例えばYouTubeのホーム画面に対して、

```text
explain what this screen is for.
```

と入力すると、

```text
This is the YouTube "Home" (main feed) page in Japanese.
...
```

のように、画面そのものの役割を説明できます。

つまり、

```text
Screen Understanding
+
GUI Grounding
+
Next Action Prediction
```

を1つのモデルで試すことができます。

画面説明を求めた場合には、

```json
{
  "action": "terminate",
  "answer": "This is the YouTube Home page ..."
}
```

のように、説明を返してタスクを終了する場合があります。

これは正常な動作です。

---

## Why Only One Action?

例えば、

```text
Search YouTube for "explainable AI".
```

というタスクを与えても、

```text
left_click
```

だけが返ってくる場合があります。

これは情報不足ではなく、Fara1.5のComputer Use Agentとしての設計によるものです。

概念的には、

```text
Screenshot 1
↓
left_click

Screenshot 2
↓
type "explainable AI"

Screenshot 3
↓
key ENTER

Screenshot 4
↓
terminate
```

というループで利用します。

このNotebookでは、実際のWebブラウザを自動操作するところまでは実装せず、

> **現在のスクリーンショットから、次の1アクションを予測する**

ところまでを実験します。

---

## Safety of This Notebook

このNotebookでは、Fara1.5-4Bが出力したComputer Use actionを実際のWebブラウザへ自動適用しません。

例えば、

```text
left_click
type
visit_url
```

などが出力されても、Notebookはそれを表示するだけです。

したがって、

```text
Screenshot
    ↓
Fara1.5-4B
    ↓
Predicted Action
```

までのモデル単体の挙動確認を目的としています。

購入、メッセージ送信、フォーム送信、ログイン操作などを自動実行するものではありません。

---

## Test Screenshot

Notebookでは、1440 × 900の簡単なブラウザ画面を自動生成します。

```text
Demo Search
```

という検索ページを作り、

```text
Search for 'explainable AI' using the search box on this page.
Choose only the next action; do not execute anything.
```

というタスクを入力します。

実際のWeb画面のスクリーンショットをGradioからアップロードして試すこともできます。

---

## Google Colab Pro

著者環境では、Google Colabの有料環境で正常に動作しました。

確認できた内容は、

- モデル読み込み
- 4bit量子化
- スクリーンショット入力
- 次アクション生成
- 画面説明
- Gradio UI

です。

今回保存したNotebook実行例では、

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

が割り当てられ、

```text
model loaded: microsoft/Fara1.5-4B
input device: cuda:0
4bit: True
GPU allocated: 3.08 GB
```

となりました。

Google Colabでは、契約プランにかかわらず常に同じGPUが割り当てられるとは限りません。

したがって、この結果は著者が実行した時点での一例です。

---

## Google Colab Free

Google Colab無料版の **T4 GPU** でも試しました。

著者環境では、

- Fara1.5-4Bの4bitモデル読み込み
- Notebookセルからの単発スクリーンショット推論
- Computer Use actionの生成

までは動作しました。

つまり、

> **無料版T4でも、Fara1.5-4Bのモデル単体を動かし、スクリーンショットから次アクションを生成するところまでは確認できました。**

一方、Gradio UIから同じ推論を実行した場合には正常に完了しませんでした。

有料版ではGradio UIからの推論も動作しています。

この違いについては、無料版T4のGPUメモリ容量に対して、Gradio経由の画像保持、Processorによる画像tensor化、Vision処理、KV cache、生成時の一時的なtensorなどが追加され、推論時のピークGPUメモリが不足した可能性が高いと考えています。

ただし、今回の実験だけから原因をGPUメモリだけに完全に限定することはできません。

現時点での実測結果は、

| Environment | Model Load | Single Screenshot Test | Gradio |
|---|---:|---:|---:|
| Colab Free / T4 | ✅ | ✅ | ❌ |
| Colab Paid / larger GPU | ✅ | ✅ | ✅ |

となりました。

---

## GPU Memory

Fara1.5-4Bでは、モデル本体以外にもGPUメモリが使われます。

```text
4bit model weights
+
Vision input
+
pixel_values
+
KV cache
+
generation buffers
+
temporary tensors
```

Gradioを利用する場合には、さらにアップロードされた画像などがUI側で保持されます。

そのため、

> **モデルがGPUに載ること**

と、

> **Gradioを含めたマルチモーダル推論が安定して実行できること**

は同じではありません。

これはGemma 4 E4B-itの実験でも確認できたポイントです。

---

## Trying to Reduce Memory on Free Colab

無料版T4でGradioまで試したい場合には、以下の方法でGPUメモリ使用量を減らせる可能性があります。

### Reduce `max_new_tokens`

Fara1.5-4Bは1回につき次の1アクションを出力するため、長い出力は通常必要ありません。

現在の、

```python
max_new_tokens=256
```

を、

```python
max_new_tokens=96
```

あるいは、

```python
max_new_tokens=64
```

程度まで減らすことができます。

### Clear CUDA Cache

推論後に、

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

を実行することで、再利用可能なCUDA cacheを解放できる場合があります。

### Delete Temporary Tensors

生成関数内で、

```python
del inputs
del outputs
del generated_ids
```

のように、不要になったtensorへの参照を削除することも有効です。

ただし、これらはGPUメモリそのものを増やす方法ではありません。

推論時のピークメモリがT4のVRAMを超える場合には、より大きなGPUが必要になります。

---

## Gradio UI

Notebookの最後では、Gradioを利用して簡単なUIを作ります。

入力は、

- Browser Screenshot
- Task

です。

例えば、

```text
explain what this screen is for.
```

と入力すると画面説明を試すことができます。

また、

```text
Search YouTube for "explainable AI".
```

と入力すると、次に行うべきComputer Use actionを試すことができます。

このUIは、

> **Fara1.5が操作を実行するUIではなく、操作を予測するUI**

です。

---

## Browser-Focused Model

Fara1.5-4Bは、主にWebブラウザのComputer Useを目的としたモデルです。

そのため、

- Webページ
- 検索画面
- Webフォーム
- Webアプリ
- Webメール
- ECサイト

などとの相性が良いと考えられます。

一方、

- macOS Finder
- Logic Pro
- MainStage
- デスクトップ版Office
- 専用業務アプリ

などの一般的なOS / Desktop GUIについて、Webブラウザと同じ精度で利用できることを前提にはしていません。

MacだからWeb操作しか出ないのではなく、

> **モデル自体がWebブラウザ向けComputer Use Agentとして設計されている**

と理解するのが適切です。

---

## Possible Research Extension

Fara1.5-4BのScreen Understanding能力は、Computer Use以外にも応用できる可能性があります。

例えば、一定間隔で取得したスクリーンショットに対して、

```text
Screenshot
    ↓
Fara1.5-4B
    ↓
Screen Activity Semantic Metadata
```

として、

```json
{
  "service": "Web Mail",
  "screen_type": "compose",
  "activity": "attaching a file",
  "intent": "sending information",
  "ui_target": "attachment button"
}
```

のような意味情報を付与できます。

これを時系列化すると、

```text
Screen Activity Semantic Logging
            +
Temporal Anomaly Detection
```

という研究・システム構成へ発展させることも考えられます。

この場合、Faraを異常検知器そのものとして使うのではなく、

> **Screen Activity Semantic Encoder**

として利用するのがポイントです。

---

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、GPUメモリ、CUDA、PyTorch、Transformers、bitsandbytes、Gradioなどは変更される可能性があります。

そのため、将来同じNotebookを実行した場合に、同じGPUや同じ結果が得られることを保証するものではありません。

特に無料版Colabでは、利用可能なGPUやリソースが固定されていません。

また、Fara1.5-4BはComputer Use Agentです。

実ブラウザを自動操作するシステムへ発展させる場合には、

- sandbox
- permission control
- confirmation before irreversible actions
- authentication information
- privacy
- security

などを別途慎重に設計する必要があります。

各モデルを利用する際には、配布元が示す最新のライセンスおよび利用条件を確認してください。

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

- Fara1.5-4B:
  `microsoft/Fara1.5-4B`
- Notebook:
  `Fara1_5_4B_Colab.ipynb`
- 『ブラウザで動かすLLM実装入門』:
  https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月14日時点のものです。Google Colabのランタイム、GPU、GPUメモリ、CUDA、PyTorch、Transformers、bitsandbytes、Gradioなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新のライセンスおよび利用条件を確認してください。
