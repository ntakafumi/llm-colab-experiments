# LFM2.5-VL-1.6B on Google Colab Free

Liquid AIの軽量Vision-Language Model **LFM2.5-VL-1.6B** を、Google Colab無料版のT4 GPU上で動かす実験です。

今回のNotebookでは、

```text
LiquidAI/LFM2.5-VL-1.6B
```

をFP16で読み込み、

- 画像理解
- 日本語での画像質問応答
- OCR
- ユーザー画像のアップロード
- 画像入力tensorの診断
- Gradioによる画像質問UI

までを試します。

著者環境では、**Google Colab無料版 / NVIDIA T4 GPUでスムーズに動作することを確認しました。**

---

## Model

- Model: `LiquidAI/LFM2.5-VL-1.6B`
- Developer: Liquid AI
- Task: Image-Text-to-Text / Vision-Language Model
- Parameters: 1.6B class
- Precision: FP16
- Runtime: Transformers
- Environment: Google Colab Free
- GPU: NVIDIA T4
- Interface: Notebook + Gradio

LFM2.5-VL-1.6Bは、画像とテキストを入力として扱える軽量VLMです。

今回の実験では、モデルサイズが1.6Bと小さいため、Google Colab無料版T4でも4bit量子化を使わず、**FP16のままモデル本来の性能を確認する**方針としました。

---

## Notebook

`LFM2_5_VL_1_6B.ipynb`

Notebookの構成は、拙著で利用しているGoogle Colab Notebookとできるだけ同じ流れにしています。

```text
Google ColabのGPU確認
        ↓
Google Drive / Hugging Face cache設定
        ↓
LFM2.5-VL-1.6BをFP16で読み込む
        ↓
公開画像で画像理解を確認
        ↓
画像tensorが正しく生成されているか診断
        ↓
日本語で画像質問
        ↓
ユーザー画像をアップロードして確認
        ↓
OCRテスト
        ↓
Gradio画像質問UI
        ↓
GPUメモリ確認
```

---

## Why FP16?

最初の試行ではbitsandbytesによるNF4 4bit量子化も検討しました。

しかし、LFM2.5-VL-1.6Bは1.6Bクラスと比較的小さいため、無料版T4でもFP16で十分に扱えます。

今回の実測では、モデルロード直後のmemory footprintは約、

```text
2.97 GB
```

でした。

そのため、

```text
1.6B model
+
T4 14.56 GB
```

という組み合わせでは、無理に4bit量子化する必要はありませんでした。

むしろ、

> **まずFP16でモデル本来の画像理解性能を確認する**

ことを優先しています。

---

## Environment Setup

Colab標準環境をできるだけ壊さないようにします。

特に、

```text
Pillow
NumPy
PyTorch
Gradio
```

は、むやみにアップグレードしません。

追加・更新する主なパッケージは、

```python
%pip -q install -U "transformers>=5.1" accelerate
```

です。

以前の試行ではPillowを不用意に更新したことで、

```text
ImportError:
cannot import name '_Ink' from 'PIL._typing'
```

という不整合が発生しました。

そのため、このNotebookではColab標準のPillowをそのまま利用します。

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
    / "hf_cache_lfm25_vl_1_6b"
)
```

一度ダウンロードしたモデルを再利用できるため、Colabセッションを作り直した場合にも便利です。

---

## Load LFM2.5-VL-1.6B

モデルは、

```python
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)

MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"
```

として読み込みます。

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    dtype=torch.float16,
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)

model.eval()
```

著者環境では、

```text
model loaded: LiquidAI/LFM2.5-VL-1.6B
input device: cuda:0
memory footprint: 2.97 GB
GPU allocated: 2.97 GB
GPU reserved : 2.97 GB
```

となりました。

無料版T4でもかなり余裕があります。

---

## Verify That the Image Is Really Passed to the Model

VLMでは、

> **画像を指定したつもりでも、実際には画像tensorがモデルへ渡っていない**

という問題が起きる場合があります。

そこで、このNotebookでは推論前にprocessorの出力を確認します。

```python
print(
    list(diagnostic_inputs.keys())
)
```

さらに各tensorのshapeも表示します。

実際の実行では、

```text
attention_mask
pixel_values
pixel_attention_mask
spatial_shapes
```

などが生成され、

```text
pixel_values (7, 1024, 768)
pixel_attention_mask (7, 1024)
```

を確認できました。

つまり、画像が単なるテキストplaceholderではなく、**実際にvision inputとしてモデルへ渡っている**ことを確認できます。

これはVLMを評価する上で重要です。

---

## Image Understanding

画像はchat templateへ直接渡します。

```python
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
    },
]
```

その後、

```python
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
```

としてモデル入力を作成します。

日本語で、

```text
この画像に何が写っているか、
日本語で具体的に説明してください。
```

のように質問できます。

---

## User Image Upload

Notebookから自分の画像をアップロードして確認できます。

```python
from google.colab import files

uploaded = files.upload()
```

アップロードした画像について、

```text
この画像を注意深く見てください。
写っている対象の数、種類、
服装や色、背景を区別して、
日本語で簡潔に説明してください。
```

のような質問を行います。

公開サンプルだけでなく、任意の画像で実際の認識性能を確認できます。

---

## OCR Test

Notebook内でOCR用のテスト画像も自動生成します。

入力例:

```text
LOCAL AI EXPERIMENT
Model: LFM2.5-VL-1.6B
Environment: Google Colab Free / T4
Task: Image Understanding and OCR
Price: 3,000 yen
Discount: 20%
```

これに対し、

```text
画像内の文字をできるだけ正確に読み取り、
内容を日本語で整理してください。
```

と質問します。

著者環境では、

```text
LOCAL AI EXPERIMENT
モデル：LFM2.5-VL-1.6B
環境：Google Colab Free / T4
タスク：画像理解とOCR
価格：3,000円
割引：20%
```

という形で、内容をほぼ正しく読み取ることができました。

---

## Gradio Vision Chat

Notebookの後半ではGradioを利用します。

```text
画像
+
日本語の質問
        ↓
LFM2.5-VL-1.6B
        ↓
日本語回答
```

というシンプルな画像質問UIです。

Gradio UIでは、

- 画像アップロード
- 日本語質問
- 回答表示

を1画面で行えます。

著者環境では、

```text
Gradio 6.20.0
```

で起動し、Google Colab無料版T4から画像質問UIまで正常に動作しました。

---

## Google Colab Free / T4

今回の最も重要な結果です。

著者環境では、

```text
Google Colab Free
NVIDIA T4
VRAM: 14.56 GB
```

で、

```text
Model Load          ⚪︎
FP16                 ⚪︎
Image Understanding ⚪︎
OCR                  ⚪︎
User Image Upload    ⚪︎
Gradio               ⚪︎
```

までスムーズに動作しました。

モデル利用後のGPUメモリも、

```text
GPU allocated: 2.99 GB
GPU reserved : 3.02 GB
GPU free      : 11.40 GB
GPU total     : 14.56 GB
```

程度でした。

つまり、

> **LFM2.5-VL-1.6Bは、無料版Google Colab T4でもかなり余裕を持って動かせる軽量VLM**

という結果になりました。

---

## Why This Is Interesting

これまでの大型モデルでは、

```text
Qwen3.8-27B
→ 4bitでも無料版T4ではロード困難
```

のように、無料版GPUのVRAMが大きな制約になりました。

一方、LFM2.5-VL-1.6Bでは、

```text
FP16
→ 約3GB
→ T4でも余裕
```

です。

つまり、

> **最新モデルを試すために、必ずしも巨大モデルや強い量子化が必要なわけではない**

ことが分かります。

特にVision-Language Modelでは、軽量なモデルでも、

```text
Image Understanding
OCR
Japanese Question Answering
Gradio UI
```

まで無料版Colab上で実用的に試すことができます。

---

## FP16 vs 4bit

今回の実験では、4bitよりFP16を選択しました。

理由は、

```text
FP16でも約3GB
```

だからです。

このサイズであれば、

```text
量子化によるメモリ削減
```

よりも、

```text
モデル本来の性能を保つ
実装をシンプルにする
量子化由来の影響を減らす
```

方が重要です。

無料版Colabだからといって、すべてのモデルを4bitにする必要はありません。

---

## About the GGUF Version

Liquid AIからは、

```text
LiquidAI/LFM2.5-VL-1.6B-GGUF
```

も公開されています。

ただし、このNotebookではGGUF版は使用していません。

VLMのGGUFは通常のテキストLLM GGUFよりも複雑で、

```text
LLM
+
Vision Encoder / Projector
+
Multimodal runtime
+
Chat template
```

の対応が必要です。

そのため今回は、

> **まず公式Transformers版をFP16で無料版T4上に動かし、モデルの能力を確認する**

ことを優先しました。

GGUF + llama.cpp multimodal版については、別の比較実験として扱う予定です。

---

## A Note on Model Output

軽量VLMであるため、画像の説明が常に完全に正確とは限りません。

また、生成時に同じ表現を繰り返すなど、language generation側の癖が見られる場合があります。

そのため、

```text
モデルが動作する
```

ことと、

```text
すべての画像を常に正確に理解する
```

ことは分けて評価する必要があります。

このNotebookでは、画像tensorそのものが正しく入力されていることを確認したうえで、実際のモデル性能を評価できるようにしています。

---

## GPU Memory Cleanup

複数回推論する場合には、

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

を利用して不要なCUDA cacheを解放します。

ただし今回のモデルは約3GB程度なので、T4では比較的余裕があります。

---

## What We Learned

今回の実験結果をまとめると、

```text
LFM2.5-VL-1.6B
+
FP16
+
Google Colab Free / T4
```

で、

```text
Model loading       ○
Image input         ○
Japanese response   ○
OCR                 ○
Gradio              ○
GPU headroom        大きい
```

という結果でした。

特に、

```text
GPU allocated ≈ 3GB
GPU total     ≈ 14.56GB
```

であり、無料版T4でもかなり余裕があります。

大型モデルを量子化して限界まで動かす実験とは逆に、

> **小型の最新VLMを、量子化せず無料版Colabで快適に動かす**

という方向も非常に実用的です。

---

## Important Notes

このNotebookは、著者が実際にGoogle Colab無料版T4 GPU上で動作確認した実験記録です。

Google ColabのGPU、GPUメモリ、CUDA、PyTorch、Transformers、Gradioなどの環境は今後変更される可能性があります。

特にGoogle Colab無料版では、常にT4 GPUが割り当てられるとは限りません。

また、モデルの出力品質は入力画像、prompt、生成条件等によって変化します。

各モデルを利用する際には、配布元の最新ライセンスおよび利用条件を確認してください。

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

- LFM2.5-VL-1.6B: `LiquidAI/LFM2.5-VL-1.6B`
- GGUF version: `LiquidAI/LFM2.5-VL-1.6B-GGUF`
- Notebook: `LFM2_5_VL_1_6B.ipynb`
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月17日時点のものです。Google Colabのランタイム、GPU、GPUメモリ、CUDA、PyTorch、Transformers、Gradioなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新ライセンスおよび利用条件を確認してください。
