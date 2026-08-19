# Microsoft Mage-VL on Google Colab Free

Microsoftの **`microsoft/Mage-VL`** を、Google Colab無料版のNVIDIA T4 GPUで動かす実験です。

今回のNotebookでは、

```text
Google Colab Free / T4
        ↓
PyTorch 2.10 + cu128
        ↓
prebuilt causal-conv1d / mamba-ssm
        ↓
Mage-VLをNF4 4bitでロード
        ↓
画像理解
        ↓
自分の画像
        ↓
8-frame video understanding
        ↓
codec-native video understanding
        ↓
Gradio UI
        ↓
GPUメモリ確認
```

までを試します。

著者環境では、**Google Colab無料版 / NVIDIA T4 GPUで、Mage-VLをNF4 4bitのままスムーズに動作させることができました。**

実際に、

- 画像理解
- 日本語での画像説明
- ユーザー画像の認識
- 8-frame video understanding
- codec-native video understanding
- Gradio UI

まで確認できました。

最終的なGPUメモリは、

```text
GPU allocated: 3.41 GB
GPU reserved : 3.45 GB
GPU free      : 10.98 GB
GPU total     : 14.56 GB
```

でした。

つまり、無料版T4でもかなり余裕を持って動作しています。

---

## Model

- Model: `microsoft/Mage-VL`
- Developer: Microsoft
- Task: Image / Video Understanding
- Type: Codec-native Vision-Language Model
- Language backbone: Qwen3-4B-Instruct-2507
- Vision encoder: Mage-ViT
- Precision in this Notebook: NF4 4bit
- Runtime: Hugging Face Transformers
- Environment: Google Colab Free
- GPU: NVIDIA Tesla T4
- License: Apache-2.0

Official model:

https://huggingface.co/microsoft/Mage-VL

Official GitHub:

https://github.com/microsoft/Mage

---

## What is Mage-VL?

Mage-VLは、Microsoftが公開している **codec-native streaming multimodal foundation model** です。

通常のVision-Language Modelでは、動画を一定間隔でframe samplingし、それぞれのframeを画像として処理します。

概念的には、

```text
Video
 ↓
Uniform Frame Sampling
 ↓
Vision Encoder
 ↓
Visual Tokens
 ↓
LLM
```

です。

一方、Mage-VLはvideo codecが持つ、

```text
I-frame
P-frame
Motion Vector
Residual
```

といった情報を利用して、

> **実際に変化した場所だけを重点的に見る**

という設計を取ります。

Microsoftの公式説明では、visual token数を75%以上削減しながらspatio-temporal contextを維持し、uniform frame samplingと比較して最大3.5倍のwall-clock inference speedupを報告しています。

---

## Mage-ViT

Mage-VLのvisual encoderは **Mage-ViT** です。

Mage-ViTは、既存の巨大image-text pretrained ViTをそのまま利用するのではなく、from-scratchで学習されたCodec-ViTです。

```text
I-frame
→ 全patchを保持

P-frame
→ codecがbitを使った重要領域のみ保持
```

という考え方でvisual tokenを削減します。

また、16×16 patch gridと3D rotary positional encodingを利用し、空間・時間情報を保持します。

---

## One Checkpoint for Image and Video

Mage-VLは、

```text
microsoft/Mage-VL
```

という1つのcheckpointで、

- image understanding
- frame-sampled video
- H.264 / HEVC codec-native video
- neural codec video
- streaming
- event-gated commentary

まで扱う設計です。

今回のNotebookでは、そのうち、

```text
Image Understanding
Frame-sampled Video
Traditional Codec-native Video
```

まで実際に試しました。

---

# Notebook

Notebook:

```text
Microsoft_Mage_VL.ipynb
```

今回のNotebookは、拙著で利用しているGoogle Colab Notebookとできるだけ同じ流れになるようにしています。

```text
環境確認
↓
必要なライブラリ
↓
モデルロード
↓
推論関数
↓
実際の入力
↓
Gradio UI
↓
GPUメモリ確認
```

という構成です。

---

# Environment

著者環境では、最初に、

```text
Python: 3.12.13
PyTorch: 2.11.0+cu128
CUDA: 12.8
GPU: Tesla T4
GPU memory: 14.56 GB
```

が割り当てられました。

しかし、Mage-VLが必要とする`mamba-ssm`について、現行ColabのPyTorch 2.11向けprebuilt wheelが使いにくかったため、Notebook内でPyTorchを2.10へ固定しています。

再起動後の環境は、

```text
Python: 3.12.13
PyTorch: 2.10.0+cu128
CUDA: 12.8
```

でした。

---

# Why PyTorch 2.10?

Mage-VLのcustom model codeでは`mamba_ssm`を利用します。

最初に、

```bash
pip install mamba-ssm
```

を試すと、Colab上でCUDA extensionのsource buildに入り、非常に時間がかかりました。

そこで最終版では、

```text
PyTorch 2.10
+
CUDA 12
+
Python 3.12
```

向けのprebuilt wheelを利用します。

具体的には、

```text
causal-conv1d 1.6.2.post1
mamba-ssm     2.3.2.post1
```

を利用しました。

実際のNotebook出力では、

```text
causal_conv1d: 1.6.2.post1
mamba_ssm: 2.3.2.post1
Mamba CUDA extensions imported successfully
```

となり、正常に読み込めました。

---

# Runtime Restart

Torchを2.11から2.10へ入れ替えるため、途中でランタイムを再起動します。

Notebookでは、

```text
コード1
↓
コード2
↓
コード3で再起動
↓
再接続
↓
コード4から続行
```

という流れです。

TorchのCUDA extensionはPyTorch ABIへ依存するため、Torchを入れ替えた後、そのまま古いPython processを使わないようにしています。

---

# Transformers

再起動後に利用した環境では、

```text
Transformers: 5.15.0
Pillow: 12.3.0
```

でした。

また、FlashAttentionは使わず、

```python
attn_implementation="sdpa"
```

を指定しています。

これにより、Colab標準のPyTorch SDPAを利用します。

---

# NF4 4bit

無料版T4で余裕を持って動かすため、Mage-VLはbitsandbytes NF4 4bitで読み込みました。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

モデルロードは、

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
```

です。

---

# Model Load Result

Google Colab無料版T4で、モデルは正常にロードできました。

実際の結果は、

```text
model loaded: microsoft/Mage-VL
input device: cuda:0
4bit: True
GPU allocated: 3.39 GB
GPU reserved : 3.42 GB
```

でした。

かなり軽いです。

---

# Image Understanding

まず、公開サンプル画像を入力しました。

Processorの出力は、

```text
pixel_values
image_grid_thw
patch_positions
input_ids
attention_mask
```

でした。

具体的には、

```text
pixel_values       (8192, 768)
image_grid_thw     (1, 3)
patch_positions    (8192, 3)
input_ids          (1, 2088)
attention_mask     (1, 2088)
```

となっています。

つまり、画像由来のtensorが正しく生成され、Mage-VLへ渡っていることを確認できます。

---

## Image Understanding Result

質問:

```text
この画像に何が写っているか、
日本語で具体的に説明してください。
```

出力:

```text
この画像には、白と黒の混合毛色を持つ犬が写っています。
犬の顔は正面に向け、大きな瞳が見えており、
口元には少し笑みが浮かんでいます。
背景には、赤と青の色調が特徴的な絨毯が掛かっています。
```

日本語でも自然に画像内容を説明できました。

---

# User Image Test

次に、自分で用意した画像をアップロードしました。

画像は、

```text
黒い猫のキャラクター
+
黄色の背景
```

です。

Processorの出力は、

```text
pixel_values       (320, 768)
image_grid_thw     (1, 3)
patch_positions    (320, 3)
input_ids          (1, 128)
attention_mask     (1, 128)
```

でした。

回答は、

```text
対象：黒い猫のキャラクター
背景：黄色の壁
文字：なし
状況：黒い猫のキャラクターが黄色の壁の前で立っている。
```

となり、かなり正確に認識しています。

---

# Frame-sampled Video

次に動画理解を試しました。

今回使用したのは、

```text
soccer-broadcast.mp4
```

という短いサンプル動画です。

まず動画から、

```text
8 frames
```

を均等に抽出しました。

```text
sampled frames: 8
```

となり、その8枚をMage-VLへ入力しました。

---

## Frame-sampled Video Result

質問:

```text
この動画で何が起きているか、日本語で説明してください。
```

回答は、

```text
この動画は、BBC Sportが放送する英語とアルジェリアの国際A级試合の
試合後の分析やレポートを提供するためのスタジアムでの
アナウンサーのインタビューの様子を描いたものです。

試合終了後、アナウンサーたちは試合の結果や試合の様子について
話し合う場面が描かれています。
```

となりました。

細部には誤認識の可能性がありますが、

```text
スタジアム
スポーツ放送
試合後の分析
アナウンサー
```

という大枠は捉えています。

---

# Codec-native Video

Mage-VLの最も特徴的な機能も試しました。

追加で、

```text
ffmpeg
ffprobe
codec-video-prep
```

をインストールします。

その後、

```python
video_backend="codec"
```

として、traditional codec pathを利用します。

---

## Codec-native Video Result

同じ動画をcodec-native経路で入力したところ、正常に推論まで完走しました。

出力は、

```text
0.0秒から27.0秒までの動画は、
スポーツイベントのスタジアムで行われている
ラジオ番組の取材や分析の様子を描いたものです。
```

という内容から始まり、

```text
スタジアム
観客席
放送ロゴ
```

などを認識しました。

一方で、

```text
同じフレーズを繰り返す
ロゴ名を誤認する
```

といった小型モデルらしい生成上の課題も見られました。

そのため、

> **codec-nativeだから必ず内容理解精度まで上がる**

とは言えません。

codec-nativeの主目的は、video processingを効率化することです。

---

# Frame Sampling vs Codec-native

今回、同じ動画について、

```text
Frame-sampled
vs
Codec-native
```

の両方を無料版T4で動かすことができました。

これが今回の実験で特に面白かった点です。

```text
Frame Sampling
→ 動画から8枚抽出して理解

Codec-native
→ codecのmotion / residual情報を利用して重要領域を選択
```

という違いがあります。

今後は、

```text
処理時間
visual token数
VRAM
回答品質
```

を同じ動画で比較すると、さらに面白いと思います。

---

# Gradio UI

Notebookの最後では、画像質問用のGradio UIも用意しました。

```text
Image
+
Question
 ↓
Mage-VL
 ↓
Answer
```

というシンプルな構成です。

実行環境では、

```text
Gradio: 6.24.0
```

でpublic URLが生成され、正常に動作しました。

---

# Final GPU Memory

画像・動画・codec-native inference・Gradioまで実行した後でも、

```text
GPU allocated: 3.41 GB
GPU reserved : 3.45 GB
GPU free      : 10.98 GB
GPU total     : 14.56 GB
```

でした。

つまり、

> **Google Colab無料版T4でも、Mage-VLはかなり余裕を持って動作しました。**

---

# What Worked

今回の結果は、

```text
Microsoft Mage-VL
+
NF4 4bit
+
Google Colab Free / T4
```

で、

```text
Model Load                 ✅
Image Understanding        ✅
Japanese Output            ✅
User Image                 ✅
8-frame Video              ✅
Codec-native Video         ✅
Gradio                     ✅
```

でした。

---

# Important Compatibility Notes

今回、最初からスムーズに動いたわけではありません。

特に問題になったのは、

```text
mamba-ssm
```

でした。

Colab標準の、

```text
PyTorch 2.11
Python 3.12
CUDA 12.8
```

では、prebuilt wheelとの対応が難しく、source buildが長時間かかりました。

そこで、

```text
Torch 2.11
↓
Torch 2.10 + cu128へ固定
↓
causal-conv1d prebuilt wheel
↓
mamba-ssm prebuilt wheel
```

へ変更しました。

この構成で最終的に安定して動作しました。

---

# Dependency Warning

基本ライブラリを更新した際、Colab側から、

```text
google-colab requires requests==2.32.4
gcsfs requires fsspec==2025.3.0
```

といったdependency warningが表示されました。

今回の実験ではMage-VL推論を最後まで実行できましたが、Colabの標準ライブラリとのversion差には注意が必要です。

将来的には依存関係をさらに固定した方が再現性は高くなる可能性があります。

---

# Mage-VL vs Mage-Flow

MicrosoftのMage familyには、

```text
Mage-VL
→ Image / Video Understanding

Mage-Flow
→ Image Generation / Editing
```

があります。

今回、Mage-FlowについてもGoogle Colab無料版で試しましたが、著者環境では安定した実行には至りませんでした。

そのため、このREADMEでは **実際に無料版T4で動作確認できたMage-VLのみ** を扱っています。

Mage-Flowについては、今後より軽量なruntimeや量子化構成が整った時点で改めて試す予定です。

---

# Why This Is Interesting

今回特に面白かったのは、

> **無料版T4で、画像だけでなくcodec-native videoまで動いた**

ことです。

最近のVLMは大型化しており、無料版Colabではモデルロードだけでも厳しいケースがあります。

一方、Mage-VLでは、

```text
4bit model
GPU allocated ≈ 3.4 GB
```

程度で済みました。

さらに、

```text
Image
Video
Codec-native Video
Gradio
```

まで一つのNotebookで試せています。

これは、動画理解の仕組みを学ぶ教材としても面白いと思います。

---

# Repository Structure

想定する構成は、

```text
Mage_VL/
├── README.md
└── Microsoft_Mage_VL.ipynb
```

です。

---

# Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

今回のNotebookは、上記書籍の公式サンプルそのものではありません。

書籍刊行後に公開された新しいopen-weight / open-modelを、書籍で扱ったGoogle Colabでの実装方法をベースに追加検証したものです。

書籍では、

- Google ColabでローカルLLMを動かす方法
- Transformers
- 量子化
- GPUメモリ
- Gradio
- RAG
- ファインチューニング
- Function Calling

などを、実際のNotebookを動かしながら学べるように構成しています。

今回のMage-VLでも、

```text
Google Colab
↓
モデルロード
↓
量子化
↓
推論関数
↓
Gradio
```

という考え方は同じです。

本Notebookで興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

もぜひご覧ください。

---

# Links

- Microsoft Mage: https://github.com/microsoft/Mage
- Mage-VL: https://huggingface.co/microsoft/Mage-VL
- Mage-VL paper: https://arxiv.org/abs/2607.24904
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月20日時点のGoogle Colab無料版 / NVIDIA T4環境に基づきます。Google ColabのGPU、PyTorch、CUDA、Transformers、Mamba、bitsandbytes、Gradio、Microsoft側のcustom model code等は今後変更される可能性があります。
