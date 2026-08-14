# MiniMax-H3 NF4 on Google Colab

MiniMaxの **MiniMax-H3** を、DiffSynth-StudioのNF4量子化版とlow-VRAM機能を利用してGoogle Colab上で動かす実験です。

今回のNotebookでは `DiffSynth-Studio/MiniMax-H3-NF4` を利用し、テキストプロンプトから **Video + Audio** を同時生成します。外部の動画生成APIは使用せず、Google Colab上でローカル推論します。

## Model

- Model: MiniMax-H3
- Quantized model: `DiffSynth-Studio/MiniMax-H3-NF4`
- Framework: DiffSynth-Studio
- Task: Text-to-Video + Audio generation
- Quantization: NF4
- Low-VRAM method: DiffSynth VRAM Management
- Offload: Disk / CPU
- Inference: Local inference on Google Colab

MiniMax-H3は、テキストプロンプトから映像だけでなく音声も含めて生成できるモデルです。

今回の実験では通常精度のモデルをそのまま利用するのではなく、DiffSynth-Studioから公開されているNF4量子化版を利用します。

## Notebook

`MiniMax_H3_NF4_DiffSynth_Colab_LowVRAM_BookStyle.ipynb`

Notebookの構成は、拙著で利用しているGoogle Colab Notebookとできるだけ同じ流れにしています。

```text
Google ColabのGPU確認
        ↓
Google Drive / ModelScope cache設定
        ↓
DiffSynth-Studioのインストール
        ↓
MiniMax-H3 NF4モデル読み込み
        ↓
Low-VRAM / offload設定
        ↓
Text-to-Video + Audio生成
        ↓
生成動画を表示
```

まずGradioなどのUIは利用せず、Notebookセルから1本の動画を最後まで生成できるかを確認することを優先しています。

## Why NF4?

MiniMax-H3は非常に大きなモデルです。通常版をGoogle ColabのGPUメモリへそのまま載せることは現実的ではありません。

そこで今回利用するのが、

```text
NF4 quantization
+
DiffSynth VRAM Management
+
offload
```

です。

NF4によってモデル重みを低bit化し、さらにDiffSynth-StudioのVRAM Managementを利用することで、必要なモデル部分だけをGPUへ移動しながら推論します。

## Model Components

今回のNF4版では、主に次のモデルファイルを利用します。

```text
minimax-h3-fl2va-nf4.safetensors
minimax-h3-text-encoder-nf4.safetensors
video_vae_nf4.safetensors
audio_vae_nf4.safetensors
```

つまり、MiniMax-H3 FL2VA、Text Encoder、Video VAE、Audio VAEをNF4量子化した構成です。

## Required Packages

今回の実験では、DiffSynth-Studio本体に加えて次のパッケージが必要でした。

```text
modelscope
av
bitsandbytes
```

Notebookでは、

```python
%pip -q install -U modelscope av bitsandbytes
```

を追加します。

`av`がない場合には `ModuleNotFoundError: No module named 'av'`、`bitsandbytes`がない場合にはNF4 backendの読み込み時にエラーになります。

## Google Drive Cache

MiniMax-H3 NF4は量子化版でもダウンロードするモデルファイルが大きいため、Google Drive上にModelScope cacheを保存します。

```python
CACHE_DIR = PROJECT_DIR / "Program" / "modelscope_cache_minimax_h3"
```

一度取得しておけば、次回以降の巨大ファイルの再ダウンロードを避けられます。

## Low-VRAM Inference

DiffSynth-Studioでは、モデル全体をGPUへ常駐させず、必要な部分だけをGPUへ移動するVRAM Managementを利用できます。

今回のNotebookでは、

```python
OFFLOAD_MODE = "disk"
```

を基本設定としています。

Disk offloadでは概念的に、

```text
Disk
 ↕
CPU
 ↕
GPU
```

とモデルを移動しながら計算します。GPUメモリ使用量を抑えられる一方、ディスクI/Oや転送が増えるため生成速度は大幅に低下します。

## VRAM Limit

DiffSynth-Studioへ、モデル側が利用するGPUメモリ量の上限を指定できます。

無料版T4で最終的に完走できた際には、

```python
VRAM_LIMIT_GB = 9.0
```

を利用しました。

モデル部分にGPUメモリを使い切らせず、attentionやNF4 dequantizationなど推論中の一時tensor用の余裕を残すことが重要でした。

## Text-to-Video + Audio

生成は次のように行います。

```python
video, audio = pipe(
    prompt=prompt,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_INFERENCE_STEPS,
    seed=SEED,
    tiled=True,
)
```

その後、

```python
write_video_audio(
    video=video,
    audio=audio,
    output_path=str(OUTPUT_PATH),
    fps=24,
    audio_sample_rate=32000,
)
```

として、映像と音声を1本のMP4へ書き出します。

## Google Colab Pro

Google Colabの有料環境では、著者環境の **G4 GPU** で、

```python
HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
```

という設定でVideo + Audio生成を最後まで完走しました。

```text
480 × 832
124 frames
50 inference steps
```

という、DiffSynth-Studioの公式例に近い条件で実際に生成できました。

生成には時間がかかりますが、Colab Pro環境ではMiniMax-H3 NF4によるVideo + Audio生成が可能でした。

## Google Colab Free

Google Colab無料版の **T4 GPU** でも限界を試しました。

今回のT4では利用可能GPUメモリは約14.56GiBでした。

試行結果は次の通りです。

| Resolution | Frames | Steps | Result |
|---|---:|---:|---|
| 448 × 256 | 56 | 12 | × OOM |
| 320 × 192 | 39 | 8 | × OOM（途中まで進行） |
| 320 × 192 | 22 | 6 | × OOM（50%まで進行） |
| **288 × 160** | **22** | **6** | **⚪︎ Completed** |

最後に、

```python
VRAM_LIMIT_GB = 9.0
OFFLOAD_MODE = "disk"

HEIGHT = 160
WIDTH = 288
NUM_FRAMES = 22
NUM_INFERENCE_STEPS = 6
```

まで条件を下げることで、

> **Google Colab無料版T4でもVideo + Audio生成を最後まで完走できました。**

## Important: "It Runs" Is Not the Same as "It Is Practical"

今回の無料版T4での成功について、最も重要な点です。

> **無料版Google ColabでもMiniMax-H3 NF4は動作しました。**

しかし、

> **無料版で実用的な品質の動画生成ができた**

という意味ではありません。

無料版で最後まで完走した設定は、

```text
288 × 160
22 frames
6 inference steps
```

です。

これはモデルの動作確認を目的として極端に条件を落としたものです。

実際に生成された動画は、**実用的に使える品質からは程遠いものでした。**

したがって、無料版T4については、

```text
動作可能
```

とは評価できますが、

```text
実用可能
```

とは評価しません。

今回確認できたのは、

> **NF4 + disk offload + VRAM limitを利用すれば、無料版T4でもMiniMax-H3のVideo + Audio生成パイプラインを最後まで動かすこと自体は可能**

という点です。

これは技術的な動作検証としては重要な結果ですが、生成品質を求める場合にはより大きなGPU環境が必要です。

## Colab Pro vs Free

今回の実測をまとめると、

| Environment | Settings | Result | Quality |
|---|---|---|---|
| **Colab Pro / G4** | 832×480 / 124 frames / 50 steps | ⚪︎ Completed | 実験・品質評価が可能 |
| **Colab Free / T4** | 288×160 / 22 frames / 6 steps / VRAM limit 9GB | ⚪︎ Completed | **動作確認レベル** |

無料版では **動かすこと自体を目標** にし、有料版では **実際の生成結果を評価する** という使い分けが現実的です。

## Why Did Free T4 Run Out of Memory?

Disk offloadを利用しても、推論中のすべてのデータをdiskへ逃がせるわけではありません。

例えば、

- attentionのQ / K / V
- latent
- intermediate tensors
- NF4 dequantized weights
- Video / Audio features

などは計算時にGPU上へ必要になります。

実際に無料版T4で発生したOOMでは、`scaled_dot_product_attention`で追加メモリを確保できないケースと、bitsandbytesの`_dequant_linear_fallback`でNF4重みを一時的にdequantizeするためのメモリを確保できないケースがありました。

つまり、

> **モデル重みをdiskへoffloadできても、推論時のピークVRAMは別に必要**

ということです。

## Why VRAM_LIMIT_GB = 9.0?

最初はGPU総容量から2GB程度を引いた値をVRAM limitとしていました。

しかしT4では、モデル側がGPUメモリを使いすぎると、attention、NF4 dequantization、temporary tensors用の空き領域がなくなります。

そこで、

```python
VRAM_LIMIT_GB = 9.0
```

まで下げ、より積極的にdisk offloadを行わせました。

速度を犠牲にしてでもGPUメモリの余白を確保することが、無料版T4での完走には重要でした。

## Inference Steps

OOMが各step内のattentionやdequantizationで起きる場合、step数を減らしても1 stepあたりのピークVRAMは大きく変わりません。

そのためVRAM削減には、

```text
Resolution
Frames
VRAM limit
```

の方が重要でした。

一方、step数を減らすことで生成時間は短縮できます。無料版では6 stepsまで減らして、まず完走することを優先しました。

## Japanese Prompt

日本語の複雑なプロンプトについても試しました。

MiniMax-H3はVideo + Audioを同時生成できるため、シーン、人物、動作、カメラワーク、発話、環境音、時間的なイベントなどを1つのプロンプトへ記述できます。

ただし今回試した範囲では、複雑な日本語プロンプトや日本語発話について、プロンプト通りの結果にならないケースがありました。

そのため、

> **日本語プロンプトを入力できること**

と、

> **複雑な日本語指示や日本語音声を高精度に再現できること**

は分けて評価する必要があります。

## GPU Memory Cleanup

動画生成を何度も続けるとGPU memoryが増えていく場合があります。

生成後には不要な生成結果を削除してCUDA cacheを解放します。

```python
del video
del audio

import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

ただし `del pipe` まで行うとモデル自体を解放してしまうため、連続生成では`pipe`は保持します。

## What We Learned

今回の実験で特に重要だったのは、

```text
Quantized
=
Small enough for practical use
```

ではないという点です。

NF4とdisk offloadによって、無料版T4でもモデルをロードし、生成を開始し、最終的には極端に軽い設定で完走することまでできました。

しかし、

```text
Model loading
Generation starts
Generation completes
Practical quality
```

はそれぞれ別の段階です。

今回の結果は、

```text
Free T4
→ Model loading:          ○
→ Generation starts:     ○
→ Generation completes:  ○
→ Practical quality:     ×
```

でした。

一方、Colab Pro G4では480×832 / 124 frames / 50 stepsまで実行できました。

この違いは、巨大な生成モデルをローカル環境で扱う際のGPUメモリの重要性を非常に分かりやすく示しています。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、GPUメモリ、CUDA、PyTorch、DiffSynth-Studio、bitsandbytes、ModelScopeなどは今後変更される可能性があります。

特にGoogle Colabでは、無料版・有料版ともに常に同じGPUが割り当てられるわけではありません。

また、今回の無料版T4での成功は、

```text
288 × 160
22 frames
6 steps
VRAM limit 9GB
disk offload
```

という非常に軽い条件での結果です。

これをもって「MiniMax-H3が無料版Colabで実用的に利用できる」とするものではありません。

各モデルを利用する際には、配布元が示す最新のライセンスおよび利用条件を確認してください。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルそのものではありません。

書籍著者が、刊行後に公開された新しいモデルを個人的に試した追加実験です。

本Notebookでより興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

をぜひお手にとってください。

## Links

- MiniMax-H3: `MiniMax/MiniMax-H3`
- MiniMax-H3 NF4: `DiffSynth-Studio/MiniMax-H3-NF4`
- DiffSynth-Studio: `modelscope/DiffSynth-Studio`
- Notebook: `MiniMax_H3_NF4_DiffSynth_Colab_LowVRAM_BookStyle.ipynb`
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月14日時点のものです。Google Colabのランタイム、GPU、GPUメモリ、CUDA、PyTorch、DiffSynth-Studio、bitsandbytes、ModelScopeなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新のライセンスおよび利用条件を確認してください。
