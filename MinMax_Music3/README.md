# MiniMax Music 3 on Google Colab

MiniMax AI が公開している音楽生成モデル **MiniMax Music 3** を、Google Colab 上で動かす実験です。

今回の Notebook では、公開重み `MiniMaxAI/MiniMax-Music3` を Hugging Face から読み込み、歌詞と Music Description から楽曲を生成します。
外部の音楽生成 API は使用せず、Google Colab 上でローカル推論します。

この実験は、以下のリポジトリの方針に従い、

> 「このモデルは Google Colab で、どこまで現実的に動かせるのか？」

を確認するためのものです。

- Repository: https://github.com/ntakafumi/llm-colab-experiments
- Model: https://huggingface.co/MiniMaxAI/MiniMax-Music3

---

## Model

- Model: `MiniMaxAI/MiniMax-Music3`
- Task: Text / Lyrics to Music
- Framework: Hugging Face Diffusers `ModularPipeline`
- Inference: Local inference on Google Colab
- Inputs:
  - Lyrics
  - Music Description
- Maximum song length: up to approximately 5 minutes
- Section tags:
  - `[Intro]`
  - `[Verse]`
  - `[Pre-Chorus]`
  - `[Chorus]`
  - `[Post-Chorus]`
  - `[Bridge]`
  - `[Instrumental]`
  - `[Solo]`
  - `[Outro]`

MiniMax Music 3 は、長距離の楽曲構造を扱う Global LLM、局所的な音響情報を扱う Local LLM、Flow Matching / Flow-VAE 系の音声生成部分などから構成されています。

公式モデルカードでは、Global LLM は 8B、Local LLM は 0.6B と説明されています。

---

## Notebook

`Ch05_MiniMax_Music3_Colab_Gradio.ipynb`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/llm-colab-experiments/blob/main/MiniMax-Music3/Ch05_MiniMax_Music3_Colab_Gradio.ipynb)

Notebook の構成は、拙著

**『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』**

で利用している Google Colab Notebook の流れをできるだけ踏襲しています。

```text
Google Colab の GPU 確認
        ↓
Google Drive / Hugging Face cache 設定
        ↓
Diffusers など必要ライブラリのインストール
        ↓
MiniMax Music 3 のモデル読み込み
        ↓
通常ロード / Low VRAM モードの切り替え
        ↓
Lyrics + Music Description から楽曲生成
        ↓
WAV 保存・Notebook 上で再生
        ↓
Gradio によるブラウザ UI
```

このリポジトリは書籍の公式サンプルコードではなく、刊行後に登場したモデルを著者が個人的に試すための実験用リポジトリです。

---

## Diffusers

MiniMax Music 3 は Hugging Face Diffusers の Modular Pipeline として公開されています。

この Notebook では、実験時点で MiniMax 公式モデルカードに記載されている Diffusers のコミットを使用しています。

```python
%pip -q install -U \
  git+https://github.com/huggingface/diffusers@dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d \
  transformers accelerate soundfile
```

MiniMax Music 3 は公開直後のモデルであり、Diffusers 側の実装は今後変更される可能性があります。

---

## Model Loading

Notebook では GPU VRAM を確認し、通常ロードと Low VRAM モードを切り替えます。

```python
LOW_VRAM_MODE = "auto"
```

`auto` の場合、

```python
use_low_vram = gpu_mem_gb < 22.0
```

として、VRAM が約 22GB 未満の場合に Low VRAM モードを利用します。

### Normal mode

十分な GPU メモリがある場合は、モデルを通常ロードして CUDA へ移します。

```python
pipe = ModularPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
)

pipe.load_components(dtype=DTYPE)
pipe.to("cuda")
```

### Low VRAM mode

VRAM が小さい環境では、MiniMax 公式モデルカードで紹介されている CPU offload と language model の group offloading を利用します。

```python
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

pipe = ModularPipeline.from_pretrained(
    MODEL_ID,
    components_manager=manager,
    cache_dir=str(CACHE_DIR),
)

pipe.load_components(dtype=DTYPE)

apply_group_offloading(
    pipe.language_model,
    onload_device=torch.device("cuda"),
    offload_type="leaf_level",
    use_stream=True,
)
```

MiniMax 公式では、

- full precision: 24GB 未満の VRAM に収まる
- automatic CPU offload: 約 22GB VRAM
- language model を layer-by-layer で streaming offload: 8GB GPU でも収まる

と説明されています。

ただし、これは主に **GPU VRAM 使用量**についての説明です。Google Colab では GPU VRAM 以外に、CPU RAM、ストレージ、一時的なモデル展開、CPU/GPU 間転送なども実行可否に影響します。

---

## Music Generation

生成には、

- `prompt`: Music Description
- `lyrics`: 歌詞
- `audio_duration`: 生成秒数
- `seed`: 乱数 seed

を指定します。

例:

```python
test_lyrics = """[Verse]
夜明けの街を歩いて
まだ見えない明日を探す

[Chorus]
光の向こうへ
もう一度走り出そう
"""

test_prompt = (
    "Genre: Japanese electronic pop. BPM: 128. Key: A minor. "
    "Emotional and energetic female lead vocal with clear articulation. "
    "Arrangement: bright synthesizers, punchy electronic drums, warm bass, "
    "and a wide uplifting chorus with layered backing vocals."
)
```

生成は次のように行います。

```python
audio = pipe(
    prompt=prompt,
    lyrics=lyrics,
    audio_duration=audio_duration,
    generator=torch.Generator("cuda").manual_seed(seed),
    output="audios",
)[0]
```

---

## Japanese Lyrics and Music Description

今回の実験では、**日本語歌詞の生成は可能**でした。

一方、Music Description を日本語で記述した場合は、英語で指定した場合よりも意図したスタイルを反映しにくいケースがありました。

そのため、この Notebook の標準例では、

```text
Lyrics            : Japanese
Music Description : English
```

という組み合わせを利用しています。

これは今回の実験条件における観察結果であり、日本語 Music Description が利用できないことを意味するものではありません。

---

## WAV Output

生成結果は WAV ファイルとして保存します。

Diffusers のバージョンによって `audio` が PyTorch Tensor または NumPy array として返る可能性があるため、Notebook では両方に対応しています。

```python
if torch.is_tensor(audio):
    audio_np = audio.detach().float().cpu().numpy()
else:
    audio_np = np.asarray(audio)

sf.write(
    str(out_file),
    audio_np.T,
    pipe.sampling_rate,
)
```

なお、MiniMax 公式モデルカードでは出力を **32 kHz / 16-bit stereo WAV** と説明しています。

一方、今回使用した Diffusers Modular Pipeline では、実行時に

```text
sampling rate: 44100
```

と表示される環境を確認しました。

この Notebook では固定値を使わず、実際の

```python
pipe.sampling_rate
```

を利用して WAV を保存します。

---

## Gradio UI

Notebook の最後では、Gradio を利用してブラウザから楽曲生成を実行できます。

入力項目は、

- Music Description
- Lyrics
- Duration
- Seed

です。

```text
Music Description
        +
Lyrics
        +
Duration / Seed
        ↓
Generate Music
        ↓
MiniMax Music 3
        ↓
WAV
        ↓
Gradio 上で再生
```

長時間生成では GPU / CPU メモリと生成時間が増えるため、最初は 10〜20 秒程度で動作確認することを推奨します。

---

## Google Colab Paid Plan

今回の実験では、**Google Colab の有料環境でモデル読み込みから楽曲生成まで完走**しました。

通常ロード時には、

```text
dtype: torch.bfloat16
Low VRAM mode: False
```

となる GPU 環境で実行できています。

20 秒の日本語歌詞 + 英語 Music Description を入力し、推論および WAV 保存まで確認しました。

したがって、この Notebook については、

> Google Colab の十分な GPU メモリを持つ有料環境では MiniMax Music 3 のローカル楽曲生成が可能

と評価しています。

ただし、Google Colab で割り当てられる GPU は固定ではないため、同じ有料プランでも毎回同じ実行環境になるとは限りません。

---

## Google Colab Free / T4

Google Colab 無料版の **NVIDIA T4 GPU** でも Low VRAM モードを試しました。

MiniMax 公式モデルカードでは、language model を layer-by-layer で streaming offload することで 8GB GPU でも収まると説明されています。

しかし、今回の Google Colab 無料版 T4 環境では、

> **モデル読み込み途中でエラーとなり、楽曲生成まで到達できませんでした。**

したがって、今回の実測結果は次の通りです。

| Environment | GPU / Mode | Result |
|---|---|---|
| Google Colab Paid | 通常ロード可能な GPU | ⚪︎ Model load + Music generation completed |
| Google Colab Free | NVIDIA T4 / Low VRAM | × Model loading did not complete |

ここで重要なのは、

> **「MiniMax Music 3 は T4 では動かない」と断定しているわけではない**

という点です。

公式には 8GB GPU 向けの Low VRAM 手順が示されています。しかし Google Colab 無料版では、GPU VRAM だけでなく CPU RAM やモデル読み込み時の一時メモリなども制約になります。

今回確認できたのは、

> **実験時点の Google Colab 無料版 T4 環境では、Low VRAM 設定を利用してもモデル読み込みを完了できなかった**

という結果です。

Colab の割り当てメモリや Diffusers の実装が変われば、結果が変わる可能性があります。

---

## Colab Paid vs Free

今回の実測を簡単にまとめると、

| Environment | Model loading | Generation | Evaluation |
|---|---:|---:|---|
| Colab Paid | G4 GPU | ⚪︎| 実用的に実験可能 |
| Colab Free / T4 | ❌ | 未到達 | 今回の条件では困難 |

MiniMax Music 3 は、公開重みを Google Colab で直接動かせる点で非常に興味深いモデルですが、モデル全体は大きく、無料版 Colab ではかなり厳しい条件になります。

---

## Why Can T4 Fail Even with Low VRAM?

Low VRAM は、モデル全体を GPU に常駐させず、CPU と GPU の間で必要な部分を移動することで GPU VRAM の使用量を下げます。

概念的には、

```text
CPU RAM
   ↕
GPU VRAM
```

とモデルを移動しながら処理します。

そのため GPU VRAM の節約にはなりますが、その代わりに、

- CPU RAM
- CPU/GPU 間転送
- モデル読み込み時の一時メモリ
- Hugging Face cache
- safetensors / checkpoint の展開
- Colab ランタイム全体のメモリ制約

などの影響を受けます。

つまり、

> GPU に 16GB VRAM があること

と、

> MiniMax Music 3 のロード処理全体が無料版 Colab で完走できること

は同じではありません。

---

## Model Download Size

MiniMax Music 3 の Hugging Face リポジトリは非常に大きく、実験時点ではリポジトリ全体で数十 GB 規模です。

モデルロード時には複数の大きな checkpoint / model component を取得します。

そのため Notebook では Hugging Face cache を Google Drive に保存できるようにしています。

```python
USE_DRIVE_CACHE = True
```

一度ダウンロードしておくことで、次回以降の巨大ファイルの再取得を減らせます。

ただし、Google Drive の空き容量にも注意してください。

---

## Important: Experimental Notebook

MiniMax Music 3 は公開されたばかりのモデルで、Diffusers 側の対応も更新が続いています。

そのため、この Notebook は、

> **2026年8月時点で Google Colab 上で MiniMax Music 3 を動かしてみた実験記録**

として公開します。

将来、

- Diffusers の正式対応
- モデル構成の変更
- Low VRAM 実装の改善
- Colab の GPU / RAM 条件変更

などによって、同じコードでも結果が変わる可能性があります。

---

## Limitations

MiniMax 公式モデルカードでは、以下の制約が示されています。

- CUDA が必要
- 現時点では non-streaming generation のみ
- tokenized text prompt は最大 5,000 tokens
- audio generation は最大 9,000 acoustic frames
- BPM、Key、Instrumentation、Lyrics、Song structure などは生成制御であり、指定内容が必ず厳密に再現されるとは限らない

また、この Notebook での日本語 Music Description の評価は簡単な試行に基づくものであり、体系的な性能評価ではありません。

---

## References

- MiniMax Music 3  
  https://huggingface.co/MiniMaxAI/MiniMax-Music3

- MiniMax Music 3 GitHub  
  https://github.com/MiniMax-AI/MiniMax-Music3

- Hugging Face Diffusers  
  https://github.com/huggingface/diffusers

- llm-colab-experiments  
  https://github.com/ntakafumi/llm-colab-experiments

---

## Relationship to the Book

この Notebook は、以下の書籍の公式サンプルコードではありません。

**『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』**

書籍で扱った Google Colab 上でモデルを動かす基本的な考え方を土台にしながら、刊行後に公開された MiniMax Music 3 を著者が個人的に試すための独立した実験です。

---

## Notes

- モデルの利用条件・ライセンスは MiniMax Music 3 の配布元を必ず確認してください。
- Google Colab の GPU、CPU RAM、ストレージ条件は固定ではありません。
- Hugging Face / Diffusers の更新によって Notebook の修正が必要になる場合があります。
- 大容量モデルのため、Google Drive および Colab ローカルストレージの空き容量に注意してください。
- 本コードは研究・学習・実験用途を想定しています。
