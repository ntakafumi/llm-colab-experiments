# Gemma 4 E4B-it on Google Colab

Googleの **Gemma 4 E4B-it** を、Google Colab上でローカル実行する実験です。

今回のNotebookでは、Google公式の軽量QAT版

```text
google/gemma-4-E4B-it-qat-mobile-transformers
```

を利用し、テキスト・画像・音声・動画のマルチモーダル入力を試します。

外部APIは使用せず、Hugging Faceからモデルをダウンロードし、Google Colab上でローカル推論します。

## Model

- Model: `google/gemma-4-E4B-it-qat-mobile-transformers`
- Base model: Gemma 4 E4B-it
- Model type: Multimodal LLM
- Distribution: Official QAT / mobile Transformers checkpoint
- Input: Text / Image / Audio / Video
- Inference: Local inference on Google Colab

Gemma 4 E4B-itは、テキストだけでなく画像・音声・動画を入力できるマルチモーダルモデルです。

今回使用するQAT版は、Google公式の量子化済みcheckpointです。通常版をColab上で読み込み時に4bit化するのではなく、モデル構造に合わせて事前に量子化された公式checkpointを利用しています。

## Notebook

`Gemma4_E4B_Multimodal_Colab_v4_fixed.ipynb`

Notebookの構成は、書籍で利用しているPhi-4 Notebookにできるだけ近づけています。

- Google ColabのGPU確認
- Google DriveへのHugging Face cache保存
- Gemma 4 E4B-it QATモデルの読み込み
- 日本語テキストによる簡単な動作確認
- 画像理解
- 音声認識
- NASA通信音声の理解
- 動画からのフレーム抽出
- 複数フレームによる動画理解
- Gradioによる画像・音声UI

## Environment Tested

著者が確認した範囲では、以下の結果になりました。

### Google Colab Free

**Google Colab無料版のT4 GPUで、モデルの読み込みとテキスト生成まで動作を確認しました。**

つまり、このQAT版Gemma 4 E4B-itは、無料版Colabでもモデル自体をGPUへ載せてローカル推論できます。

ただし、T4のGPUメモリには大きな余裕があるわけではありません。著者環境では、モデルの読み込み、テキスト生成、画像入力、音声入力までは条件によって動作しましたが、複数回連続して生成したり、画像・音声などのマルチモーダル入力を追加したりすると、GPUメモリ不足になる場合がありました。

特にマルチモーダル入力では、モデル重み以外にも、

- KV cache
- 入力token
- 出力token
- Vision Encoder用のtensor
- 画像feature
- 音声feature
- 中間buffer

などがGPUメモリを使用します。

そのため、

> **無料版T4でもモデル自体は動作するが、マルチモーダル入力や連続生成ではVRAM不足になりやすい。**

というのが、著者が実際に試した時点での結果です。

無料版では、まず以下のような軽い条件から試すことを推奨します。

- テキストのみ
- 画像は1枚
- 音声は短め
- `max_new_tokens` は128〜256程度
- 推論ごとに不要なtensorを解放する
- 動画はフレーム数を少なめにする

### Google Colab Pro

Google Colab Proでは、より大きなGPUメモリを持つGPUが割り当てられた場合、無料版より安定して画像・音声・動画を試すことができます。

今回のようなマルチモーダル実験では、単にモデル重みがGPUへ載るかだけでなく、推論時の追加メモリも重要です。

そのため、画像・音声・動画を安定して扱う場合は、Colab Pro以上の環境を推奨します。

ただし、Google Colabでは契約プランにかかわらず、常に特定のGPUが割り当てられることが保証されているわけではありません。

## Why Official QAT Model?

今回のNotebookでは、通常版Gemma 4 E4B-itをbitsandbytesで読み込み時に4bit量子化する方法ではなく、Google公式のQAT版を利用しています。

モデルIDは、

```python
MODEL_ID = "google/gemma-4-E4B-it-qat-mobile-transformers"
```

です。

QATはQuantization-Aware Trainingの略で、量子化を前提として学習・調整されたモデルです。

この公式QAT checkpointでは、モデル全体を一律に4bit化するのではなく、モデル内部の構造に応じて異なる精度が利用されています。

そのため、Google ColabのようにGPUメモリが限られた環境では、通常版を後から量子化するより扱いやすい構成です。

## Model Loading

モデルはTransformersから次のように読み込みます。

```python
from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=str(CACHE_DIR),
)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
    cache_dir=str(CACHE_DIR),
)
```

通常のテキストLLMではTokenizerを中心に利用しますが、Gemma 4では画像・音声なども扱うため、`AutoProcessor` を利用します。

## Text Input

テキストだけを利用する場合は、通常のチャットモデルと同様に質問できます。

Notebookでは最初の動作確認として、

```text
日本語で1文だけ自己紹介してください。
```

という簡単な質問を実行しています。

無料版Colabでも、まずテキストのみでモデルが正常に読み込まれていることを確認するのが安全です。

## Image Input

Gemma 4 E4B-itは画像入力に対応しています。

NotebookではNASAの公開画像を利用し、

```text
この画像に何が写っているか、日本語で簡潔に説明してください。
```

という質問を行います。

画像入力では、テキストだけの場合よりVision Encoderなどの追加処理が必要になるため、GPUメモリ使用量も増加します。

無料版T4では、まず画像1枚から試すことを推奨します。

## Audio Input

Gemma 4 E4B-itは音声入力にも対応しています。

Notebookでは、

- British Englishの朗読
- NASA STS-1 air-to-ground communication

という2種類の音声を利用します。

最初の音声は比較的聞き取りやすい英語音声です。2つ目のNASA通信は、実環境の通信音声であり、より難しい音声理解の例として利用しています。

音声は16kHz / mono WAVへ変換してからモデルへ入力します。

## Video Input

Gemma 4 E4B-it自体は動画入力をサポートしています。

ただし、今回のGoogle Colab / Transformers環境では、動画ファイルをVideoProcessorへ直接渡した場合に、

```text
AttributeError: 'tuple' object has no attribute 'to'
```

というエラーが発生しました。

そのため公開Notebookでは、動画ファイルを直接Processorへ渡さず、

```text
60秒動画
 ↓
32フレームを均等抽出
 ↓
複数画像としてGemma 4へ入力
```

という方法を利用しています。

Gemma 4の動画理解も内部的には動画をフレーム列として処理するため、この方法では処理内容を明示的に確認できます。

ただし、32枚の画像を一度に入力するとGPUメモリ使用量が大きくなります。

無料版T4では、`32 frames`ではなく`8 frames`や`16 frames`程度から試す方が安全です。

## GPU Memory

今回の実験では、無料版T4でもモデル自体はロードできました。

しかし、マルチモーダルモデルではモデル重みだけでGPUメモリ使用量を判断することはできません。

特に、

```text
Text
 ↓
Image
 ↓
Audio
 ↓
Multiple Images / Video Frames
```

と入力情報が増えるにつれて、推論時の一時的なGPUメモリ使用量も増加します。

また、PyTorchでは一度確保したCUDA memoryをreserved memoryとして保持する場合があります。

そのため、1回目の推論は成功しても、複数回生成した後にCUDA out of memoryになる場合があります。

必要に応じて、

```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

を利用すると改善する場合があります。

ただし、現在参照されているtensorは `empty_cache()` だけでは解放されないため、不要な変数も削除する必要があります。

## Colab Environment Notes

今回、Google Colab上でGemma 4を試す過程では、Pythonパッケージの互換性に関する問題も発生しました。

特に、

```text
PyTorch CUDA version
TorchAudio CUDA version
```

が異なることで、`AutoProcessor` のimport時にエラーになるケースがありました。

Gemma 4の公式例ではTorchaudioは必須ではないため、このNotebookでは `torchaudio` を利用せず、音声処理には `librosa` を利用しています。

またNumPy / SciPyについても、Colab環境によってバイナリ互換性の問題が発生する場合があります。

公開Notebookでは、できるだけColab標準環境を維持し、必要なパッケージだけ追加する方針にしています。

## Test Media

このNotebookでは、GitHubや技術記事で公開しやすいように、再利用条件が明確な素材を利用しています。

### Image

**ISS-42 Earth from the ISS.jpg**

- NASA / Samantha Cristoforetti
- Public Domain
- via Wikimedia Commons

### Audio 1

**Recording of speaker of British English (Received Pronunciation).ogg**

- P. Roach / International Phonetic Association
- CC BY-SA 3.0
- via Wikimedia Commons
- 元音声36秒から先頭30秒を実験用に利用

### Audio 2

**STS-001-1-10.ogg**

- NASA
- CC0 1.0
- via Wikimedia Commons
- STS-1 air-to-ground communication

### Video

**T - 60 Seconds with Chris Cassidy.webm**

- NASA Johnson Space Center
- Public Domain
- via Wikimedia Commons
- 元動画から先頭60秒を実験用に利用

Wikimedia Commons側でHTTP 429が発生する場合があるため、Notebookのダウンロード処理ではretryとexponential backoffを利用しています。

## Gradio UI

Notebookの最後では、Gradioを利用した簡単なマルチモーダルUIを用意しています。

無料版ColabのGPUメモリ制約を考慮し、公開版UIでは主に、

- テキスト
- 画像
- 音声

を入力できる構成にしています。

動画については、Notebook内の専用セルでフレーム抽出して試します。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、メモリ、CUDA環境、Pythonパッケージ、Transformersなどは変更される可能性があります。

そのため、将来同じNotebookを実行した場合に、同じ結果が再現されることを保証するものではありません。

特に無料版Colabでは、利用できるGPUやGPUメモリが固定されていません。

本READMEに記載している無料版の結果は、著者環境でT4 GPUが割り当てられた際の実行結果です。

各モデルを利用する際には、Hugging Face上の配布元が示す最新のライセンスおよび利用条件を確認してください。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルそのものではありません。

書籍著者が、刊行後に公開された新しいLLMを個人的に試した追加実験です。

本Notebookでより興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

をぜひお手にとってください。

## Links

- Gemma 4 E4B-it QAT mobile Transformers: `google/gemma-4-E4B-it-qat-mobile-transformers`
- Notebook: `Gemma4_E4B_Multimodal_Colab_v4_fixed.ipynb`
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月時点のものです。Google Colabのランタイム、GPU、メモリ、CUDA、PyTorch、Transformersなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新のライセンスおよび利用条件を確認してください。
