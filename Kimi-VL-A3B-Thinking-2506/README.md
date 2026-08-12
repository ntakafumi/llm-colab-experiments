# Kimi-VL-A3B-Thinking-2506 on Google Colab

Moonshot AIのKimi-VL-A3B-Thinking-2506を4bit量子化した `SoybeanMilk/Kimi-VL-A3B-Thinking-2506-BNB-4bit` を、Google Colab上でTransformers + bitsandbytesを用いて動かす実験です。

## Model

- Model: `SoybeanMilk/Kimi-VL-A3B-Thinking-2506-BNB-4bit`
- Base model: `moonshotai/Kimi-VL-A3B-Thinking-2506`
- Model type: Vision-Language Model
- Quantization: bitsandbytes 4bit
- Model files: 約9.5GB
- Input: Text / Image + Text

今回使用するのは、Kimi-VL-A3B-Thinking-2506をbitsandbytes 4bitで量子化したモデルです。

Notebookでは、量子化設定がモデル側のconfigに保存されているため、`BitsAndBytesConfig` を読み込み時にもう一度指定せず、そのまま4bitモデルを読み込んでいます。

## Notebook

`Kimi_VL_A3B_Thinking_2506_Colab_BNB4bit.ipynb`

Notebookの構成は、できるだけシンプルにしています。

- Google ColabのGPU確認
- PillowとTransformersの実行環境設定
- Google DriveへのHugging Faceキャッシュ保存
- Transformersによる4bitモデルの読み込み
- Kimi-VL用Processorの読み込み
- 日本語でのテキスト動作確認
- 画像を用いたVision-Language推論
- Gradioによる画像・テキスト入力UI

## Environment Tested

著者が確認した範囲では、以下の結果になりました。

### Google Colab Free

未確認です。

Kimi-VLの4bitモデル自体は約9.5GBですが、モデルの読み込みにはGPUメモリだけでなく、システムメモリやモデルロード時の一時的なメモリ使用量も影響します。

そのため、無料版Colabで常に実行できることを保証するものではありません。

### Google Colab Pro

著者が実行したGoogle Colab環境では、**NVIDIA RTX PRO 6000 Blackwell Server Edition** で動作を確認しました。

実行時に確認された環境は以下です。

```text
Python: 3.12.13
PyTorch: 2.11.0+cu128
Transformers: 4.48.2
Pillow: 11.3.0
bitsandbytes: 0.50.0
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
GPU memory: 95.0 GB
```

モデル読み込み後のGPUメモリ使用量は、著者環境では以下でした。

```text
4bit: True
GPU allocated: 9.94 GB
GPU reserved : 9.97 GB
```

なお、Google Colabでは利用可能なGPUが実行時期や契約プランによって変わるため、常に同じGPUが割り当てられるわけではありません。

## 4bit Quantization

Notebookでは、あらかじめbitsandbytes 4bitで量子化されたモデルを利用しています。

```python
MODEL_ID = "SoybeanMilk/Kimi-VL-A3B-Thinking-2506-BNB-4bit"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)
```

量子化情報はモデル側のconfigに保存されているため、Notebook側では `BitsAndBytesConfig` を二重に指定していません。

実際の読み込み時にも、

```text
4bit: True
```

となることを確認しています。

モデルファイルは2つのsafetensors shardに分割されており、Notebook実行時には約5.00GBと約4.48GBのファイルがダウンロードされました。

## Reasoning Mode

Kimi-VL-A3B-Thinking-2506は、モデル名のとおりThinkingを行うモデルです。

Notebook内の応答生成関数では、Thinking部分と最終回答部分を分離し、通常の利用では最終回答を優先して表示するようにしています。

```python
show_thinking=False
```

Thinkingタグが含まれる場合は、

```text
◁think▷
...
◁/think▷
```

より後ろの最終回答部分を取り出します。

今回のNotebookでは、一般的なチャットモデルとして使いやすくするため、Gradio UIではThinking部分を表示しない設定にしています。

## Vision-Language Input

Kimi-VLはテキストだけでなく、画像とテキストを組み合わせた入力にも対応しています。

Notebookでは、画像が指定された場合に、

```python
content.append({"type": "image", "image": ""})
content.append({"type": "text", "text": user_text})
```

のようにメッセージを構築し、`AutoProcessor` を用いて画像とテキストをモデルへ入力しています。

Gradio UIも、設定項目を増やしすぎず、

- 画像入力
- テキスト入力
- Send
- Clear
- 回答表示

のみのシンプルな構成にしています。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、メモリ、CUDA環境、Pythonパッケージは変更される可能性があります。  
そのため、将来同じNotebookを実行した場合に、同じ結果が再現されることを保証するものではありません。

Kimi-VLでは `trust_remote_code=True` を使用しており、Hugging Faceからモデル固有のPythonコードを読み込みます。

また、Google Colab環境でPillowとTransformersの依存関係が衝突するケースがあったため、このNotebookではPillowとTransformersのバージョンを明示的に設定しています。

最初の環境設定セルを実行すると、依存関係を反映するためColabランタイムを一度再起動します。  
再接続後は、環境チェックのセルから実行してください。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルではありません。  
書籍著者が、刊行後に公開された新しいLLMを個人的に試した追加実験です。

## Links

- Hugging Face: `SoybeanMilk/Kimi-VL-A3B-Thinking-2506-BNB-4bit`
- Base Model: `moonshotai/Kimi-VL-A3B-Thinking-2506`
- Notebook: `Kimi_VL_A3B_Thinking_2506_Colab_BNB4bit.ipynb`
