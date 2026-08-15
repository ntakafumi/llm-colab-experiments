# Qwen3.8-27B on Google Colab

Qwenの **Qwen3.8-27B** をGoogle Colab上でbitsandbytesによるNF4 4bit量子化を利用して動かす実験です。

今回のNotebookでは `Qwen/Qwen3.8-27B` をHugging Faceから取得し、4bit量子化、日本語生成、Thinkingを無効化した通常チャット、Gradio UIまでを試します。外部LLM APIは使用しません。

## Model

- Model: `Qwen/Qwen3.8-27B`
- Parameters: 27B class / dense
- Inference: Local inference on Google Colab
- Quantization: bitsandbytes NF4 4bit
- Interface: Transformers + Gradio

27B denseモデルなので、BF16のまま一般的なColab GPUへ載せるのは困難です。そこで書籍のPhi-4実装と同じ考え方でNF4 4bitを利用します。

## Notebook

`Qwen3_8_27B.ipynb`

```text
GPU確認
→ Google Drive / Hugging Face cache
→ Qwen3.8-27BをNF4 4bitでロード
→ 日本語動作確認
→ 推論テスト
→ Gradio Chat
→ GPUメモリ確認
```

## Environment

```python
%pip -q install -U transformers accelerate bitsandbytes
```

TorchとGradioは可能な限りColab既定環境を利用します。

## Google Drive Cache

Qwen3.8-27Bのcheckpointは非常に大きいため、Hugging Face cacheをGoogle Driveへ保存します。

```python
CACHE_DIR = PROJECT_DIR / "Program" / "hf_cache_qwen38_27b"
```

無料版Colabで取得した際には、

```text
Download complete: 47.7GB
Reconstruction complete: 55.6GB
Fetching 18 files: 100%
```

となりました。4bit推論を行う場合でも、取得する元checkpoint自体は非常に大きいため、Driveには十分な空き容量が必要です。

## NF4 4bit

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)
```

モデルは、

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)
```

として読み込みます。

27Bを単純に4bitとして計算しても、重みだけで概算約13.5GBです。実際にはquantization metadata、CUDA context、KV cache、activation、一時tensor等も必要なので、「27Bを4bitにすれば16GB GPUへ必ず載る」わけではありません。

## Thinking Mode

Qwen系ではThinking modeが利用される場合があります。通常チャットでは推論過程を表示させないため、chat templateで、

```python
enable_thinking=False
```

を指定します。

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_dict=True,
    return_tensors="pt",
)
```

これにより通常のGradio Chatでは最終回答だけを表示します。

## Japanese Chat

簡単な日本語応答に加えて、

```text
定価3000円の本を20%引きで購入し、
その価格に10%の消費税がかかります。
支払額を計算過程とともに簡潔に説明してください。
```

のような推論テストも行います。

有料版Colabでは4bitモデルの読み込み、日本語生成、Thinking OFF、Gradio UIまで動作を確認しました。

## Gradio UI

書籍のPhi-4 Notebookと同様に、シンプルなローカルチャットUIを利用します。

```text
Qwen3.8-27B — Local 4bit Chat
```

有料版ではGradioからの日本語チャットまで正常に動作しました。

## Google Colab Paid

著者環境のGoogle Colab有料版では、

```text
Qwen/Qwen3.8-27B
+
bitsandbytes NF4 4bit
```

で、

```text
Model Load      ⚪︎
Japanese Chat   ⚪︎
Thinking OFF    ⚪︎
Gradio          ⚪︎
```

まで確認できました。

## Google Colab Free / T4

無料版T4でも試しました。

checkpointのダウンロードと再構築自体は成功しました。

```text
Download complete: 47.7GB
Reconstruction complete: 55.6GB
Fetching 18 files: 100%
```

しかし4bitモデル読み込み時に、

```text
ValueError:
Some modules are dispatched on the CPU or the disk.
Make sure you have enough GPU RAM to fit the quantized model.
```

となりました。

`device_map="auto"`でモデル全体をT4へ配置できず、一部moduleをCPUまたはdiskへdispatchする必要が生じたためです。

## Why Free T4 Failed

概念的には、

```text
27B model
↓
NF4 4bit
↓
weights alone ≈ 13.5GB
↓
T4 VRAM ≈ 14–15GiB
↓
metadata / CUDA / cache / activation等の余裕がない
↓
device_map="auto" がCPU/disk配置を必要とする
↓
通常のbnb 4bit loaderではロード停止
```

となりました。

したがって無料版では「ダウンロード失敗」ではなく、**55.6GB checkpoint取得成功後の4bitモデル配置段階でVRAM不足**となっています。

## Paid vs Free

| Environment | Download | 4bit Load | Japanese Generation | Gradio |
|---|---:|---:|---:|---:|
| **Colab Paid / larger GPU** | ⚪︎ | ⚪︎ | ⚪︎ | ⚪︎ |
| **Colab Free / T4** | ⚪︎ | × | — | — |

## Could CPU / Disk Offload Work?

エラーメッセージでは、CPU / diskへ一部moduleをoffloadする別構成も示唆されています。そのため、custom `device_map`等を使えばさらに先へ進める可能性はあります。

ただし本Notebookでは検証していません。

今回の目的は、書籍のPhi-4実装に近い通常の、

```text
Transformers
+
bitsandbytes NF4
+
device_map="auto"
```

でどこまで動くかを確認することです。

無料版T4については、**通常のNF4 4bit方式ではLoad failed**という実測結果を記録します。

## Why Not Force It Further?

CPU / disk offloadまで追い込むと、GPU↔CPU↔Diskの転送が増え、推論速度やCPU RAM、実装の複雑さが大きく変わります。

今回は通常のLLMチャットとして使うことが目的なので、MiniMax-H3のように「とにかく完走させる」方向には進めず、無料版では通常4bit実装の限界で区切っています。

## Important: 4bit Does Not Mean Small Enough

今回の重要なポイントは、

```text
27B
→ 4bit
→ T4なら余裕
```

とはならないことです。

量子化によって重みは大幅に小さくできますが、

```text
Model weights
```

と、

```text
Memory required to actually run the model
```

は同じではありません。

## What We Learned

```text
Colab Paid
→ Load:       ○
→ Generation: ○
→ Gradio:     ○

Colab Free T4
→ Download:   ○
→ Load:       ×
```

27B denseモデルは、4bit量子化してもT4 16GBクラスでは非常に厳しい境界にあります。一方、有料版のより大きなGPUでは、書籍のPhi-4とほぼ同じシンプルな構成でローカルチャットまで実行できました。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、GPUメモリ、CUDA、PyTorch、Transformers、bitsandbytes等は今後変更される可能性があります。無料版・有料版とも常に同じGPUが割り当てられるわけではありません。

今回の無料版T4での結果は、

```text
official checkpoint download → success
bitsandbytes NF4 4bit load   → failed
```

です。CPU / disk offloadを利用した別構成は本Notebookでは検証していません。

各モデルを利用する際には、配布元の最新ライセンスおよび利用条件を確認してください。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは上記書籍の公式サンプルそのものではなく、書籍著者が刊行後に公開された新しいopen-weightモデルを試した追加実験です。

本Notebookでより興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

をぜひお手にとってください。

## Links

- Qwen3.8-27B: `Qwen/Qwen3.8-27B`
- Notebook: `Qwen3_8_27B.ipynb`
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月15日時点のものです。Google Colabのランタイム、GPU、GPUメモリ、CUDA、PyTorch、Transformers、bitsandbytesなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新のライセンスおよび利用条件を確認してください。
