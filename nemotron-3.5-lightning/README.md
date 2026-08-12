# NVIDIA Nemotron 3.5 Lightning on Google Colab

NVIDIAの `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` を、Google Colab上でbitsandbytesによる4bit量子化を用いて動かす実験です。

## Model

- Model: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Total parameters: 約30B
- Active parameters: 約3B
- Architecture: MoEを含むハイブリッド構成
- Quantization in this Notebook: bitsandbytes 4bit / NF4

今回使用するのはNVFP4版ではなく、BF16版です。

NVFP4版はNVIDIA ModelOpt向けの量子化設定を含んでおり、通常のTransformers + bitsandbytes構成ではそのまま扱いにくかったため、BF16版を読み込み時に4bit量子化する方法を採用しています。

## Notebook

`Nemotron35_Lightning_Colab.ipynb`

Notebookの構成は、できるだけシンプルにしています。

- Google ColabのGPU確認
- Google DriveへのHugging Faceキャッシュ保存
- Transformersによるモデル読み込み
- bitsandbytesによる4bit量子化
- 日本語での簡単な動作確認
- GradioによるチャットUI

## Environment Tested

著者が確認した範囲では、以下の結果になりました。

### Google Colab Free

著者環境では動作しませんでした。

Nemotron 3.5 Lightningは30B total / 3B activeのMoEモデルですが、推論時に有効になるパラメータ数が約3Bであっても、30B規模のモデル重みを保持するためのメモリが不要になるわけではありません。

そのため、無料版Colabに割り当てられるGPUやシステムメモリでは、今回の方法では実行できませんでした。

### Google Colab Pro

Google Colab Proの **G4 GPUランタイム** で動作を確認しました。

モデル読み込み時にBF16版をそのまま利用するのではなく、bitsandbytesで4bit量子化しています。

## 4bit Quantization

Notebookでは、以下のようなbitsandbytes設定を利用しています。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

この設定により、BF16モデルを読み込み時に4bitへ量子化し、必要なGPUメモリを削減しています。

ただし、4bit量子化すればどのようなサイズのLLMでもColabで動くというわけではありません。  
モデルロード時にはGPUメモリだけでなく、システムメモリや一時的なメモリ使用量も影響します。

## Reasoning Mode

Nemotron 3.5 Lightningでは、通常の回答とは別にreasoning / thinkingを生成するモードがあります。

通常のチャット用途では、Notebook内で次のように設定しています。

```python
enable_thinking=False
```

これを指定しない場合、簡単な質問でも

```text
Here's a thinking process:
...
```

のような推論過程が出力される場合があります。

今回のNotebookでは、一般的なチャットモデルとして使いやすくするため、thinkingを無効にしています。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、メモリ、CUDA環境、Pythonパッケージは変更される可能性があります。  
そのため、将来同じNotebookを実行した場合に、同じ結果が再現されることを保証するものではありません。

また、無料版Colabについても「常に動かない」という意味ではありません。  
本READMEに記載しているのは、著者が実験した時点の環境では動作しなかった、という結果です。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルではありません。  
書籍著者が、刊行後に公開された新しいLLMを個人的に試した追加実験です。

## Links

- Hugging Face: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`
- Notebook: `Nemotron35_Lightning_Colab.ipynb`
