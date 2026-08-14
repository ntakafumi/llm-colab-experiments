# DeepSeek-R1-0528-Qwen3-8B on Google Colab

DeepSeekの `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` を、Google Colab上でbitsandbytesによる4bit量子化を用いて動かす実験です。

外部APIは使用せず、Hugging Faceからモデルをダウンロードし、Google Colab上でローカル推論します。

## Model

- Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Parameters: 約8B
- Base architecture: Qwen3-8B
- Model type: Reasoning LLM
- Quantization in this Notebook: bitsandbytes 4bit / NF4
- Input: Text
- Inference: Local inference on Google Colab

今回使用するのは、DeepSeek-R1-0528の8Bモデルです。

より巨大なDeepSeekモデルではなく、Google Colab上で実際にローカル推論できるサイズの公式DeepSeekモデルとして、このモデルを利用しています。

Notebookでは、Hugging Face上のモデルを読み込む際にbitsandbytesによる4bit量子化を行い、GPUメモリ使用量を削減しています。

## Notebook

`DeepSeek-R1-0528.ipynb`

Notebookの構成は、できるだけシンプルにしています。

- Google ColabのGPU確認
- Google DriveへのHugging Faceキャッシュ保存
- TransformersによるTokenizerとモデルの読み込み
- bitsandbytesによる4bit量子化
- 日本語での簡単な動作確認
- Reasoning / Thinking部分の分離
- GradioによるチャットUI
- Thinking表示のON / OFF

## Environment Tested

著者が確認した範囲では、以下の結果になりました。

### Google Colab Free

**動作を確認しました。**

Google Colab無料版でも、4bit量子化したDeepSeek-R1-0528-Qwen3-8Bを読み込み、ローカル推論することができました。

ただし、回答生成速度は遅めです。

このモデルはReasoningモデルであり、質問によっては最終回答を生成する前に長いThinkingを生成します。そのため、8Bの4bitモデルがGPUメモリに収まっていても、通常の8Bチャットモデルと比べて回答までに時間がかかる場合があります。

特に長いReasoningを必要とする質問では、生成token数が増えるため、無料版Colabでは待ち時間が長くなります。

したがって、

> **Google Colab無料版でも動作可能。ただし生成速度は遅い。**

というのが、著者が実際に試した時点での結果です。

### Google Colab Pro

Google Colab Proでは、より高速なGPUが割り当てられた場合、無料版より快適に利用できる可能性があります。

ただし、Google Colabでは契約プランにかかわらず、常に特定のGPUが割り当てられることが保証されているわけではありません。

今回のモデルは8Bを4bit量子化して利用するため、モデルサイズの面では比較的Colabで扱いやすいモデルです。

## 4bit Quantization

Notebookでは、以下のようなbitsandbytes設定を利用しています。

```python
COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)
```

モデルは次のように4bitで読み込みます。

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=str(CACHE_DIR),
    low_cpu_mem_usage=True,
)
```

この方法では、元のモデルをHugging Faceから取得し、読み込み時にbitsandbytesのNF4 4bit量子化を適用します。

4bit量子化によってGPUメモリ使用量を大幅に削減できるため、8BクラスのReasoningモデルをGoogle Colab無料版でも試すことが可能になります。

ただし、4bit量子化によって削減されるのは主としてモデル重みのメモリ使用量です。

実際の推論時には、

- KV cache
- 入力token
- 生成token
- 中間tensor
- 会話履歴

などにもGPUメモリが使われます。

そのため、長い入力や長いReasoningを生成すると、必要なメモリと生成時間は増加します。

## Reasoning Mode

DeepSeek-R1-0528-Qwen3-8BはReasoningモデルです。

生成結果には、最終回答とは別にThinking部分が含まれる場合があります。

Notebookでは、以下のような形式を検出し、

```text
<think>
...
</think>
```

Thinking部分と最終回答を分離しています。

通常のチャットではThinking部分を表示せず、最終回答だけを表示します。

Gradio UIでは、

```text
Thinkingも表示する
```

というチェックボックスを用意しており、必要に応じてReasoning部分も確認できます。

Notebook内の生成設定は、基本的に次の値を利用しています。

```python
temperature=0.6
top_p=0.95
```

また、生成token数が大きいほど回答に時間がかかるため、無料版Colabでは最初は短めの生成設定で試すことを推奨します。

## Google Colab Freeで遅くなる理由

Google Colab無料版でモデルが動いたとしても、必ずしも高速に生成できるわけではありません。

特にDeepSeek-R1系では、回答を生成する前にReasoningを長く生成する場合があります。

たとえば、通常のチャットモデルが数十tokenで回答できる質問でも、Reasoningモデルでは内部的に数百token以上を生成する可能性があります。

生成速度が同じであれば、生成するtoken数が増えるほど回答までの時間も長くなります。

そのため、このモデルでは、

> モデルがGPUメモリに載るか

だけでなく、

> 何tokenのReasoningを生成するか

も体感速度に大きく影響します。

無料版Colabでは、まず `max_new_tokens` を512〜1024程度に抑えて試すと扱いやすくなります。

## Gradio Chat UI

Notebookでは、Gradioを利用した簡単なチャットUIも用意しています。

UIでは、

- 質問入力
- Thinking表示のON / OFF
- Send
- Clear
- チャット履歴

を利用できます。

外部のDeepSeek APIへ質問を送るのではなく、Google Colab上に読み込んだモデルが回答を生成します。

したがって、このNotebookは **DeepSeek-R1-0528-Qwen3-8Bのローカル推論実験**です。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、メモリ、CUDA環境、Pythonパッケージは変更される可能性があります。  
そのため、将来同じNotebookを実行した場合に、同じ結果が再現されることを保証するものではありません。

また、無料版Colabについても、常に同じGPUや計算資源が割り当てられるわけではありません。

本READMEに記載している「無料版で動作した」という結果は、著者が実験した時点のGoogle Colab環境で実際に動作を確認した、という意味です。

Reasoningモデルは生成token数が多くなる場合があるため、無料版では特に応答に時間がかかることがあります。

各モデルを利用する際には、Hugging Face上の配布元が示す最新のライセンスおよび利用条件を確認してください。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルではありません。  
書籍著者が、刊行後に公開された新しいLLMを個人的に試した追加実験です。

Amazon:

https://www.amazon.co.jp/dp/4295024813

## Links

- Hugging Face: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
- Hugging Face: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- Notebook: `DeepSeek_R1_0528_Qwen3_8B_Colab_4bit_Chat.ipynb`
- 『ブラウザで動かすLLM実装入門』Amazon: https://www.amazon.co.jp/dp/4295024813

---

※本READMEの実行結果は2026年8月時点のものです。Google Colabのランタイム、GPU、メモリ、CUDA、PyTorch、Transformers、bitsandbytesなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新のライセンスおよび利用条件を確認してください。
