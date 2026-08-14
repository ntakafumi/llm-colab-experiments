# Meta Muse Glimmer 30B on Google Colab

Metaの **Muse Glimmer 30B** を、Google Colab上で **GGUF + llama.cpp** を用いて動かす実験です。

これまでのNotebookでは主にTransformers + bitsandbytesを利用してきましたが、Muse Glimmerでは公開直後からGGUF版とllama.cppによるローカル推論環境が整っていたため、今回は実行方式を変更しています。

外部APIは使用せず、Hugging Faceから量子化済みGGUFをダウンロードし、Google Colab上でローカル推論します。

## Model

- Model: `Muse Glimmer 30B`
- Parameters: 約30B
- Model type: Dense LLM
- Distribution format in this Notebook: GGUF
- Quantization: Q4 class GGUF
- Inference engine: `llama.cpp`
- Input: Text
- Inference: Local inference on Google Colab

今回使用するのは、Muse Glimmer 30Bの量子化済みGGUFです。

NotebookではHugging Face上のGGUFリポジトリから利用可能なモデルファイルを取得し、Q4系のファイルを優先して自動選択します。

著者環境では、

```text
Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf
```

を利用して動作しました。

Q4級GGUFは約18GB前後あるため、8Bクラスの4bitモデルよりかなり大きなモデルです。

## Why GGUF + llama.cpp?

これまでの実験では、

```text
Transformers
+
bitsandbytes
+
4bit quantization
```

という構成を主に利用してきました。

Muse Glimmerでは、公開直後からGGUF版とllama.cpp側の対応が進んでいたため、今回は **GGUF + llama.cpp** を採用しています。

GGUFは、ローカルLLMで広く利用されている量子化済みモデル形式です。

またllama.cppを利用すると、GPU VRAMへ載せるlayer数を調整できるため、

```text
GPU
+
CPU
```

へモデルを分割して推論することもできます。

このNotebookでは、Google Colab上でCUDA対応のllama.cppをビルドし、

```text
GGUF
 ↓
llama-server
 ↓
OpenAI互換API
 ↓
Gradio
```

という構成でチャットUIを実現しています。

## Notebook

`Muse_Glimmer_30B_Colab_GGUF_Chat.ipynb`

Notebookの構成は、以下のようになっています。

- Google ColabのGPU確認
- Google DriveへのGGUF保存
- llama.cppのCUDA対応ビルド
- Hugging Face上のGGUFファイル自動検出
- Q4系GGUFのダウンロード
- llama-serverの起動
- 日本語での簡単な動作確認
- OpenAI互換APIによる推論
- GradioによるチャットUI

公開直後のモデルでも壊れにくいように、GGUFファイル名を完全固定せず、候補から自動選択するようにしています。

## Environment Tested

著者が確認した範囲では、以下の結果になりました。

### Google Colab Free

本Notebookでは動作対象外としています。

Muse Glimmer 30BのQ4級GGUFは約18GB前後あり、Google Colab無料版で一般的に利用される16GB級GPUでは、モデル全体をGPUへロードすることができません。

llama.cppでは、

```python
GPU_LAYERS = 30
```

のようにGPUへ載せるlayer数を減らし、残りをCPUへoffloadすることは可能です。

そのため、無料版Colabでも理論上は動作する可能性があります。

しかし、

- 30B denseモデルである
- GPUへ載らないlayerをCPUで処理する必要がある
- システムメモリも使用する
- GGUF自体が約18GBある
- 推論速度が大幅に低下する可能性が高い

という理由から、無料版での実用的な利用は想定していません。

したがって本Notebookでは、

> **Google Colab無料版は対象外。Colab Pro以上を推奨。**

という位置づけにしています。

### Google Colab Pro

**動作を確認しました。**

著者環境では、Google Colab ProのGPUランタイムでMuse Glimmer 30BのQ4級GGUFを読み込み、ローカル推論とGradioチャットまで動作しました。

今回のNotebookでは、

```python
GPU_LAYERS = 999
```

として、可能な限り全layerをGPUへoffloadしています。

また、context lengthは最初から最大にせず、

```python
CONTEXT_SIZE = 8192
```

としています。

長いcontextを利用するとKV cacheが増え、GPUメモリ使用量も増えるため、Google Colabではまず8K程度から試す構成にしています。

## GGUF Model Loading

Notebookでは、Hugging Face上のGGUFファイルを一覧取得し、Q4系を優先して自動選択します。

```python
priority_keywords = [
    "kquant-dynamic",
    "q4_k_xl",
    "q4_k_m",
    "q4",
]
```

この方法により、公開直後にGGUFファイル名が多少変更されても、利用可能なQ4系モデルを検出しやすくしています。

モデルはGoogle Driveへ保存するため、初回ダウンロード後は同じファイルを再利用できます。

## llama.cpp

Notebookでは、最新版のllama.cppをGoogle Colab上で取得し、CUDA対応でビルドします。

```bash
cmake \
  -S llama.cpp \
  -B llama.cpp/build \
  -DGGML_CUDA=ON \
  -DLLAMA_CURL=ON \
  -DCMAKE_BUILD_TYPE=Release
```

その後、

```text
llama-server
```

を起動します。

Muse Glimmerは公開直後のモデルであるため、古いllama.cppでは未対応になる可能性があります。

そのためNotebookでは、固定された古いbinaryを使用せず、実行時に最新版を取得してビルドする構成にしています。

## llama-server and OpenAI Compatible API

Muse Glimmerの推論は、Pythonから直接モデルオブジェクトを呼び出すのではなく、`llama-server`を利用しています。

llama-serverを起動すると、

```text
/v1/chat/completions
```

というOpenAI互換APIを利用できます。

NotebookではPythonの `requests` を使って、

```python
requests.post(
    base_url + "/v1/chat/completions",
    json=payload,
)
```

のようにローカルサーバへリクエストしています。

ここで利用しているのは外部APIではありません。

Google Colab内で起動したllama-serverへ、同じColab環境からHTTPリクエストを送っています。

この構成により、

```text
Local GGUF model
 ↓
llama.cpp
 ↓
OpenAI-compatible API
 ↓
Application
```

という、ローカルLLMサーバとしての構成も確認できます。

## Dynamic Port Selection

初期版ではllama-serverを、

```text
127.0.0.1:8080
```

で起動していました。

しかしGoogle Colab環境では、8080番ポートが別のプロセスに利用されている場合があり、

```text
couldn't bind HTTP server socket
```

というエラーが発生することがありました。

v2では、

```python
s.bind(("127.0.0.1", 0))
```

として、OSに空いているTCPポートを自動選択させています。

これにより、以前起動したサーバやColab内部のプロセスとのポート競合を避けやすくしています。

## Gradio Chat UI

Notebookの最後では、Gradioを利用したシンプルなチャットUIを起動します。

UIは、

- チャット履歴
- 質問入力
- Send
- Clear

だけの構成です。

Muse Glimmerへの推論は、Gradioからllama-serverのOpenAI互換APIへ送信されます。

Gradioはバージョンによって `gr.Chatbot` の引数が異なる場合があるため、v2では、

```python
inspect.signature(gr.Chatbot).parameters
```

を確認し、`type="messages"` が利用可能な場合だけ指定するようにしています。

## Important Notes

このNotebookは、著者が実際にGoogle Colab上で試した実験記録です。

Google ColabのGPU、メモリ、CUDA環境、Pythonパッケージ、llama.cpp、GGUF配布状況は変更される可能性があります。

そのため、将来同じNotebookを実行した場合に、同じ結果が再現されることを保証するものではありません。

またMuse Glimmerは公開直後のモデルであり、llama.cpp側の対応状況も今後変更される可能性があります。

GGUFファイルについても、配布元で量子化方式やファイル名が変更される可能性があります。

本Notebookではできるだけ最新環境へ追従しやすい構成にしていますが、実行時にはHugging Faceおよびllama.cppの最新情報を確認してください。

各モデルを利用する際には、配布元の最新のライセンスおよび利用条件も確認してください。

## Related Book

『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』

このNotebookは、上記書籍の公式サンプルではありません。

書籍著者が、刊行後に公開された新しいLLMを個人的に試した追加実験です。

本記事やNotebookでより興味を持っていただけましたら、

[「ブラウザで動かすLLM実装入門　Google Colaboratoryで実践するLLM・RAG・ファインチューニング」(インプレス, 2026/8/4)](https://amzn.asia/d/0dV4aYtC)

をぜひお手にとってください。

## Links

- Muse Glimmer
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Notebook: `Muse_Glimmer_30B_Colab_GGUF_Chat_v2.ipynb`
- 『ブラウザで動かすLLM実装入門』: https://amzn.asia/d/0dV4aYtC

---

※本READMEの実行結果は2026年8月時点のものです。Google Colabのランタイム、GPU、メモリ、CUDA、Pythonパッケージ、llama.cpp、GGUFなどの環境は今後変更される可能性があります。また、各モデルを利用する際には、配布元の最新のライセンスおよび利用条件を確認してください。
