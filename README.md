# llm-colab-experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ntakafumi/llm-colab-experiments)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)

Google Colabで、比較的新しいオープンLLMを実際に動かしてみるための**個人的な実験置き場**です。

書籍『ブラウザで動かすLLM実装入門』の著者が、時間のあるときに気になったモデルを触りながら、

> **「このモデルはGoogle Colabで、どこまで現実的に動かせるのか？」**

を試していきます。

このリポジトリは、書籍の公式サンプルコードとは独立しています。

---

## About

オープンLLMは非常に速いペースで公開・更新されています。

しかし、モデルが公開されたからといって、すべてのモデルをGoogle Colabで簡単に動かせるわけではありません。モデルサイズ、量子化方式、GPUメモリ、システムメモリ、ライブラリの対応状況などによって、実行できる条件は大きく変わります。

このリポジトリでは、**動いた例だけを集めるのではなく、動かなかった条件や試行錯誤も含めて記録する**ことを重視します。

「最新LLMをすべて追いかける」ことは目的にしていません。  
Google Colabで試してみる意味がありそうなモデルを見つけたときに、無理のないペースで少しずつ追加していきます。

---

## How to use

各実験は、モデルごとのサブディレクトリに分けています。

```text
llm-colab-experiments/
├── README.md
├── <experiment-name>/
│   ├── README.md
│   └── <notebook>.ipynb
├── <experiment-name>/
│   ├── README.md
│   └── <notebook>.ipynb
└── ...
```

使い方はシンプルです。

1. 興味のある実験ディレクトリを開く
2. そのディレクトリの `README.md` で、モデル・実行条件・注意点を確認する
3. **Open in Colab** バッジからNotebookを開く
4. 必要に応じてColabのランタイムを変更して、上から順に実行する

各実験のREADMEには、できるだけ次の情報を残します。

- 使用したモデルと配布元
- Notebookで採用したロード方法・量子化方法
- 動作確認したGoogle Colabのランタイム
- 無料版Colabでの確認結果
- 実行時に気づいた制約や注意点
- モデル固有の設定
- 実験時点の日付

---

## Repository policy

### 1. 成功例だけにしない

無料版Colabでは動かない、特定のGPUでしか動かない、ライブラリの組み合わせによって失敗する、といった結果も記録します。

### 2. 「動く／動かない」を断定しすぎない

Google Colabの利用可能なGPU、メモリ、ソフトウェア環境は固定ではありません。

そのため、このリポジトリでの結果は、

> **実験した時点の著者環境ではどうだったか**

という記録として扱います。

### 3. Notebookはできるだけ読みやすくする

モデルを動かすために必要以上に複雑な構成にはせず、可能な範囲で、

- Transformers
- bitsandbytes
- Gradio
- Google Driveのキャッシュ

など、一般的な構成で読めるNotebookを目指します。

ただし、モデルによってはvLLM、専用量子化形式、独自ランタイムなどが必要になる場合があります。

### 4. 再現性よりも「実験時点の記録」を優先する場合がある

新しいモデルや新しいライブラリは更新が速く、数か月後に同じコードがそのまま動くとは限りません。

必要に応じてNotebookやREADMEを更新しますが、古い実験結果そのものにも意味があるため、可能な限り実験条件を残します。

---

## Google Colabについて

このリポジトリではGoogle Colabを主な実行環境として使います。

無料版で動作することを前提にはしていません。  
特にパラメータ数の大きいLLMでは、4bit量子化などを利用してもGPUメモリやシステムメモリが不足する場合があります。

また、MoE（Mixture-of-Experts）モデルでは「Active Parameters」が小さくても、モデル全体の重みを保持するためのメモリが必要になる場合があります。

---

## Relationship to the book

このリポジトリは、以下の書籍の**公式サンプルコードではありません**。

**『ブラウザで動かすLLM実装入門  
Google Colaboratoryで実践するLLM・RAG・ファインチューニング』**

- Amazon: https://www.amazon.co.jp/dp/4295024813

書籍で扱った基本的な考え方を土台にしながら、刊行後に登場したモデルを著者が個人的に試すための独立した実験リポジトリです。

---

## Notes

- 各モデルの利用条件・ライセンスは、必ず配布元のライセンスを確認してください。
- Notebookの実行には、大容量のモデルダウンロードやGoogle Driveの空き容量が必要になる場合があります。
- Google Colabの仕様変更やPythonパッケージの更新により、過去のNotebookが動作しなくなる可能性があります。
- 本リポジトリのコードは研究・学習・実験用途を想定しています。

---

## License

このリポジトリ自体のライセンスは、公開時に別途明示します。  
モデル、データセット、外部ライブラリについては、それぞれの配布元のライセンスが適用されます。
