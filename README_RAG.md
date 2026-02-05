# Ollama + Weaviate RAGシステム 完全ガイド

## 目次

- [はじめに](#はじめに)
- [システム構成](#システム構成)
- [クイックスタート](#クイックスタート)
- [詳細な使い方](#詳細な使い方)
- [RAGAS評価](#ragas評価)
- [データ管理](#データ管理)
- [高度な使い方](#高度な使い方)
- [トラブルシューティング](#トラブルシューティング)
- [FAQ（よくある質問）](#faqよくある質問)
- [システム要件](#システム要件)
- [参考情報](#参考情報)

---

## はじめに

### RAGシステムとは？

**RAG（Retrieval-Augmented Generation：検索拡張生成）**は、大量のドキュメントから関連情報を検索し、その情報を基にLLMが回答を生成する技術です。

### このシステムでできること

✅ **PDFやテキストファイルを質問可能に**
- PDFやMarkdownファイルを読み込んで質問に答えられる
- 複数のドキュメントを横断して検索可能

✅ **完全ローカル実行**
- すべての処理がDocker内で完結
- ローカル環境にPythonをインストール不要
- データは外部に送信されない（プライバシー保護）

✅ **意味的検索（セマンティック検索）**
- キーワード一致ではなく、意味で検索
- 表現が違っても同じ意味なら検索できる

✅ **システムの品質評価**
- RAGAS による定量的な評価が可能

### システムの特徴

- **日本語対応**: ELYZA-JP-8B モデルで日本語に最適化
- **GPU対応**: NVIDIA GPU で高速推論
- **追記型DB**: 同じファイルを再登録しても重複しない
- **複数フォーマット対応**: PDF, TXT, MD に対応

---

## システム構成

### アーキテクチャ図

```
┌──────────────────────────────────────────────────────────┐
│              Docker Compose環境（完全ローカル）           │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Ollama     │  │   Weaviate   │  │ Python App    │  │
│  │  Container   │◄─┤   Container  │◄─┤   Container   │  │
│  │              │  │              │  │               │  │
│  │ GPU有効化    │  │ ベクトルDB   │  │ ・インデックス│  │
│  │ elyza-jp-8b  │  │ HNSW Index   │  │ ・質問応答    │  │
│  │ mxbai-embed  │  │ LSM Tree     │  │ ・RAGAS評価   │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│         ↓                  ↓                              │
│    LLM推論 +         ベクトル検索                         │
│    Embedding生成     (セマンティック)                      │
└──────────────────────────────────────────────────────────┘
```

### 使用技術

| コンポーネント | 役割 | 詳細 |
|--------------|------|-----|
| **Ollama** | LLM推論エンジン | ELYZA-JP-8B で日本語回答生成<br>kun432/cl-nagoya-ruri-large でベクトル化 |
| **Weaviate** | ベクトルデータベース | HNSW インデックスで高速検索<br>LSM Tree でデータ永続化 |
| **Python App** | RAG処理 | ドキュメント読込・検索・評価 |

### データフロー

```
【インデックス作成】
PDF/テキスト → 読込 → チャンク化 → Embedding化 → Weaviateに保存
                                    ↓ (Ollama)
                           1024次元ベクトル

【質問応答】
質問 → Embedding化 → ベクトル検索 → 関連チャンク取得 → LLM生成 → 回答
        ↓ (Ollama)     ↓ (Weaviate)                      ↓ (Ollama)
    1024次元ベクトル   類似度計算                    コンテキスト付き
```

---

## クイックスタート

### 前提条件

- Docker & Docker Compose がインストール済み
- （オプション）NVIDIA GPU + nvidia-docker

### 5ステップで開始

```bash
# 1. コンテナ起動（初回は時間がかかります）
docker compose up -d

# 2. PDFやテキストファイルを配置
cp your_document.pdf data/documents/

# 3. ドキュメントをインデックス化
./run_indexer.sh data/documents/your_document.pdf

# 4. 質問応答システムを起動
./run_query.sh

# 5. 質問を入力してEnter
質問: このドキュメントの要約を教えてください
```

---

## 詳細な使い方

### 1. システム起動

```bash
# すべてのコンテナを起動
docker compose up -d

# 起動確認
docker compose ps

# 期待される出力:
# NAME       STATUS        PORTS
# ollama     Up X minutes  0.0.0.0:11434->11434/tcp
# weaviate   Up X minutes  0.0.0.0:8080->8080/tcp
```

**初回起動時の注意**:
- Ollamaモデルのダウンロードで数分かかります
- ネットワークが遅い場合、10分程度かかることも

### 2. ドキュメントのインデックス作成

#### 対応ファイル形式

- **PDF**: `.pdf`
- **テキスト**: `.txt`
- **Markdown**: `.md`, `.markdown`

#### 基本的な使い方

```bash
# 1つのファイルをインデックス化
./run_indexer.sh data/documents/<filename>
```

**例:**
```bash
# PDFファイル
./run_indexer.sh data/documents/manual.pdf

# テキストファイル
./run_indexer.sh data/documents/notes.txt

# Markdownファイル
./run_indexer.sh data/documents/README.md
```

#### 複数ファイルの一括インデックス化

```bash
# すべてのPDFファイル
for pdf in data/documents/*.pdf; do
    ./run_indexer.sh "$pdf"
done

# すべてのファイル（形式問わず）
for file in data/documents/*; do
    if [[ -f "$file" ]]; then
        ./run_indexer.sh "$file"
    fi
done
```

#### インデックス作成の仕組み

1. **ファイル読込**: PDF/テキストを読み込み
2. **テキスト抽出**: PDFの場合はテキスト抽出
3. **クリーニング**: 余分な空白や改行を整理
4. **チャンク化**: 500文字ごとに分割（50文字オーバーラップ）
5. **Embedding化**: Ollama (kun432/cl-nagoya-ruri-large) でベクトル化
6. **保存**: Weaviate に保存（既存データは自動削除）

#### 出力例

```
ドキュメント RAG インデックス作成
ファイル: data/documents/sample.pdf

✓ PDFロード完了: 26チャンク

既存コレクション 'PDFDocuments' を使用
既存データを削除: 26件 (ソース: data/documents/sample.pdf)
インデックス作成中... (26件)
Processing ━━━━━━━━━━━━━━━━ 100% 0:00:03
✓ 26件のドキュメントをインデックス化完了

インデックス作成完了！
```

### 3. 質問応答

#### 対話型質問システムの起動

```bash
./run_query.sh
```

#### 使い方

```
╭────────────────────────────────────────╮
│ PDF RAG 質問システム                   │
│ 質問を入力してください（終了: 'exit'） │
╰────────────────────────────────────────╯

質問: RAGシステムとは何ですか？

検索中...
╭──────────────────────────────────── 回答 ────────────────────────────────────╮
│ RAG（Retrieval-Augmented Generation）は、検索拡張生成と呼ばれる技術で、      │
│ 大量のドキュメントから関連情報を検索し、その情報を基にLLMが回答を生成します。 │
╰──────────────────────────────────────────────────────────────────────────────╯

ソース: data/documents/guide.md

質問: exit
```

#### 終了方法

以下のいずれかを入力：
- `exit`
- `quit`
- `終了`
- または `Ctrl+C`

#### 質問のコツ

**Good 質問例:**
- ✅ 「このドキュメントの主な内容は何ですか？」
- ✅ 「〇〇について詳しく教えてください」
- ✅ 「〇〇と△△の違いは何ですか？」

**Bad 質問例:**
- ❌ 「はい」「いいえ」で答えられる質問
- ❌ ドキュメントに記載されていない内容の質問
- ❌ 極端に短い質問（「これ」「あれ」など）

---

## RAGAS評価

### RAGAS とは？

**RAGAS（RAG Assessment）**は、RAGシステムの性能を定量的に評価するフレームワークです。

### 評価指標

| 指標 | 説明 | 良いスコアの意味 |
|-----|------|---------------|
| **Faithfulness<br>（忠実性）** | 回答が検索されたコンテキストに基づいているか | 回答がコンテキストから逸脱していない |
| **Answer Relevancy<br>（回答関連性）** | 回答が質問に適切に答えているか | 質問に対して的確な回答 |
| **Context Recall<br>（コンテキスト再現率）** | 必要な情報を検索できているか | 関連情報を漏れなく取得 |
| **Context Precision<br>（コンテキスト精度）** | 検索されたコンテキストの品質 | 無関係な情報が少ない |

**スコア範囲**: 0.0 〜 1.0（高いほど良い）

### 使い方

#### 1. テストデータセット生成

```bash
./run_generate_testset.sh
```

このコマンドは：
- Weaviate内のドキュメントを確認
- 各ドキュメントに対する質問と正解を生成
- `evaluation/testset.json` に保存

**生成例:**
```json
[
  {
    "source": "data/documents/file1.txt",
    "question": "ファイル1の内容は何ですか？",
    "ground_truth": "テストファイル1です。"
  }
]
```

#### 2. 評価実行

```bash
./run_evaluate.sh
```

**所要時間**: 数分（テストケース数とLLMの速度に依存）

#### 3. 結果確認

```
====== 評価結果 ======

                  RAGAS評価スコア
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ メトリクス         ┃ スコア ┃ 説明           ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 忠実性             │ 0.8500 │ ...            │
│ 回答関連性         │ 0.7800 │ ...            │
│ コンテキスト再現率 │ 0.9200 │ ...            │
│ コンテキスト精度   │ 0.8100 │ ...            │
└────────────────────┴────────┴────────────────┘

総合スコア: 0.8400
✓ 優秀なRAGシステムです！
```

#### 評価基準

- **0.8以上**: 優秀 ✓
- **0.6〜0.8**: 改善の余地あり ⚠
- **0.6未満**: 大幅な改善が必要 ✗

### カスタムテストセット

手動でテストケースを作成する場合：

```json
[
  {
    "source": "data/documents/your_file.pdf",
    "question": "あなたの質問",
    "ground_truth": "期待される正解"
  }
]
```

`evaluation/testset.json` に保存して `./run_evaluate.sh` を実行。

---

## データ管理

### 重複防止機能

**同じファイルを複数回インデックス化しても重複しません。**

```bash
# 1回目: 新規登録
./run_indexer.sh data/documents/file1.txt
# → 1件追加

# 2回目: 既存データを自動削除して更新
./run_indexer.sh data/documents/file1.txt
# → 既存1件削除 → 新規1件追加
```

### 複数ドキュメントの蓄積

異なるファイルは蓄積されます：

```bash
./run_indexer.sh data/documents/file1.txt  # → 1件
./run_indexer.sh data/documents/file2.txt  # → 2件（合計）
./run_indexer.sh data/documents/file3.pdf  # → 3件（合計）
```

### データの確認

```bash
# Weaviateのスキーマ確認
curl http://localhost:8080/v1/schema

# 登録されているオブジェクト数
curl http://localhost:8080/v1/objects?class=PDFDocuments | jq '.totalResults'
```

### データの削除

特定のドキュメントを削除する場合は、再インデックスせずに放置するか、Weaviateのコレクション全体をリセット：

```bash
# コレクション全体をリセット（全データ削除）
docker compose run --rm python-app python -c "
import weaviate
from config import Config
from urllib.parse import urlparse

parsed_url = urlparse(Config.WEAVIATE_URL)
client = weaviate.connect_to_local(host=parsed_url.hostname, port=parsed_url.port)

if client.collections.exists(Config.WEAVIATE_COLLECTION_NAME):
    client.collections.delete(Config.WEAVIATE_COLLECTION_NAME)
    print('コレクション削除完了')

client.close()
"
```

### ディレクトリ構造（データ関連）

```
.
├── data/
│   └── documents/          # ←ここにファイルを配置
│       ├── sample.pdf
│       ├── notes.txt
│       └── guide.md
│
├── evaluation/             # 評価関連
│   ├── testset.json        # テストデータセット
│   └── results.json        # 評価結果
│
└── volumes/                # 永続化データ（自動生成）
    ├── ollama/             # Ollamaモデルデータ（約5-10GB）
    └── weaviate/           # Weaviateベクトルデータ
```

---

## 高度な使い方

### 設定のカスタマイズ

`.env` ファイルまたは `docker-compose.yml` の environment セクションを編集：

```bash
# .env ファイル例

# Ollama設定
OLLAMA_API_URL=http://ollama:11434
OLLAMA_EMBEDDING_MODEL=kun432/cl-nagoya-ruri-large  # 変更可能
OLLAMA_LLM_MODEL=elyza-jp-8b

# Weaviate設定
WEAVIATE_URL=http://weaviate:8080
WEAVIATE_COLLECTION_NAME=PDFDocuments

# RAG設定
CHUNK_SIZE=500           # チャンクサイズ（300-1200推奨）
CHUNK_OVERLAP=50         # オーバーラップ（10-100推奨）
TOP_K_RESULTS=5          # 検索結果数（3-10推奨）
```

**変更後は再ビルドが必要**:
```bash
docker compose down
docker compose build python-app
docker compose up -d
```

### コンテナ内で直接実行

```bash
# Python-appコンテナ内でシェル起動
docker compose run --rm python-app /bin/bash

# コンテナ内で
python scripts/indexer.py data/documents/sample.pdf
python scripts/query.py
python scripts/generate_testset.py
python scripts/evaluate.py
```

### 別のEmbeddingモデルを試す

```bash
# Ollamaで別のEmbeddingモデルをダウンロード
docker compose exec ollama ollama pull nomic-embed-text

# .env を編集
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# 再起動
docker compose restart python-app

# 既存データを再インデックス化
./run_indexer.sh data/documents/your_file.pdf
```

---

## トラブルシューティング

### コンテナが起動しない

```bash
# ログを確認
docker compose logs ollama
docker compose logs weaviate

# ポートが使用中かチェック
lsof -i :11434  # Ollama
lsof -i :8080   # Weaviate

# コンテナを完全にリセット
docker compose down -v
docker compose up -d
```

### Weaviate接続エラー

**症状**: `Connection to Weaviate failed`

**原因**: Weaviateが起動していない、またはネットワークエラー

**解決方法**:
```bash
# Weaviateの状態確認
docker compose ps weaviate

# Weaviateのログ確認
docker compose logs weaviate

# Weaviateが起動するまで少し待つ（初回は30秒程度）
sleep 30

# 再度インデックス作成を試す
./run_indexer.sh data/documents/your_file.pdf
```

### Ollama接続エラー

**症状**: `Connection to Ollama failed`

**原因**: Ollamaが起動していない、モデル未登録

**解決方法**:
```bash
# Ollamaの状態確認
docker compose ps ollama

# モデルリスト確認
docker compose exec ollama ollama list

# 必要なモデルがない場合
docker compose exec ollama ollama pull kun432/cl-nagoya-ruri-large
docker compose exec ollama ollama create elyza-jp-8b -f /models/elyza/Modelfile
```

### Pythonアプリのエラー

**症状**: スクリプト実行時にエラー

**解決方法**:
```bash
# イメージを再ビルド
docker compose build python-app

# キャッシュを無視して再ビルド
docker compose build --no-cache python-app

# ログ確認
docker compose logs python-app
```

### インデックス作成が遅い

**原因**:
- 大きなPDFファイル
- GPUが使用されていない
- ネットワークが遅い

**解決方法**:
```bash
# GPU使用状況を確認
docker compose exec ollama nvidia-smi

# チャンクサイズを増やす（速度優先）
# .env で CHUNK_SIZE=1000 に変更
```

### 検索精度が低い

**症状**: 質問に対して的外れな回答が返る

**原因**:
- ドキュメント内容が不足
- Embeddingモデルが合っていない
- チャンクサイズが不適切

**解決方法**:
```bash
# 1. チャンクサイズを調整（小さくすると精度向上）
# .env で CHUNK_SIZE=300 に変更

# 2. 検索結果数を増やす
# .env で TOP_K_RESULTS=10 に変更

# 3. 別のEmbeddingモデルを試す
# multilingual-e5-large など
```

### GPUが認識されない

```bash
# nvidia-dockerの確認
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# docker-compose.ymlのGPU設定を確認
# deploy.resources.reservations.devices が正しいか確認
```

---

## FAQ（よくある質問）

### Q1. ローカル環境にPythonは必要ですか？

**A:** 不要です。すべてDockerコンテナ内で実行されます。

### Q2. インターネット接続は必要ですか？

**A:** 初回のモデルダウンロード時のみ必要です。その後はオフラインで動作します。

### Q3. 同じファイルを複数回インデックス化するとどうなりますか？

**A:** 既存データは自動的に削除され、新しいデータで置き換えられます。重複しません。

### Q4. 複数のPDFファイルから検索できますか？

**A:** はい。複数ファイルをインデックス化すれば、すべてのファイルから横断検索できます。

### Q5. GPUがなくても動きますか？

**A:** 動きます。ただし、推論速度は遅くなります。

### Q6. どのくらいのドキュメント数まで対応できますか？

**A:** Weaviateは数百万件のドキュメントに対応できます。ただし、ローカル環境のメモリとストレージに依存します。

### Q7. 英語のドキュメントも使えますか？

**A:** はい。kun432/cl-nagoya-ruri-largeは多言語対応で、ELYZA-JP-8Bも英語に対応しています。

### Q8. PDFの画像やグラフは検索できますか？

**A:** テキスト部分のみ検索可能です。画像やグラフ内の文字は検索できません。

### Q9. API経由で使用できますか？

**A:** 現在はCLI専用です。FastAPIなどでAPIサーバー化することは可能です。

### Q10. RAGAS評価でスコアが低い場合はどうすればいいですか？

**A:** 以下を試してください：
1. ドキュメントの内容を充実させる
2. チャンクサイズを調整（300-500文字推奨）
3. 別のEmbeddingモデルを試す
4. テストデータセットを改善する

### Q11. ファイルを削除したい場合は？

**A:** 現在、個別ファイルの削除機能はありません。コレクション全体をリセットするか、上書きインデックス化で対応してください。

### Q12. コンテナのログはどこで見れますか？

**A:**
```bash
# すべてのログ
docker compose logs -f

# 特定のコンテナ
docker compose logs -f ollama
docker compose logs -f weaviate
docker compose logs -f python-app
```

---

## システム要件

### 最小要件

- **OS**: Linux, macOS, Windows (with WSL2)
- **Docker**: 20.10以降
- **Docker Compose**: 2.0以降
- **メモリ**: 8GB以上推奨
- **ストレージ**: 20GB以上の空き容量

### 推奨要件

- **GPU**: NVIDIA GPU（CUDA対応）
- **メモリ**: 16GB以上
- **ストレージ**: SSD 50GB以上

### モデルサイズ

- **ELYZA-JP-8B**: 約4.6GB
- **kun432/cl-nagoya-ruri-large**: 約669MB
- **Weaviateデータ**: ドキュメント量に依存（1ドキュメント約100-200KB）

---

## 参考情報

### ディレクトリ構造（完全版）

```
.
├── docker-compose.yml       # Docker Compose設定
├── Dockerfile               # Pythonアプリ用Dockerfile
├── requirements.txt         # Python依存関係
├── .env                     # 環境変数
├── CLAUDE.md               # Claude Code用プロジェクトガイド
├── README_RAG.md           # このファイル
│
├── run_indexer.sh          # インデックス作成ヘルパー
├── run_query.sh            # 質問応答ヘルパー
├── run_generate_testset.sh # テストセット生成ヘルパー
├── run_evaluate.sh         # RAGAS評価ヘルパー
│
├── scripts/                # Pythonスクリプト
│   ├── config.py           # 設定管理
│   ├── pdf_loader.py       # PDFローダー
│   ├── text_loader.py      # テキストローダー
│   ├── indexer.py          # インデックス作成
│   ├── query.py            # RAG質問応答
│   ├── generate_testset.py # テストセット生成
│   └── evaluate.py         # RAGAS評価
│
├── data/                   # データディレクトリ
│   └── documents/          # ドキュメントファイル格納場所
│       ├── *.pdf           # PDFファイル
│       ├── *.txt           # テキストファイル
│       └── *.md            # Markdownファイル
│
├── evaluation/             # 評価関連（自動生成）
│   ├── testset.json        # テストデータセット
│   └── results.json        # 評価結果
│
├── volumes/                # 永続化データ（自動生成）
│   ├── ollama/             # Ollamaモデルデータ
│   └── weaviate/           # Weaviateベクトルデータ
│
└── models/                 # ローカルモデル
    └── elyza/              # ELYZAモデル
        ├── Llama-3-ELYZA-JP-8B-q4_k_m.gguf  # モデルファイル（4.6GB）
        └── Modelfile       # Ollamaモデル定義
```

### コマンド一覧

| コマンド | 説明 |
|---------|------|
| `docker compose up -d` | コンテナ起動 |
| `docker compose down` | コンテナ停止 |
| `docker compose ps` | コンテナ状態確認 |
| `docker compose logs -f` | ログ確認 |
| `./run_indexer.sh <file>` | ドキュメントをインデックス化 |
| `./run_query.sh` | 質問応答システム起動 |
| `./run_generate_testset.sh` | テストセット生成 |
| `./run_evaluate.sh` | RAGAS評価実行 |
| `docker compose exec ollama ollama list` | Ollamaモデル一覧 |
| `curl http://localhost:8080/v1/schema` | Weaviateスキーマ確認 |

### 関連リンク

- **Ollama**: https://ollama.com/
- **Weaviate**: https://weaviate.io/
- **RAGAS**: https://github.com/explodinggradients/ragas
- **ELYZA-JP-8B**: https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B
- **kun432/cl-nagoya-ruri-large**: https://ollama.com/library/kun432/cl-nagoya-ruri-large

### ライセンス

各コンポーネントのライセンスに従います：
- **Ollama**: MIT License
- **Weaviate**: BSD 3-Clause License
- **ELYZA-JP-8B**: Apache 2.0 License
- **kun432/cl-nagoya-ruri-large**: Apache 2.0 License
- **RAGAS**: Apache 2.0 License

---

## サポート

問題が発生した場合：

1. まず[トラブルシューティング](#トラブルシューティング)を確認
2. [FAQ](#faqよくある質問)を確認
3. ログを確認: `docker compose logs -f`
4. GitHubのIssueで質問

**ログの取得方法**:
```bash
# すべてのログをファイルに保存
docker compose logs > logs.txt 2>&1
```

---

**最終更新**: 2026-01-25
