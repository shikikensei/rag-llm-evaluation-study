# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このリポジトリは、Ollama + WeaviateベースのRAG（Retrieval-Augmented Generation）システムです。ドキュメントファイル（PDF、テキスト、Markdown）をベクトル化してセマンティック検索を行い、LLMで質問に回答します。

すべての処理はDockerコンテナ内で完結し、ローカル環境を汚さずに利用できます。

## アーキテクチャ

### Docker Compose構成

- **ollama サービス**: Ollama APIサーバー（ポート11434で公開）
  - GPU（NVIDIA）を使用してモデルを実行
  - モデルデータは `./volumes/ollama` にバインドマウントされ永続化される
  - LLMモデル: ELYZA-JP-8B（日本語特化）
  - Embeddingモデル: mxbai-embed-large（多言語対応）

- **weaviate サービス**: ベクトルデータベース（ポート8080で公開）
  - ドキュメントのベクトル検索を提供
  - データは `./volumes/weaviate` に永続化される
  - Ollamaと統合してEmbedding生成とLLM推論を実行

- **python-app サービス**: RAGスクリプト実行環境
  - PDF/テキストファイルのインデックス作成
  - 対話型質問応答システム
  - 必要時のみ起動（profilesで管理）

### ディレクトリ構造

- `models/elyza/`: ELYZAモデル関連ファイル
  - `Llama-3-ELYZA-JP-8B-q4_k_m.gguf`: 量子化されたLLMモデルファイル（約4.6GB）
  - `Modelfile`: Ollamaモデル定義ファイル
- `volumes/ollama/`: Ollamaの永続化データ（モデルキャッシュ等）
- `volumes/weaviate/`: Weaviateの永続化データ（ベクトルデータ）
- `data/documents/`: ドキュメントファイル格納場所（PDF、TXT、MD）
- `scripts/`: RAGシステムのPythonスクリプト
  - `config.py`: 設定管理
  - `pdf_loader.py`: PDFローダー
  - `text_loader.py`: テキストローダー
  - `indexer.py`: インデックス作成（重複防止機能付き）
  - `query.py`: RAG質問応答
  - `generate_testset.py`: RAGAS評価用テストデータセット生成
  - `evaluate.py`: RAGAS評価実行
- `evaluation/`: RAGAS評価関連ファイル
  - `testset.json`: 評価用テストデータセット
  - `results.json`: 評価結果

### 過去の構成

`docker-compose.yml.bak`には、以前のOpen WebUI + Ollama構成が保存されています。

## 基本コマンド

### RAGシステムの使い方

#### 1. 環境の起動

```bash
docker-compose up -d
```

#### 2. ドキュメントのインデックス作成

```bash
# PDFファイル
./run_indexer.sh data/documents/sample.pdf

# テキストファイル
./run_indexer.sh data/documents/readme.txt

# Markdownファイル
./run_indexer.sh data/documents/guide.md
```

対応形式: `.pdf`, `.txt`, `.md`, `.markdown`

#### 3. 質問応答

```bash
./run_query.sh
```

対話型UIで質問を入力すると、インデックス化したドキュメントの内容に基づいて回答が生成されます。
終了するには `exit` または `終了` と入力してください。

#### 4. 環境の停止

```bash
docker-compose down
```

### その他のコマンド

#### Ollama APIの確認

```bash
curl http://localhost:11434/api/version
```

#### モデルの確認

```bash
docker compose exec ollama ollama list
```

#### Weaviateの確認

```bash
curl http://localhost:8080/v1/schema
```

#### ログの確認

```bash
# すべてのコンテナ
docker-compose logs -f

# 特定のコンテナ
docker-compose logs -f ollama
docker-compose logs -f weaviate
```

## RAG評価システム（RAGAS）

このシステムにはRAGASを使用した精度評価機能が組み込まれています。

### テストデータセット生成

```bash
./run_generate_testset.sh
```

インデックス化されたドキュメントから自動的にテストケースを生成します。
生成されたテストデータは `evaluation/testset.json` に保存されます。

### 評価の実行

```bash
./run_evaluate.sh
```

以下の4つの指標でRAGシステムの精度を評価します:

- **Faithfulness（忠実性）**: 回答が検索コンテキストに基づいているか
- **Answer Relevancy（回答関連性）**: 回答が質問に適切に答えているか
- **Context Recall（コンテキスト再現率）**: 必要な情報を検索できているか
- **Context Precision（コンテキスト精度）**: 検索されたコンテキストの品質

評価結果は `evaluation/results.json` に保存され、総合スコアが表示されます:
- 0.8以上: 優秀なRAGシステム
- 0.6-0.8: 改善の余地あり
- 0.6未満: 大幅な改善が必要

### 注意事項

- 評価を実行する前に、必ずテストデータセットを生成してください
- 評価にはELYZA-JP-8Bモデルを使用します（GPU推論）
- 評価には数分程度かかります

## データ管理

### 重複防止機能

このシステムは**append mode（追記モード）**を採用しており、自動的に重複を防止します:

```bash
# 同じファイルを複数回インデックス化しても大丈夫
./run_indexer.sh data/documents/file1.txt  # 初回登録
./run_indexer.sh data/documents/file1.txt  # 重複なし（古いデータを削除して再登録）
```

**動作の詳細**:
1. ファイルをインデックス化する際、同じソースファイルの既存データを自動削除
2. 新しいデータを追加
3. 異なるファイルのデータは保持される（accumulate）

### データの確認

```bash
# Weaviateに保存されているドキュメント数を確認
curl http://localhost:8080/v1/objects | jq '.objects | length'

# コレクション情報を確認
curl http://localhost:8080/v1/schema | jq
```

### データの削除

```bash
# すべてのインデックスデータを削除
docker compose exec weaviate curl -X DELETE http://localhost:8080/v1/schema/PDFDocuments

# または、Weaviateのボリュームを削除して再起動
docker compose down
rm -rf volumes/weaviate
docker compose up -d
```

## GPU要件

このプロジェクトはNVIDIA GPUの使用を前提としています。Docker環境でNVIDIA Container Toolkitが適切に設定されている必要があります。

## 詳細情報

より詳しい使い方、トラブルシューティング、FAQ等については `README_RAG.md` を参照してください。以下の情報が含まれています:

- システムアーキテクチャの詳細図
- クイックスタートガイド
- RAGAS評価の詳細ガイド
- トラブルシューティング（7つの一般的な問題と解決策）
- FAQ（12の質問と回答）
- 高度な使い方（チャンクサイズ調整、カスタム設定等）
- システム要件とスケール制限
- 完全なディレクトリ構造とコマンドリファレンス
