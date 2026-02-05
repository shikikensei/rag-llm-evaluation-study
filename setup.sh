#!/bin/bash
# RAGシステム初回セットアップスクリプト

set -e

echo "========================================="
echo "  RAGシステム セットアップ開始"
echo "========================================="
echo ""

# Step 1: Docker環境起動
echo "[1/3] Dockerコンテナを起動中..."
docker compose up -d ollama weaviate

# Ollamaが完全に起動するまで待機
echo "Ollamaの起動を待機中..."
sleep 10

# Step 2: 必要なモデルをダウンロード
echo ""
echo "[2/3] 必要なモデルをダウンロード中..."
echo "  - Embeddingモデル: mxbai-embed-large (669MB)"
echo "  - LLMモデル: dsasai/llama3-elyza-jp-8b (4.9GB)"
echo ""
echo "※ 初回は数分〜10分程度かかります"
echo ""

# Embeddingモデル
if docker compose exec ollama ollama list | grep -q "mxbai-embed-large"; then
    echo "✓ mxbai-embed-large は既にインストール済み"
else
    echo "→ mxbai-embed-large をダウンロード中..."
    docker compose exec ollama ollama pull mxbai-embed-large
fi

# LLMモデル
if docker compose exec ollama ollama list | grep -q "dsasai/llama3-elyza-jp-8b"; then
    echo "✓ dsasai/llama3-elyza-jp-8b は既にインストール済み"
else
    echo "→ dsasai/llama3-elyza-jp-8b をダウンロード中..."
    docker compose exec ollama ollama pull dsasai/llama3-elyza-jp-8b
fi

# Step 3: 完了
echo ""
echo "[3/3] セットアップ完了確認..."
echo ""
docker compose exec ollama ollama list

echo ""
echo "========================================="
echo "  ✓ セットアップ完了！"
echo "========================================="
echo ""
echo "次のステップ:"
echo "  1. ドキュメントを配置: cp your_file.pdf data/documents/"
echo "  2. インデックス作成: ./run_indexer.sh data/documents/your_file.pdf"
echo "  3. 質問応答開始: ./run_query.sh"
echo ""
