#!/bin/bash
# RAG評価用のテストデータセットを生成するスクリプト

echo "RAG評価用テストデータセット生成中..."
echo "---"

# python-appコンテナを一時起動してテストセット生成
docker compose run --rm python-app python scripts/generate_testset.py
