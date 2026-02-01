#!/bin/bash
# RAG質問応答システムを起動するスクリプト

echo "PDF RAG 質問システムを起動中..."
echo "---"

# python-appコンテナを一時起動して対話型スクリプト実行
docker compose run --rm python-app python scripts/query.py
