#!/bin/bash
# RAGASを使用してRAGシステムの精度評価を実行するスクリプト

echo "RAGASによるRAGシステム評価を実行中..."
echo "---"

# テストセットの存在確認
if [ ! -f "evaluation/testset.json" ]; then
    echo "エラー: evaluation/testset.json が見つかりません"
    echo "先に ./run_generate_testset.sh を実行してテストセットを生成してください"
    exit 1
fi

# python-appコンテナを一時起動して評価実行
docker compose run --rm python-app python scripts/evaluate.py
