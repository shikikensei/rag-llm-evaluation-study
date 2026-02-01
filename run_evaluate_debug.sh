#!/bin/bash
# RAGASデバッグ評価を実行するスクリプト

echo "RAGASデバッグ評価を実行中..."
echo "各テストケースの詳細情報が表示されます"
echo "---"

# テストセットの存在確認
if [ ! -f "evaluation/testset.json" ]; then
    echo "エラー: evaluation/testset.json が見つかりません"
    echo "先に ./run_generate_testset.sh を実行してテストセットを生成してください"
    exit 1
fi

# python-appコンテナを一時起動してデバッグ評価実行
docker compose run --rm \
    -e OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-elyza-jp-8b}" \
    python-app python scripts/evaluate_debug.py
