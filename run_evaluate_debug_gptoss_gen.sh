#!/bin/bash
# RAGASデバッグ評価を実行するスクリプト（gpt-oss:20b生成 + ELYZA評価版）

echo "RAGASデバッグ評価を実行中（gpt-oss:20b生成 + ELYZA評価版）..."
echo "回答生成: gpt-oss:20b (大型汎用モデル)"
echo "RAGAS評価: ELYZA-JP-8B (日本語特化、高速)"
echo "---"

# テストセットの存在確認
if [ ! -f "evaluation/testset.json" ]; then
    echo "エラー: evaluation/testset.json が見つかりません"
    echo "先に ./run_generate_testset.sh を実行してテストセットを生成してください"
    exit 1
fi

# python-appコンテナを一時起動してデバッグ評価実行
docker compose run --rm python-app python scripts/evaluate_debug_gptoss_gen.py
