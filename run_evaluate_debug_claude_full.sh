#!/bin/bash
# RAGASデバッグ評価（Full Claude版）を実行するスクリプト

echo "RAGASデバッグ評価を実行中（Full Claude版）..."
echo "回答生成: Claude Sonnet 4"
echo "RAGAS評価: Claude Sonnet 4"
echo "---"

# ANTHROPIC_API_KEYの確認
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "エラー: ANTHROPIC_API_KEY環境変数が設定されていません"
    echo "以下のコマンドでAPIキーを設定してください:"
    echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
    exit 1
fi

# テストセットの存在確認
if [ ! -f "evaluation/testset.json" ]; then
    echo "エラー: evaluation/testset.json が見つかりません"
    echo "先に ./run_generate_testset.sh を実行してテストセットを生成してください"
    exit 1
fi

# python-appコンテナを一時起動してFull Claude版デバッグ評価実行
docker compose run --rm \
    -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    python-app python scripts/evaluate_debug_claude_full.py
