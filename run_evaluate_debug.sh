#!/bin/bash
# RAGASデバッグ評価を実行するスクリプト
# 検索モードとalpha値を引数で指定可能
#
# 使用例:
#   ./run_evaluate_debug.sh                         # デフォルト設定
#   ./run_evaluate_debug.sh --mode vector           # ベクトル検索
#   ./run_evaluate_debug.sh --mode hybrid           # ハイブリッド検索
#   ./run_evaluate_debug.sh --mode hybrid --alpha 0.7  # カスタムalpha

echo "RAGASデバッグ評価を実行中..."
echo "---"

# テストセットの存在確認
if [ ! -f "evaluation/testset.json" ]; then
    echo "エラー: evaluation/testset.json が見つかりません"
    echo "先に ./run_generate_testset.sh を実行してテストセットを生成してください"
    exit 1
fi

# python-appコンテナを一時起動してデバッグ評価実行
# 引数をそのまま渡す
docker compose run --rm \
    -e OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-dsasai/llama3-elyza-jp-8b}" \
    python-app python scripts/evaluate_debug.py "$@"
