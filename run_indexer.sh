#!/bin/bash
# PDFファイルのインデックスを作成するスクリプト

if [ -z "$1" ]; then
    echo "使い方: ./run_indexer.sh <ドキュメントファイルパス>"
    echo "対応形式: PDF (.pdf), テキスト (.txt, .md, .markdown)"
    echo "例: ./run_indexer.sh data/documents/sample.pdf"
    echo "例: ./run_indexer.sh data/documents/readme.md"
    exit 1
fi

PDF_PATH=$1

# PDFファイルの存在確認
if [ ! -f "$PDF_PATH" ]; then
    echo "エラー: ファイル '$PDF_PATH' が見つかりません"
    exit 1
fi

echo "ドキュメントインデックス作成中: $PDF_PATH"
echo "---"

# python-appコンテナを一時起動してスクリプト実行
docker compose run --rm python-app python scripts/indexer.py "$PDF_PATH"
