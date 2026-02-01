FROM python:3.12-slim

WORKDIR /app

# 必要なシステムパッケージをインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# スクリプトディレクトリをコピー
COPY scripts /app/scripts

# 作業ディレクトリを設定
WORKDIR /app

# デフォルトコマンド（対話型シェル）
CMD ["/bin/bash"]
