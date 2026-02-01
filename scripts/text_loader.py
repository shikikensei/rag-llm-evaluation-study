from typing import List
import re
import os


class TextLoader:
    """テキストファイルを読み込み、テキストをチャンク化するクラス"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: チャンクの最大文字数
            chunk_overlap: チャンク間のオーバーラップ文字数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, file_path: str) -> List[dict]:
        """
        テキストファイルを読み込み、チャンク化されたドキュメントを返す

        Args:
            file_path: テキストファイルのパス

        Returns:
            ドキュメントのリスト（各ドキュメントはcontent, metadataを含む）
        """
        # ファイル名を取得
        filename = os.path.basename(file_path)

        # テキストファイルを読み込み（UTF-8でエンコーディング）
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # UTF-8で読めない場合は、他のエンコーディングを試す
            with open(file_path, 'r', encoding='shift_jis') as f:
                text = f.read()

        # テキストのクリーニング
        text = self._clean_text(text)

        # チャンク化
        chunks = self._chunk_text(text)

        documents = []
        for chunk_idx, chunk in enumerate(chunks):
            documents.append({
                "content": chunk,
                "metadata": {
                    "source": file_path,
                    "filename": filename,
                    "chunk": chunk_idx
                }
            })

        return documents

    def _clean_text(self, text: str) -> str:
        """
        テキストのクリーニング

        Args:
            text: クリーニング対象のテキスト

        Returns:
            クリーニング済みテキスト
        """
        # 複数の空白を1つに（改行は保持）
        text = re.sub(r'[ \t]+', ' ', text)
        # 3つ以上の連続する改行を2つに
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 前後の空白を削除
        text = text.strip()
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """
        テキストをチャンクに分割

        Args:
            text: 分割対象のテキスト

        Returns:
            チャンクのリスト
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # チャンクを取得
            chunk = text[start:end]

            # 空でない場合のみ追加
            if chunk.strip():
                chunks.append(chunk)

            # 次のチャンクの開始位置（オーバーラップ考慮）
            start = end - self.chunk_overlap

        return chunks
