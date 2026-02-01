from typing import List
from pypdf import PdfReader
import re


class PDFLoader:
    """PDFファイルを読み込み、テキストを抽出してチャンク化するクラス"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: チャンクの最大文字数
            chunk_overlap: チャンク間のオーバーラップ文字数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, pdf_path: str) -> List[dict]:
        """
        PDFを読み込み、チャンク化されたドキュメントを返す

        Args:
            pdf_path: PDFファイルのパス

        Returns:
            ドキュメントのリスト（各ドキュメントはcontent, metadataを含む）
        """
        reader = PdfReader(pdf_path)

        documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            text = self._clean_text(text)

            # チャンク化
            chunks = self._chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": pdf_path,
                        "page": page_num + 1,
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
        # 複数の空白を1つに
        text = re.sub(r'\s+', ' ', text)
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
            chunks.append(text[start:end])
            start = end - self.chunk_overlap

        return chunks
