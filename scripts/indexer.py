import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from typing import List
import os
from rich.console import Console
from rich.progress import track
from config import Config
from pdf_loader import PDFLoader
from text_loader import TextLoader

console = Console()


class WeaviateIndexer:
    """Weaviateにドキュメントをインデックス化するクラス"""

    def __init__(self):
        """Weaviateクライアントを初期化"""
        # WEAVIATE_URLから接続（http://weaviate:8080 または http://localhost:8080）
        from urllib.parse import urlparse
        parsed_url = urlparse(Config.WEAVIATE_URL)
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 8080

        self.client = weaviate.connect_to_local(
            host=host,
            port=port
        )
        self.collection_name = Config.WEAVIATE_COLLECTION_NAME

    def create_collection_if_not_exists(self):
        """コレクションが存在しない場合のみ作成"""

        # 既存コレクションがあればスキップ
        if self.client.collections.exists(self.collection_name):
            console.print(f"[dim]既存コレクション '{self.collection_name}' を使用[/dim]")
            return

        # コレクション作成
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                api_endpoint=Config.OLLAMA_API_URL,
                model=Config.OLLAMA_EMBEDDING_MODEL
            ),
            generative_config=Configure.Generative.ollama(
                api_endpoint=Config.OLLAMA_API_URL,
                model=Config.OLLAMA_LLM_MODEL
            ),
            properties=[
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="ドキュメントの内容"
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                    description="ソースファイルパス"
                ),
                Property(
                    name="filename",
                    data_type=DataType.TEXT,
                    description="ファイル名"
                ),
                Property(
                    name="page",
                    data_type=DataType.INT,
                    description="ページ番号（PDFのみ、テキストファイルは0）"
                ),
                Property(
                    name="chunk",
                    data_type=DataType.INT,
                    description="チャンク番号"
                )
            ]
        )

        console.print(f"[green]✓ コレクション '{self.collection_name}' を作成[/green]")

    def remove_documents_by_source(self, source_path: str):
        """
        指定されたソースパスと一致するドキュメントを削除

        Args:
            source_path: 削除するドキュメントのソースパス
        """
        collection = self.client.collections.get(self.collection_name)

        # ソースパスで検索して削除
        try:
            result = collection.data.delete_many(
                where=Filter.by_property("source").equal(source_path)
            )

            if result.successful > 0:
                console.print(f"[yellow]既存データを削除: {result.successful}件 (ソース: {source_path})[/yellow]")
            else:
                console.print(f"[dim]既存データなし (ソース: {source_path})[/dim]")
        except Exception as e:
            console.print(f"[yellow]警告: 既存データの削除に失敗: {e}[/yellow]")

    def index_documents(self, documents: List[dict]):
        """
        ドキュメントをインデックス化

        Args:
            documents: インデックス化するドキュメントのリスト
        """
        collection = self.client.collections.get(self.collection_name)

        console.print(f"[blue]インデックス作成中... ({len(documents)}件)[/blue]")

        with collection.batch.dynamic() as batch:
            for doc in track(documents, description="Processing"):
                # ファイル名を取得（存在しない場合はソースから抽出）
                filename = doc["metadata"].get("filename", os.path.basename(doc["metadata"]["source"]))

                # ページ番号（PDFのみ、テキストファイルは0）
                page = doc["metadata"].get("page", 0)

                batch.add_object(
                    properties={
                        "content": doc["content"],
                        "source": doc["metadata"]["source"],
                        "filename": filename,
                        "page": page,
                        "chunk": doc["metadata"]["chunk"]
                    }
                )

        console.print(f"[green]✓ {len(documents)}件のドキュメントをインデックス化完了[/green]")

    def close(self):
        """Weaviate接続をクローズ"""
        self.client.close()


def main(file_path: str):
    """
    メイン処理：ファイルをロードしてWeaviateにインデックス化

    Args:
        file_path: ドキュメントファイルのパス（PDF, TXT, MDなど）
    """
    # ファイル拡張子を取得
    _, ext = os.path.splitext(file_path.lower())

    # 対応している拡張子かチェック
    supported_extensions = {'.pdf', '.txt', '.md', '.markdown', '.text'}
    if ext not in supported_extensions:
        console.print(f"[red]エラー: 未対応のファイル形式です: {ext}[/red]")
        console.print(f"[yellow]対応形式: {', '.join(supported_extensions)}[/yellow]")
        return

    console.print(f"[bold cyan]ドキュメント RAG インデックス作成[/bold cyan]")
    console.print(f"ファイル: {file_path}\n")

    # ファイル形式に応じてローダーを選択
    if ext == '.pdf':
        loader = PDFLoader(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        file_type = "PDF"
    else:
        # .txt, .md, .markdown, .text
        loader = TextLoader(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        file_type = "テキスト"

    # ドキュメントロード
    documents = loader.load(file_path)
    console.print(f"[green]✓ {file_type}ロード完了: {len(documents)}チャンク[/green]\n")

    # Weaviateインデックス化
    indexer = WeaviateIndexer()

    # コレクションを作成（存在しない場合のみ）
    indexer.create_collection_if_not_exists()

    # 同じソースの既存ドキュメントを削除（重複防止）
    indexer.remove_documents_by_source(file_path)

    # 新しいドキュメントを追加
    indexer.index_documents(documents)
    indexer.close()

    console.print(f"\n[bold green]インデックス作成完了！[/bold green]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        console.print("[red]使い方: python indexer.py <ファイルパス>[/red]")
        console.print("[yellow]対応形式: PDF (.pdf), テキスト (.txt, .md, .markdown)[/yellow]")
        sys.exit(1)

    main(sys.argv[1])
