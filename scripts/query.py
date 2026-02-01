import weaviate
from rich.console import Console
from rich.panel import Panel
from config import Config

console = Console()


class RAGQuery:
    """RAG検索・質問応答を行うクラス"""

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

    def search(self, query: str, top_k: int = None):
        """
        セマンティック検索を実行

        Args:
            query: 検索クエリ
            top_k: 取得する結果数（Noneの場合はConfig.TOP_K_RESULTSを使用）

        Returns:
            検索結果のリスト
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS

        collection = self.client.collections.get(self.collection_name)

        response = collection.query.near_text(
            query=query,
            limit=top_k,
            return_metadata=['distance']
        )

        results = []
        for obj in response.objects:
            results.append({
                "content": obj.properties["content"],
                "source": obj.properties["source"],
                "page": obj.properties["page"],
                "chunk": obj.properties["chunk"],
                "distance": obj.metadata.distance
            })

        return results

    def generate_answer(self, query: str, top_k: int = None):
        """
        RAG生成（検索 + LLM生成）を実行

        Args:
            query: 質問
            top_k: 取得する検索結果数（Noneの場合はConfig.TOP_K_RESULTSを使用）

        Returns:
            生成結果のリスト（検索結果 + 生成テキスト）
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS

        collection = self.client.collections.get(self.collection_name)

        # Weaviateの生成機能を使用
        response = collection.generate.near_text(
            query=query,
            limit=top_k,
            single_prompt="""あなたは親切なアシスタントです。以下のコンテキストを使用して質問に答えてください。

コンテキスト: {content}

質問: """ + query + """

回答:"""
        )

        # 検索結果
        search_results = []
        for obj in response.objects:
            search_results.append({
                "content": obj.properties["content"],
                "source": obj.properties["source"],
                "page": obj.properties["page"],
                "generated": obj.generated
            })

        return search_results

    def close(self):
        """Weaviate接続をクローズ"""
        self.client.close()


def main():
    """対話型RAG質問インターフェース"""
    console.print(Panel.fit(
        "[bold cyan]PDF RAG 質問システム[/bold cyan]\n"
        "質問を入力してください（終了: 'exit'）",
        border_style="cyan"
    ))

    rag = RAGQuery()

    try:
        while True:
            query = console.input("\n[bold yellow]質問:[/bold yellow] ")

            if query.lower() in ['exit', 'quit', '終了']:
                break

            if not query.strip():
                continue

            console.print("\n[dim]検索中...[/dim]")

            # RAG生成
            results = rag.generate_answer(query)

            if not results:
                console.print("[red]関連するドキュメントが見つかりませんでした。[/red]")
                continue

            # 最初の結果の生成テキストを表示
            console.print(Panel(
                results[0]["generated"],
                title="[bold green]回答[/bold green]",
                border_style="green"
            ))

            # ソース情報（ページ番号がある場合のみ表示）
            source_info = f"ソース: {results[0]['source']}"
            if results[0].get('page', 0) > 0:
                source_info += f" (ページ {results[0]['page']})"
            console.print(f"\n[dim]{source_info}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]終了します[/yellow]")
    finally:
        rag.close()


if __name__ == "__main__":
    main()
