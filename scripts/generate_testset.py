"""
RAG評価用のテストデータセットを生成するスクリプト
"""
import weaviate
from urllib.parse import urlparse
from config import Config
from rich.console import Console
import json
from typing import List, Dict

console = Console()


def generate_testset_from_documents() -> List[Dict]:
    """
    Weaviateに登録されているドキュメントからテストデータセットを生成

    Returns:
        テストケースのリスト
    """
    # Weaviate接続
    parsed_url = urlparse(Config.WEAVIATE_URL)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 8080

    client = weaviate.connect_to_local(host=host, port=port)
    collection = client.collections.get(Config.WEAVIATE_COLLECTION_NAME)

    # 全ドキュメントを取得
    result = collection.query.fetch_objects(limit=100)

    console.print(f"[blue]取得したドキュメント数: {len(result.objects)}件[/blue]")

    # ドキュメントをソースごとにグループ化
    docs_by_source = {}
    for obj in result.objects:
        source = obj.properties.get('source', 'unknown')
        content = obj.properties.get('content', '')

        if source not in docs_by_source:
            docs_by_source[source] = []
        docs_by_source[source].append(content)

    console.print(f"[blue]ソース数: {len(docs_by_source)}件[/blue]\n")

    # テストケースを生成
    testset = []

    # 各ドキュメントに対して質問を生成
    test_cases = [
        {
            "source": "data/documents/file1.txt",
            "question": "ファイル1の内容は何ですか？",
            "ground_truth": "テストファイル1です（更新版）。追記型のテストを行い、内容を更新しました。"
        },
        {
            "source": "data/documents/file2.txt",
            "question": "ファイル2について教えてください",
            "ground_truth": "テストファイル2です。異なるファイルを追加します。"
        },
        {
            "source": "data/documents/test.txt",
            "question": "志岐の誕生日はいつですか？",
            "ground_truth": "3月15日です。"
        },
        {
            "source": "data/documents/sample.txt",
            "question": "RAGシステムの主な特徴は何ですか？",
            "ground_truth": "セマンティック検索（意味に基づく検索）、文脈理解（ドキュメント全体の文脈を考慮）、柔軟な対応（PDF、テキスト、Markdownなど複数フォーマット対応）の3つです。"
        },
        {
            "source": "data/documents/sample.txt",
            "question": "このRAGシステムの技術スタックを教えてください",
            "ground_truth": "Ollama（LLM推論エンジン）、Weaviate（ベクトルデータベース）、ELYZA-JP-8B（日本語特化LLMモデル）、mxbai-embed-large（多言語Embeddingモデル）を使用しています。"
        }
    ]

    # 実際に存在するソースのテストケースのみ追加
    for test_case in test_cases:
        if test_case["source"] in docs_by_source:
            testset.append(test_case)

    client.close()

    console.print(f"[green]✓ テストケース生成完了: {len(testset)}件[/green]")

    return testset


def save_testset(testset: List[Dict], output_path: str = "evaluation/testset.json"):
    """
    テストデータセットをJSONファイルに保存

    Args:
        testset: テストケースのリスト
        output_path: 保存先パス
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(testset, f, ensure_ascii=False, indent=2)

    console.print(f"[green]✓ テストセットを保存: {output_path}[/green]")


def main():
    """メイン処理"""
    console.print("[bold cyan]RAG評価用テストデータセット生成[/bold cyan]\n")

    # テストセット生成
    testset = generate_testset_from_documents()

    # 保存
    save_testset(testset)

    # 内容表示
    console.print("\n[bold]生成されたテストケース:[/bold]")
    for i, test_case in enumerate(testset, 1):
        console.print(f"\n{i}. [yellow]{test_case['question']}[/yellow]")
        console.print(f"   正解: {test_case['ground_truth'][:50]}...")

    console.print(f"\n[bold green]完了！[/bold green]")


if __name__ == "__main__":
    main()
