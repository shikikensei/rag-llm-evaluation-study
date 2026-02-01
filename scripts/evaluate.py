"""
RAGASを使用してRAGシステムの精度を評価するスクリプト
"""
import json
import weaviate
from urllib.parse import urlparse
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import track
from config import Config
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas import evaluate
# RAGAS 0.4でevaluate()関数を使う場合はレガシーメトリクスを使用
from ragas.metrics import (
    faithfulness,
    # answer_relevancy,  # RAGAS 0.4との互換性問題のため一時的に無効化
    context_recall,
    context_precision,
)
from datasets import Dataset
import pandas as pd

console = Console()


class LegacyEmbeddingsWrapper:
    """RAGAS 0.4のOpenAIEmbeddingsをレガシーメトリクス用にラップ"""

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def embed_query(self, text: str):
        """レガシーメトリクスが期待するembed_queryメソッド（同期版）"""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 既にイベントループが実行中の場合は新しいループで実行
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(self._embeddings.embed_text(text))

    def embed_documents(self, texts: List[str]):
        """レガシーメトリクスが期待するembed_documentsメソッド（同期版）"""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(self._embeddings.embed_texts(texts))


class RAGEvaluator:
    """RAGシステムの評価を行うクラス"""

    def __init__(self):
        """初期化"""
        # Weaviate接続
        parsed_url = urlparse(Config.WEAVIATE_URL)
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 8080

        self.weaviate_client = weaviate.connect_to_local(host=host, port=port)
        self.collection_name = Config.WEAVIATE_COLLECTION_NAME

        # OllamaをOpenAI互換APIとして使用（RAGAS 0.4対応）
        # Docker環境内ではollamaホスト名でアクセス
        ollama_base_url = Config.OLLAMA_API_URL.replace("http://ollama:", "http://ollama:")
        if not ollama_base_url.endswith("/v1"):
            ollama_base_url = ollama_base_url.rstrip("/") + "/v1"

        # AsyncOpenAIクライアント作成（RAGAS用）
        openai_client = AsyncOpenAI(
            api_key="ollama",  # Ollamaは認証不要
            base_url=ollama_base_url
        )

        # 通常のテキスト生成用のOpenAIクライアント（同期版）
        self.openai_sync_client = OpenAI(
            api_key="ollama",
            base_url=ollama_base_url
        )

        # RAGAS 0.4のllm_factoryでLLM作成（メトリクス評価用）
        self.llm = llm_factory(
            model=Config.OLLAMA_LLM_MODEL,
            provider="openai",
            client=openai_client
        )

        # RAGAS 0.4のOpenAIEmbeddingsを直接使用
        embeddings = OpenAIEmbeddings(
            model=Config.OLLAMA_EMBEDDING_MODEL,
            client=openai_client
        )

        # レガシーメトリクスとの互換性のためラップ
        self.embeddings = LegacyEmbeddingsWrapper(embeddings)

    def retrieve_contexts(self, question: str, top_k: int = 3) -> List[str]:
        """
        質問に対してコンテキストを検索

        Args:
            question: 質問
            top_k: 取得する結果数

        Returns:
            コンテキストのリスト
        """
        collection = self.weaviate_client.collections.get(self.collection_name)

        response = collection.query.near_text(
            query=question,
            limit=top_k
        )

        contexts = []
        for obj in response.objects:
            content = obj.properties.get("content", "")
            contexts.append(content)

        return contexts

    def generate_answer(self, question: str, contexts: List[str]) -> str:
        """
        コンテキストと質問から回答を生成

        Args:
            question: 質問
            contexts: コンテキストのリスト

        Returns:
            生成された回答
        """
        context_text = "\n\n".join(contexts)

        prompt = f"""以下のコンテキストを使用して質問に答えてください。

コンテキスト:
{context_text}

質問: {question}

回答:"""

        # OpenAIクライアント（Ollama互換）を直接使用して回答生成
        response = self.openai_sync_client.chat.completions.create(
            model=Config.OLLAMA_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def run_evaluation(self, testset: List[Dict]) -> Dict:
        """
        テストセットを使用して評価を実行

        Args:
            testset: テストケースのリスト

        Returns:
            評価結果
        """
        console.print("[blue]RAG評価を実行中...[/blue]\n")

        # 各テストケースに対してRAGを実行
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for test_case in track(testset, description="Processing"):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            # コンテキスト検索
            retrieved_contexts = self.retrieve_contexts(question)

            # 回答生成
            answer = self.generate_answer(question, retrieved_contexts)

            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(ground_truth)

        # データセット作成（RAGAS 0.4ハイブリッド形式）
        # レガシーメトリクスはquestion/answer/contexts、一部メトリクスはreferenceを要求
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
            "reference": ground_truths  # context_recallとcontext_precisionはreferenceを要求
        }

        dataset = Dataset.from_dict(data)

        console.print("\n[blue]RAGAS評価指標を計算中...[/blue]")

        # RAGAS評価実行（レガシーメトリクスを使用）
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                # answer_relevancy,  # 互換性問題のため除外
                context_recall,
                context_precision,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )

        return result

    def close(self):
        """リソースをクローズ"""
        self.weaviate_client.close()


def load_testset(testset_path: str = "evaluation/testset.json") -> List[Dict]:
    """
    テストセットをロード

    Args:
        testset_path: テストセットのパス

    Returns:
        テストケースのリスト
    """
    with open(testset_path, 'r', encoding='utf-8') as f:
        testset = json.load(f)

    console.print(f"[green]✓ テストセットをロード: {len(testset)}件[/green]\n")
    return testset


def display_results(result):
    """
    評価結果を表示

    Args:
        result: RAGAS評価結果
    """
    console.print("\n[bold cyan]====== 評価結果 ======[/bold cyan]\n")

    # スコアテーブル
    table = Table(title="RAGAS評価スコア")
    table.add_column("メトリクス", style="cyan")
    table.add_column("スコア", style="green")
    table.add_column("説明", style="dim")

    metrics_info = {
        "faithfulness": ("忠実性", "回答が検索コンテキストに基づいているか"),
        # "answer_relevancy": ("回答関連性", "回答が質問に適切に答えているか"),
        "context_recall": ("コンテキスト再現率", "必要な情報を検索できているか"),
        "context_precision": ("コンテキスト精度", "検索されたコンテキストの品質"),
    }

    # EvaluationResultオブジェクトを辞書に変換
    result_dict = {}
    if hasattr(result, 'to_pandas'):
        # DataFrameに変換してから辞書に
        df = result.to_pandas()
        result_dict = df.to_dict('records')[0] if len(df) > 0 else {}
    elif hasattr(result, '__dict__'):
        result_dict = result.__dict__
    else:
        result_dict = dict(result)

    for metric_name, score in result_dict.items():
        if metric_name in metrics_info and isinstance(score, (int, float)):
            label, description = metrics_info[metric_name]
            table.add_row(label, f"{score:.4f}", description)

    console.print(table)

    # 総合評価
    scores = [v for v in result_dict.values() if isinstance(v, (int, float))]
    avg_score = sum(scores) / len(scores) if scores else 0
    console.print(f"\n[bold]総合スコア: {avg_score:.4f}[/bold]")

    if avg_score >= 0.8:
        console.print("[green]✓ 優秀なRAGシステムです！[/green]")
    elif avg_score >= 0.6:
        console.print("[yellow]⚠ 改善の余地があります[/yellow]")
    else:
        console.print("[red]✗ 大幅な改善が必要です[/red]")


def save_results(result, output_path: str = "evaluation/results.json"):
    """
    評価結果を保存

    Args:
        result: 評価結果
        output_path: 保存先パス
    """
    # 結果を辞書に変換
    result_dict = {}
    if hasattr(result, 'to_pandas'):
        df = result.to_pandas()
        result_dict = df.to_dict('records')[0] if len(df) > 0 else {}
    elif hasattr(result, '__dict__'):
        result_dict = {k: v for k, v in result.__dict__.items() if isinstance(v, (int, float, str, bool, type(None)))}
    else:
        result_dict = dict(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]✓ 評価結果を保存: {output_path}[/green]")


def main():
    """メイン処理"""
    console.print("[bold cyan]RAGASによるRAGシステム評価[/bold cyan]\n")

    # テストセットをロード
    testset = load_testset()

    # 評価実行
    evaluator = RAGEvaluator()
    result = evaluator.run_evaluation(testset)
    evaluator.close()

    # 結果表示
    display_results(result)

    # 結果保存
    save_results(result)

    console.print("\n[bold green]評価完了！[/bold green]")


if __name__ == "__main__":
    main()
