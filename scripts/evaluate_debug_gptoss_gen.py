"""
RAGASメトリクスの詳細なデバッグ情報を表示するスクリプト（gpt-oss:20b生成 + ELYZA評価版）
各テストケースごとにメトリクスのスコアと計算過程を表示
回答生成にgpt-oss:20b、RAGAS評価にELYZA-JP-8Bを使用
"""
import json
import os
import weaviate
from urllib.parse import urlparse
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track, Progress
from config import Config
from openai import AsyncOpenAI, OpenAI
from ragas.llms import llm_factory
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    # answer_relevancy,  # RAGAS 0.4との互換性問題のため除外
    context_recall,
    context_precision,
)
from datasets import Dataset

console = Console()


class RAGEvaluatorDebug:
    """RAGシステムの評価を行うクラス（デバッグログ付き）"""

    def __init__(self):
        """初期化"""
        # Weaviate接続
        parsed_url = urlparse(Config.WEAVIATE_URL)
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 8080

        self.weaviate_client = weaviate.connect_to_local(host=host, port=port)
        self.collection_name = Config.WEAVIATE_COLLECTION_NAME

        # OllamaをOpenAI互換APIとして使用（RAGAS 0.4対応）
        ollama_base_url = Config.OLLAMA_API_URL.replace("http://ollama:", "http://ollama:")
        if not ollama_base_url.endswith("/v1"):
            ollama_base_url = ollama_base_url.rstrip("/") + "/v1"

        # 回答生成用モデル（gpt-oss:20b）
        self.generation_model = "gpt-oss:20b"
        console.print(f"[green]✓ 回答生成モデル: {self.generation_model}[/green]")

        # 評価用モデル（ELYZA-JP-8B）
        self.evaluation_model = "elyza-jp-8b"
        console.print(f"[green]✓ RAGAS評価モデル: {self.evaluation_model}[/green]")

        # AsyncOpenAIクライアント作成（RAGAS評価用 - ELYZA-JP-8B）
        openai_client_eval = AsyncOpenAI(
            api_key="ollama",
            base_url=ollama_base_url,
            timeout=600.0,  # 10分
            max_retries=3
        )

        # 通常のテキスト生成用のOpenAIクライアント（同期版 - gpt-oss:20b）
        self.openai_sync_client = OpenAI(
            api_key="ollama",
            base_url=ollama_base_url,
            timeout=600.0,  # 10分
            max_retries=3
        )

        # RAGAS 0.4のllm_factoryでLLM作成（評価用はELYZA-JP-8B）
        self.llm = llm_factory(
            model=self.evaluation_model,
            provider="openai",
            client=openai_client_eval
        )

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
        コンテキストと質問から回答を生成（gpt-oss:20b使用）

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

        # OpenAIクライアントを直接使用して回答生成（gpt-oss:20b）
        response = self.openai_sync_client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def run_evaluation_with_debug(self, testset: List[Dict]) -> tuple:
        """
        テストセットを使用して評価を実行（デバッグ情報付き）

        Args:
            testset: テストケースのリスト

        Returns:
            (評価結果, デバッグ情報のリスト)
        """
        console.print("[blue]RAG評価を実行中（デバッグモード）...[/blue]\n")

        # 各テストケースに対してRAGを実行
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        debug_info = []

        for idx, test_case in enumerate(track(testset, description="Processing"), 1):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            # コンテキスト検索
            retrieved_contexts = self.retrieve_contexts(question, top_k=Config.TOP_K_RESULTS)

            # 回答生成（gpt-oss:20b）
            answer = self.generate_answer(question, retrieved_contexts)

            questions.append(question)
            answers.append(answer)
            contexts.append(retrieved_contexts)
            ground_truths.append(ground_truth)

            # デバッグ情報を保存
            debug_info.append({
                "index": idx,
                "question": question,
                "retrieved_contexts": retrieved_contexts,
                "answer": answer,
                "ground_truth": ground_truth,
                "num_contexts": len(retrieved_contexts)
            })

        # データセット作成（RAGAS 0.4ハイブリッド形式）
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
            "reference": ground_truths
        }

        dataset = Dataset.from_dict(data)

        console.print("\n[blue]RAGAS評価指標を計算中...[/blue]")

        # RAGAS評価実行（answer_relevancyを除外）
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                # answer_relevancy,  # 互換性問題のため除外
                context_recall,
                context_precision,
            ],
            llm=self.llm,
        )

        return result, debug_info

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


def display_debug_info(debug_info: List[Dict], result):
    """
    デバッグ情報を表示

    Args:
        debug_info: デバッグ情報のリスト
        result: RAGAS評価結果
    """
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]詳細デバッグログ[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    # 評価結果をDataFrameに変換
    df = result.to_pandas()

    for i, info in enumerate(debug_info):
        console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
        console.print(f"[bold yellow]テストケース #{info['index']}: {info['question'][:60]}...[/bold yellow]")
        console.print(f"[bold yellow]{'='*80}[/bold yellow]\n")

        # 質問
        console.print(Panel(
            info['question'],
            title="[bold cyan]質問[/bold cyan]",
            border_style="cyan"
        ))

        # 検索されたコンテキスト
        console.print(f"\n[bold green]検索されたコンテキスト（{info['num_contexts']}件）:[/bold green]")
        for ctx_idx, context in enumerate(info['retrieved_contexts'], 1):
            console.print(f"\n[dim]--- コンテキスト {ctx_idx} ---[/dim]")
            console.print(f"[dim]{context[:300]}...[/dim]")

        # LLMの回答
        console.print(f"\n[bold blue]LLMの回答:[/bold blue]")
        console.print(Panel(
            info['answer'],
            border_style="blue"
        ))

        # グランドトゥルース
        console.print(f"\n[bold magenta]グランドトゥルース（期待される正解）:[/bold magenta]")
        console.print(Panel(
            info['ground_truth'],
            border_style="magenta"
        ))

        # 評価スコア
        if i < len(df):
            row = df.iloc[i]
            console.print(f"\n[bold white]評価スコア:[/bold white]")

            table = Table(show_header=True, header_style="bold")
            table.add_column("指標", style="cyan")
            table.add_column("スコア", style="green", justify="right")
            table.add_column("説明", style="dim")

            metrics_info = {
                "faithfulness": "忠実性（回答がコンテキストに基づいているか）",
                # "answer_relevancy": "回答関連性（質問に適切に答えているか）",
                "context_recall": "コンテキスト再現率（必要な情報を検索できているか）",
                "context_precision": "コンテキスト精度（検索されたコンテキストの品質）",
            }

            for metric, description in metrics_info.items():
                if metric in row:
                    score = row[metric]
                    if isinstance(score, (int, float)) and not (score != score):  # NaNチェック
                        table.add_row(metric, f"{score:.4f}", description)
                    else:
                        table.add_row(metric, "N/A", description)

            console.print(table)

        console.print("\n")


def display_summary(result):
    """
    評価結果のサマリーを表示

    Args:
        result: RAGAS評価結果
    """
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]評価サマリー[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")

    # DataFrameに変換
    df = result.to_pandas()

    # 全体の平均スコア
    table = Table(title="全体平均スコア", show_header=True, header_style="bold cyan")
    table.add_column("メトリクス", style="cyan")
    table.add_column("平均スコア", style="green", justify="right")
    table.add_column("説明", style="dim")

    metrics_info = {
        "faithfulness": ("忠実性", "回答が検索コンテキストに基づいているか"),
        # "answer_relevancy": ("回答関連性", "回答が質問に適切に答えているか"),
        "context_recall": ("コンテキスト再現率", "必要な情報を検索できているか"),
        "context_precision": ("コンテキスト精度", "検索されたコンテキストの品質"),
    }

    avg_scores = {}
    for metric in metrics_info.keys():
        if metric in df.columns:
            valid_scores = df[metric].dropna()
            if len(valid_scores) > 0:
                avg_score = valid_scores.mean()
                avg_scores[metric] = avg_score
                label, description = metrics_info[metric]
                table.add_row(label, f"{avg_score:.4f}", description)

    console.print(table)

    # 総合評価
    if avg_scores:
        overall_avg = sum(avg_scores.values()) / len(avg_scores)
        console.print(f"\n[bold]総合スコア: {overall_avg:.4f}[/bold]")

        if overall_avg >= 0.8:
            console.print("[green]✓ 優秀なRAGシステムです！[/green]")
        elif overall_avg >= 0.6:
            console.print("[yellow]⚠ 改善の余地があります[/yellow]")
        else:
            console.print("[red]✗ 大幅な改善が必要です[/red]")


def save_debug_log(debug_info: List[Dict], result, output_path: str = "evaluation/debug_log.json"):
    """
    デバッグログを保存

    Args:
        debug_info: デバッグ情報のリスト
        result: RAGAS評価結果
        output_path: 保存先パス
    """
    # 評価結果をDataFrameに変換
    df = result.to_pandas()

    # デバッグ情報と評価スコアを結合
    full_log = []
    for i, info in enumerate(debug_info):
        log_entry = info.copy()

        if i < len(df):
            row = df.iloc[i]
            log_entry["scores"] = {
                "faithfulness": float(row["faithfulness"]) if "faithfulness" in row and not (row["faithfulness"] != row["faithfulness"]) else None,
                # "answer_relevancy": float(row["answer_relevancy"]) if "answer_relevancy" in row and not (row["answer_relevancy"] != row["answer_relevancy"]) else None,
                "context_recall": float(row["context_recall"]) if "context_recall" in row and not (row["context_recall"] != row["context_recall"]) else None,
                "context_precision": float(row["context_precision"]) if "context_precision" in row and not (row["context_precision"] != row["context_precision"]) else None,
            }

        full_log.append(log_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_log, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]✓ デバッグログを保存: {output_path}[/green]")


def main():
    """メイン処理"""
    console.print("[bold cyan]RAGASによるRAGシステム評価（gpt-oss:20b生成 + ELYZA評価版・デバッグモード）[/bold cyan]\n")
    console.print("[yellow]📌 回答生成: gpt-oss:20b (大型汎用モデル)[/yellow]")
    console.print("[yellow]📌 RAGAS評価: ELYZA-JP-8B (日本語特化、高速)[/yellow]\n")

    # テストセットをロード
    testset = load_testset()

    # 評価実行
    evaluator = RAGEvaluatorDebug()
    result, debug_info = evaluator.run_evaluation_with_debug(testset)
    evaluator.close()

    # デバッグ情報を表示
    display_debug_info(debug_info, result)

    # サマリー表示
    display_summary(result)

    # デバッグログを保存（gpt-oss生成版）
    save_debug_log(debug_info, result, output_path="evaluation/debug_log_gptoss_gen.json")

    console.print("\n[bold green]評価完了！[/bold green]")
    console.print("[dim]結果はevaluation/debug_log_gptoss_gen.jsonに保存されました[/dim]")


if __name__ == "__main__":
    main()
