"""
RAGASメトリクスの詳細なデバッグ情報を表示するスクリプト
各テストケースごとにメトリクスのスコアと計算過程を表示
"""
import json
import time
import argparse
import weaviate
from datetime import datetime, timezone, timedelta
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

# JST (日本標準時) タイムゾーン
JST = timezone(timedelta(hours=9))


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

        # AsyncOpenAIクライアント作成（RAGAS用）
        # タイムアウトを600秒（10分）に設定してgpt-oss:20bのような大型モデルに対応
        openai_client = AsyncOpenAI(
            api_key="ollama",
            base_url=ollama_base_url,
            timeout=600.0,  # 10分
            max_retries=3
        )

        # 通常のテキスト生成用のOpenAIクライアント（同期版）
        self.openai_sync_client = OpenAI(
            api_key="ollama",
            base_url=ollama_base_url,
            timeout=600.0,  # 10分
            max_retries=3
        )

        # RAGAS 0.4のllm_factoryでLLM作成
        self.llm = llm_factory(
            model=Config.OLLAMA_LLM_MODEL,
            provider="openai",
            client=openai_client
        )

    def vector_search(self, question: str, top_k: int = 3) -> List[Dict]:
        """ベクトル検索（セマンティック検索）"""
        collection = self.weaviate_client.collections.get(self.collection_name)
        response = collection.query.near_text(
            query=question,
            limit=top_k,
            return_metadata=['distance']
        )

        results = []
        for obj in response.objects:
            distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 1.0
            results.append({
                "content": obj.properties.get("content", ""),
                "vector_score": 1 / (1 + distance)
            })
        return results

    def bm25_search(self, question: str, top_k: int = 3) -> List[Dict]:
        """BM25検索（キーワード検索）"""
        collection = self.weaviate_client.collections.get(self.collection_name)
        response = collection.query.bm25(
            query=question,
            limit=top_k,
            return_metadata=['score']
        )

        results = []
        for obj in response.objects:
            bm25_score = obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0
            results.append({
                "content": obj.properties.get("content", ""),
                "bm25_score": bm25_score
            })
        return results

    def hybrid_search(self, question: str, top_k: int = 3, alpha: float = 0.5) -> List[Dict]:
        """手動ハイブリッド検索"""
        # 両方の検索を実行
        vector_results = self.vector_search(question, top_k=top_k * 2)
        bm25_results = self.bm25_search(question, top_k=top_k * 2)

        # コンテンツでマージ
        merged = {}
        for result in vector_results:
            key = result["content"]
            merged[key] = {**result, "bm25_score": 0}

        for result in bm25_results:
            key = result["content"]
            if key in merged:
                merged[key]["bm25_score"] = result["bm25_score"]
            else:
                merged[key] = {**result, "vector_score": 0}

        # ハイブリッドスコアを計算
        max_bm25 = max([r.get("bm25_score", 0) for r in merged.values()] or [1])
        for key in merged:
            v_score = merged[key].get("vector_score", 0)
            b_score = merged[key].get("bm25_score", 0)
            b_score_norm = b_score / max_bm25 if max_bm25 > 0 else 0
            merged[key]["hybrid_score"] = alpha * v_score + (1 - alpha) * b_score_norm

        # スコアでソート
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.get("hybrid_score", 0),
            reverse=True
        )
        return sorted_results[:top_k]

    def retrieve_contexts(self, question: str, top_k: int = 3, search_mode: str = None, alpha: float = None) -> List[str]:
        """
        質問に対してコンテキストを検索

        Args:
            question: 質問
            top_k: 取得する結果数
            search_mode: "vector", "bm25", "hybrid" (Noneの場合はConfig.SEARCH_MODEを使用)
            alpha: ハイブリッド検索の重み (Noneの場合はConfig.HYBRID_ALPHAを使用)

        Returns:
            コンテキストのリスト
        """
        if search_mode is None:
            search_mode = getattr(Config, 'SEARCH_MODE', 'vector')
        if alpha is None:
            alpha = getattr(Config, 'HYBRID_ALPHA', 0.5)

        # 検索実行
        if search_mode == "hybrid":
            results = self.hybrid_search(question, top_k, alpha)
        elif search_mode == "bm25":
            results = self.bm25_search(question, top_k)
        else:  # vector
            results = self.vector_search(question, top_k)

        # コンテンツのみを抽出
        contexts = [result["content"] for result in results]
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

        # OpenAIクライアントを直接使用して回答生成
        response = self.openai_sync_client.chat.completions.create(
            model=Config.OLLAMA_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content

    def run_evaluation_with_debug(self, testset: List[Dict], search_mode: str = None, alpha: float = None) -> tuple:
        """
        テストセットを使用して評価を実行（デバッグ情報付き）

        Args:
            testset: テストケースのリスト
            search_mode: 検索モード ("vector", "bm25", "hybrid")
            alpha: ハイブリッド検索の重み

        Returns:
            (評価結果, デバッグ情報のリスト)
        """
        if search_mode is None:
            search_mode = getattr(Config, 'SEARCH_MODE', 'vector')
        if alpha is None:
            alpha = getattr(Config, 'HYBRID_ALPHA', 0.5)

        console.print(f"[blue]RAG評価を実行中（デバッグモード）...[/blue]")
        console.print(f"[dim]検索モード: {search_mode}, α={alpha}[/dim]\n")

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
            retrieved_contexts = self.retrieve_contexts(question, top_k=Config.TOP_K_RESULTS, search_mode=search_mode, alpha=alpha)

            # 回答生成
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


def save_debug_log(debug_info: List[Dict], result, testset_count: int, search_mode: str, alpha: float, execution_time: str, output_dir: str = "evaluation"):
    """
    デバッグログを保存（タイムスタンプ付きファイル名とメタデータ）

    Args:
        debug_info: デバッグ情報のリスト
        result: RAGAS評価結果
        testset_count: テストセット件数
        search_mode: 検索モード
        alpha: alpha値
        execution_time: 実行時間
        output_dir: 出力ディレクトリ
    """
    # タイムスタンプ付きファイル名を生成
    timestamp_str = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    output_filename = f"debug_log_{timestamp_str}.json"
    output_path = f"{output_dir}/{output_filename}"

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

    # メタデータと結果を含む構造
    save_data = {
        "metadata": {
            "timestamp": datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST"),
            "testset_file": "evaluation/testset.json",
            "testset_count": testset_count,
            "search_mode": search_mode,
            "alpha": alpha,
            "execution_time": execution_time
        },
        "test_results": full_log
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]✓ デバッグログを保存: {output_path}[/green]")


def main():
    """メイン処理"""
    # コマンドライン引数を解析
    parser = argparse.ArgumentParser(
        description="RAGASによるRAGシステム評価（デバッグモード） - 検索設定を指定可能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルト設定で評価（Config.SEARCH_MODE, Config.HYBRID_ALPHAを使用）
  python evaluate_debug.py

  # ベクトル検索で評価
  python evaluate_debug.py --mode vector

  # ハイブリッド検索で評価（デフォルトα=0.5）
  python evaluate_debug.py --mode hybrid

  # ハイブリッド検索（α=0.7、セマンティック寄り）
  python evaluate_debug.py --mode hybrid --alpha 0.7

  # BM25検索で評価
  python evaluate_debug.py --mode bm25
        """
    )
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "bm25"],
        help="検索モード（指定なし=Config.SEARCH_MODEを使用）"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="ハイブリッド検索のalpha値（0.0-1.0、指定なし=Config.HYBRID_ALPHAを使用）"
    )

    args = parser.parse_args()

    # alpha値の検証
    if args.alpha is not None and not (0.0 <= args.alpha <= 1.0):
        console.print("[red]エラー: alpha値は0.0から1.0の範囲で指定してください[/red]")
        return

    # 設定を取得（コマンドライン引数 > Config）
    search_mode = args.mode if args.mode else getattr(Config, 'SEARCH_MODE', 'vector')
    alpha = args.alpha if args.alpha is not None else getattr(Config, 'HYBRID_ALPHA', 0.5)

    # 評価設定を表示
    mode_names = {"vector": "ベクトル検索", "hybrid": "ハイブリッド検索", "bm25": "BM25検索"}
    mode_display = mode_names.get(search_mode, search_mode)

    title_text = "[bold cyan]RAGASによるRAGシステム評価（デバッグモード）[/bold cyan]\n\n"
    title_text += f"検索モード: {mode_display}"
    if search_mode == "hybrid":
        title_text += f" (α={alpha})"

    console.print(Panel.fit(title_text, border_style="cyan"))

    # 実行開始時刻を記録
    start_time = time.time()

    # テストセットをロード
    testset = load_testset()

    # 評価実行
    evaluator = RAGEvaluatorDebug()
    result, debug_info = evaluator.run_evaluation_with_debug(testset, search_mode=search_mode, alpha=alpha)
    evaluator.close()

    # 実行時間を計算
    end_time = time.time()
    execution_time_sec = end_time - start_time
    execution_time_str = f"{int(execution_time_sec // 60)}分{int(execution_time_sec % 60)}秒"

    # デバッグ情報を表示
    display_debug_info(debug_info, result)

    # サマリー表示
    display_summary(result)

    # デバッグログを保存
    save_debug_log(debug_info, result, len(testset), search_mode, alpha, execution_time_str)

    console.print("\n[bold green]評価完了！[/bold green]")


if __name__ == "__main__":
    main()
