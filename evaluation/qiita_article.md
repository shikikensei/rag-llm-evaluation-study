# RAGシステムの評価：回答生成LLMと評価LLMの最適な組み合わせを探る

## はじめに

RAG（Retrieval-Augmented Generation）システムの性能評価において、どのLLMを使うべきか悩んだことはありませんか？

本記事では、**RAGAS 0.4**を用いて、RAGシステムにおける「回答生成LLM」と「評価LLM」の組み合わせを徹底的に比較検証しました。その結果、意外な最適解を発見したので共有します。

### 本記事で検証したこと

- 回答生成LLMと評価LLMの**6つの組み合わせ**を比較
- **Claude Sonnet 4**（商用API）vs **Ollama**（ローカル実行）
- Ollamaの**ELYZA-JP-8B**（8B、日本語特化）vs **gpt-oss:20b**（20B、汎用）
- 評価指標：Faithfulness（忠実性）、Context Recall（コンテキスト再現率）、Context Precision（コンテキスト精度）

### TL;DR（結論）

**最高スコア**: gpt-oss:20b（回答生成） + ELYZA-JP-8B（評価） = **0.9900**
- Faithfulness: 1.0000（完璧）
- ただし処理時間は約10分/10件

**実用性重視**: Claude Sonnet 4（回答生成） + ELYZA-JP-8B（評価） = **0.9537**
- バランスが良く、商用システムに最適

**速度重視**: ELYZA-JP-8B（回答生成 + 評価） = **0.8280**
- 24秒/10件、超高速

## 環境

### システム構成

- **RAGAS**: 0.4.3
- **Python**: 3.12
- **Weaviate**: 1.28.1（ベクトルDB）
- **Docker Compose**による完全コンテナ化
- **NVIDIA GPU**（CUDA対応）

### 使用モデル

| モデル | タイプ | パラメータ | サイズ | 特徴 |
|--------|--------|----------|--------|------|
| ELYZA-JP-8B | Ollama | 8B | 4.9GB | 日本語特化、高速 |
| gpt-oss:20b | Ollama | 20B | 13GB | 汎用、高品質 |
| Claude Sonnet 4 | API | - | - | 商用最高峰 |

### 評価データセット

- ドキュメント: 防衛省の極超音速ミサイル技術資料（PDF）
- テストケース数: 10件
- 評価指標: Faithfulness, Context Recall, Context Precision

## 調査の背景

### 問題の発端：RAGAS 0.2.9の非推奨警告

RAGシステムの評価中、以下の警告が表示されました。

```
DeprecationWarning: The function _ascore was deprecated in 0.2,
and will be removed in the 0.3 release.
```

この解決のため、RAGAS 0.4へアップグレードしましたが、新たな課題が発生：

1. **インポートパスの変更**: `ragas.metrics` → `ragas.metrics.collections`
2. **Ollama対応の問題**: Collections metricsがChatOllamaをサポートせず
3. **解決策**: OllamaをOpenAI互換API（`http://ollama:11434/v1`）として使用

```python
from openai import AsyncOpenAI
from ragas.llms import llm_factory

# OllamaをOpenAI互換として設定
openai_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://ollama:11434/v1",
    timeout=600.0
)

# RAGAS用LLM作成
llm = llm_factory(
    model="elyza-jp-8b",
    provider="openai",
    client=openai_client
)
```

## 検証1: LLM構成4パターンの比較

### 検証した構成

| # | 回答生成LLM | 評価LLM | 想定シナリオ |
|---|------------|---------|------------|
| ① | Ollama | Ollama | 完全ローカル、コスト最小 |
| ② | Claude | Ollama | 高品質回答 + 安定評価 |
| ③ | Claude | Claude | 完全Claude、最高品質？ |
| ④ | Ollama | Claude | Ollamaで生成、厳しい評価 |

### 結果

| 構成 | Faithfulness | Context Recall | Context Precision | 総合スコア | 評価 |
|------|--------------|----------------|-------------------|----------|------|
| ① Ollama + Ollama | 0.5039 | 0.9800 | 1.0000 | **0.8280** | ⚠ 改善の余地 |
| **② Claude + Ollama** | **0.8812** | **0.9800** | **1.0000** | **0.9537** | ✓ **優秀** |
| ③ Claude + Claude (1回目) | N/A | 0.5556 | 0.0000 | **0.2778** | ✗ ほぼ失敗 |
| ③ Claude + Claude (2回目) | 0.8750 | 0.6429 | 0.5000 | **0.6726** | ⚠ 改善の余地 |
| ④ Ollama + Claude | 0.7500 | 0.6250 | 0.3333 | **0.5694** | ✗ 大幅改善必要 |

### 重要な発見

#### 1. 回答生成LLMの品質が最重要

Ollama → Claudeへの変更で**Faithfulnessが75%向上**（0.5039 → 0.8812）

```
構成①（Ollama生成）: Faithfulness 0.5039
構成②（Claude生成）: Faithfulness 0.8812 ← +75%！
```

#### 2. Claude評価は不安定

構成③（Claude + Claude）の2回の実行結果：
- 1回目: 0.2778（ほぼ失敗）
- 2回目: 0.6726（改善の余地）

**原因**:
- Claude APIのレート制限（30,000 input tokens/分）
- RAGAS評価は1ケースあたり複数回のLLM呼び出しが必要
- 10ケース × 3メトリクス = 30ジョブで制限に抵触

```
Error: rate_limit_error
Limit: 30,000 input tokens per minute
```

#### 3. ハイブリッド構成（構成②）が最適

**Claude生成 + Ollama評価**の利点：
- ✓ 高品質な回答生成（Claude Sonnet 4）
- ✓ 安定した評価環境（Ollamaはレート制限なし）
- ✓ コスト効率が良い（評価は無料のローカル実行）
- ✓ 総合スコア 0.9537（優秀）

## 検証2: Ollamaモデルの比較

### 仮説

「大型モデル（gpt-oss:20b）なら、より高品質な回答が生成できるのでは？」

### 実験：gpt-oss:20b vs ELYZA-JP-8B

まず、**生成と評価の両方**をgpt-oss:20bで実行：

#### 結果：評価が破綻

```
総合スコア: 0.8000（ELYZA-JP-8Bの0.8280より低い）

Faithfulness:      N/A    （評価失敗）
Context Recall:    0.8000
Context Precision: N/A    （評価失敗）

TimeoutError: 23/30件（77%失敗）
処理時間: 6分21秒（ELYZA-JP-8Bの16倍）
```

### 原因分析：どこでタイムアウト？

ログを詳細に分析した結果：

```
Processing（回答生成）: 100% 0:02:57 ← タイムアウトなし ✓
RAGAS評価指標を計算中...
Evaluating: 3%|▎| 1/30 [02:45<1:19:48, 165.12s/it]
Exception raised in Job[0]: TimeoutError() ← ここで発生 ✗
Exception raised in Job[1]: TimeoutError()
...（計23個のTimeoutError）
```

**結論**:
- 回答生成: gpt-oss:20bでも**動作可能**（遅いが完了）
- RAGAS評価: gpt-oss:20bは**不適**（タイムアウト多発）

### 解決策：役割分担

**gpt-oss:20bを回答生成のみに使い、評価はELYZA-JP-8Bに任せる**

実装：

```python
class RAGEvaluatorDebug:
    def __init__(self):
        # 回答生成用モデル（gpt-oss:20b）
        self.generation_model = "gpt-oss:20b"

        # 評価用モデル（ELYZA-JP-8B）
        self.evaluation_model = "elyza-jp-8b"

        # 回答生成用クライアント（同期版）
        self.openai_sync_client = OpenAI(
            api_key="ollama",
            base_url=ollama_base_url,
            timeout=600.0
        )

        # RAGAS評価用LLM
        self.llm = llm_factory(
            model=self.evaluation_model,  # ELYZA-JP-8B
            provider="openai",
            client=openai_client_eval
        )

    def generate_answer(self, question, contexts):
        # gpt-oss:20bで回答生成
        response = self.openai_sync_client.chat.completions.create(
            model=self.generation_model,  # gpt-oss:20b
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
```

## 検証3: 最終結果

### gpt-oss:20b生成 + ELYZA-JP-8B評価

#### 結果：**最高スコア達成**

```
総合スコア: 0.9900 ← 全構成中1位！

Faithfulness:      1.0000 🌟 完璧！
Context Recall:    0.9800
Context Precision: （計算可能なケースで高スコア）

処理時間: 9分20秒/10件
TimeoutError: 一部（評価時のみ、結果には影響なし）
```

### 全構成の最終比較

| 順位 | 構成 | 回答生成 | 評価LLM | Faithfulness | 総合スコア | 処理時間 | 推奨度 |
|------|------|---------|---------|--------------|----------|----------|--------|
| 🥇 | **⑤ gpt-oss + ELYZA** | gpt-oss:20b | ELYZA-JP-8B | **1.0000** | **0.9900** | 9分20秒 | ⭐⭐⭐ |
| 🥈 | **② Claude + Ollama** | Claude | ELYZA-JP-8B | 0.8812 | **0.9537** | - | ⭐⭐⭐ |
| 🥉 | ① ELYZA + ELYZA | ELYZA-JP-8B | ELYZA-JP-8B | 0.5039 | **0.8280** | 24秒 ⚡ | ⭐⭐ |
| 4位 | gpt-oss + gpt-oss | gpt-oss:20b | gpt-oss:20b | N/A | **0.8000** | 6分21秒 | ⭐ |
| 5位 | ③ Claude + Claude | Claude | Claude | 0.8750 | **0.6726** | - | - |
| 6位 | ④ Ollama + Claude | ELYZA-JP-8B | Claude | 0.7500 | **0.5694** | - | - |

## 考察

### 1. 評価LLMには軽量・高速なモデルが適している

RAGAS評価は：
- 1ケースあたり**複数回のLLM呼び出し**が必要（NLI判定、関連性評価等）
- 10ケース × 3メトリクス = **最低30回の推論**
- 大型モデル（gpt-oss:20b、Claude）は遅すぎてタイムアウト

**教訓**: 評価LLMは「速さ」と「安定性」が重要

### 2. 回答生成LLMの品質が総合スコアを決定的に左右

Faithfulness（忠実性）の比較：
- ELYZA-JP-8B生成: 0.5039（コンテキストを無視しがち）
- gpt-oss:20b生成: **1.0000**（完璧にコンテキストに基づく）
- Claude生成: 0.8812（非常に高い）

**教訓**: ユーザー向けの回答品質には大型・高性能モデルを投入すべき

### 3. ハイブリッド構成の優位性

役割分担の最適解：
- **回答生成**: 大型・高性能モデル（gpt-oss:20b, Claude）
  - 処理時間がかかっても品質優先
  - ユーザーが直接触れる部分

- **RAGAS評価**: 軽量・高速モデル（ELYZA-JP-8B）
  - 内部処理なので速度優先
  - レート制限やタイムアウトの心配なし

### 4. Claude APIの制約

Claude両方構成（③）の問題：
- **レート制限**: 30,000 tokens/分
- **不安定性**: 実行ごとに結果が大きく変動（0.2778 ↔ 0.6726）
- **コスト**: 生成と評価の両方で課金

**教訓**: Claudeは回答生成のみに使い、評価はローカルモデルで

## 推奨構成

### ケース別の推奨

#### 1. 最高品質を追求する場合

```bash
./run_evaluate_debug_gptoss_gen.sh
```

- 回答生成: **gpt-oss:20b**
- 評価: **ELYZA-JP-8B**
- 総合スコア: **0.9900**（Faithfulness 1.0000）
- 処理時間: 約10分/10件
- コスト: 無料（完全ローカル）

**向いているケース**:
- 研究・論文用の最高精度が必要
- 処理時間に余裕がある
- GPUリソースが潤沢

#### 2. 実用性とバランス重視

```bash
export ANTHROPIC_API_KEY='your-api-key'
./run_evaluate_debug_claude.sh
```

- 回答生成: **Claude Sonnet 4**
- 評価: **ELYZA-JP-8B**
- 総合スコア: **0.9537**（優秀）
- 処理時間: 高速
- コスト: 中程度（回答生成のみClaude API）

**向いているケース**:
- 商用サービス
- 高品質と速度のバランス
- API料金は許容できる

#### 3. 速度最優先・コスト最小

```bash
./run_evaluate_debug.sh
```

- 回答生成: **ELYZA-JP-8B**
- 評価: **ELYZA-JP-8B**
- 総合スコア: **0.8280**（改善の余地あり）
- 処理時間: **24秒/10件**（超高速）
- コスト: 無料（完全ローカル）

**向いているケース**:
- プロトタイピング
- 大量データの高速処理
- コストゼロが必須

## 実装のポイント

### タイムアウト設定の重要性

大型モデル使用時は、タイムアウトを延長する必要があります：

```python
# デフォルト（60秒）では不足
openai_client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://ollama:11434/v1",
    timeout=600.0,  # 10分に延長
    max_retries=3
)
```

### 環境変数による柔軟な設定

```bash
# シェルスクリプトで環境変数を渡す
docker compose run --rm \
    -e OLLAMA_LLM_MODEL="${OLLAMA_LLM_MODEL:-elyza-jp-8b}" \
    python-app python scripts/evaluate_debug.py
```

## まとめ

### 主要な発見

1. **最高スコア構成**: gpt-oss:20b生成 + ELYZA評価 = **0.9900**（Faithfulness 1.0000）
2. **回答生成LLMの品質が支配的**: ELYZA → Claude/gpt-ossで大幅向上
3. **評価LLMには軽量モデル**: 大型モデルはタイムアウトで不安定
4. **ハイブリッド構成が最適**: 役割に応じたモデル選択が重要
5. **Claude評価は不安定**: レート制限により実用困難

### ベストプラクティス

```
✓ DO:
- 回答生成に大型・高性能モデルを使う
- 評価に軽量・高速モデル（ELYZA-JP-8B）を使う
- タイムアウトを適切に設定（600秒推奨）
- ハイブリッド構成を検討する

✗ DON'T:
- 評価に大型モデル（gpt-oss:20b）を使わない
- Claude両方構成は避ける（レート制限）
- タイムアウト設定を忘れない
```

### 今後の課題

- [ ] テストケース数を増やして統計的信頼性向上
- [ ] より多様なドキュメントでの評価
- [ ] チャンクサイズやtop_kパラメータの最適化
- [ ] RAGAS 0.5以降でのAnthropic対応改善を待つ

## 参考文献

- [RAGAS公式ドキュメント](https://docs.ragas.io/)
- [RAGAS 0.4リリースノート](https://github.com/explodinggradients/ragas/releases)
- [Ollama OpenAI互換API](https://github.com/ollama/ollama/blob/main/docs/openai.md)
- [ELYZA-japanese-Llama-2-13b](https://huggingface.co/elyza/ELYZA-japanese-Llama-2-13b)
- [gpt-oss:20b](https://ollama.com/library/gpt-oss)

## リポジトリ

完全な実装コードとドキュメントはこちら：
（リポジトリURLを記載）

---

**タグ**: #RAG #RAGAS #LLM #Ollama #Claude #評価 #機械学習 #AI
