# RAGシステム評価 - LLM構成比較レポート

## 調査概要

本調査では、RAGシステムにおける回答生成LLMと評価LLMの組み合わせが、RAGAS評価スコアに与える影響を検証しました。

### 調査期間
2026年1月実施

### 評価対象
- テストケース数: 10件
- 評価指標: Faithfulness（忠実性）、Context Recall（コンテキスト再現率）、Context Precision（コンテキスト精度）
- 使用モデル:
  - Ollama: ELYZA-JP-8B（日本語特化、ローカル実行）
  - Claude: Sonnet 4（高性能商用モデル）

## 評価結果サマリー

### 4構成の比較表

| 構成 | 回答生成LLM | 評価LLM | Faithfulness | Context Recall | Context Precision | 総合スコア | 評価 |
|------|------------|---------|--------------|----------------|-------------------|----------|------|
| ① | Ollama | Ollama | 0.5039 | 0.9800 | 1.0000 | **0.8280** | ⚠ 改善の余地 |
| ② | **Claude** | **Ollama** | **0.8812** | **0.9800** | **1.0000** | **0.9537** | ✓ **優秀** |
| ③-1 | Claude | Claude | N/A | 0.5556 | 0.0000 | **0.2778** | ✗ ほぼ失敗 |
| ③-2 | Claude | Claude | 0.8750 | 0.6429 | 0.5000 | **0.6726** | ⚠ 改善の余地 |
| ④ | Ollama | Claude | 0.7500 | 0.6250 | 0.3333 | **0.5694** | ✗ 大幅改善必要 |

### スコア基準
- **0.8以上**: 優秀なRAGシステム
- **0.6-0.8**: 改善の余地あり
- **0.6未満**: 大幅な改善が必要

## 詳細分析

### 構成① Ollama生成 + Ollama評価（ベースライン）
**総合スコア: 0.8280**

```
Faithfulness:      0.5039 ⚠ 低い
Context Recall:    0.9800 ✓ 優秀
Context Precision: 1.0000 ✓ 完璧
```

**特徴**:
- Context系指標は優秀（検索性能は良好）
- Faithfulness（忠実性）が低い → Ollamaの回答生成品質に課題
- コスト: 無料（ローカル実行）
- 安定性: 非常に高い

### 構成② Claude生成 + Ollama評価（推奨構成）
**総合スコア: 0.9537** ⭐

```
Faithfulness:      0.8812 ✓ 大幅改善（+75%）
Context Recall:    0.9800 ✓ 優秀
Context Precision: 1.0000 ✓ 完璧
```

**特徴**:
- **全指標で高スコア達成**
- Faithfulness が 0.5039 → 0.8812 に劇的改善
- 唯一「優秀なRAGシステム」評価を獲得
- コスト: 中程度（回答生成のみClaude API使用）
- 安定性: 非常に高い

**メリット**:
- ユーザー向け回答の質が高い（Claude生成）
- 評価の安定性が高い（Ollama評価）
- レート制限の影響を受けにくい
- コストパフォーマンスが良い

### 構成③ Claude生成 + Claude評価（不安定）

#### 1回目実行
**総合スコア: 0.2778** ✗

```
Faithfulness:      N/A    （評価失敗）
Context Recall:    0.5556 ⚠ 低下
Context Precision: 0.0000 ✗ 失敗
```

#### 2回目実行（再試行）
**総合スコア: 0.6726** ⚠

```
Faithfulness:      0.8750 ✓ 良好
Context Recall:    0.6429 △ 低下
Context Precision: 0.5000 △ 低下
```

**特徴**:
- **実行ごとに結果が大きく変動**（0.2778 → 0.6726）
- レート制限エラー（429）が頻発
  - `30,000 input tokens per minute` の制限
  - 10テストケース × 3メトリクス = 30以上のAPI呼び出し
- `InstructorRetryException`、`max_tokens length limit` エラー発生
- Context系指標が大幅に低下

**問題点**:
- API料金が高い（回答生成 + 評価の両方）
- レート制限により評価が不完全になる
- 再現性が低く、本番運用に不適

### 構成④ Ollama生成 + Claude評価
**総合スコア: 0.5694** ✗

```
Faithfulness:      0.7500 △ 改善（構成①比）
Context Recall:    0.6250 △ 低下
Context Precision: 0.3333 ⚠ 大幅低下
```

**特徴**:
- Ollama評価（構成①）より全体的に厳しい評価
- Claudeが評価LLMとして不安定
- Context Precisionが特に低い（0.3333）

## 重要な発見

### 1. 回答生成LLMの品質が最重要
- Ollama → Claude への変更で **Faithfulness が 75% 向上**（0.5039 → 0.8812）
- ユーザーが直接触れる回答の質に最も影響を与える要素

### 2. 評価LLMには安定性が求められる
- Ollama評価: 安定、高速、レート制限なし
- Claude評価: 不安定、レート制限あり、コスト高

### 3. Claude評価の技術的課題
- RAGAS 0.4のAnthropic対応がまだ実験的
- 短時間に多数のAPI呼び出しが発生しレート制限に抵触
- 評価結果の再現性が低い

### 4. ハイブリッド構成の優位性
**構成②（Claude生成 + Ollama評価）が最適な理由**:
- 高品質な回答生成（Claude）
- 安定した評価環境（Ollama）
- コスト効率が良い（評価は無料）
- レート制限の心配がない

## 結論と推奨事項

### 推奨構成
**構成② Claude生成 + Ollama評価**

実行コマンド:
```bash
./run_evaluate_debug_claude.sh
```

実装ファイル:
- スクリプト: `scripts/evaluate_debug_claude.py`
- シェル: `run_evaluate_debug_claude.sh`

### 推奨する理由
1. **最高の総合スコア**: 0.9537（唯一の「優秀」評価）
2. **安定性**: 実行ごとの結果のブレが少ない
3. **コストパフォーマンス**: 回答生成のみClaude APIを使用
4. **実用性**: レート制限の影響を受けにくい

### 非推奨構成
- **構成③（Claude両方）**: 不安定、高コスト、レート制限
- **構成④（Ollama生成 + Claude評価）**: Claude評価の不安定性

### 今後の改善案

#### 短期的改善
- RAGAS 0.4の次バージョンでAnthropic対応の安定化を待つ
- Claude API のレート制限緩和を検討

#### 長期的改善
- テストケース数を増やして統計的信頼性を向上
- より多様なドキュメントでの評価
- チャンクサイズやtop_kパラメータの最適化

## 技術メモ

### 使用環境
- RAGAS: 0.4.3
- Python: 3.12
- Ollama Model: ELYZA-JP-8B
- Claude Model: claude-sonnet-4-20250514
- Weaviate: 4.9.4

### 評価データ
- テストセット: `evaluation/testset.json`
- デバッグログ:
  - `evaluation/debug_log.json` (構成①)
  - `evaluation/debug_log_claude.json` (構成②)
  - `evaluation/debug_log_claude_full.json` (構成③)
  - `evaluation/debug_log_ollama_gen.json` (構成④)

### レート制限詳細（構成③で発生）
```
Error: rate_limit_error
Limit: 30,000 input tokens per minute
Organization: 497c27b2-a4e9-4760-bee0-88b7d73292d2
```

## 補足調査: Ollamaモデル比較

### ELYZA-JP-8B vs gpt-oss:20b

Ollama生成 + Ollama評価構成において、日本語特化モデル（ELYZA-JP-8B）と大型汎用モデル（gpt-oss:20b）を比較しました。

| モデル | パラメータ | サイズ | Faithfulness | Context Recall | Context Precision | 総合スコア | 処理時間 | TimeoutError |
|--------|----------|--------|--------------|----------------|-------------------|----------|----------|--------------|
| **ELYZA-JP-8B** | 8B | 4.9GB | 0.5039 | 0.9800 | 1.0000 | **0.8280** | 0:00:24 ⚡ | 0件 ✓ |
| gpt-oss:20b | 20B | 13GB | N/A | 0.8000 | N/A | **0.8000** | 0:06:21 🐌 | 23/30件 ⚠ |

### 評価結果

**ELYZA-JP-8Bが優位**:
- 処理速度が**16倍高速**
- Context系指標が高い（特にContext Precision: 1.0000）
- タイムアウトエラーなし（安定性が高い）
- 総合スコアが上回る（0.8280 vs 0.8000）

**gpt-oss:20bの問題点**:
- モデルサイズが大きく推論が遅い（13GB）
- RAGAS評価で大量のタイムアウト発生（23/30件）
- FaithfulnessとContext Precisionが評価不能（N/A）
- タイムアウトを600秒（10分）に延長しても改善不十分

### 結論

**日本語RAGタスクにはELYZA-JP-8Bが最適**:
- 日本語特化により効率的
- 高速・安定・低リソース
- モデルサイズと性能のバランスが良い

gpt-oss:20bは汎用性は高いものの、RAGAS評価のような短時間に多数の推論が必要なタスクでは、大きすぎて実用的ではありません。

### 技術的詳細

**タイムアウト設定の影響**:
```python
# デフォルト（約60秒）
Context Recall: 0.6667
TimeoutError: 27/30件

# 延長後（600秒 = 10分）
Context Recall: 0.8000  (+20%改善)
TimeoutError: 23/30件  (依然として77%失敗)
```

タイムアウトを延ばしても根本的な解決にはならず、モデルサイズが大きすぎることが本質的な問題です。

## 付録: 各構成の実行方法

### 構成① Ollama生成 + Ollama評価
```bash
./run_evaluate_debug.sh
```

### 構成② Claude生成 + Ollama評価（推奨）
```bash
export ANTHROPIC_API_KEY='your-api-key'
./run_evaluate_debug_claude.sh
```

### 構成③ Claude生成 + Claude評価
```bash
export ANTHROPIC_API_KEY='your-api-key'
./run_evaluate_debug_claude_full.sh
```

### 構成④ Ollama生成 + Claude評価
```bash
export ANTHROPIC_API_KEY='your-api-key'
./run_evaluate_debug_ollama_gen.sh
```

---

**レポート作成日**: 2026年1月30日
**調査実施者**: RAG System Evaluation Team
**バージョン**: 1.0
