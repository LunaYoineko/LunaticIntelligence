# LunaLLM

Concept Network + Tsetlin Machine + Hebbian Synapses による認知アーキテクチャ LLM。  
ニューラルネットワーク不使用、GPU不要。Pure cognition on CPU.

## 特徴

- **Concept Network**: コーパスから単語を抽出し、概念グラフを構築
- **Tsetlin Machine**: 概念活性化パターンから意図分類・推論
- **Hebbian Synapses**: 使用頻度に基づくシナプス強化
- **Response Catalog**: 学習済みコーパスから直接応答を返す高速照合
- **Grammar Engine**: 概念組み合わせによる多様な応答生成

## 対象環境

- 2 CPU cores (AMD EPYC-Milan)
- 1.9GB RAM
- GPU不要

## クイックスタート

```bash
# Nim 2.2.10以降が必要
nimble install

# 学習（42k会話で約1秒）
MODE=observe OBSERVE_DATA=corpus_combined.txt DB=luna_cognitive.db ./src/lunallm

# チャット（コーパスファイル不要）
MODE=chat DB=luna_cognitive.db ./src/lunallm

# デバッグ
MODE=debug DB=luna_cognitive.db ./src/lunallm
```

## コーパス形式

```
入力テキスト|応答テキスト
```

例:
```
おはよう|おはようございます
元気？|元気です
何してる？|特に nothing
```

## DB構成

最小限のテーブル構成（軽量配布対応）:

- `concept_nodes`: 概念ノード（単語+カテゴリ+頻度）
- `concept_edges`: 概念間エッジ（関係性+重み）
- `tm_states`: Tsetlin Machine状態
- `synapses`: シナプス強度
- `response_catalog`: 応答カタログ（入力→出力マッピング）
- `model_config`: 設定値

episodes, working_memory, tokenizer_vocab はDBに保存しない（メモリ上のみ）。

## パフォーマンス

| 処理 | 時間 |
|------|------|
| 42kコーパス学習 | 約1秒 |
| DBサイズ | 約320KB |
| メモリ使用量 | 約150MB |

## アーキテクチャ

```
corpus → [単語抽出] → [概念グラフ構築] → [TM学習] → [Hebbian強化] → [カタログ構築]
                                                                              ↓
user input → [概念活性化] → [カタログ照合 or 文法生成] → response
```

## ライセンス

MIT
