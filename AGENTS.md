# LunaLLM Agents Guide

## ビルド

```bash
nim c -d:release src/lunallm.nim
```

## 実行

```bash
# 学習
MODE=observe OBSERVE_DATA=corpus_combined.txt DB=luna_cognitive.db ./src/lunallm

# チャット
MODE=chat DB=luna_cognitive.db ./src/lunallm

# デバッグ
MODE=debug DB=luna_cognitive.db ./src/lunallm
```

## プロジェクト構成

- `src/types.nim` - 全型定義
- `src/concept_graph.nim` - 概念グラフ（単語抽出、エッジ構築）
- `src/tsetlin.nim` - Tsetlin Machine（学習・推論）
- `src/cognitive_loop.nim` - 認知パイプライン全体
- `src/generator.nim` - 応答生成
- `src/grammar.nim` - 日本語文法エンジン
- `src/storage.nim` - SQLite永続化
- `src/tokenizer.nim` - 単語分割（BPE不要、高頻度辞書）
- `src/lunallm.nim` - メインバイナリ

## コーディング規約

- Nim 2.2.10+
- メモリ管理: refc
- リリースビルド: `-d:release`
- DB: SQLite（軽量配布対応）
- コメント: 必要最小限
- エラー処理: `except CatchableError`

## テスト

```bash
# 個別テスト
nim c -r test_concepts.nim
nim c -r test_extract.nim
nim c -r test_vocab.nim
```

## 注意事項

- `concept` はNimの予約語
- エンコードはUTF-8
- CJK文字範囲: U+3040-U+309F, U+30A0-U+30FF, U+4E00-U+9FFF
