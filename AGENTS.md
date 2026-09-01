# LunaticIntelligence Agents Guide

## ビルド

```bash
nim c -d:release -d:ssl src/lunatic.nim
nim c -d:release -d:ssl --mm:orc --threads:off src/server.nim
nim c -d:release -d:ssl --mm:orc --threads:off src/mcp_server.nim
```

## 実行

```bash
# 学習
./src/lunatic observe --data corpus_combined.txt --db lunatic_cognitive.db

# チャット
./src/lunatic chat --db lunatic_cognitive.db

# デバッグ
./src/lunatic debug --db lunatic_cognitive.db

# サーバー
./src/lunatic serve --port 8080 --db lunatic_cognitive.db

# 互換（環境変数）
MODE=observe OBSERVE_DATA=corpus_combined.txt DB=lunatic_cognitive.db ./src/lunatic
MODE=chat DB=lunatic_cognitive.db ./src/lunatic
MODE=debug DB=lunatic_cognitive.db ./src/lunatic
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
- `src/code_structure.nim` - 軽量Nim AST解析
- `src/db_compress.nim` - CatelliteCompressor ラッパー
- `src/server.nim` - HTTPサーバー (/lunatic/chat)
- `src/mcp_server.nim` - MCPサーバー (lunatic_*)
- `src/lunatic.nim` - メインバイナリ

## コーディング規約

- Nim 2.2.10+
- メモリ管理: orc
- スレッド: off
- リリースビルド: `-d:release -d:ssl`
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
