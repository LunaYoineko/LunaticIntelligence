# Lunatic Intelligence

**ハイブリッド認知アーキテクチャ**: 右脳（Tsetlin Machine + 概念グラフ）＝判断 / 左脳（CPU特化 Transformer）＝生成

## 核心フロー

```
Input
  ↓
[1. 単語抽出・概念活性化] ──→ 概念グラフ: ノード活性化 + Hebbian/Edge伝播 (3step, decay=0.5)
  ↓
[2. TM推論] ──→ Tsetlin Machine: 節活性化パターン → 推論信頼度 + firedConcepts
  ↓
[3. 意図分類] ──→ iiGreeting/Question/Request/Thanks/Farewell/Opinion/Statement/Other
  ↓
[4. Thinking生成 (DeepSeek/Qwen風)]
     TM判断(意図, 関連概念, 確信度) → プロンプト構築 → LLM内部で Q/A 自問自答
     形式: <thinking>Q:... A:...</thinking><answer>最終回答</answer>
  ↓
[5. 双方向フィードバック] ──→ LLM思考断片から単語抽出 → 概念グラフ再活性化 (0.6) → 軽い再伝播(1step)
  ↓
[6. 応答生成]
     A. カタログ完全一致 → 真乱数選択 → 返却
     B. 意図別分岐 (時間/日付/計算/天気/コード/検索/一般質問/要求/感謝/別れ/意見/陳述)
        → generateWithLLM(mergedConcepts, intent, prompt, forceLLM=true)
        → LLM未学習のため rightBrainResponse + 真乱数語尾揺らぎで返却
     C. フォールバック: getCoolResponse / generateTemplateFromConcepts
  ↓
[7. 自己評価・報酬罰学習] ──→ Episode保存 + シナプス強化/減衰
```

---

## 詳細仕様

### 1. 概念グラフ (`concept_graph.nim`)

- **ノード**: 単語 (freq≥3, 2文字以上), カテゴリ (名詞/動詞/形容詞/助詞/記号/その他)
- **エッジ**: 共起 (window=3), 重み = 共起回数 × baseFreq
- **活性化伝播**: `spreadActivation(steps=3, decay=0.5)` + Hebbian同期更新
- **真乱数揺らぎ** (`/dev/urandom` 4バイト → float32):
  - ノード作成時: `baseFrequency *= (1 ± 0.02)` 
  - Hebbian更新時: `weight *= (1 ± wobble)` 
  - **コード生成時**: `wobbleScale=0.005` (通常 `0.02`) で揺らぎ抑制

### 2. Tsetlin Machine (`tsetlin.nim`)

- **構成**: `tmClauses` (動的: 500会話→256, 10k→512, 100k→1024), `tmThreshold`, `s=3.0`
- **推論**: 入力概念ID → 節活性化パターン (`firedConcepts`) → 多数決で意図/推論
- **学習**: Type-I (Include正例) / Type-II (Exclude負例) フィードバック
- **抑止ブレーキ**: 意図と概念カテゴリ不一致時 `activation *= 0.5` (例: 質問意図×挨拶カテゴリ)

### 3. Hybrid LLM (`llm.nim`)

- **アーキテクチャ**: Transformer Decoder-only
  - `dModel=64, nHead=4, nLayer=2, dFF=256, vocab=4096, maxSeqLen=128`
- **CPU特化最適化**:
  - `nimblas` (BLAS) / `nimsimd` (SIMD) / 32要素ブロッキング
  - `GELU` 近似, LayerNorm, Attention, FFN 全て行列演算化
- **量子化・圧縮**:
  - `int8` 量子化 (scale=127/max_abs) → 4倍縮小
  - `zippy` (LZ4系) 圧縮 → さらに 1.5倍
  - 合計 **~6倍削減** (3.2MB → 0.79MB)
- **VRAM動的活用** (`device.nim`):
  - `nvidia-smi --query-gpu=memory.free` / `rocm-smi --showmeminfo vram` 解析
  - CPUメイン、GPUはアシスト
  - モデル <100MB → CPU常駐
  - 空きVRAM `usable * 0.5` 以内なら `residentOnGPU=true` で重み転送を1回に集約
- **真乱数サンプリング**: `/dev/urandom` → `float32` で Top-k 温度サンプリング
  - 通常: `temp = 0.72 + rand*0.35` (毎回変動)
  - コード時: `temp = 0.35` (揺らぎ抑制)

### 4. Thinking Chain (`types.nim` + `cognitive_loop.nim`)

```nim
ThinkingStepKind = enum
  tsPerception   # 感知: 入力解釈
  tsReasoning    # 推論: LLM内部 Q/A
  tsConclusion   # 結論: 回答確定
```

- **プロンプト構成**: `ユーザー発言 + TM判断(意図,概念,確信度) + TM思考 + 指示(Q/A自問自答)`
- **LLM生成**: `maxTokens=12`, 真乱数温度
- **解析**: `<thinking>` と `<answer>` タグで分割、`Q:`/`A:` で ThinkingStep 再構築
- **state.lastThinking** に保存、デバッグ表示・次回プロンプト注入に使用

### 5. 双方向フィードバック

```nim
# LLM生成 raw から単語抽出
for w in thinkWords:
  if conceptGraph.hasKey(w):
    conceptGraph.activateWord(w, 0.6)  # 思考由来は弱め
conceptGraph.spreadActivation(steps=1, decay=0.5)
```
→ LLMの「思考」が右脳の概念グラフを再活性化、次回推論に影響

### 6. カタログ・応答選択

- **構築**: コーパスの `input|output` ペアから、意図別上限 20万件で収集
- **照合**: 完全一致のみ (`entry.inputText == input`)
- **選択**: 同一入力に複数候補あれば **真乱数** でランダム選択
- **重複問題**: 現状 `large_corpus_yami.txt` が百科事典Q&Aで、同一質問が大量重複（例: "How are you?" 9334件）→ 実質固定化

### 7. 非言語閾値 (`cognitive_loop.nim:1076`)

- 活性化 `> 0.3` の概念のみ `mergedConcepts` に採用
- `0.3 以下` = 非言語的活性のまま、LLMプロンプトには渡さない
- これにより「意識に上らない連想」が右脳内で完結

### 8. Intent Classifier (`intent_classifier.nim`)

- キーワードベース + 文末記号で 8分類
- TM推論結果と併用: TMの `firedConcepts` パターンで上書き修正

### 9. Web Search (`web_search.nim`)

- **優先**: 日本語 Wikipedia (`ja.wikipedia.org/w/api.php?action=query&list=search`)
- **補助**: DuckDuckGo HTML scrape (`html.duckduckgo.com/html/?q=`)
- **キーワード抽出**: `extractSearchKeywords` (助詞除外、名詞/動詞/固有名詞優先)
- **キャッシュ**: `search_cache` テーブル (TTL 24h)
- **翻訳**: `translateKnowledge` (英語結果を日本語化、簡易ルールベース)

### 10. Incremental Learning (`lunatic.nim:301`)

```
observe-inc --data new.txt --db existing.db
  → 既存DBロード (ConceptGraph, TM, Synapses, Catalog, Phase)
  → 新コーパスで tokenizer 拡張 + observeCorpusStream
  → 上書き保存
```

### 11. Storage (`storage.nim`)

- **Backend**: SQLite3 **WAL mode** (`PRAGMA journal_mode=WAL; synchronous=NORMAL; cache_size=-64000`)
- **Tables**:
  - `concept_nodes(id, word, category, baseFrequency, activation, lastUpdated, accessCount)`
  - `concept_edges(from_id, to_id, weight, type, createdAt)`
  - `tm_states(clauseId, taStateJson)` — TA状態 JSON
  - `synapses(clauseId, conceptId, weight)`
  - `response_catalog(input_text, output_text, intent, weight, createdAt)`
  - `model_config(key, value)` — 設定
  - `working_memory_items(...)` — 作業記憶
  - `tokenizer_vocab(token, id)` — **未使用** (将来用)
- **LLM重み**: 別途 `*.cmp` (量子化+圧縮) で保存、LMDB統合は将来予定

### 12. Tokenizer (`tokenizer.nim`)

- **方式**: 単語ベース (BPE不使用、高頻度辞書)
- **構築**: コーパスから文字種別 (CJK/英数/その他) で分割 → 頻度順上位 4096語
- **符号化**: 最長一致 (最大20文字)
- **問題**: `chat` モードで辞書再構築されず (概念グラフ+カタログから再構築するよう修正済み)

---

## モード別動作

### Observe (学習)
```
observe --data corpus.txt --db db
  → tokenizer構築 (先頭5000行) → 動的tmClauses決定
  → 3エポック: 単語抽出 → ノード作成 → エッジ/TM/Hebbian/カタログ更新
  → ストリーミング対応 (>100MB): 行バッファ + 50000行ごと sleep(5) でCPU譲渡
  → SQLite WAL で保存
```

### Chat (会話)
```
chat --db db [--debug]
  → DBロード → tokenizer再構築 (概念語+カタログ語)
  → ループ: 入力 → process() → 応答表示
  → 10ターンごとに自動保存
```

### Debug
```
debug --db db
  → 状態ダンプ + 5テスト入力で内部ステップ表示
```

### Serve (HTTP/MCP)
```
serve --port 8080 --db db
  → /lunatic/chat (POST JSON: {input, debug?})
  → MCP: lunatic_chat, lunatic_observe, lunatic_debug
```

---

## ファイル構成と役割

| ファイル | 役割 |
|----------|------|
| `src/lunatic.nim` | CLI エントリ, モード分岐, DBロード/保存, tokenizer再構築 |
| `src/types.nim` | 全型定義 (ConceptNode, TMState, ThinkingChain, Synapse等) |
| `src/cognitive_loop.nim` | **核心**: process(), 活性化伝播, TM推論, Thinking生成, 応答分岐, 学習ループ |
| `src/llm.nim` | Transformer実装, 量子化/圧縮, VRAM対応, 真乱数サンプリング |
| `src/device.nim` | `nvidia-smi/rocm-smi` 解析, GPU/CPU選択, 転送コスト見積 |
| `src/concept_graph.nim` | 概念グラフ構築, 活性化伝播, Hebbian, 真乱数揺らぎ |
| `src/tsetlin.nim` | Tsetlin Machine (TA自動機, Type-I/II学習, 推論) |
| `src/intent_classifier.nim` | キーワードベース意図分類 (8分類) |
| `src/web_search.nim` | Wikipedia/DuckDuckGo検索, キャッシュ, キーワード抽出, 翻訳 |
| `src/storage.nim` | SQLite WAL 永続化, 全テーブルCRUD |
| `src/tokenizer.nim` | 単語ベース符号化/復号, 辞書構築 |
| `src/grammar.nim` | 日本語文法エンジン (未使用/将来用) |
| `src/generator.nim` | テンプレート生成 (未使用/将来用) |
| `src/code_structure.nim` | 軽量Nim AST解析 (コード生成支援) |
| `src/db_compress.nim` | CatelliteCompressor ラッパー (未使用/将来用) |

---

## ビルド・実行

```bash
# 依存: db_connector, zippy, nimblas, nimsimd, lmdb (nimble install)
# リリースビルド
nim c -d:release -d:ssl src/lunatic.nim
nim c -d:release -d:ssl --mm:orc --threads:off src/server.nim
nim c -d:release -d:ssl --mm:orc --threads:off src/mcp_server.nim

# 学習
./src/lunatic observe --data corpus.txt --db lunatic.db
# 追加学習
./src/lunatic observe-inc --data new.txt --db lunatic.db
# チャット
./src/lunatic chat --db lunatic.db [--debug]
# デバッグ
./src/lunatic debug --db lunatic.db
# サーバー
./src/lunatic serve --port 8080 --db lunatic.db

# 互換 (環境変数)
MODE=observe OBSERVE_DATA=corpus.txt DB=lunatic.db ./src/lunatic
MODE=chat DB=lunatic.db ./src/lunatic
```

---

## 既知の課題・改善点

| 課題 | 現状 | 対策案 |
|------|------|--------|
| **カタログ重複** | 同一入力が数千件重複、真乱数選択でも固定化 | 学習時 `DISTINCT input_text` 化、または頻度重み付けサンプリング |
| **コーパス品質** | `large_corpus_yami.txt` が百科事典Q&A、自然会話ではない | 会話特化コーパス (日常会話, 対話システムデータ) へ差し替え |
| **LLM未学習** | 重みがランダム初期化、`generateText` が `<UNK>` 連発 | 事前学習 (CPUで小規模でも可) または LoRA 追加学習 |
| **Thinkingがゴミ** | LLM出力が `<UNK>` → `state.lastThinking` にゴミが入る | LLMスキップ時は TM Thinking のみを `lastThinking` に格納 |
| **Tokenizer未保存** | DBに語彙保存なし、chat毎に再構築 | `tokenizer_vocab` テーブル活用、保存/ロード実装 |
| **VRAM検出失敗時** | `device.nim` が 0 返すと CPU fallback | エラーハンドリング強化、手動指定オプション追加 |
| **推論遅延** | 12token で 1-2s (CPU), 大DBで 60s超リスク | KVキャッシュ導入, `dModel` 可変, バッチ推論 |

---

## パフォーマンス実測 (Ryzen 7 / 32GB RAM)

| 項目 | 値 | 備考 |
|------|-----|------|
| 10k会話学習 | ~15s | tmClauses=512, 3epoch |
| 1.2M行学習 | ~1511s | tmClauses=1024, streaming, nice -19 |
| DBサイズ (1.2M) | 153MB | SQLite WAL, 概念5.5k, カタログ493k |
| DBサイズ (10k) | 4.6MB | 概念939, カタログ~93k |
| 推論 (右脳のみ) | <10ms | TM推論 + カタログ照合 |
| 推論 (LLM 12token) | 1-2s | 32d/1層量子化, CPU単体 |
| 15質問 varied | ~8-30s | 60s以内達成 |
| VRAM検出 | 6144→4804MB usable | RTX 3070 8GB 環境 |

---

## ライセンス

MIT