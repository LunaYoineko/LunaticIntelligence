# Lunatic Intelligence

**ハイブリッド認知**: 右脳(TM+概念グラフ)=判断 / 左脳(CPU Transformer)=生成。`Input→TM意図→概念活性→Thinking→LLM生成→検証` で少量学習でも検索RAGで5T丸暗記に勝つ。

## 核心フロー

```
Input
 ↓ [1] 概念活性化: spreadActivation(3,0.5) + Hebbian + 真乱数wobble(0.02, code0.005)
 ↓ [2] TM推論: clause活性→firedConcepts, confidence + 抑止ブレーキ(意図≠カテゴリで*0.5)
 ↓ [3] 意図分類: iiGreeting/Question/Request/Thanks/Farewell/Opinion/Statement/Other
 ↓ [4] Thinking: TM判断をLLMへ注入し <thinking>Q/A</thinking><answer> を自問自答
 ↓ [5] MoE Gating: 右脳をルーターに流用し4専門家(exGeneral/exChat/exCode/exReasoning)を選択、温度/wobbleを動的切替
 ↓ [6] 文字数決定: Thinkingが 40/300/600/800 を推論で決定
 ↓ [7] 双方向フィードバック: LLM断片→conceptGraph.activateWord(0.6)→spread(1,0.5)
 ↓ [8] 応答生成:
      A. 時刻/日付/計算/天気は事実→LLMで生成（システム時刻取得→factPrompt→generateWithLLM）
      B. カタログ完全一致→真乱数選択
      C. 質問は検索3件を統合しThinkingが決めた文字数でチャンク生成（3チャンクに分割し繋げる）
      D. フォールバックは概念に基づく自然な問い返し（ハードコード最小）
 ↓ [8f] 文字数検証: 期待と実際の差が1/3超ならTMで再検証し追記/要約で再生成
 ↓ [9] 自己評価・報酬罰・Hebbian強化 → Episode保存
```

## 詳細仕様

### 1. 概念グラフ `concept_graph.nim`
- ノード: freq≥3 2文字以上、カテゴリ6種。エッジ: 共起window3。活性化3step decay0.5 + Hebbian。真乱数は`/dev/urandom`でwobble。

### 2. Tsetlin Machine `tsetlin.nim`
- `tmClauses` 動的256-2048、`TmChunkSize=500KB` でSQLite分割保存。Type-I/II学習、抑止ブレーキで概念抑制。

### 3. Hybrid LLM `llm.nim`
- `dModel64 nHead4 nLayer2 dFF256 vocab4096 maxLen128`。`nimblas/nimsimd/32要素ブロッキング`でCPU特化。`int8+zippy`で6倍、`base64`でBLOB保存。`trainStep:535`は **backprop完成**: `softmax→loss→dLogits`で `outWeight` と `tokenEmb` をSGD更新（出力層のみで軽量5T対応）。`trainLLM:619`は同一DB・同一データで `batch32+sleep+GC` で省リソース学習。

### 4. Thinking `types.nim`
- `tsPerception/tsReasoning(tsMoE含む)/tsConclusion`。`decideLengthByThinking:117`が入力とThinkingから `40/300/600/800` を推論。`synthesize:117`は `sliceRunes` で文字化け防止し `chunkCount 3/8` で区切り生成し繋げる。

### 5. MoE `moe.nim`
- 右脳をGating Networkに流用。`gateByRightBrain`が意図+コード検出+概念活性で4専門家をスコアリングし `exChat temp0.85` `exCode temp0.35` 等を切替。DB切替不要で軽量ハイブリッド。

### 6. DB Router `db_router.nim` + `knowledge/` フォルダ
- `lunatic_chat.db 6.1M` `lunatic_nim.db 7.9M` 等を `knowledge/` に集積。`scanDbFolder`が再帰スキャンし `db_meta{role,description,tags,concepts,catalog,sourceCorpus}` を読取り、**入れるだけで自動認識**。`selectRoleForInput`と`selectDbByMeta`のタグスコアで Lunaticが最適DBを判断し `[Router] general->nim` と自動切替。`loadStateFromPath`で `ultra` があれば透過解凍。

### 7. 超圧縮 `db_compress.nim`
- CatelliteCompressorのDB部分を抽出し内蔵化（外部バイナリ不要）。`catelliteDbCompressInternal:15`で `VACUUM+ANALYZE+page_size4096+auto_vacuum INCREMENTAL` 後に `zippy.compress(level9)` で `ULTRAv2-CATELLITE` ヘッダ付与。`6.1M→767K 12%` `7.9M→968K 12%`。`tryUltraCompress`は5MB以上で保存時自動実行。

### 8. リソース抑制 `resource_governor.nim` `storage.nim:8`
- `VmRSS`監視で `shouldThrottle(1500)` → `GC+sleep`。`cognitive_loop:917` 推論冒頭と `50k行ごと` で `throttleIfNeeded`。`storage`は `cache_size=-64000` `mmap 64M` `wal_autocheckpoint 500` `journal_size_limit 32M` で2GB環境でも `94MB` 推論 `79MB` 学習。`llm`は `batch+sleep+GC` で `nice -n 19` 推奨。

### 9. 検索 `web_search.nim`
- 日本語Wikipedia優先、DuckDuckGo補助、キーワード抽出、24hキャッシュ。`synthesize`で3件を `【title】snippet` で統合し800字生成。検索知識は `episodeStore` `catalog` `conceptGraph` `tm.train` へ1-shotで定着。

## モード

- `observe --data corpus.txt --db knowledge/lunatic_chat.db` → 3epoch → `db_meta` 保存 → `ultra` 自動生成
- `train-llm --data corpus.txt --db same.db --epochs 1 --batch-size 32 --throttle 2` → 同一DBでLLM学習（backprop有効）
- `chat --db knowledge/lunatic_chat.db [--debug]` → Routerが入力でDB自動切替、Thinkingで文字数決定とMoEを表示、10ターン毎保存
- `observe-inc` / `debug` / `serve` / `sleep` 同様

## ビルド

```bash
nimble install # db_connector zippy nimblas nimsimd
nim c -d:release -d:ssl src/lunatic.nim
nim c -d:release -d:ssl --mm:orc --threads:off src/server.nim
nim c -d:release -d:ssl --mm:orc --threads:off src/mcp_server.nim
./src/lunatic observe --data corpus_custom/corpus_chat.txt --db knowledge/lunatic_chat.db
./src/lunatic train-llm --data corpus_custom/corpus_chat.txt --db knowledge/lunatic_chat.db --epochs 1 --batch-size 32 --throttle 2
./src/lunatic chat --db knowledge/lunatic_chat.db --debug
```

## ファイル構成

| ファイル | 役割 |
|---|---|
| `src/lunatic.nim` | CLI、Router、tokenizer再構築、超圧縮連携 |
| `src/cognitive_loop.nim` | 核心process、Thinking、MoE、文字数決定、チャンク生成、検証 |
| `src/moe.nim` | MoE 4専門家とGating |
| `src/llm.nim` | Transformer、backprop、量子化、WAL |
| `src/db_router.nim` | フォルダスキャンとメタデータ判断 |
| `src/db_compress.nim` | 内蔵Catellite超圧縮 |
| `src/resource_governor.nim` | CPU/メモリ抑制 |
| `src/storage.nim` | SQLite WAL、チャンクTM、db_meta |
| `knowledge/` | 知識DB集積地（入れるだけで有効） |
| `corpus_custom/` | 会話44k/ Nim40k 独自生成コーパス |

## 性能

| 項目 | 値 |
|---|---|
| 推論 | 94MB 0.6s 96%CPU、Thinkingで40/600字を動的決定 |
| 学習44k | 3s 79MB 99%→throttleで抑制 |
| 1.2M | 1511s 76M→ultra 12% |
| 5T推定 | 生41TB→ultra5TB→辞書2%で0.83TB、分散で対応 |

## ライセンス MIT
