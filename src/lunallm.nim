import os, strutils, times, tables, sequtils, algorithm, math, unicode
import types, tokenizer, concept_graph, working_memory, tsetlin,
       generator, cognitive_loop, storage

# ---------------------------------------------------------------------------
# LunaLLM - Cognitive Architecture (v2)
# ---------------------------------------------------------------------------
# 認知科学に基づくアーキテクチャ
# 知識はデータベースに、論理はルールに、表現はテンプレートに、個性はシナプスに

proc loadTrainingData(path: string): seq[string] =
  result = @[]
  if not fileExists(path):
    echo "Data not found: " & path
    return
  for line in path.lines:
    let trimmed = line.strip()
    if trimmed.len == 0: continue
    let pipePos = trimmed.find('|')
    if pipePos >= 0:
      let a = trimmed[0..<pipePos].strip()
      let b = trimmed[pipePos+1..^1].strip()
      if a.len > 0 and b.len > 0:
        result.add(a & "|" & b)

proc buildTokenizerFromCorpus(corpus: seq[string]; maxVocab: int = 4096): Tokenizer =
  var vocabCount: Table[string, int]
  for text in corpus:
    for rune in text.toRunes:
      let s = $rune
      vocabCount[s] = vocabCount.getOrDefault(s, 0) + 1

  result.vocab = @[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
  result.tokenToId = initTable[string, int]()
  result.tokenToId[PAD_TOKEN] = PAD_ID
  result.tokenToId[UNK_TOKEN] = UNK_ID
  result.tokenToId[EOS_TOKEN] = EOS_ID

  var sorted = toSeq(vocabCount.pairs)
  sorted.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))

  var idx = 3
  for (token, _) in sorted:
    if token.len > 0 and idx < maxVocab:
      result.vocab.add(token)
      result.tokenToId[token] = idx
      inc idx

# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
proc main() =
  echo "===================================="
  echo " LunaLLM - Cognitive Architecture v2"
  echo "===================================="
  echo " Concept Network + TM Reasoning + Hebbian Synapses"
  echo " No neural network. No GPU. Pure cognition."
  echo ""

  let mode = getEnv("MODE", "chat")
  let dbPath = getEnv("DB", "luna_cognitive.db")
  let debugMode = paramCount() >= 1 and paramStr(1) == "--debug"

  if mode == "observe":
    # === 観察モード ===
    echo "=== Observe Mode: Corpus Learning ==="

    let dataPath = getEnv("OBSERVE_DATA", "ollama_2000.txt")
    if not fileExists(dataPath):
      echo "Data not found: " & dataPath
      quit(1)

    # 大規模コーパスはストリーミング処理
    let fileSize = getFileSize(dataPath)
    let useStreaming = fileSize > 100_000_000  # 100MB以上

    if useStreaming:
      echo "Large corpus detected (" & $(fileSize div 1_000_000) & "MB), using streaming mode"

      # 認知状態初期化
      var cfg = CognitiveConfig(
        wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
        activationThreshold: 0.1, tmClauses: 64, tmThreshold: 0.3,
        tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
        thinkingEnabled: true, evalEnabled: true,
        rewardRate: 0.05, punishRate: 0.03
      )
      var state = initCognitiveState(cfg)

      # ストリーミング観察
      state.observeCorpusStream(dataPath)

      # 永続化
      echo "Saving..."
      var sdb = openStorage(dbPath)
      sdb.saveConfig(cfg)
      sdb.saveConceptGraph(state.conceptGraph)
      sdb.saveTM(state.tm)
      sdb.saveSynapses(state.bridge)
      sdb.saveCatalog(state.catalog)
      sdb.savePhase(state.phase)
      sdb.close()
      echo "Saved to: " & dbPath
      echo "Done!"

    else:
      # 小規模コーパスは従来方式
      let corpus = loadTrainingData(dataPath)
      if corpus.len == 0:
        echo "No data found in: " & dataPath
        quit(1)

      echo "Corpus: " & $corpus.len & " conversations"

      # トカナイザ構築（サブセットで学習）
      let tokCorpus = if corpus.len > 5000: corpus[0..<5000] else: corpus
      var tok = buildTokenizer(tokCorpus, 4096)
      echo "Tokenizer: " & $tok.vocab.len & " vocab"

      # 認知状態初期化
      var cfg = CognitiveConfig(
        wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
        activationThreshold: 0.1, tmClauses: 64, tmThreshold: 0.3,
        tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
        thinkingEnabled: true, evalEnabled: true,
        rewardRate: 0.05, punishRate: 0.03
      )
      var state = initCognitiveState(cfg)
      state.tokenizer = tok

      # コーパス観察
      state.observeCorpus(corpus)

      # 永続化（概念グラフ、TM、synapses、カタログ）
      echo "Saving..."
      var sdb = openStorage(dbPath)
      sdb.saveConfig(cfg)
      sdb.saveConceptGraph(state.conceptGraph)
      sdb.saveTM(state.tm)
      sdb.saveSynapses(state.bridge)
      sdb.saveCatalog(state.catalog)
      sdb.savePhase(state.phase)
      sdb.close()
      echo "Saved to: " & dbPath
      echo "Done!"

  elif mode == "chat":
    # === チャットモード ===
    echo "=== Chat Mode ==="

    if not fileExists(dbPath):
      echo "No model found at: " & dbPath
      echo "Run MODE=observe first"
      quit(1)

    var sdb = openStorage(dbPath)
    let cfg = sdb.loadConfig()
    var state = initCognitiveState(cfg)
    state.conceptGraph = sdb.loadConceptGraph()
    sdb.loadTM(state.tm)
    state.bridge = sdb.loadSynapses()
    state.catalog = sdb.loadCatalog()
    state.phase = sdb.loadPhase()
    sdb.close()

    echo "Concepts: " & $state.conceptGraph.conceptCount()
    echo "Catalog: " & $state.catalog.entries.len & " entries"
    echo "Synapses: " & $state.bridge.synapses.len
    echo "Phase: " & $state.phase
    echo ""
    echo "Type 'exit' to quit"
    echo ""

    while true:
      stdout.write "You: "
      stdout.flushFile()
      let input = stdin.readLine()
      if input.strip() == "exit": break
      if input.strip().len == 0: continue

      let response = state.process(input)
      echo "Luna: " & (if response.len > 0: response else: "(empty)")

      # シンキングチェーン表示（--debug のみ）
      if debugMode and state.lastThinking.steps.len > 0:
        echo ""
        echo "  [Thinking]"
        for step in state.lastThinking.steps:
          echo "    " & $step.kind & ": " & step.description &
               " (conf=" & $formatFloat(step.confidence, ffDecimal, 3) & ")"
          for d in step.details:
            echo "      - " & d
        echo "    Total confidence: " & $formatFloat(state.lastThinking.totalConfidence, ffDecimal, 3)

      # 自己評価表示（--debug のみ）
      if debugMode:
        echo "  [Eval] " & state.lastEval.reason
        if state.lastEval.contradictions.len > 0:
          for c in state.lastEval.contradictions:
            echo "    Contradiction: " & c
        echo "  [Reward/Punish] total_reward=" & $formatFloat(state.totalReward, ffDecimal, 3) &
             " total_punish=" & $formatFloat(state.totalPunish, ffDecimal, 3)
        echo ""

      # 定期保存（10会話ごと）
      if state.episodeStore.episodes.len mod 10 == 0:
        var saveDb = openStorage(dbPath)
        saveDb.saveConceptGraph(state.conceptGraph)
        saveDb.saveTM(state.tm)
        saveDb.saveSynapses(state.bridge)
        saveDb.saveCatalog(state.catalog)
        saveDb.close()

  elif mode == "debug":
    # === デバッグモード ===
    echo "=== Debug: Cognitive State ==="

    if not fileExists(dbPath):
      echo "No model found"
      quit(1)

    var sdb = openStorage(dbPath)
    let cfg = sdb.loadConfig()
    var state = initCognitiveState(cfg)
    state.conceptGraph = sdb.loadConceptGraph()
    sdb.loadTM(state.tm)
    state.bridge = sdb.loadSynapses()
    state.catalog = sdb.loadCatalog()
    sdb.close()

    echo "Concepts: " & $state.conceptGraph.conceptCount()
    echo "Edges: " & $state.conceptGraph.edgeCount()
    echo "Catalog: " & $state.catalog.entries.len & " entries"
    echo "Synapses: " & $state.bridge.synapses.len
    echo ""

    # テスト入力
    let testInputs = ["おはよう", "元気？", "何してる？", "日本語で話して", "ありがとう"]
    for input in testInputs:
      echo "--- Input: " & input & " ---"
      let response = state.process(input)
      echo "  Response: " & (if response.len > 0: response else: "(empty)")

      # シンキングチェーン表示
      if state.lastThinking.steps.len > 0:
        echo "  [Thinking]"
        for step in state.lastThinking.steps:
          echo "    " & $step.kind & ": " & step.description &
               " (conf=" & $formatFloat(step.confidence, ffDecimal, 3) & ")"
        echo "    Total confidence: " & $formatFloat(state.lastThinking.totalConfidence, ffDecimal, 3)

      # 自己評価表示
      echo "  [Eval] " & state.lastEval.reason
      echo "  [Reward] " & $formatFloat(state.totalReward, ffDecimal, 3) &
           " [Punish] " & $formatFloat(state.totalPunish, ffDecimal, 3)
      echo ""

  elif mode == "sleep":
    # === 睡眠モード: 背景統合 ===
    echo "=== Sleep Mode: Consolidation ==="

    if not fileExists(dbPath):
      echo "No model found"
      quit(1)

    var sdb = openStorage(dbPath)
    let cfg = sdb.loadConfig()
    var state = initCognitiveState(cfg)
    state.conceptGraph = sdb.loadConceptGraph()
    state.bridge = sdb.loadSynapses()
    sdb.close()

    # シナプス減衰
    echo "Decaying synapses..."
    let now = epochTime()
    for syn in state.bridge.synapses.mitems:
      let elapsed = now - syn.lastActivated
      let days = elapsed / 86400.0
      syn.strength = syn.strength * pow(0.5, days.float32 / syn.halfLifeDays)

    # 弱いシナプス除去
    let before = state.bridge.synapses.len
    state.bridge.synapses.keepItIf(it.strength > 0.01)
    echo "  Synapses: " & $before & " -> " & $state.bridge.synapses.len

    # 保存
    var saveDb = openStorage(dbPath)
    saveDb.saveSynapses(state.bridge)
    saveDb.close()
    echo "Sleep complete."

  else:
    echo "Unknown mode: " & mode
    echo "Use: MODE=observe|chat|debug|sleep"

main()
