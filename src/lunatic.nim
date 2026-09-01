import os, strutils, times, tables, sequtils, algorithm, math, unicode, parseopt, asyncdispatch, asynchttpserver, json, httpclient
import types, tokenizer, concept_graph, working_memory, tsetlin,
       generator, cognitive_loop, storage

const VERSION* = "0.1.0"

var gServerState: CognitiveState
var gServerCfg: CognitiveConfig
var gServerDbPath = "lunatic_cognitive.db"
var gServerHasState = false
var gServerPort = 8080

proc initServerState() {.gcsafe.} =
  {.cast(gcsafe).}:
    if fileExists(gServerDbPath):
      var sdb = openStorage(gServerDbPath)
      gServerCfg = sdb.loadConfig()
      gServerState = initCognitiveState(gServerCfg)
      gServerState.conceptGraph = sdb.loadConceptGraph()
      sdb.loadTM(gServerState.tm)
      gServerState.bridge = sdb.loadSynapses()
      gServerState.catalog = sdb.loadCatalog()
      gServerState.phase = sdb.loadPhase()
      sdb.close()
      gServerHasState = true
    else:
      gServerCfg = CognitiveConfig(
        wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
        activationThreshold: 0.1, tmClauses: 8192, tmThreshold: 0.3,
        tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
        thinkingEnabled: true, evalEnabled: true,
        rewardRate: 0.05, punishRate: 0.03
      )
      gServerState = initCognitiveState(gServerCfg)
      gServerHasState = true

proc handleChatServer(input: string): string {.gcsafe.} =
  {.cast(gcsafe).}:
    if not gServerHasState:
      initServerState()
    return gServerState.process(input)

proc handleRequestServer(req: Request) {.async, gcsafe.} =
  let path = req.url.path
  if req.reqMethod == HttpGet and path == "/health":
    await req.respond(Http200, "{\"status\":\"ok\"}", newHttpHeaders([("Content-Type","application/json")]))
    return
  if req.reqMethod == HttpPost and (path == "/lunatic/chat" or path == "/luna/chat" or path == "/chat" or path == "/v1/chat/completions"):
    var body = req.body
    var inputText = ""
    try:
      let j = parseJson(body)
      if j.hasKey("message"):
        inputText = j["message"].getStr()
      elif j.hasKey("input"):
        inputText = j["input"].getStr()
      elif j.hasKey("messages"):
        let msgs = j["messages"]
        if msgs.kind == JArray and msgs.len > 0:
          let last = msgs[msgs.len-1]
          if last.hasKey("content"):
            inputText = last["content"].getStr()
      elif j.hasKey("prompt"):
        inputText = j["prompt"].getStr()
      else:
        inputText = body
    except CatchableError:
      inputText = body
    if inputText.strip().len == 0:
      await req.respond(Http400, "{\"error\":\"empty input\"}", newHttpHeaders([("Content-Type","application/json")]))
      return
    let resp = handleChatServer(inputText)
    var outJson: JsonNode
    if path == "/v1/chat/completions":
      outJson = %*{
        "id": "chatcmpl-lunatic",
        "object": "chat.completion",
        "created": int(epochTime()),
        "model": "lunatic",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": resp}, "finish_reason": "stop"}]
      }
    else:
      outJson = %*{"response": resp}
    await req.respond(Http200, $outJson, newHttpHeaders([("Content-Type","application/json")]))
    return
  await req.respond(Http404, "{\"error\":\"not found\"}", newHttpHeaders([("Content-Type","application/json")]))

proc serverMain() {.async, gcsafe.} =
  {.cast(gcsafe).}:
    initServerState()
    var server = newAsyncHttpServer()
    echo "LunaticIntelligence server listening on port " & $gServerPort & " db=" & gServerDbPath
    await server.serve(Port(gServerPort), handleRequestServer)

type
  CliOptions* = object
    command*: string
    dbPath*: string
    dataPath*: string
    port*: int
    debugMode*: bool
    showHelpFlag*: bool
    showVersionFlag*: bool

proc showHelp*() =
  echo "LunaticIntelligence v" & VERSION & " - Cognitive Architecture"
  echo ""
  echo "Usage:"
  echo "  lunatic <command> [options]"
  echo ""
  echo "Commands:"
  echo "  observe            Learn from corpus (full training)"
  echo "  observe-inc        Incremental learning on existing DB"
  echo "  chat               Interactive chat"
  echo "  debug              Show cognitive state"
  echo "  sleep              Consolidation / decay"
  echo "  serve              Start HTTP server"
  echo ""
  echo "Options:"
  echo "  --db <path>     Database path (default: lunatic_cognitive.db)"
  echo "  --data <path>   Corpus path for observe"
  echo "  --port <n>      Port for serve (default: 8080)"
  echo "  --debug         Enable debug output"
  echo "  --help          Show help"
  echo "  --version       Show version"
  echo ""
  echo "Incremental learning:"
  echo "  ./src/lunatic observe-inc --data new_corpus.txt --db existing.db"
  echo ""
  echo "Environment compatibility:"
  echo "  MODE=observe|chat|debug|sleep DB=lunatic_cognitive.db OBSERVE_DATA=corpus.txt ./src/lunatic"
  echo "  ./src/lunatic observe --data corpus.txt --db lunatic_cognitive.db"

proc parseCli*(): CliOptions =
  result.command = ""
  result.dbPath = getEnv("DB", "lunatic_cognitive.db")
  if result.dbPath == "luna_cognitive.db":
    result.dbPath = "lunatic_cognitive.db"
  result.dataPath = getEnv("OBSERVE_DATA", "corpus_combined.txt")
  result.port = 8080
  if existsEnv("PORT"):
    try: result.port = parseInt(getEnv("PORT"))
    except: discard
  result.debugMode = false
  result.showHelpFlag = false
  result.showVersionFlag = false
  var positional: seq[string] = @[]
  let params = commandLineParams()
  var idx = 0
  while idx < params.len:
    let arg = params[idx]
    if arg == "--":
      inc idx
      continue
    elif arg == "--help" or arg == "-h":
      result.showHelpFlag = true
    elif arg == "--version" or arg == "-v":
      result.showVersionFlag = true
    elif arg == "--debug":
      result.debugMode = true
    elif arg.startsWith("--db="):
      result.dbPath = arg[5..^1]
    elif arg == "--db" and idx+1 < params.len:
      result.dbPath = params[idx+1]
      inc idx
    elif arg.startsWith("--data="):
      result.dataPath = arg[7..^1]
    elif arg == "--data" and idx+1 < params.len:
      result.dataPath = params[idx+1]
      inc idx
    elif arg.startsWith("--port="):
      try: result.port = parseInt(arg[7..^1])
      except: discard
    elif arg == "--port" and idx+1 < params.len:
      try: result.port = parseInt(params[idx+1])
      except: discard
      inc idx
    else:
      if not arg.startsWith("-"):
        positional.add(arg)
      elif arg == "--debug":
        result.debugMode = true
    inc idx
  if positional.len > 0:
    result.command = positional[0].toLower()
    for pos in positional:
      if pos == "--debug":
        result.debugMode = true
  if result.command.len == 0:
    let mode = getEnv("MODE", "")
    if mode.len > 0:
      result.command = mode.toLower()
      if paramCount() >= 1 and paramStr(1) == "--debug":
        result.debugMode = true
  if result.command.len == 0:
    result.command = "chat"
  if result.dbPath == "luna_cognitive.db" or result.dbPath == "luna_large.db":
    result.dbPath = result.dbPath.replace("luna_", "lunatic_")

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

proc runObserve(opts: CliOptions) =
  echo "=== Observe Mode: Corpus Learning ==="
  let dataPath = opts.dataPath
  if not fileExists(dataPath):
    echo "Data not found: " & dataPath
    quit(1)
  let fileSize = getFileSize(dataPath)
  let useStreaming = fileSize > 100_000_000
  if useStreaming:
    echo "Large corpus detected (" & $(fileSize div 1_000_000) & "MB), using streaming mode"
    # 32GB RAM活用: 大規模でも 2048 clausesで十分、8192はDB肥大化で60s目標に反する
    var cfg = CognitiveConfig(
      wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
      activationThreshold: 0.1, tmClauses: 2048, tmThreshold: 0.3,
      tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 10000, topKEpisodes: 3,
      thinkingEnabled: true, evalEnabled: true,
      rewardRate: 0.05, punishRate: 0.03
    )
    var state = initCognitiveState(cfg)
    state.observeCorpusStream(dataPath)
    echo "Saving..."
    var sdb = openStorage(opts.dbPath)
    sdb.saveConfig(cfg)
    sdb.saveConceptGraph(state.conceptGraph)
    sdb.saveTM(state.tm)
    sdb.saveSynapses(state.bridge)
    sdb.saveCatalog(state.catalog)
    sdb.savePhase(state.phase)
    sdb.close()
    echo "Saved to: " & opts.dbPath
    echo "Done!"
  else:
    let corpus = loadTrainingData(dataPath)
    if corpus.len == 0:
      echo "No data found in: " & dataPath
      quit(1)
    echo "Corpus: " & $corpus.len & " conversations"
    let tokCorpus = if corpus.len > 5000: corpus[0..<5000] else: corpus
    var tok = buildTokenizer(tokCorpus, 4096)
    echo "Tokenizer: " & $tok.vocab.len & " vocab"
    # 動的にTM規模を決定（60s目標 + 32GB RAM活用）
    let tmClausesDyn = if corpus.len < 500: 64 elif corpus.len < 5000: 256 elif corpus.len < 50000: 512 else: 1024
    let maxEpDyn = if corpus.len < 1000: 2000 elif corpus.len < 10000: 5000 else: 10000
    echo "Config: tmClauses=" & $tmClausesDyn & " maxEpisodes=" & $maxEpDyn & " (32GB RAM対応、60s目標で動的調整)"
    var cfg = CognitiveConfig(
      wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
      activationThreshold: 0.1, tmClauses: tmClausesDyn, tmThreshold: 0.3,
      tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: maxEpDyn, topKEpisodes: 3,
      thinkingEnabled: true, evalEnabled: true,
      rewardRate: 0.05, punishRate: 0.03
    )
    var state = initCognitiveState(cfg)
    state.tokenizer = tok
    state.observeCorpus(corpus)
    echo "Saving..."
    var sdb = openStorage(opts.dbPath)
    sdb.saveConfig(cfg)
    sdb.saveConceptGraph(state.conceptGraph)
    sdb.saveTM(state.tm)
    sdb.saveSynapses(state.bridge)
    sdb.saveCatalog(state.catalog)
    sdb.savePhase(state.phase)
    sdb.close()
    echo "Saved to: " & opts.dbPath
    echo "Done!"

proc runObserveIncremental(opts: CliOptions) =
  echo "=== Observe Mode: Incremental Learning ==="
  if not fileExists(opts.dbPath):
    echo "Database not found: " & opts.dbPath
    echo "Run normal observe first, then use incremental for additional data"
    quit(1)
  if not fileExists(opts.dataPath):
    echo "Data not found: " & opts.dataPath
    quit(1)
  
  let fileSize = getFileSize(opts.dataPath)
  let useStreaming = getFileSize(opts.dataPath) > 100_000_000
  
  # Load existing model
  echo "Loading existing model from: " & opts.dbPath
  var sdb = openStorage(opts.dbPath)
  let cfg = sdb.loadConfig()
  var state = initCognitiveState(cfg)
  state.conceptGraph = sdb.loadConceptGraph()
  sdb.loadTM(state.tm)
  state.bridge = sdb.loadSynapses()
  state.catalog = sdb.loadCatalog()
  state.phase = sdb.loadPhase()
  sdb.close()
  
  echo "Existing model loaded:"
  # トークナイザ辞書を概念グラフから構築
  var incTokCorpus: seq[string] = @[]
  for node in state.conceptGraph.nodes:
    if node.word.len >= 2: incTokCorpus.add(node.word)
  for entry in state.catalog.entries:
    if entry.inputText.len > 0: incTokCorpus.add(entry.inputText)
    if entry.outputText.len > 0: incTokCorpus.add(entry.outputText)
  if incTokCorpus.len > 0:
    state.tokenizer = buildTokenizer(incTokCorpus, 4096)
    echo "  Tokenizer: " & $state.tokenizer.vocab.len & " vocab"
  echo "  Concepts: " & $state.conceptGraph.conceptCount()
  echo "  Catalog: " & $state.catalog.entries.len & " entries"
  echo "  Synapses: " & $state.bridge.synapses.len
  echo "  Phase: " & $state.phase
  
  let corpus = loadTrainingData(opts.dataPath)
  if corpus.len == 0:
    echo "No data found in: " & opts.dataPath
    quit(1)
  
  echo "Additional corpus: " & $corpus.len & " conversations"
  
  # Learn incrementally
  if useStreaming:
    state.observeCorpusStream(opts.dataPath)
  else:
    # Extend tokenizer with new vocabulary
    let newTokCorpus = if corpus.len > 5000: corpus[0..<5000] else: corpus
    var newTok = buildTokenizer(newTokCorpus, 4096)
    # Merge vocabularies (simple approach: just use existing tokenizer)
    # For better incremental, we'd merge vocabularies
    state.observeCorpus(corpus)
  
  echo "Saving updated model..."
  var saveDb = openStorage(opts.dbPath)
  saveDb.saveConfig(cfg)
  saveDb.saveConceptGraph(state.conceptGraph)
  saveDb.saveTM(state.tm)
  saveDb.saveSynapses(state.bridge)
  saveDb.saveCatalog(state.catalog)
  saveDb.savePhase(state.phase)
  saveDb.close()
  echo "Saved to: " & opts.dbPath
  echo "Done! Model updated incrementally."

proc runChat(opts: CliOptions) =
  echo "=== Chat Mode ==="
  if not fileExists(opts.dbPath):
    echo "No model found at: " & opts.dbPath
    echo "Run observe first: ./src/lunatic observe --data corpus.txt --db " & opts.dbPath
    quit(1)
  var sdb = openStorage(opts.dbPath)
  let cfg = sdb.loadConfig()
  var state = initCognitiveState(cfg)
  state.conceptGraph = sdb.loadConceptGraph()
  sdb.loadTM(state.tm)
  state.bridge = sdb.loadSynapses()
  state.catalog = sdb.loadCatalog()
  state.phase = sdb.loadPhase()
  sdb.close()
  # トークナイザ辞書を概念グラフから構築（chat/observe-incで必須）
  var chatTokCorpus: seq[string] = @[]
  for node in state.conceptGraph.nodes:
    if node.word.len >= 2: chatTokCorpus.add(node.word)
  for entry in state.catalog.entries:
    if entry.inputText.len > 0: chatTokCorpus.add(entry.inputText)
    if entry.outputText.len > 0: chatTokCorpus.add(entry.outputText)
  if chatTokCorpus.len > 0:
    state.tokenizer = buildTokenizer(chatTokCorpus, 4096)
    echo "Tokenizer: " & $state.tokenizer.vocab.len & " vocab (from concept graph + catalog)"
  else:
    echo "Warning: empty corpus, tokenizer will only have PAD/UNK/EOS"
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
    var input: string
    try:
      input = stdin.readLine()
    except EOFError:
      break
    if input.strip() == "exit": break
    if input.strip().len == 0: continue
    let response = state.process(input)
    echo "Lunatic: " & (if response.len > 0: response else: "(empty)")
    if opts.debugMode and state.lastThinking.steps.len > 0:
      echo ""
      echo "  [Thinking]"
      for step in state.lastThinking.steps:
        echo "    " & $step.kind & ": " & step.description &
             " (conf=" & $formatFloat(step.confidence, ffDecimal, 3) & ")"
        for d in step.details:
          echo "      - " & d
      echo "    Total confidence: " & $formatFloat(state.lastThinking.totalConfidence, ffDecimal, 3)
    if opts.debugMode:
      echo "  [Intent] " & $state.lastIntent
      echo "  [Eval] " & state.lastEval.reason
      if state.lastEval.contradictions.len > 0:
        for c in state.lastEval.contradictions:
          echo "    Contradiction: " & c
      echo "  [Reward/Punish] total_reward=" & $formatFloat(state.totalReward, ffDecimal, 3) &
           " total_punish=" & $formatFloat(state.totalPunish, ffDecimal, 3)
      echo ""
    if state.episodeStore.episodes.len mod 10 == 0:
      var saveDb = openStorage(opts.dbPath)
      saveDb.saveConceptGraph(state.conceptGraph)
      saveDb.saveTM(state.tm)
      saveDb.saveSynapses(state.bridge)
      saveDb.saveCatalog(state.catalog)
      saveDb.close()

proc runDebug(opts: CliOptions) =
  echo "=== Debug: Cognitive State ==="
  if not fileExists(opts.dbPath):
    echo "No model found"
    quit(1)
  var sdb = openStorage(opts.dbPath)
  let cfg = sdb.loadConfig()
  var state = initCognitiveState(cfg)
  state.conceptGraph = sdb.loadConceptGraph()
  sdb.loadTM(state.tm)
  state.bridge = sdb.loadSynapses()
  state.catalog = sdb.loadCatalog()
  sdb.close()
  echo "Concepts: " & $state.conceptGraph.conceptCount()
  # トークナイザ辞書を構築
  var dbgTokCorpus: seq[string] = @[]
  for node in state.conceptGraph.nodes:
    if node.word.len >= 2: dbgTokCorpus.add(node.word)
  for entry in state.catalog.entries:
    if entry.inputText.len > 0: dbgTokCorpus.add(entry.inputText)
    if entry.outputText.len > 0: dbgTokCorpus.add(entry.outputText)
  if dbgTokCorpus.len > 0:
    state.tokenizer = buildTokenizer(dbgTokCorpus, 4096)
    echo "Tokenizer: " & $state.tokenizer.vocab.len & " vocab"
  echo "Edges: " & $state.conceptGraph.edgeCount()
  echo "Catalog: " & $state.catalog.entries.len & " entries"
  echo "Synapses: " & $state.bridge.synapses.len
  echo ""
  let testInputs = ["おはよう", "元気？", "何してる？", "日本語で話して", "ありがとう"]
  for input in testInputs:
    echo "--- Input: " & input & " ---"
    let response = state.process(input)
    echo "  Response: " & (if response.len > 0: response else: "(empty)")
    if state.lastThinking.steps.len > 0:
      echo "  [Thinking]"
      for step in state.lastThinking.steps:
        echo "    " & $step.kind & ": " & step.description &
             " (conf=" & $formatFloat(step.confidence, ffDecimal, 3) & ")"
      echo "    Total confidence: " & $formatFloat(state.lastThinking.totalConfidence, ffDecimal, 3)
    echo "  [Eval] " & state.lastEval.reason
    echo "  [Reward] " & $formatFloat(state.totalReward, ffDecimal, 3) &
         " [Punish] " & $formatFloat(state.totalPunish, ffDecimal, 3)
    echo ""

proc runSleep(opts: CliOptions) =
  echo "=== Sleep Mode: Consolidation ==="
  if not fileExists(opts.dbPath):
    echo "No model found"
    quit(1)
  var sdb = openStorage(opts.dbPath)
  let cfg = sdb.loadConfig()
  var state = initCognitiveState(cfg)
  state.conceptGraph = sdb.loadConceptGraph()
  state.bridge = sdb.loadSynapses()
  sdb.close()
  echo "Decaying synapses..."
  let now = epochTime()
  for syn in state.bridge.synapses.mitems:
    let elapsed = now - syn.lastActivated
    let days = elapsed / 86400.0
    syn.strength = syn.strength * pow(0.5, days.float32 / syn.halfLifeDays)
  let before = state.bridge.synapses.len
  state.bridge.synapses.keepItIf(it.strength > 0.01)
  echo "  Synapses: " & $before & " -> " & $state.bridge.synapses.len
  var saveDb = openStorage(opts.dbPath)
  saveDb.saveSynapses(state.bridge)
  saveDb.close()
  echo "Sleep complete."

proc runServer(opts: CliOptions) =
  gServerDbPath = opts.dbPath
  gServerPort = opts.port
  waitFor serverMain()

proc main() =
  echo "===================================="
  echo " LunaticIntelligence v" & VERSION & " - Cognitive Architecture"
  echo "===================================="
  echo " Concept Network + TM Reasoning + Hebbian Synapses"
  echo " No neural network. No GPU. Pure cognition."
  echo ""
  let opts = parseCli()
  if opts.showHelpFlag:
    showHelp()
    quit(0)
  if opts.showVersionFlag:
    echo "LunaticIntelligence v" & VERSION
    quit(0)
  case opts.command
  of "observe": runObserve(opts)
  of "observe-inc": runObserveIncremental(opts)
  of "chat": runChat(opts)
  of "debug": runDebug(opts)
  of "sleep": runSleep(opts)
  of "serve": runServer(opts)
  else:
    echo "Unknown command: " & opts.command
    showHelp()
    quit(1)

main()
