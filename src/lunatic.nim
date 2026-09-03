{.push warning[UnusedImport]: off, hint[XDeclaredButNotUsed]: off.}
import os, strutils, times, tables, sequtils, algorithm, math, unicode, parseopt, asyncdispatch, asynchttpserver, json, httpclient
import types, tokenizer, concept_graph, working_memory, tsetlin,
       generator, cognitive_loop, storage, llm, db_router, db_compress, intent_classifier

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

proc handleChatServer(input: string; systemPrompt: string = ""): string {.gcsafe.} =
  {.cast(gcsafe).}:
    if not gServerHasState:
      initServerState()
    if systemPrompt.len > 0:
      return gServerState.process(input, systemPrompt)
    else:
      let envPrompt = getEnv("SYSTEM_PROMPT", "")
      if envPrompt.len > 0:
        return gServerState.process(input, envPrompt)
      return gServerState.process(input)

proc handleRequestServer(req: Request) {.async, gcsafe.} =
  let path = req.url.path
  if req.reqMethod == HttpGet and path == "/health":
    await req.respond(Http200, "{\"status\":\"ok\"}", newHttpHeaders([("Content-Type","application/json")]))
    return
  if req.reqMethod == HttpPost and (path == "/lunatic/chat" or path == "/luna/chat" or path == "/chat" or path == "/v1/chat/completions"):
    var body = req.body
    var inputText = ""
    var systemPrompt = ""
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
      if j.hasKey("system"):
        systemPrompt = j["system"].getStr()
      elif j.hasKey("system_prompt"):
        systemPrompt = j["system_prompt"].getStr()
      elif j.hasKey("systemPrompt"):
        systemPrompt = j["systemPrompt"].getStr()
      elif j.hasKey("instruction"):
        systemPrompt = j["instruction"].getStr()
      if j.hasKey("messages") and j["messages"].kind == JArray:
        for m in j["messages"]:
          if m.hasKey("role") and m["role"].getStr() == "system" and m.hasKey("content"):
            systemPrompt = m["content"].getStr()
            break
    except CatchableError:
      inputText = body
    if inputText.strip().len == 0:
      await req.respond(Http400, "{\"error\":\"empty input\"}", newHttpHeaders([("Content-Type","application/json")]))
      return
    let resp = handleChatServer(inputText, systemPrompt)
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
    epochs*: int
    batchSize*: int
    throttleMs*: int

proc showHelp*() =
  echo "LunaticIntelligence v" & VERSION & " - Cognitive Architecture"
  echo ""
  echo "Usage:"
  echo "  lunatic <command> [options]"
  echo ""
  echo "Commands:"
  echo "  observe            Learn from corpus (dual-path: right brain concepts + left brain LLM)"
  echo "  observe-inc        Incremental learning on existing DB"
  echo "  train-llm          Train LLM on corpus (CPU throttled)"
  echo "  chat               Interactive chat"
  echo "  debug              Show cognitive state"
  echo "  sleep              Consolidation / decay"
  echo "  serve              Start HTTP server"
  echo ""
  echo "Options:"
  echo "  --db <path>        Database path (default: lunatic_cognitive.db)"
  echo "  --data <path>      Corpus path for observe/train-llm"
  echo "  --port <n>         Port for serve (default: 8080)"
  echo "  --debug            Enable debug output"
  echo "  --epochs <n>       Epochs for observe/train-llm (default: 3-5)"
  echo "  --batch-size <n>   Batch size for train-llm (default: 32)"
  echo "  --throttle <ms>    CPU throttle per batch for train-llm (default: 10)"
  echo "  --lr <float>       Learning rate for LLM (default: 0.001)"
  echo "  --help             Show help"
  echo "  --version          Show version"
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
  result.epochs = 3
  result.batchSize = 32
  result.throttleMs = 10
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
    elif arg == "--epochs" and idx+1 < params.len:
      try: result.epochs = parseInt(params[idx+1])
      except: discard
      inc idx
    elif arg == "--batch-size" and idx+1 < params.len:
      try: result.batchSize = parseInt(params[idx+1])
      except: discard
      inc idx
    elif arg == "--throttle" and idx+1 < params.len:
      try: result.throttleMs = parseInt(params[idx+1])
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

proc buildTokenizerFromCorpus(corpus: seq[string]; maxVocab: int = 4096): Tokenizer {.used.} =
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
  echo "=== Observe Mode: Dual-Path Corpus Learning (Right Brain + Left Brain) ==="
  let dataPath = opts.dataPath
  if not fileExists(dataPath):
    echo "Data not found: " & dataPath
    quit(1)
  let fileSize = getFileSize(dataPath)
  let useStreaming = fileSize > 100_000_000
  
  # 設定: デュアルパス学習用に最適化
  var cfg = CognitiveConfig(
    wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
    activationThreshold: 0.1, tmClauses: if useStreaming: 2048 else: 1024, tmThreshold: 0.3,
    tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 10000, topKEpisodes: 3,
    thinkingEnabled: true, evalEnabled: true,
    rewardRate: 0.05, punishRate: 0.03
  )
  var state = initCognitiveState(cfg)
  
  if useStreaming:
    echo "Large corpus detected (" & $(fileSize div 1_000_000) & "MB), using streaming dual-path mode"
  else:
    echo "Standard corpus, using dual-path mode"
    # トークナイザーは observeCorpusDualPath 内で構築される
  
  # デュアルパス統合学習実行（右脳: 概念+TM+Hebbian / 左脳: LLM次トークン予測）
  let epochs = if useStreaming: 3 else: 5
  state.observeCorpusDualPath(dataPath, opts.dbPath, maxEpochs=epochs, 
                               llmLearningRate=0.001f32, cpuThrottleMs=if useStreaming: 10 else: 5)
  
  # メタデータ保存（observeCorpusDualPathで重み保存済み、メタのみ追加）
  var sdb = openStorage(opts.dbPath)
  let roleGuess = if opts.dbPath.contains("nim"): "nim" elif opts.dbPath.contains("chat"): "chat" elif opts.dbPath.contains("code"): "code" else: "general"
  let desc = "Dual-Path Corpus: " & extractFilename(dataPath) & " concepts:" & $state.conceptGraph.conceptCount() & " catalog:" & $state.catalog.entries.len
  var meta = initTable[string,string]()
  meta["role"] = roleGuess
  meta["description"] = desc
  meta["tags"] = if roleGuess=="nim": "nim,code,proc,import" elif roleGuess=="chat": "chat,greeting,thanks,日常会話" else: "general,encyclopedia,qa"
  meta["concepts"] = $state.conceptGraph.conceptCount()
  meta["catalog"] = $state.catalog.entries.len
  meta["createdAt"] = $now()
  meta["sourceCorpus"] = dataPath
  meta["training_mode"] = "dual_path"
  meta["llm_trained"] = "true"
  sdb.saveDbMeta(meta)
  sdb.close()
  
  # 超圧縮を自動実行
  discard tryUltraCompress(opts.dbPath)
  echo "Saved to: " & opts.dbPath & " (meta role=" & roleGuess & " ultra=" & getUltraRatio(opts.dbPath) & ")"
  echo "Done! Dual-path training complete."

proc runObserveIncremental(opts: CliOptions) =
  echo "=== Observe Mode: Incremental Learning (Auto-Discover) ==="
  var dbPath = opts.dbPath
  if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
    if dirExists("knowledge"):
      for entry in walkDir("knowledge"):
        if entry.path.endsWith(".db"):
          dbPath = "knowledge/" & entry.path
          break
    if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
      echo "Database not found. Run normal observe first."
      quit(1)
  if not fileExists(opts.dataPath):
    echo "Data not found: " & opts.dataPath
    quit(1)
  
  let fileSize = getFileSize(opts.dataPath)
  let useStreaming = getFileSize(opts.dataPath) > 100_000_000
  
  # Load existing model
  echo "Loading existing model from: " & dbPath
  var sdb = openStorage(dbPath)
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
  
  # Learn incrementally with dual-path (right brain + left brain)
  let epochs = if useStreaming: 2 else: 3
  state.observeCorpusDualPath(opts.dataPath, dbPath, maxEpochs=epochs, 
                               llmLearningRate=0.001f32, cpuThrottleMs=if useStreaming: 10 else: 5)
  
  echo "Saved to: " & dbPath
  echo "Done! Model updated incrementally with dual-path learning."

proc loadStateFromPath(dbPath: string; useUltraDecompress: bool = true): CognitiveState =
  var actualPath = dbPath
  if useUltraDecompress and dbPath.endsWith(".ultra"):
    let tmp = dbPath.replace(".ultra", "")
    if fileExists(dbPath) and not fileExists(tmp):
      discard decompressDB(dbPath, tmp)
      actualPath = tmp
    elif fileExists(tmp):
      actualPath = tmp
  elif useUltraDecompress and fileExists(dbPath & ".ultra"):
    # 超圧縮版があれば自動解凍して使用（なければ通常DB）
    discard decompressDB(dbPath & ".ultra", "")
  var sdb = openStorage(actualPath)
  let cfg = sdb.loadConfig()
  result = initCognitiveState(cfg)
  result.llmDBPath = actualPath
  result.conceptGraph = sdb.loadConceptGraph()
  sdb.loadTM(result.tm)
  result.bridge = sdb.loadSynapses()
  result.catalog = sdb.loadCatalog()
  result.phase = sdb.loadPhase()
  # tokenizerは永続化されたLLM語彙を優先
  var tokLoaded = false
  try:
    let ltok = sdb.loadTokenizer()
    if ltok.vocab.len > 10:
      result.tokenizer = ltok
      tokLoaded = true
  except: discard
  if not tokLoaded:
    var lstore = openLLMWeightStore(actualPath)
    var ltok = Tokenizer(vocab: @[], tokenToId: initTable[string,int]())
    if loadLLMTokenizer(lstore, ltok):
      result.tokenizer = ltok
      tokLoaded = true
    closeLLMWeightStore(lstore)
  if not tokLoaded:
    var tokCorpus: seq[string] = @[]
    for node in result.conceptGraph.nodes:
      if node.word.len >= 2: tokCorpus.add(node.word)
    for entry in result.catalog.entries:
      if entry.inputText.len > 0: tokCorpus.add(entry.inputText)
      if entry.outputText.len > 0: tokCorpus.add(entry.outputText)
    if tokCorpus.len > 0:
      result.tokenizer = buildTokenizer(tokCorpus, 4096)
  sdb.close()

proc runChat(opts: CliOptions) =
  echo "=== Chat Mode (Auto-Discover DBs from knowledge/) ==="
  # ルーター初期化: --db が指定されていない場合は knowledge/ フォルダから自動検出
  var basePath = opts.dbPath
  if basePath == "lunatic_cognitive.db" and not fileExists(basePath):
    # デフォルトDBが存在しない場合、knowledge/ から探索
    if dirExists("knowledge"):
      basePath = "knowledge"
      echo "No default DB found. Auto-discovering from knowledge/ folder..."
    else:
      echo "No model found. Run observe first: ./src/lunatic observe --data corpus.txt --db lunatic_cognitive.db"
      quit(1)
  
  var router = initRouter(basePath)
  let available = router.getAllAvailableRoles()
  if available.len == 0:
    echo "No model found at: " & basePath & " nor in knowledge/ folder"
    echo "Run observe first: ./src/lunatic observe --data corpus.txt --db lunatic_cognitive.db"
    quit(1)
  echo "Router available DBs:"
  for role in available:
    if role == drCustom: continue
    let p = router.roleToPath[role]
    if p.len==0 or not fileExists(p): continue
    let sz = getFileSize(p)
    echo "  " & $role & " -> " & p & " (" & $(sz div 1024) & "KB)"
  for path, meta in router.customPathToMeta.pairs:
    if fileExists(path) and path notin router.roleToPath.values.toSeq:
      echo "  custom:" & meta.role & " -> " & path & " (" & $(getFileSize(path) div 1024) & "KB) tags:" & meta.tags.join(",")
  # デフォルト状態（general または最初の利用可能DB）
  var states: Table[DbRole, CognitiveState]
  var cfgs: Table[DbRole, CognitiveConfig]
  for role in available:
    if role == drCustom: continue
    let p = router.roleToPath[role]
    if p.len==0 or not fileExists(p): continue
    try:
      let st = loadStateFromPath(p, true)
      states[role] = st
      cfgs[role] = st.cfg
      echo "Loaded " & $role & ": " & $st.conceptGraph.conceptCount() & " concepts, " & $st.catalog.entries.len & " entries"
    except CatchableError as e:
      echo "Failed to load " & $role & ": " & e.msg
  # custom DBは遅延ロード（フォルダに入れるだけで使われる）
  # フォールバック用
  var currentRole = if drGeneral in states: drGeneral else: available[0]
  var state = states[currentRole]
  var intentClassifier = initIntentClassifier()
  let cliSystemPrompt = getEnv("SYSTEM_PROMPT", "")
  if cliSystemPrompt.len > 0:
    echo "System prompt: " & cliSystemPrompt
  echo ""
  echo "Router active. Inputに応じて LunaticがDBを自動切替します。"
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
    # Lunatic判断でDB切替（フォルダ内のDBは入れるだけで自動認識）
    let sel = router.getStorageForInput(input, intentClassifier)
    var targetRole = sel.role
    var targetPath = sel.path
    # customは遅延ロード
    if targetRole == drCustom and targetPath.len>0 and targetPath notin router.roleToPath.values.toSeq:
      if drCustom notin states:
        try:
          let st = loadStateFromPath(targetPath, true)
          states[drCustom] = st
          echo "  [Router] lazy load custom: " & targetPath & " (" & $st.conceptGraph.conceptCount() & " concepts)"
        except CatchableError as e:
          echo "  [Router] failed to load custom " & targetPath & ": " & e.msg
          targetRole = currentRole
          targetPath = router.roleToPath.getOrDefault(currentRole, "")
    if targetRole in states and targetRole != currentRole:
      echo "  [Router] " & $currentRole & " -> " & $targetRole & " (" & targetPath & ")"
      currentRole = targetRole
      state = states[currentRole]
    elif targetRole == drCustom and drCustom in states and currentRole != drCustom:
      echo "  [Router] " & $currentRole & " -> custom (" & targetPath & ")"
      currentRole = drCustom
      state = states[currentRole]
    let response = if cliSystemPrompt.len > 0: state.process(input, cliSystemPrompt) else: state.process(input)
    states[currentRole] = state
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
      let savePath = router.roleToPath.getOrDefault(currentRole, opts.dbPath)
      var saveDb = openStorage(savePath)
      saveDb.saveConceptGraph(state.conceptGraph)
      saveDb.saveTM(state.tm)
      saveDb.saveSynapses(state.bridge)
      saveDb.saveCatalog(state.catalog)
      saveDb.close()
      # 超圧縮が有効ならバックグラウンドで再圧縮（失敗しても無視）
      if getFileSize(savePath) > 5*1024*1024:
        discard tryUltraCompress(savePath)

proc runDebug(opts: CliOptions) =
  echo "=== Debug: Cognitive State (Auto-Discover) ==="
  var dbPath = opts.dbPath
  if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
    if dirExists("knowledge"):
      # knowledge/ から最初のDBを探す
      for entry in walkDir("knowledge"):
        if entry.path.endsWith(".db"):
          dbPath = "knowledge/" & entry.path
          break
    if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
      echo "No model found. Run observe first."
      quit(1)
  
  var sdb = openStorage(dbPath)
  let cfg = sdb.loadConfig()
  var state = initCognitiveState(cfg)
  state.llmDBPath = dbPath
  state.conceptGraph = sdb.loadConceptGraph()
  sdb.loadTM(state.tm)
  state.bridge = sdb.loadSynapses()
  state.catalog = sdb.loadCatalog()
  # 永続化されたトークナイザを優先して復元（LLM語彙と一致させる）
  var loadedTok = false
  try:
    let ltok = sdb.loadTokenizer()
    if ltok.vocab.len > 10:
      state.tokenizer = ltok
      loadedTok = true
  except: discard
  if not loadedTok:
    # LLM専用語彙を試す
    var lstore = openLLMWeightStore(dbPath)
    var ltok = Tokenizer(vocab: @[], tokenToId: initTable[string,int]())
    if loadLLMTokenizer(lstore, ltok):
      state.tokenizer = ltok
      loadedTok = true
    closeLLMWeightStore(lstore)
  sdb.close()
  echo "Concepts: " & $state.conceptGraph.conceptCount()
  # フォールバック: 概念グラフから構築（旧DB互換）
  if not loadedTok:
    var dbgTokCorpus: seq[string] = @[]
    for node in state.conceptGraph.nodes:
      if node.word.len >= 2: dbgTokCorpus.add(node.word)
    for entry in state.catalog.entries:
      if entry.inputText.len > 0: dbgTokCorpus.add(entry.inputText)
      if entry.outputText.len > 0: dbgTokCorpus.add(entry.outputText)
    if dbgTokCorpus.len > 0:
      state.tokenizer = buildTokenizer(dbgTokCorpus, 4096)
  echo "Tokenizer: " & $state.tokenizer.vocab.len & " vocab (from " & (if loadedTok: "persisted LLM vocab" else: "concept graph") & ")"
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
  echo "=== Sleep Mode: Consolidation (Auto-Discover) ==="
  var dbPath = opts.dbPath
  if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
    if dirExists("knowledge"):
      for entry in walkDir("knowledge"):
        if entry.path.endsWith(".db"):
          dbPath = "knowledge/" & entry.path
          break
    if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
      echo "No model found. Run observe first."
      quit(1)
  
  var sdb = openStorage(dbPath)
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

proc runTrainLLM(opts: CliOptions) =
  echo "=== Train-LLM Mode (same dataset, same DB, throttled) ==="
  if not fileExists(opts.dataPath):
    echo "Data not found: " & opts.dataPath
    quit(1)
  # 同一DBを使用・WAL有効・バッチ毎sleepでCPU/メモリ抑制
  # nice値が低い場合は自動でthrottle強化
  let effThrottle = if opts.throttleMs < 0: 0 else: opts.throttleMs
  echo "Dataset: " & opts.dataPath & " -> DB: " & opts.dbPath
  echo "Params: epochs=" & $opts.epochs & " batch=" & $opts.batchSize & " throttle=" & $effThrottle & "ms"
  # 既存DBがあればそのまま追記学習、なければ新規
  llm.trainLLM(opts.dataPath, opts.dbPath, maxEpochs=opts.epochs, batchSize=opts.batchSize, cpuThrottleMs=effThrottle)
  echo "train-llm done. Single file DB: " & opts.dbPath

proc runServer(opts: CliOptions) =
  var dbPath = opts.dbPath
  if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
    if dirExists("knowledge"):
      for entry in walkDir("knowledge"):
        if entry.path.endsWith(".db"):
          dbPath = "knowledge/" & entry.path
          break
    if dbPath == "lunatic_cognitive.db" and not fileExists(dbPath):
      echo "No model found. Run observe first."
      quit(1)
  
  gServerDbPath = dbPath
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
  of "train-llm": runTrainLLM(opts)
  of "chat": runChat(opts)
  of "debug": runDebug(opts)
  of "sleep": runSleep(opts)
  of "serve": runServer(opts)
  else:
    echo "Unknown command: " & opts.command
    showHelp()
    quit(1)

main()
{.pop.}
