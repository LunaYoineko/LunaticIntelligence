import json, strutils, os, times, tables
import types, concept_graph, storage, cognitive_loop

var gState: CognitiveState
var gCfg: CognitiveConfig
var gDbPath = "lunatic_cognitive.db"
var gInitialized = false

proc initMcpState(dbPath: string) =
  gDbPath = dbPath
  if fileExists(dbPath):
    var sdb = openStorage(dbPath)
    gCfg = sdb.loadConfig()
    gState = initCognitiveState(gCfg)
    gState.conceptGraph = sdb.loadConceptGraph()
    sdb.loadTM(gState.tm)
    gState.bridge = sdb.loadSynapses()
    gState.catalog = sdb.loadCatalog()
    gState.phase = sdb.loadPhase()
    sdb.close()
  else:
    gCfg = CognitiveConfig(
      wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
      activationThreshold: 0.1, tmClauses: 64, tmThreshold: 0.3,
      tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
      thinkingEnabled: true, evalEnabled: true,
      rewardRate: 0.05, punishRate: 0.03
    )
    gState = initCognitiveState(gCfg)
  gInitialized = true

proc handleTool(name: string; args: JsonNode): JsonNode =
  let tool = name.replace("luna_", "lunatic_")
  if tool == "lunatic_process" or tool == "lunatic_chat":
    var input = ""
    if args.hasKey("input"): input = args["input"].getStr()
    elif args.hasKey("message"): input = args["message"].getStr()
    elif args.hasKey("text"): input = args["text"].getStr()
    if not gInitialized: initMcpState(gDbPath)
    let resp = gState.process(input)
    return %*{"content": [{"type": "text", "text": resp}]}
  elif tool == "lunatic_status":
    if not gInitialized: initMcpState(gDbPath)
    return %*{"content": [{"type": "text", "text": "Concepts: " & $gState.conceptGraph.conceptCount() & " Catalog: " & $gState.catalog.entries.len}]}
  elif tool == "lunatic_observe":
    var dataPath = ""
    if args.hasKey("data"): dataPath = args["data"].getStr()
    elif args.hasKey("path"): dataPath = args["path"].getStr()
    if dataPath.len == 0:
      return %*{"content": [{"type": "text", "text": "missing data path"}]}
    if not gInitialized: initMcpState(gDbPath)
    if fileExists(dataPath):
      let sz = getFileSize(dataPath)
      if sz > 100_000_000:
        gState.observeCorpusStream(dataPath)
      else:
        var corpus: seq[string] = @[]
        for line in dataPath.lines:
          let t = line.strip()
          if t.len > 0: corpus.add(t)
        gState.observeCorpus(corpus)
      var sdb = openStorage(gDbPath)
      sdb.saveConfig(gCfg)
      sdb.saveConceptGraph(gState.conceptGraph)
      sdb.saveTM(gState.tm)
      sdb.saveSynapses(gState.bridge)
      sdb.saveCatalog(gState.catalog)
      sdb.savePhase(gState.phase)
      sdb.close()
      return %*{"content": [{"type": "text", "text": "observed " & dataPath}]}
    else:
      return %*{"content": [{"type": "text", "text": "file not found: " & dataPath}]}
  else:
    return %*{"content": [{"type": "text", "text": "unknown tool: " & name}]}

proc main() =
  var dbPath = getEnv("DB", "lunatic_cognitive.db")
  for i in 1..paramCount():
    let p = paramStr(i)
    if p.startsWith("--db="): dbPath = p[5..^1]
    elif p == "--db" and i < paramCount(): dbPath = paramStr(i+1)
  initMcpState(dbPath)
  while true:
    var line: string
    try:
      line = stdin.readLine()
    except EOFError:
      break
    if line.strip().len == 0: continue
    var req: JsonNode
    try:
      req = parseJson(line)
    except CatchableError:
      continue
    var id = req.getOrDefault("id")
    var methodName = ""
    if req.hasKey("method"): methodName = req["method"].getStr()
    var params = newJObject()
    if req.hasKey("params"): params = req["params"]
    var resultNode = newJObject()
    if methodName == "initialize":
      resultNode = %*{
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "lunatic", "version": "0.1.0"}
      }
    elif methodName == "tools/list":
      resultNode = %*{
        "tools": [
          {"name": "lunatic_process", "description": "Process input through LunaticIntelligence", "inputSchema": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}},
          {"name": "lunatic_status", "description": "Get cognitive state status", "inputSchema": {"type": "object", "properties": {}}},
          {"name": "lunatic_observe", "description": "Observe corpus", "inputSchema": {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]}},
          {"name": "luna_process", "description": "Alias for lunatic_process", "inputSchema": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}},
          {"name": "luna_status", "description": "Alias", "inputSchema": {"type": "object", "properties": {}}},
          {"name": "luna_observe", "description": "Alias", "inputSchema": {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]}}
        ]
      }
    elif methodName == "tools/call":
      var toolName = ""
      var args = newJObject()
      if params.hasKey("name"): toolName = params["name"].getStr()
      if params.hasKey("arguments"): args = params["arguments"]
      resultNode = handleTool(toolName, args)
    else:
      resultNode = %*{"content": [{"type": "text", "text": "unknown method: " & methodName}]}
    var resp = %*{"jsonrpc": "2.0", "id": id, "result": resultNode}
    echo $resp
    flushFile(stdout)

when isMainModule:
  main()
