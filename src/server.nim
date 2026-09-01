import asyncdispatch, asynchttpserver, json, strutils, os, times, tables
import types, storage, cognitive_loop

var globalState: CognitiveState
var globalCfg: CognitiveConfig
var dbPathGlobal = "lunatic_cognitive.db"
var hasState = false

proc initState() =
  if fileExists(dbPathGlobal):
    var sdb = openStorage(dbPathGlobal)
    globalCfg = sdb.loadConfig()
    globalState = initCognitiveState(globalCfg)
    globalState.conceptGraph = sdb.loadConceptGraph()
    sdb.loadTM(globalState.tm)
    globalState.bridge = sdb.loadSynapses()
    globalState.catalog = sdb.loadCatalog()
    globalState.phase = sdb.loadPhase()
    sdb.close()
    hasState = true
  else:
    globalCfg = CognitiveConfig(
      wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
      activationThreshold: 0.1, tmClauses: 64, tmThreshold: 0.3,
      tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
      thinkingEnabled: true, evalEnabled: true,
      rewardRate: 0.05, punishRate: 0.03
    )
    globalState = initCognitiveState(globalCfg)
    hasState = true

proc handleChat(input: string): string {.gcsafe.} =
  {.cast(gcsafe).}:
    if not hasState:
      initState()
    return globalState.process(input)

proc handleRequest(req: Request) {.async, gcsafe.} =
  let path = req.url.path
  if req.reqMethod == HttpGet and path == "/health":
    await req.respond(Http200, """{"status":"ok"}""", newHttpHeaders([("Content-Type","application/json")]))
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
          elif last.hasKey("text"):
            inputText = last["text"].getStr()
      elif j.hasKey("prompt"):
        inputText = j["prompt"].getStr()
      else:
        inputText = body
    except CatchableError:
      inputText = body
    if inputText.strip().len == 0:
      await req.respond(Http400, """{"error":"empty input"}""", newHttpHeaders([("Content-Type","application/json")]))
      return
    let resp = handleChat(inputText)
    var outJson: JsonNode
    if path == "/v1/chat/completions":
      outJson = %*{
        "id": "chatcmpl-lunatic",
        "object": "chat.completion",
        "created": int(epochTime()),
        "model": "lunatic",
        "choices": [
          {
            "index": 0,
            "message": {"role": "assistant", "content": resp},
            "finish_reason": "stop"
          }
        ]
      }
    else:
      outJson = %*{"response": resp, "output": resp}
    await req.respond(Http200, $outJson, newHttpHeaders([("Content-Type","application/json")]))
    return
  await req.respond(Http404, """{"error":"not found"}""", newHttpHeaders([("Content-Type","application/json")]))

proc main() {.async, gcsafe.} =
  var port = 8080
  for i in 1..paramCount():
    let p = paramStr(i)
    if p.startsWith("--port="):
      port = parseInt(p[7..^1])
    elif p.startsWith("--db="):
      dbPathGlobal = p[5..^1]
    elif p == "--port" and i < paramCount():
      port = parseInt(paramStr(i+1))
    elif p == "--db" and i < paramCount():
      dbPathGlobal = paramStr(i+1)
  if existsEnv("DB"):
    dbPathGlobal = getEnv("DB")
  if existsEnv("PORT"):
    try: port = parseInt(getEnv("PORT"))
    except: discard
  initState()
  var server = newAsyncHttpServer()
  echo "LunaticIntelligence server listening on port " & $port & " db=" & dbPathGlobal
  await server.serve(Port(port), handleRequest)

when isMainModule:
  waitFor main()
