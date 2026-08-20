import db_connector/db_sqlite, strutils, sequtils, algorithm, math, tables, times
import types, tokenizer, tsetlin, concept_graph, working_memory

type
  StorageDB* = object
    db*: DbConn

proc openStorage*(path: string): StorageDB =
  result.db = open(path, "", "", "")
  result.db.exec(sql"PRAGMA journal_mode=WAL")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS model_config (key TEXT PRIMARY KEY, value TEXT)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS tokenizer_vocab (id INTEGER, token TEXT)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS concept_nodes (id INTEGER PRIMARY KEY, word TEXT, category INTEGER, base_frequency REAL, access_count INTEGER)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS concept_edges (from_id INTEGER, to_id INTEGER, relation INTEGER, weight REAL, hebbian_count INTEGER)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS tm_states (layer_idx INTEGER, clause_idx INTEGER, states BLOB)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS synapses (clause_id INTEGER, concept_id INTEGER, strength REAL, activation_count INTEGER, last_activated REAL, half_life_days REAL)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS response_catalog (intent INTEGER, keyword TEXT, input_text TEXT, output_text TEXT, weight REAL)")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS working_memory_items (concept_id INTEGER, activation REAL, source TEXT)")

proc close*(sdb: StorageDB) =
  sdb.db.close()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
proc saveConfig*(sdb: StorageDB; cfg: CognitiveConfig) =
  sdb.db.exec(sql"DELETE FROM model_config")
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "wmCapacity", $cfg.wmCapacity)
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "spreadSteps", $cfg.spreadSteps)
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "spreadDecay", $formatFloat(cfg.spreadDecay, ffDecimal, 6))
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "activationThreshold", $formatFloat(cfg.activationThreshold, ffDecimal, 6))
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "tmClauses", $cfg.tmClauses)
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "tmThreshold", $formatFloat(cfg.tmThreshold, ffDecimal, 6))
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "tmSParam", $formatFloat(cfg.tmSParam, ffDecimal, 6))
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "halfLifeDays", $formatFloat(cfg.halfLifeDays, ffDecimal, 6))
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "maxEpisodes", $cfg.maxEpisodes)
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "topKEpisodes", $cfg.topKEpisodes)
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "thinkingEnabled", if cfg.thinkingEnabled: "1" else: "0")
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "evalEnabled", if cfg.evalEnabled: "1" else: "0")
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "rewardRate", $formatFloat(cfg.rewardRate, ffDecimal, 6))
  sdb.db.exec(sql"INSERT INTO model_config VALUES (?, ?)", "punishRate", $formatFloat(cfg.punishRate, ffDecimal, 6))

proc loadConfig*(sdb: StorageDB): CognitiveConfig =
  result = CognitiveConfig(
    wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
    activationThreshold: 0.1, tmClauses: 64, tmThreshold: 0.3,
    tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
    thinkingEnabled: true, evalEnabled: true,
    rewardRate: 0.05, punishRate: 0.03
  )
  for row in sdb.db.fastRows(sql"SELECT key, value FROM model_config"):
    let key = row[0]
    let val = row[1]
    case key
    of "wmCapacity": result.wmCapacity = parseInt(val)
    of "spreadSteps": result.spreadSteps = parseInt(val)
    of "spreadDecay": result.spreadDecay = parseFloat(val)
    of "activationThreshold": result.activationThreshold = parseFloat(val)
    of "tmClauses": result.tmClauses = parseInt(val)
    of "tmThreshold": result.tmThreshold = parseFloat(val)
    of "tmSParam": result.tmSParam = parseFloat(val)
    of "halfLifeDays": result.halfLifeDays = parseFloat(val)
    of "maxEpisodes": result.maxEpisodes = parseInt(val)
    of "topKEpisodes": result.topKEpisodes = parseInt(val)
    of "thinkingEnabled": result.thinkingEnabled = val == "1" or val == "true"
    of "evalEnabled": result.evalEnabled = val == "1" or val == "true"
    of "rewardRate": result.rewardRate = parseFloat(val)
    of "punishRate": result.punishRate = parseFloat(val)

  # 未設定のフィールドにデフォルト値を設定
  if result.wmCapacity == 0: result.wmCapacity = 7
  if result.spreadSteps == 0: result.spreadSteps = 3
  if result.spreadDecay == 0: result.spreadDecay = 0.5
  if result.activationThreshold == 0: result.activationThreshold = 0.1
  if result.tmClauses == 0: result.tmClauses = 64
  if result.tmThreshold == 0: result.tmThreshold = 8.0
  if result.tmSParam == 0: result.tmSParam = 3.0
  if result.halfLifeDays == 0: result.halfLifeDays = 7.0
  if result.maxEpisodes == 0: result.maxEpisodes = 5000
  if result.topKEpisodes == 0: result.topKEpisodes = 3
  if result.rewardRate == 0: result.rewardRate = 0.05
  if result.punishRate == 0: result.punishRate = 0.03
  # thinkingEnabled/evalEnabledはデフォルトtrue

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
proc saveTokenizer*(sdb: StorageDB; tok: Tokenizer) =
  sdb.db.exec(sql"DELETE FROM tokenizer_vocab")
  for i in 0..<tok.vocab.len:
    sdb.db.exec(sql"INSERT INTO tokenizer_vocab VALUES (?, ?)", i, tok.vocab[i])

proc loadTokenizer*(sdb: StorageDB): Tokenizer =
  result.vocab = @[]
  result.tokenToId = initTable[string, int]()
  for row in sdb.db.fastRows(sql"SELECT id, token FROM tokenizer_vocab"):
    let id = parseInt(row[0])
    let token = row[1]
    while result.vocab.len <= id:
      result.vocab.add("")
    result.vocab[id] = token
  for i in 0..<result.vocab.len:
    if result.vocab[i].len > 0:
      result.tokenToId[result.vocab[i]] = i

# ---------------------------------------------------------------------------
# ConceptGraph
# ---------------------------------------------------------------------------
proc saveConceptGraph*(sdb: StorageDB; graph: ConceptGraph) =
  sdb.db.exec(sql"DELETE FROM concept_nodes")
  sdb.db.exec(sql"BEGIN")
  for node in graph.nodes:
    sdb.db.exec(sql"INSERT INTO concept_nodes VALUES (?, ?, ?, ?, ?)",
      node.id, node.word, node.category.ord, node.baseFrequency, node.accessCount)
  sdb.db.exec(sql"COMMIT")
  sdb.db.exec(sql"DELETE FROM concept_edges")
  sdb.db.exec(sql"BEGIN")
  for edge in graph.edges:
    sdb.db.exec(sql"INSERT INTO concept_edges VALUES (?, ?, ?, ?, ?)",
      edge.fromId, edge.toId, edge.relation.ord, edge.weight, edge.hebbianCount)
  sdb.db.exec(sql"COMMIT")

proc loadConceptGraph*(sdb: StorageDB): ConceptGraph =
  result = initConceptGraph()
  for row in sdb.db.fastRows(sql"SELECT id, word, category, base_frequency, access_count FROM concept_nodes"):
    let id = parseInt(row[0])
    let word = row[1]
    let cat = ConceptType(parseInt(row[2]))
    let freq = parseFloat(row[3])
    let acc = parseInt(row[4])
    let nodeId = result.addNode(word, cat)
    result.nodes[nodeId].baseFrequency = freq.float32
    result.nodes[nodeId].accessCount = acc

  for row in sdb.db.fastRows(sql"SELECT from_id, to_id, relation, weight, hebbian_count FROM concept_edges"):
    let fromId = parseInt(row[0])
    let toId = parseInt(row[1])
    let rel = EdgeRelation(parseInt(row[2]))
    let weight = parseFloat(row[3])
    let hc = parseInt(row[4])
    if fromId < result.nodes.len and toId < result.nodes.len:
      let edgeId = result.edges.len
      result.edges.add(ConceptEdge(
        fromId: fromId, toId: toId, relation: rel,
        weight: weight.float32, hebbianCount: hc,
        lastCoactivated: epochTime()
      ))
      result.adjacency.mgetOrPut(fromId, @[]).add(edgeId)
      result.edgeSet[(fromId, toId)] = edgeId

# ---------------------------------------------------------------------------
# TM
# ---------------------------------------------------------------------------
proc saveTM*(sdb: StorageDB; tm: HierarchicalTM) =
  sdb.db.exec(sql"DELETE FROM tm_states")
  for li in 0..<tm.layers.len:
    let states = tm.layers[li].states
    var blob = newString(states.len)
    for i in 0..<states.len:
      blob[i] = char(states[i] + 128)
    sdb.db.exec(sql"INSERT INTO tm_states VALUES (?, ?, ?)", li, 0, blob)

proc loadTM*(sdb: StorageDB; tm: var HierarchicalTM) =
  for row in sdb.db.fastRows(sql"SELECT layer_idx, states FROM tm_states"):
    let li = parseInt(row[0])
    if li < tm.layers.len:
      let blob = row[1]
      for i in 0..<min(blob.len, tm.layers[li].states.len):
        tm.layers[li].states[i] = int8(blob[i].ord - 128)

# ---------------------------------------------------------------------------
# Synapses
# ---------------------------------------------------------------------------
proc saveSynapses*(sdb: StorageDB; bridge: SynapticBridge) =
  sdb.db.exec(sql"DELETE FROM synapses")
  for syn in bridge.synapses:
    sdb.db.exec(sql"INSERT INTO synapses VALUES (?, ?, ?, ?, ?, ?)",
      syn.clauseId, syn.conceptId, syn.strength,
      syn.activationCount, syn.lastActivated, syn.halfLifeDays)

proc loadSynapses*(sdb: StorageDB): SynapticBridge =
  result = SynapticBridge(synapses: @[], halfLifeDays: 7.0)
  for row in sdb.db.fastRows(sql"SELECT clause_id, concept_id, strength, activation_count, last_activated, half_life_days FROM synapses"):
    result.synapses.add(Synapse(
      clauseId: parseInt(row[0]),
      conceptId: parseInt(row[1]),
      strength: parseFloat(row[2]).float32,
      activationCount: parseInt(row[3]),
      lastActivated: parseFloat(row[4]),
      halfLifeDays: parseFloat(row[5]).float32
    ))

# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------
proc saveEpisodes*(sdb: StorageDB; store: EpisodeStore) =
  sdb.db.exec(sql"DELETE FROM episodes")
  sdb.db.exec(sql"BEGIN")
  for ep in store.episodes:
    var inputCids = ""
    for cid in ep.inputConceptIds:
      if inputCids.len > 0: inputCids.add(",")
      inputCids.add($cid)
    var outputCids = ""
    for cid in ep.outputConceptIds:
      if outputCids.len > 0: outputCids.add(",")
      outputCids.add($cid)
    var clausePat = ""
    for b in ep.tmClausePattern:
      clausePat.add(if b: "1" else: "0")
    sdb.db.exec(sql"INSERT INTO episodes (input_text, output_text, input_concept_ids, output_concept_ids, tm_clause_pattern, confidence, speaker, context_tag, situation, timestamp, reward, rank_val, access_count, emotional_valence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      ep.inputText, ep.outputText, inputCids, outputCids, clausePat,
      ep.confidence, $ep.speaker, ep.contextTag, ep.situation,
      ep.timestamp, ep.reward, ep.rank, ep.accessCount, ep.emotionalValence)
  sdb.db.exec(sql"COMMIT")

proc loadEpisodes*(sdb: StorageDB): seq[Episode] =
  result = @[]
  for row in sdb.db.fastRows(sql"SELECT input_text, output_text, input_concept_ids, output_concept_ids, tm_clause_pattern, confidence, speaker, context_tag, situation, timestamp, reward, rank_val, access_count, emotional_valence FROM episodes"):
    var inputCids: seq[int] = @[]
    if row[2].len > 0:
      for part in row[2].split(","):
        if part.len > 0: inputCids.add(parseInt(part))
    var outputCids: seq[int] = @[]
    if row[3].len > 0:
      for part in row[3].split(","):
        if part.len > 0: outputCids.add(parseInt(part))
    var clausePat: seq[bool] = @[]
    for ch in row[4]:
      clausePat.add(ch == '1')
    var speaker = spSystem
    if row[6] == "user": speaker = spUser
    result.add(Episode(
      inputText: row[0],
      outputText: row[1],
      inputConceptIds: inputCids,
      outputConceptIds: outputCids,
      tmClausePattern: clausePat,
      confidence: parseFloat(row[5]).float32,
      speaker: speaker,
      contextTag: row[7],
      situation: row[8],
      timestamp: parseFloat(row[9]),
      reward: parseFloat(row[10]).float32,
      rank: parseFloat(row[11]).float32,
      accessCount: parseInt(row[12]),
      emotionalValence: parseFloat(row[13]).float32
    ))

# ---------------------------------------------------------------------------
# Working Memory (restore)
# ---------------------------------------------------------------------------
proc saveWorkingMemory*(sdb: StorageDB; wm: WorkingMemory) =
  sdb.db.exec(sql"DELETE FROM working_memory_items")
  for item in wm.items:
    sdb.db.exec(sql"INSERT INTO working_memory_items VALUES (?, ?, ?)",
      item.conceptId, item.activation, item.source)

proc loadWorkingMemory*(sdb: StorageDB; capacity: int = 7): WorkingMemory =
  result = initWorkingMemory(capacity)
  for row in sdb.db.fastRows(sql"SELECT concept_id, activation, source FROM working_memory_items"):
    result.items.add(WMItem(
      conceptId: parseInt(row[0]),
      activation: parseFloat(row[1]).float32,
      timestamp: epochTime(),
      source: row[2]
    ))

# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------
proc savePhase*(sdb: StorageDB; phase: int) =
  sdb.db.exec(sql"INSERT OR REPLACE INTO model_config VALUES (?, ?)", "phase", $phase)

proc loadPhase*(sdb: StorageDB): int =
  result = 0
  for row in sdb.db.fastRows(sql"SELECT value FROM model_config WHERE key='phase'"):
    result = parseInt(row[0])

# ---------------------------------------------------------------------------
# ResponseCatalog
# ---------------------------------------------------------------------------
proc saveCatalog*(sdb: StorageDB; catalog: ResponseCatalog) =
  sdb.db.exec(sql"DELETE FROM response_catalog")
  sdb.db.exec(sql"BEGIN")
  for entry in catalog.entries:
    sdb.db.exec(sql"INSERT INTO response_catalog VALUES (?, ?, ?, ?, ?)",
      entry.intent.ord, entry.keyword, entry.inputText, entry.outputText, entry.weight)
  sdb.db.exec(sql"COMMIT")

proc loadCatalog*(sdb: StorageDB): ResponseCatalog =
  result = ResponseCatalog(entries: @[])
  for row in sdb.db.fastRows(sql"SELECT intent, keyword, input_text, output_text, weight FROM response_catalog"):
    result.entries.add(CatalogEntry(
      intent: InputIntent(parseInt(row[0])),
      keyword: row[1],
      inputText: row[2],
      outputText: row[3],
      weight: parseFloat(row[4]).float32
    ))
