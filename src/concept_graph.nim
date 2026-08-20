import tables, algorithm, math, times, strutils, unicode
import types, tokenizer

# ---------------------------------------------------------------------------
# ConceptGraph: 意味記憶ネットワーク
# ---------------------------------------------------------------------------
# 人間の概念的なつながりをグラフで表現
# Spreading Activationで関連概念を探索

proc initConceptGraph*(): ConceptGraph =
  result.nodes = @[]
  result.edges = @[]
  result.nodeIndex = initTable[string, int]()
  result.adjacency = initTable[int, seq[int]]()

# ---------------------------------------------------------------------------
# ノード追加
# ---------------------------------------------------------------------------
proc addNode*(graph: var ConceptGraph; word: string; category: ConceptType): int =
  if graph.nodeIndex.hasKey(word):
    return graph.nodeIndex[word]

  let id = graph.nodes.len
  var node = ConceptNode(
    id: id,
    word: word,
    category: category,
    activation: 0.0,
    baseFrequency: 0.0,
    lastAccessed: epochTime(),
    accessCount: 0
  )
  graph.nodes.add(node)
  graph.nodeIndex[word] = id
  graph.adjacency[id] = @[]
  return id

# ---------------------------------------------------------------------------
# エッジ追加
# ---------------------------------------------------------------------------
proc addEdge*(graph: var ConceptGraph;
              fromWord, toWord: string;
              relation: EdgeRelation;
              weight: float32 = 0.5) =
  if not graph.nodeIndex.hasKey(fromWord) or not graph.nodeIndex.hasKey(toWord):
    return

  let fromId = graph.nodeIndex[fromWord]
  let toId = graph.nodeIndex[toWord]
  let key = (fromId, toId)

  # 重複チェック（HashSet使用）
  if graph.edgeSet.hasKey(key):
    let edgeId = graph.edgeSet[key]
    graph.edges[edgeId].weight = min(1.0, graph.edges[edgeId].weight + 0.1)
    graph.edges[edgeId].hebbianCount += 1
    return

  let edgeId = graph.edges.len
  var edge = ConceptEdge(
    fromId: fromId,
    toId: toId,
    relation: relation,
    weight: weight,
    hebbianCount: 1,
    lastCoactivated: epochTime()
  )
  graph.edges.add(edge)
  graph.adjacency[fromId].add(edgeId)
  graph.edgeSet[key] = edgeId

# ---------------------------------------------------------------------------
# テキストから概念を抽出
# ---------------------------------------------------------------------------
proc categorizeWord*(word: string): ConceptType =
  # 助詞判定
  if word in ["は", "が", "を", "に", "で", "と", "も", "の", "から", "まで",
              "より", "へ", "について", "に対して", "として"]:
    return ctParticle
  # 挨拶判定
  if word in ["おはよう", "こんにちは", "こんばんは", "はじめまして",
              "ありがとう", "すみません", "ごめんなさい", "お疲れ様"]:
    return ctGreeting
  # 質問詞判定
  if word in ["何", "どこ", "いつ", "誰", "なぜ", "どう", "どの", "どんな",
              "いくら", "いくつ"]:
    return ctQuestion
  # 感情語判定
  if "嬉しい" in word or "悲しい" in word or "楽しい" in word or
     "面白い" in word or "美しい" in word or "綺麗" in word:
    return ctEmotion
  # 動詞判定（基本形）
  if word.endsWith("る") or word.endsWith("う") or word.endsWith("つ") or
     word.endsWith("す") or word.endsWith("く") or word.endsWith("ぐ"):
    return ctVerb
  # 形容詞判定
  if word.endsWith("い") or word.endsWith("ない") or word.endsWith("しい") or
     word.endsWith("らしい"):
    return ctAdj
  # その他は名詞
  return ctNoun

# ---------------------------------------------------------------------------
# コーパスから概念グラフを自動構築
# ---------------------------------------------------------------------------
proc buildFromCorpus*(graph: var ConceptGraph; corpus: seq[string]) =
  echo "Building concept graph from corpus..."

  # 意味のないトークンをフィルタ
  proc isMeaningful(word: string): bool =
    if word.len == 0: return false
    if word == PAD_TOKEN or word == UNK_TOKEN or word == EOS_TOKEN: return false
    if word in ["、", "。", "！", "？", "「", "」", "（", "）", "…", "・", "～",
                "：", "；", "，", "．", "（", "）", "【", "】", "〈", "〉",
                "《", "》", "「", "」", "『", "』", "（", "）", "［", "］"]:
      return false
    if word.startsWith("*") or word.startsWith(":") or word.startsWith("|"):
      return false
    if word.startsWith("1.") or word.startsWith("2."):
      return false
    if word.len <= 4:
      var isEmoji = false
      for r in word.toRunes:
        let cp = r.int32
        if cp >= 0x1F600 and cp <= 0x1F64F: isEmoji = true
        if cp >= 0x1F300 and cp <= 0x1F5FF: isEmoji = true
        if cp >= 0x1F680 and cp <= 0x1F6FF: isEmoji = true
        if cp >= 0x1F900 and cp <= 0x1F9FF: isEmoji = true
        if cp >= 0x2600 and cp <= 0x26FF: isEmoji = true
        if cp >= 0x2700 and cp <= 0x27BF: isEmoji = true
        if cp >= 0x1F1E0 and cp <= 0x1F1FF: isEmoji = true
      if isEmoji: return false
    let particles = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                     "な", "し", "る", "い", "す", "り", "く", "た", "て", "せ",
                     "れ", "め", "け", "ね", "へ", "お", "ご", "ごと", "こ"]
    var runeCount = 0
    for r in word.toRunes:
      runeCount += 1
    if runeCount == 1 and word in particles:
      return false
    if runeCount <= 2:
      var allAscii = true
      for ch in word:
        if ch.int32 > 127: allAscii = false
      # 日本語の短い助詞のみフィルタ（英語は許可）
      if allAscii and runeCount == 1: return false
    return true

  # コーパスから直接単語を抽出（BPEトークンではなく）
  # 助詞・助動詞で区切って、CJK文字列を単語として扱う
  let splitParticles = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                        "な", "から", "まで", "より", "って", "じゃ",
                        "です", "ます", "だ", "である", "いる", "ある",
                        "そう", "よ", "ね", "さ", "わ"]

  proc extractJapaneseWords(text: string): seq[string] =
    result = @[]
    var current = ""
    var isAlpha = false
    for rune in text.toRunes:
      let cp = rune.int32
      # CJK文字（日本語）
      if (cp >= 0x3040 and cp <= 0x309F) or
         (cp >= 0x30A0 and cp <= 0x30FF) or
         (cp >= 0x4E00 and cp <= 0x9FFF):
        if current.len > 0 and isAlpha:
          result.add(current)
          current = ""
          isAlpha = false
        current.add($rune)
      # 英語（アルファベット＋数字）
      elif (cp >= 0x0041 and cp <= 0x005A) or
           (cp >= 0x0061 and cp <= 0x007A) or
           (cp >= 0x0030 and cp <= 0x0039) or
           cp == 0x005F:  # アンダースコア
        if current.len > 0 and not isAlpha:
          result.add(current)
          current = ""
        current.add($rune)
        isAlpha = true
      else:
        if current.len > 0:
          result.add(current)
          current = ""
          isAlpha = false
    if current.len > 0:
      result.add(current)

  proc splitByParticles(word: string): seq[string] =
    # 助詞で単語を分割（「食べれば」→「食べ」+「れ」+「ば」のような場合）
    result = @[]
    var remaining = word
    while remaining.len > 0:
      var found = false
      for p in splitParticles:
        if remaining.len > p.len and remaining.endsWith(p):
          let base = remaining[0..<(remaining.len - p.len)]
          if base.len > 0:
            result.add(base)
          remaining = p
          found = true
          break
      if not found:
        result.add(remaining)
        break

  echo "  Extracting words from corpus..."
  var wordFreq: Table[string, int]
  var cachedWords: seq[seq[string]]
  cachedWords.setLen(corpus.len)

  for i in 0..<corpus.len:
    let rawWords = extractJapaneseWords(corpus[i])
    # 助詞で分割して単語リストを作成
    var uniqueWords: seq[string] = @[]
    for w in rawWords:
      let parts = splitByParticles(w)
      for part in parts:
        var runeCount = 0
        for r in part.toRunes:
          runeCount += 1
        if runeCount >= 2 and part notin uniqueWords:
          uniqueWords.add(part)
          wordFreq[part] = wordFreq.getOrDefault(part, 0) + 1
    cachedWords[i] = uniqueWords
    if (i+1) mod 10000 == 0:
      echo "    " & $(i+1) & "/" & $corpus.len

  # 頻出単語をノードとして追加（閾値: コーパスサイズに応じて調整）
  let minFreq = max(1, min(3, corpus.len div 100))
  var sortedWords: seq[(string, int)]
  for (word, freq) in wordFreq.pairs:
    if freq >= minFreq:
      sortedWords.add((word, freq))
  sortedWords.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))

  for (word, freq) in sortedWords:
    let category = categorizeWord(word)
    let nodeId = graph.addNode(word, category)
    graph.nodes[nodeId].baseFrequency = freq.float32 / sortedWords.len.float32

  echo "  Nodes: " & $graph.nodes.len

  # 共起エッジを構築
  var edgeBuilt = 0
  for words in cachedWords:
    for i in 0..<(words.len - 1):
      let w1 = words[i]
      let w2 = words[i + 1]
      if graph.nodeIndex.hasKey(w1) and graph.nodeIndex.hasKey(w2):
        let cat1 = categorizeWord(w1)
        let cat2 = categorizeWord(w2)
        var relation: EdgeRelation
        var weight: float32 = 0.3
        if cat1 == ctParticle or cat2 == ctParticle:
          relation = erRelatedTo
        elif cat1 == ctNoun and cat2 == ctVerb:
          relation = erCauses
          weight = 0.5
        elif cat1 == ctVerb and cat2 == ctNoun:
          relation = erHasProperty
          weight = 0.5
        elif cat1 == ctNoun and cat2 == ctAdj:
          relation = erHasProperty
          weight = 0.4
        elif cat1 == ctAdj and cat2 == ctNoun:
          relation = erRelatedTo
          weight = 0.4
        elif cat1 == ctVerb and cat2 == ctVerb:
          relation = erCauses
          weight = 0.4
        else:
          relation = erRelatedTo
        graph.addEdge(w1, w2, relation, weight)

    edgeBuilt += 1
    if edgeBuilt mod 5000 == 0:
      echo "  Edges: " & $graph.edges.len & " (processed " & $edgeBuilt & "/" & $corpus.len & ")"

  echo "  Edges: " & $graph.edges.len

# ---------------------------------------------------------------------------
# Spreading Activation (活性化伝播)
# ---------------------------------------------------------------------------
proc resetActivation*(graph: var ConceptGraph) =
  for i in 0..<graph.nodes.len:
    graph.nodes[i].activation = 0.0

proc activateNode*(graph: var ConceptGraph; nodeId: int; level: float32) =
  if nodeId >= 0 and nodeId < graph.nodes.len:
    graph.nodes[nodeId].activation = min(1.0, level)
    graph.nodes[nodeId].lastAccessed = epochTime()
    graph.nodes[nodeId].accessCount += 1

proc activateWord*(graph: var ConceptGraph; word: string; level: float32 = 1.0) =
  if graph.nodeIndex.hasKey(word):
    graph.activateNode(graph.nodeIndex[word], level)

proc spreadActivation*(graph: var ConceptGraph;
                       steps: int = 3;
                       decay: float32 = 0.5) =
  ## 活性化伝播: 活性化したノードからエッジ経由で隣接ノードに伝播
  for step in 0..<steps:
    var newActivations: seq[(int, float32)] = @[]

    for nodeId in 0..<graph.nodes.len:
      let currentAct = graph.nodes[nodeId].activation
      if currentAct <= 0.01: continue  # 閾値以下はスキップ

      # 隣接ノードに伝播
      for edgeId in graph.adjacency.getOrDefault(nodeId, @[]):
        let edge = graph.edges[edgeId]
        let targetId = edge.toId
        let propagated = currentAct * edge.weight * decay

        if propagated > 0.01:
          newActivations.add((targetId, propagated))

    # 伝播結果を適用
    for (nodeId, act) in newActivations:
      let current = graph.nodes[nodeId].activation
      graph.nodes[nodeId].activation = min(1.0, current + act)

# ---------------------------------------------------------------------------
# 上位概念を取得
# ---------------------------------------------------------------------------
proc getTopConcepts*(graph: ConceptGraph; topK: int = 7): seq[ConceptNode] =
  result = @[]
  for node in graph.nodes:
    if node.activation > 0.01:
      result.add(node)
  result.sort(proc(a, b: ConceptNode): int = cmp(b.activation, a.activation))
  if result.len > topK:
    result = result[0..<topK]

proc getActiveNodeIds*(graph: ConceptGraph): seq[int] =
  result = @[]
  for node in graph.nodes:
    if node.activation > 0.01:
      result.add(node.id)

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
proc conceptCount*(graph: ConceptGraph): int = graph.nodes.len
proc edgeCount*(graph: ConceptGraph): int = graph.edges.len

proc getWord*(graph: ConceptGraph; nodeId: int): string =
  if nodeId >= 0 and nodeId < graph.nodes.len:
    return graph.nodes[nodeId].word
  return ""

proc getNode*(graph: ConceptGraph; nodeId: int): ConceptNode =
  if nodeId >= 0 and nodeId < graph.nodes.len:
    return graph.nodes[nodeId]
  return ConceptNode()

proc getNodeByWord*(graph: ConceptGraph; word: string): ConceptNode =
  if graph.nodeIndex.hasKey(word):
    return graph.getNode(graph.nodeIndex[word])
  return ConceptNode()

# ---------------------------------------------------------------------------
# Hebbian エッジ強化
# ---------------------------------------------------------------------------
proc hebbianStrengthen*(graph: var ConceptGraph;
                        word1, word2: string;
                        amount: float32 = 0.05) =
  ## 同時発火した概念間のエッジを強化
  if not graph.nodeIndex.hasKey(word1) or not graph.nodeIndex.hasKey(word2):
    return
  let id1 = graph.nodeIndex[word1]
  let id2 = graph.nodeIndex[word2]
  let key = (id1, id2)

  if graph.edgeSet.hasKey(key):
    let edgeId = graph.edgeSet[key]
    graph.edges[edgeId].weight = min(1.0, graph.edges[edgeId].weight + amount)
    graph.edges[edgeId].hebbianCount += 1
    return

  # エッジがなければ新規作成
  graph.addEdge(word1, word2, erRelatedTo, 0.3 + amount)

# ---------------------------------------------------------------------------
# 出力（デバッグ用）
# ---------------------------------------------------------------------------
proc dumpActivation*(graph: ConceptGraph) =
  for node in graph.nodes:
    if node.activation > 0.01:
      echo "  [" & $node.category & "] " & node.word &
           " act=" & $formatFloat(node.activation, ffDecimal, 3) &
           " freq=" & $node.accessCount
