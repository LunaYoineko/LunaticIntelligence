import std/hashes, strutils, math, unicode, tables

# ---------------------------------------------------------------------------
# SimHash: 単語の類似比較をハミング距離で高速に行う
# ---------------------------------------------------------------------------

proc simHash*(word: string): uint64 =
  ## 単語を64bit SimHashに変換（文字n-gramのハッシュを重み付け）
  var v = newSeq[int](64)
  for rune in word.toRunes:
    let s = $rune
    let h = hash(s).uint64
    for bit in 0..<64:
      if ((h shr bit) and 1.uint64) == 1:
        v[bit] += 1
      else:
        v[bit] -= 1
  result = 0
  for bit in 0..<64:
    if v[bit] > 0:
      result = result or (1.uint64 shl bit)

proc simHashForRunes*(word: string): uint64 =
  ## 日本語対応: ルーン単位でハッシュ
  var v = newSeq[int](64)
  for rune in word.toRunes:
    let s = $rune
    let h = hash(s).uint64
    for bit in 0..<64:
      if ((h shr bit) and 1.uint64) == 1:
        v[bit] += 1
      else:
        v[bit] -= 1
  result = 0
  for bit in 0..<64:
    if v[bit] > 0:
      result = result or (1.uint64 shl bit)

proc hammingDistance*(a, b: uint64): int =
  var x = a xor b
  result = 0
  while x != 0:
    result += int(x and 1.uint64)
    x = x shr 1

proc similarity*(a, b: string): float32 =
  let ha = simHashForRunes(a.toLowerAscii())
  let hb = simHashForRunes(b.toLowerAscii())
  let dist = hammingDistance(ha, hb)
  result = 1.0 - dist.float32 / 64.0

# ---------------------------------------------------------------------------
# 概念検索での利用
# ---------------------------------------------------------------------------
import types, concept_graph

proc findSimilarConcepts*(inputWords: seq[string]; graph: ConceptGraph; threshold: float32 = 0.7): seq[ConceptNode] =
  result = @[]
  for word in inputWords:
    let hw = simHashForRunes(word.toLowerAscii())
    var bestScore = 0.0
    var bestNode: ConceptNode
    var found = false
    for node in graph.nodes:
      let hn = simHashForRunes(node.word.toLowerAscii())
      let dist = hammingDistance(hw, hn)
      let score = 1.0 - dist.float / 64.0
      if score >= threshold and score > bestScore:
        bestScore = score
        bestNode = node
        found = true
    if found:
      # 重複除外
      var dup = false
      for n in result:
        if n.word == bestNode.word: dup = true; break
      if not dup:
        result.add(bestNode)
  # スコア順にソート（類似度高い順）
  # 簡易: 入力順を維持

proc extractMeaningfulWords*(input: string; graph: ConceptGraph): seq[ConceptNode] =
  ## 入力から意味的に重要な単語のみを抽出（助詞・汎用語を除外し、概念グラフで裏付け）
  result = @[]
  # 1. 単語分割（concept_graphのcategorizeWordを利用）
  var words: seq[string] = @[]
  var cur = ""
  var isCJK = false
  for rune in input.toRunes:
    let cp = rune.int32
    let s = $rune
    if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
      if cur.len > 0 and not isCJK:
        words.add(cur); cur = ""
      cur.add(s); isCJK = true
    elif s == " " or s == "," or s == "." or s == "?" or s == "？" or s == "!" or s == "！" or s == "、" or s == "。":
      if cur.len > 0:
        words.add(cur); cur = ""; isCJK = false
    else:
      if cur.len > 0 and isCJK:
        words.add(cur); cur = ""
      cur.add(s); isCJK = false
  if cur.len > 0: words.add(cur)

  for w in words:
    let cat = categorizeWord(w)
    # 助詞・汎用語は除外、名詞・動詞・形容詞のみ
    if cat in [ctParticle, ctAbstract] and w.len < 4: continue
    if w in ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か", "な", "です", "ます"]: continue
    if w.len < 2: continue
    # 概念グラフに存在するか、SimHashで類似があれば採用
    if graph.nodeIndex.hasKey(w):
      let nid = graph.nodeIndex[w]
      result.add(graph.nodes[nid])
    else:
      # 未知語でも、SimHashで類似概念があればその概念を代用
      let similar = findSimilarConcepts(@[w], graph, 0.85)
      if similar.len > 0:
        result.add(similar[0])
      elif w.runeLen >= 2:
        # 新語として一時的に概念化（活性化はしない）
        result.add(ConceptNode(word: w, category: cat, activation: 0.5, baseFrequency: 0.1, id: -1, lastAccessed: 0, accessCount: 0))
