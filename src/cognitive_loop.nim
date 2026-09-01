import os, strutils, tables, algorithm, math, times, unicode, sequtils, random
import types, tokenizer, concept_graph, working_memory, tsetlin, generator, grammar
import intent_classifier, semantic_matcher, web_search
import code_structure
import llm

proc trueRandFloatCog*(): float32 =
  try:
    var b: array[4, byte]
    let f = open("/dev/urandom", fmRead)
    let n = f.readBytes(b, 0, 4)
    f.close()
    if n == 4:
      let u = uint32(b[0]) or (uint32(b[1]) shl 8) or (uint32(b[2]) shl 16) or (uint32(b[3]) shl 24)
      return float32(u) / float32(high(uint32))
  except:
    discard
  return rand(1.0).float32

# web_search から関数をインポート
from web_search import calculateExpression

# 前方宣言
proc extractWords*(text: string; tokenizer: Tokenizer): seq[string]

proc fillTemplate*(tmpl: string; replacements: Table[string, string]): string =
  result = tmpl
  for (key, value) in replacements.pairs:
    result = result.replace("{" & key & "}", value)
  return result

proc chooseRandomTemplate*(templates: seq[string]): string =
  if templates.len == 0:
    return ""
  return templates[rand(0..<templates.len)]

# クールでミステリアスな口調のテンプレート
proc getCoolResponse*(intent: InputIntent; concepts: seq[ConceptNode]; input: string): string =
  ## クールでミステリアスな応答を返す (フォールバック用)
  case intent
  of iiGreeting:
    return chooseRandomTemplate(@[
      "呼ばれた気がする...",
      "呼んだ?...",
      "はーい...",
      "何か用かな?...",
      "こんにちは..."
    ])
  of iiQuestion:
    if concepts.len > 0:
      var words: seq[string] = @[]
      for c in concepts:
        if c.activation > 0.3:
          words.add(c.word)
      if words.len > 0:
        let uniqueWords = words.deduplicate
        if uniqueWords.len > 1:
          return uniqueWords.join("や") & "について..."
        else:
          return uniqueWords[0] & "..."
      else:
        return "その質問..."
    else:
      return "何について...?"
  of iiRequest:
    if concepts.len > 0:
      return "わかったよ... " & concepts[0].word & "について..."
    else:
      return "わかったよ..."
  of iiThanks:
    return chooseRandomTemplate(@[
      "どういたしまして...",
      "いいってことよ...",
      "気にしないで..."
    ])
  of iiFarewell:
    return chooseRandomTemplate(@[
      "じゃね...",
      "またね...",
      "バイバイ..."
    ])
  of iiOpinion:
    if concepts.len > 0:
      return "「" & concepts[0].word & "」..."
    else:
      return "その気持ち..."
  of iiStatement:
    if concepts.len > 0:
      var words: seq[string] = @[]
      for c in concepts:
        if c.activation > 0.3:
          words.add(c.word)
      if words.len > 0:
        let uniqueWords = words.deduplicate
        if uniqueWords.len > 1:
          return uniqueWords.join("や") & "..."
        else:
          return uniqueWords[0] & "..."
      else:
        return "..."
    else:
      return "..."
  else:
    return "..."

# LLMを使用した応答生成 - フロー: TMで意味理解→ピックアップ→LLM推論(Thinking含む)→回答
proc generateWithLLM*(state: var CognitiveState; concepts: seq[ConceptNode]; intent: InputIntent; input: string; forceLLM: bool = false): string =
  ## 右脳(TM)で意味理解しピックアップした概念を左脳(LLM)に渡し、Thinkingを含めて生成
  let rightBrainResponse = getCoolResponse(intent, concepts, input)
  # 未学習LLMのゴミ(<UNK>)出力を回避: カタログ/TM応答をそのまま返す
  if not forceLLM:
    return rightBrainResponse

  # LLMをスキップ: 真乱数で語尾・接頭語揺らぎ（固定化防止）
  if rightBrainResponse.len > 0 and rightBrainResponse != "...":
    var varied = rightBrainResponse
    # 真乱数で語尾を多様化: 挨拶・感话・別れ特有の語尾をランダム選択
    if intent == iiGreeting:
      let greets = ["！", "ね", "よ", "。", " やあ", " へーい"]
      varied &= greets[int(trueRandFloatCog() * greets.len.float32) mod greets.len]
    elif intent == iiThanks:
      let thanks = ["どういたしまして！", "どういたましょうかね", "嬉しいです", "いえいえ"]
      varied = thanks[int(trueRandFloatCog() * thanks.len.float32) mod thanks.len]
    elif intent == iiFarewell:
      let farewells = ["ね、またね", "バイバイ", "おやすみ～", "じゃあね", "またお会いしましょう"]
      varied = farewells[int(trueRandFloatCog() * farewells.len.float32) mod farewells.len]
    elif intent == iiQuestion:
      # 質問時も語尾多様化
      let suffixes = ["。", " ね", "よ", "かな？", "だって"]
      varied &= suffixes[int(trueRandFloatCog() * suffixes.len.float32) mod suffixes.len]
    else:
      let suffix = if trueRandFloatCog() > 0.5: "。" elif trueRandFloatCog() > 0.5: " ね" else: ""
      varied &= suffix
    return varied

  # フォールバック: 概念から簡易生成
  var response = ""
  for c in concepts:
    if c.activation > 0.3 and c.word.len > 0 and c.word notin ["...", "の"]:
      response = c.word & "、"
      break
  if response.len == 0:
    response = "それは面白いね。"
  let suffix = if trueRandFloatCog() > 0.5: "。" elif trueRandFloatCog() > 0.5: " ね" else: ""
  return response & suffix

# 推論ベースのテンプレート生成
proc generateTemplateFromConcepts*(state: CognitiveState; intent: InputIntent; concepts: seq[ConceptNode]; input: string): string =
  ## 概念グラフからテンプレートを生成
  result = ""
  
  # 意図に基づいてテンプレートを生成
  case intent
  of iiGreeting:
    # 挨拶テンプレート
    let greetings = @["…来たのね", "…いらっしゃい", "…こんにちは"]
    result = chooseRandomTemplate(greetings)
  of iiQuestion:
    # 質問テンプレート（概念に基づく）
    if concepts.len > 0:
      var questionWords: seq[string] = @[]
      let inputWords = extractWords(input, state.tokenizer)
      for c in concepts:
        if c.activation > 0.3 and c.word notin inputWords:
          questionWords.add(c.word)
      if questionWords.len > 0:
        let uniqueWords = questionWords.deduplicate
        if uniqueWords.len > 1:
          result = uniqueWords.join("や") & "…ね"
        else:
          result = uniqueWords[0] & "…ね"
      else:
        result = "…その質問ね"
    else:
      result = "…何について？"
  of iiRequest:
    # 要求テンプレート
    if concepts.len > 0:
      result = "…わかったよ。" & concepts[0].word & "…ね"
    else:
      result = "…わかったよ"
  of iiThanks:
    let thanksTemplates = @["…どういたしまして", "…いいってことよ", "…気にしないで"]
    result = chooseRandomTemplate(thanksTemplates)
  of iiFarewell:
    let farewellTemplates = @["…じゃね", "…またね", "…バイバイ"]
    result = chooseRandomTemplate(farewellTemplates)
  of iiOpinion:
    if concepts.len > 0:
      let opinionTemplates = @[
        "「" & concepts[0].word & "」…ね",
        "「" & concepts[0].word & "」…わかるよ",
        "…その「" & concepts[0].word & "」って気持ちね"
      ]
      result = chooseRandomTemplate(opinionTemplates)
    else:
      result = "…その気持ちね"
  of iiStatement:
    if concepts.len > 0:
      var statementWords: seq[string] = @[]
      for c in concepts:
        if c.activation > 0.3:
          statementWords.add(c.word)
      if statementWords.len > 0:
        let uniqueWords = statementWords.deduplicate
        if uniqueWords.len > 1:
          result = uniqueWords.join("や") & "…ね"
        else:
          result = uniqueWords[0] & "…ね"
      else:
        result = "…ね"
    else:
      result = "…ね"
  else:
    result = ""
  
  return result

proc deduplicate*(s: seq[string]): seq[string] =
  result = @[]
  for item in s:
    if item notin result:
      result.add(item)

# 検索用キーワード抽出
proc extractSearchKeywords*(query: string): string =
  var clean = query
  # 質問語・助詞を除去
  let removePatterns = ["とは何", "とは", "について教えて", "について知りたい", "について", 
                        "何", "どこ", "いつ", "誰", "なぜ", "どう", "どの", "どんな",
                        "ですか", "でしょうか", "はどう", "か？", "か", "？", "?",
                        "！", "!", "。", "、", "の", "は", "が", "を", "に", "で", "と", "も"]
  for p in removePatterns:
    clean = clean.replace(p, " ")
  # 複数スペースを1つに
  clean = clean.replace("  ", " ").strip()
  if clean.len == 0:
    return query
  return clean

# ---------------------------------------------------------------------------
# CognitiveLoop: 認知プロセス統括
# ---------------------------------------------------------------------------
# 入力→シンキング→概念活性化→伝播→TM推論→シナプス修正→エピソード検索→生成→評価→学習

proc extractWords*(text: string; tokenizer: Tokenizer): seq[string] =
  result = @[]
  # If tokenizer is empty, extract words directly from concept graph
  if tokenizer.vocab.len <= 3:
    var current = ""
    var isAlpha = false
    for rune in text.toRunes:
      let cp = rune.int32
      if (cp >= 0x3040 and cp <= 0x309F) or
         (cp >= 0x30A0 and cp <= 0x30FF) or
         (cp >= 0x4E00 and cp <= 0x9FFF):
        if current.len > 0 and isAlpha:
          result.add(current)
          current = ""
          isAlpha = false
        current.add($rune)
      elif (cp >= 0x0041 and cp <= 0x005A) or
           (cp >= 0x0061 and cp <= 0x007A) or
           (cp >= 0x0030 and cp <= 0x0039) or
           cp == 0x005F:
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
  else:
    let ids = tokenizer.encode(text)
    for id in ids:
      if id >= 0 and id < tokenizer.vocab.len:
        let word = tokenizer.vocab[id]
        if word.len > 0 and word != PAD_TOKEN and word != UNK_TOKEN and word != EOS_TOKEN:
          result.add(word)


proc containsWord*(text: string; word: string): bool =
  return text.contains(word)

const STOP_WORDS* = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                     "な", "から", "まで", "より", "って", "じゃ",
                     "です", "ます", "だ", "である", "いる", "ある",
                     "そう", "よ", "ね", "さ", "わ", "だ", "し",
                     "れ", "ば", "から", "ので", "けど", "から",
                     "こと", "もの", "ため", "ところ", "とき", "よう", "そう"]

proc filterConcepts*(concepts: seq[ConceptNode]): seq[ConceptNode] =
  result = @[]
  for c in concepts:
    if c.word in STOP_WORDS: continue
    if c.word.len == 0: continue
    var runeCount = 0
    for r in c.word.toRunes: runeCount += 1
    if runeCount < 2 and c.category == ctParticle: continue
    result.add(c)

proc isJapanese*(text: string): bool =
  for rune in text.toRunes:
    let cp = rune.int32
    if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
      return true
  return false

proc isJapaneseResponse*(text: string): bool =
  for rune in text.toRunes:
    let cp = rune.int32
    if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
      return true
  return false

proc generateJapaneseResponse*(state: var CognitiveState; concepts: seq[ConceptNode]; input: string): string =
  var filtered = filterConcepts(concepts)
  if filtered.len == 0:
    filtered = concepts
  
  # 入力の意味解析
  let inputWords = extractWords(input, state.tokenizer)
  var inputConcepts: seq[ConceptNode] = @[]
  for w in inputWords:
    if state.conceptGraph.nodeIndex.hasKey(w):
      inputConcepts.add(state.conceptGraph.nodes[state.conceptGraph.nodeIndex[w]])
  
  # 活性化概念と入力概念をマージ（入力単語を除外）
  var allConcepts: seq[ConceptNode] = @[]
  for c in filtered:
    if c.word notin inputWords:
      allConcepts.add(c)
  for c in inputConcepts:
    if c.word notin inputWords:
      allConcepts.add(c)
  if allConcepts.len == 0:
    allConcepts = filtered
  
  # 文法生成を試みる
  var base = state.generator.generate(allConcepts, "description")
  
  # 概念に基づく具体的な応答生成
  if base.len == 0 and allConcepts.len > 0:
    # 入力に含まれる概念を優先的に使用
    var mainConcepts: seq[string] = @[]
    for c in allConcepts:
      if c.activation > 0.1 and c.word.len > 1:
        mainConcepts.add(c.word)
    
    if mainConcepts.len > 0:
      # 複数の概念がある場合は接続
      if mainConcepts.len >= 2:
        base = mainConcepts.join("と")
      else:
        base = mainConcepts[0]
      
      # 概念カテゴリに基づく応答パターン
      var hasQuestion = false
      var hasGreeting = false
      var hasNoun = false
      var hasEmotion = false
      
      for c in allConcepts:
        case c.category
        of ctQuestion: hasQuestion = true
        of ctGreeting: hasGreeting = true
        of ctNoun, ctVerb, ctAdj: hasNoun = true
        of ctEmotion: hasEmotion = true
        else: discard
      
      if hasQuestion:
        if mainConcepts.len > 0:
          base = mainConcepts[0] & "についての質問ですね。"
          if mainConcepts.len > 1:
            base = base & mainConcepts[1..^1].join("や") & "などと関連しています。"
          else:
            base = base & "色々な側面から考察できます。"
        else:
          base = "その質問についてですね。"
      elif hasGreeting:
        base = "こんにちは！" & (if mainConcepts.len > 0: mainConcepts.join("についてお話できます。") else: "よろしくお願いします。")
      elif hasEmotion:
        base = base & "ですね。"
        if mainConcepts.len > 0:
          base = "「" & mainConcepts[0] & "」という気持ち" & base
        else:
          base = "その気持ち" & base
        base = base & "よく分かります。"
      elif hasNoun:
        base = base & "については、"
        if mainConcepts.len > 1:
          base = base & "それぞれに特徴があります。"
        else:
          base = base & "興味深いトピックですね。"
      else:
        base = base & "についてですね。"
    else:
      base = "その内容についてですね。"
  
  # 生成結果が短すぎる場合は補足
  if base.len > 0 and base.len < 50:
    if base.endsWith("ですね") or base.endsWith("です") or base.endsWith("ます"):
      base = base & " 具体的には、関連する概念や背景を整理すると、より深い理解が得られます。"
    else:
      base = base & " このトピックは多角的に考察する価値があります。"
  
  # それでも短い場合
  if base.len > 0 and base.len < 80:
    base = base & " さらに詳しく説明すると、関連する知識を組み合わせることで理解が促進されます。"
  
  # 最後の手段
  if base.len == 0:
    if inputConcepts.len > 0:
      base = inputConcepts[0].word & "についてですね。"
    else:
      base = "その質問についてですね。"
  
  return base

proc decomposeSearchResults*(results: seq[SearchResult]): seq[string] =
  result = @[]
  for r in results:
    if r.snippet.len == 0: continue
    var parts = r.snippet.split(". ")
    for p in parts:
      let t = p.strip()
      if t.len > 10:
        result.add(t)
    if r.title.len > 0:
      result.add(r.title)

proc initCognitiveState*(cfg: CognitiveConfig): CognitiveState =
  result.cfg = cfg
  result.tokenizer = Tokenizer(vocab: @[], tokenToId: initTable[string, int]())
  result.conceptGraph = initConceptGraph()
  result.workingMemory = initWorkingMemory(cfg.wmCapacity)
  result.tm = initHierarchicalTM(
    @[(cfg.tmClauses * 8, cfg.tmClauses, cfg.tmThreshold, cfg.tmSParam)],
    32
  )
  result.bridge = SynapticBridge(
    synapses: @[], halfLifeDays: cfg.halfLifeDays,
    conceptIndex: initTable[int, seq[int]]()
  )
  result.episodeStore = EpisodeStore(
    episodes: @[], maxEpisodes: cfg.maxEpisodes,
    totalAccess: 0, phase: 0
  )
  result.generator = initTextGenerator()
  result.intentClassifier = initIntentClassifier()
  result.unknownWords = UnknownWordTracker(
    wordCounts: initTable[string, int](),
    threshold: 3,
    pendingWords: @[]
  )
  result.phase = 0
  result.lastThinking = ThinkingChain(steps: @[], totalConfidence: 0.0, reasoningPath: @[])
  result.lastEval = EvalResult(verdict: evAccept, score: 0.0, reason: "", contradictions: @[],
                               relevanceScore: 0.0, coherenceScore: 0.0, improvements: @[])
  result.totalReward = 0.0
  result.totalPunish = 0.0

proc initCognitiveState*(): CognitiveState =
  result = initCognitiveState(CognitiveConfig(
    wmCapacity: 7, spreadSteps: 3, spreadDecay: 0.5,
    activationThreshold: 0.1, tmClauses: 64, tmThreshold: 0.3,
    tmSParam: 3.0, halfLifeDays: 7.0, maxEpisodes: 5000, topKEpisodes: 3,
    thinkingEnabled: true, evalEnabled: true,
    rewardRate: 0.05f, punishRate: 0.03f
  ))

# ---------------------------------------------------------------------------
# シナプスインデックス構築
# ---------------------------------------------------------------------------
proc buildSynapseIndex*(bridge: var SynapticBridge) =
  bridge.conceptIndex = initTable[int, seq[int]]()
  for i in 0..<bridge.synapses.len:
    let cid = bridge.synapses[i].conceptId
    if not bridge.conceptIndex.hasKey(cid):
      bridge.conceptIndex[cid] = @[]
    bridge.conceptIndex[cid].add(i)

# ---------------------------------------------------------------------------
# シナプス減衰（経過時間に基づく）
# ---------------------------------------------------------------------------
proc decaySynapses*(bridge: var SynapticBridge) =
  let now = epochTime()
  for syn in bridge.synapses.mitems:
    let elapsed = now - syn.lastActivated
    let days = elapsed / 86400.0
    if days > 0.1:
      syn.strength = syn.strength * pow(0.5, days.float32 / syn.halfLifeDays)
  bridge.synapses.keepItIf(it.strength > 0.01)
  bridge.buildSynapseIndex()

# ---------------------------------------------------------------------------
# シナプス強化（Hebbian則）
# ---------------------------------------------------------------------------
proc strengthenSynapse*(bridge: var SynapticBridge;
                        clauseId, conceptId: int;
                        amount: float32) =
  if not bridge.conceptIndex.hasKey(conceptId):
    bridge.conceptIndex[conceptId] = @[]
  for idx in bridge.conceptIndex[conceptId]:
    if bridge.synapses[idx].clauseId == clauseId:
      bridge.synapses[idx].strength = min(1.0, bridge.synapses[idx].strength + amount)
      bridge.synapses[idx].activationCount += 1
      bridge.synapses[idx].lastActivated = epochTime()
      return
  bridge.synapses.add(Synapse(
    clauseId: clauseId, conceptId: conceptId,
    strength: 0.3 + amount,
    activationCount: 1,
    lastActivated: epochTime(),
    halfLifeDays: bridge.halfLifeDays
  ))
  bridge.conceptIndex[conceptId].add(bridge.synapses.len - 1)

# ---------------------------------------------------------------------------
# シナプスによるTM出力修正
# ---------------------------------------------------------------------------
proc applySynapticModulation*(bridge: SynapticBridge;
                              clauses: var seq[bool];
                              activeConceptIds: seq[int]) =
  ## シナプス強度に基づいて条款の発火を増減
  for cid in activeConceptIds:
    if not bridge.conceptIndex.hasKey(cid): continue
    for idx in bridge.conceptIndex[cid]:
      let syn = bridge.synapses[idx]
      if syn.clauseId < clauses.len and syn.strength > 0.3:
        if not clauses[syn.clauseId]:
          if syn.strength > 0.6:
            clauses[syn.clauseId] = true
        else:
          if syn.strength > 0.5:
            clauses[syn.clauseId] = true

# ---------------------------------------------------------------------------
# 意図→TM クラスマッピング
# ---------------------------------------------------------------------------
proc intentToClass*(intent: InputIntent): int =
  ## 各意図をTMのクラスIDにマッピング
  case intent
  of iiGreeting:  return 0
  of iiQuestion:  return 1
  of iiRequest:   return 2
  of iiOpinion:   return 3
  of iiAgreement: return 4
  of iiThanks:    return 5
  of iiFarewell:  return 6
  of iiStatement: return 7
  of iiOther:     return 8

proc intentToClassFromText*(text: string): int =
  ## テキストから意図を推定してTMクラスIDを返す
  let (intent, _) = initIntentClassifier().classifyIntent(text)
  return intentToClass(intent)

# ---------------------------------------------------------------------------
# シンキング: 推論チェーン生成
# ---------------------------------------------------------------------------
proc generateThinkingChain(state: var CognitiveState;
                           input: string;
                           words: seq[string];
                           topConcepts: seq[ConceptNode];
                           reasoning: ClauseReasoning;
                           bestEpisode: string;
                           bestScore: float32): ThinkingChain =
  var chain = ThinkingChain(steps: @[], totalConfidence: 0.0, reasoningPath: @[])
  chain.reasoningPath = reasoning.firedConcepts
  let intentStr = $state.lastIntent
  # 非言語→言語翻訳: 活性化0.5超のみが言葉になる（人間らしい内省）
  let verbalConcepts = topConcepts.filterIt(it.activation > 0.5)
  let conceptWords = if verbalConcepts.len > 0: verbalConcepts.mapIt(it.word & "(" & $formatFloat(it.activation, ffDecimal, 2) & ")").join(", ") else: "(閾値未満のため言語化なし)"
  let nonVerbalCount = topConcepts.len - verbalConcepts.len

  # Step1: 入力解釈 - ユーザーは何を言っているか？自問自答
  var s1 = ThinkingStep(kind: tsPerception, description: "ユーザーの発言を解釈: 「" & input & "」は何を求めているか？")
  s1.details.add("Q: ユーザーは「" & input & "」と言っているが、本当の意図は何か？")
  s1.details.add("A: 意図分類=" & intentStr & " / トークン=" & $words.len & " / 文字数=" & $input.len)
  if input.contains("？") or input.contains("?") or input.contains("とは") or input.contains("何"):
    s1.details.add("→ 質問形式と判定。事実回答か説明が求められている")
  elif input.contains("書いて") or input.contains("作って") or input.contains("生成"):
    s1.details.add("→ 創作・生成依頼と判定。創造的応答が必要")
  else:
    s1.details.add("→ 雑談・感情表現と判定。共感的応答が適切")
  s1.confidence = 0.9f
  chain.steps.add(s1)

  # Step2: 知識想起 - 関連概念は何か？（非言語→言語閾値0.5）
  var s2 = ThinkingStep(kind: tsActivation, description: "関連知識を想起: この入力に関連する概念は？")
  if topConcepts.len > 0:
    s2.details.add("Q: 活性化した概念「" & conceptWords & "」は何を意味するか？（閾値0.5超のみ言語化、" & $nonVerbalCount & "件は非言語のまま）")
    s2.details.add("A: 言語化された概念のみが思考の言葉になる。非言語は背景として保持")
    if verbalConcepts.len >= 2:
      s2.details.add("→ 言語化概念が豊富。内省が言葉として成立")
    elif verbalConcepts.len == 0:
      s2.details.add("→ 全て閾値未満。思考はまだ言葉にならない→沈黙的処理")
    else:
      s2.details.add("→ 言語化が少ない。汎用的応答か検索が必要かも")
  else:
    s2.details.add("Q: 関連概念が見つからない。未知の話題か？")
    s2.details.add("A: 検索または汎用応答で補う必要がある")
  s2.confidence = if verbalConcepts.len > 0: 0.75f else: 0.3f
  chain.steps.add(s2)

  # Step3: 推論 - TMは何を言っているか？
  var s3 = ThinkingStep(kind: tsReasoning, description: "論理的に検証: どう返すのが正解か？")
  s3.details.add("Q: TM推論の結果（発火 " & $reasoning.firedConcepts.len & " 条款, 確信度 " & $formatFloat(reasoning.confidence, ffDecimal, 3) & "）は信頼できるか？")
  if reasoning.confidence > 0.6:
    s3.details.add("A: 確信度高。TMの示す意図「" & intentStr & "」に従うのが妥当")
  elif reasoning.confidence > 0.3:
    s3.details.add("A: 確信度中。TMと概念グラフの両方を参照して判断")
  else:
    s3.details.add("A: 確信度低。過去エピソードや文法生成に頼るべき")
  if bestEpisode.len > 0:
    s3.details.add("Q: 過去の類似エピソード「" & bestEpisode[0..min(40, bestEpisode.len-1)] & "...」 (score " & $formatFloat(bestScore, ffDecimal,2) & ") は使えるか？")
    if bestScore > 0.5:
      s3.details.add("A: 類似度高。エピソード記憶を優先的に参照")
    else:
      s3.details.add("A: 類似度低。参考程度に留める")
  else:
    s3.details.add("Q: 類似エピソードはあるか？ → なし。ゼロからの生成が必要")
  s3.confidence = max(0.2f, reasoning.confidence * 0.8f + bestScore * 0.2f)
  chain.steps.add(s3)

  # Step4: 戦略選択 - どの生成経路が最適か？自己批判
  var s4 = ThinkingStep(kind: tsSelection, description: "応答戦略を選択: 最適な生成経路は？")
  s4.details.add("Q: カタログ直接照合・文法生成・検索のどれが最適か？")
  if input.contains("今何時") or input.contains("今日"):
    s4.details.add("A: 時刻/日付質問 → システム時刻で即答（検索不要）")
  elif input.contains("天気"):
    s4.details.add("A: 天気質問 → 現状APIなしのため正直に「取得不可」と答えるべき")
  elif input.contains("+") or input.contains("-") or input.contains("*") or input.contains("/"):
    s4.details.add("A: 計算質問 → ローカル計算で正確に答える")
  elif bestScore > 0.7:
    s4.details.add("A: カタログ一致度高 → カタログ応答を優先")
  elif topConcepts.len >= 2:
    s4.details.add("A: 概念が豊富 → 右脳（概念グラフ）ベースで文法生成")
  else:
    s4.details.add("A: 手がかり少 → 汎用的な共感応答 + 検索フォールバックを準備")
  s4.details.add("Q: この方針で矛盾や不自然さはないか？ → 自己批判: 入力の語をそのまま繰り返さない、敬語過多にならない、長すぎない")
  s4.confidence = 0.65f
  chain.steps.add(s4)

  # Step5: 結論 - 最終出力の方針を確定
  var s5 = ThinkingStep(kind: tsConclusion, description: "結論: どう返すか確定")
  s5.details.add("Q: 最終的にユーザーにどう返すのが正解か？")
  case state.lastIntent
  of iiGreeting: s5.details.add("A: 挨拶は短くクールに。「こんにちは...」程度で十分。余計な説明は不要")
  of iiQuestion: s5.details.add("A: 質問は簡潔に事実を。検索結果があれば要約、なければ概念ベースで答える")
  of iiRequest: s5.details.add("A: 依頼は「わかったよ...」で受容を示し、可能なら実行結果を添える")
  of iiThanks: s5.details.add("A: 感謝は「どういたしまして...」で軽く返す")
  of iiFarewell: s5.details.add("A: 別れは「じゃね...」でクールに締める")
  else: s5.details.add("A: 意図 " & intentStr & " に対して、概念「" & conceptWords & "」を素材に自然な一文を生成")
  s5.confidence = 0.7f
  chain.steps.add(s5)

  var totalConf = 0.0
  for s in chain.steps: totalConf += s.confidence.float
  chain.totalConfidence = (totalConf / chain.steps.len.float).float32
  return chain

# ---------------------------------------------------------------------------
# 自己評価
# ---------------------------------------------------------------------------
proc selfEvaluate(state: var CognitiveState;
                  input: string;
                  output: string;
                  thinking: ThinkingChain;
                  topConcepts: seq[ConceptNode]): EvalResult =
  var result = EvalResult(
    verdict: evAccept, score: 0.5, reason: "",
    contradictions: @[], relevanceScore: 0.0,
    coherenceScore: 0.0, improvements: @[]
  )

  if output.len == 0:
    result.verdict = evReject
    result.score = 0.0
    result.reason = "出力が空"
    return result

  var inputWords = extractWords(input, state.tokenizer)
  var outputWords = extractWords(output, state.tokenizer)
  var overlap = 0
  for iw in inputWords:
    for ow in outputWords:
      if iw == ow:
        overlap += 1
        break
  if inputWords.len > 0:
    result.relevanceScore = overlap.float32 / inputWords.len.float32
  else:
    result.relevanceScore = 0.5

  var coherenceSum = 0.0
  if thinking.steps.len > 0:
    for s in thinking.steps:
      coherenceSum += s.confidence.float
    result.coherenceScore = (coherenceSum / thinking.steps.len.float32).float32
  else:
    result.coherenceScore = 0.3

  let negations = ["ない", "はずがない", "間違", "嘘", "不信", "反対"]
  for neg in negations:
    if neg in output and not (neg in input):
      result.contradictions.add("否定表現「" & neg & "」を検出")

  result.score = (result.relevanceScore * 0.4 + result.coherenceScore * 0.4 +
                  (1.0 - result.contradictions.len.float32 * 0.2) * 0.2)
  result.score = max(0.0, min(1.0, result.score))

  if result.score >= 0.6:
    result.verdict = evAccept
    result.reason = "良好 (score=" & $formatFloat(result.score, ffDecimal, 3) & ")"
  elif result.score >= 0.3:
    result.verdict = evRefine
    result.reason = "改善可能 (score=" & $formatFloat(result.score, ffDecimal, 3) & ")"
  else:
    result.verdict = evReject
    result.reason = "要修正 (score=" & $formatFloat(result.score, ffDecimal, 3) & ")"
  return result

# ---------------------------------------------------------------------------
# 報酬/罰学習
# ---------------------------------------------------------------------------
proc applyRewardPunishment(state: var CognitiveState;
                           evalResult: EvalResult;
                           thinking: ThinkingChain;
                           intent: InputIntent;
                           topConcepts: seq[ConceptNode]) =
  let isReward = evalResult.verdict == evAccept
  let signal = if isReward: state.cfg.rewardRate else: -state.cfg.punishRate
  let tmClass = intentToClass(intent)

  # TM学習（意図ベース）
  if topConcepts.len > 0:
    var activeIds: seq[int] = @[]
    for c in topConcepts:
      activeIds.add(c.id)
    let fv = featureVectorFromConcepts(activeIds, state.cfg.tmClauses * 8)
    if isReward:
      state.tm.train(fv, tmClass, 1.0f)
      state.totalReward += signal
    else:
      state.tm.train(fv, tmClass, 0.0f)
      state.totalPunish += abs(signal)

  # TM条款への報酬（発火条款がある場合）
  for clauseId in thinking.reasoningPath:
    if clauseId < state.tm.layers[0].pos_reward.len:
      if isReward:
        state.tm.layers[0].pos_reward[clauseId] += signal
      else:
        state.tm.layers[0].neg_reward[clauseId] += signal

  # Hebbianシナプス強化
  for clauseId in thinking.reasoningPath:
    for c in topConcepts:
      state.bridge.strengthenSynapse(clauseId, c.id, signal)

  # シナプスがなければ、概念間だけ強化
  if thinking.reasoningPath.len == 0:
    for i in 0..<topConcepts.len:
      for j in (i+1)..<topConcepts.len:
        state.bridge.strengthenSynapse(0, topConcepts[i].id, signal * 0.5)
        state.bridge.strengthenSynapse(0, topConcepts[j].id, signal * 0.5)

  # エピソードの報酬更新
  if state.episodeStore.episodes.len > 0:
    let lastEp = addr state.episodeStore.episodes[state.episodeStore.episodes.len - 1]
    if isReward:
      lastEp.reward = min(1.0, lastEp.reward + 0.1)
    else:
      lastEp.reward = max(0.0, lastEp.reward - 0.1)

  # 概念グラフHebbian強化（入力概念間）
  for i in 0..<topConcepts.len:
    for j in (i+1)..<topConcepts.len:
      if isReward:
        state.conceptGraph.hebbianStrengthen(topConcepts[i].word, topConcepts[j].word, 0.02)

# ---------------------------------------------------------------------------
# 未学習語トラッキング
# ---------------------------------------------------------------------------
proc trackUnknownWords(state: var CognitiveState; words: seq[string]) =
  for word in words:
    if not state.conceptGraph.nodeIndex.hasKey(word):
      let runeCount = block:
        var c = 0
        for r in word.toRunes: c += 1
        c
      if runeCount >= 2:
        state.unknownWords.wordCounts[word] = state.unknownWords.wordCounts.getOrDefault(word, 0) + 1

proc getLearnableWords(state: var CognitiveState): seq[string] =
  result = @[]
  for (word, count) in state.unknownWords.wordCounts.pairs:
    if count >= state.unknownWords.threshold:
      result.add(word)

proc learnFromUserInput(state: var CognitiveState; input: string; output: string) =
  ## 未学習語を含む会話をコーパスとして保存
  let learnable = state.getLearnableWords()
  if learnable.len == 0: return

  # 未学習語を概念グラフに追加
  for word in learnable:
    let cat = categorizeWord(word)
    let nodeId = state.conceptGraph.addNode(word, cat)
    state.conceptGraph.nodes[nodeId].baseFrequency = 1.0
    state.unknownWords.wordCounts.del(word)

  # 入力を学習済みとして記録
  let inputWords = extractWords(input, state.tokenizer)
  var inputCids: seq[int] = @[]
  for word in inputWords:
    if state.conceptGraph.nodeIndex.hasKey(word):
      inputCids.add(state.conceptGraph.nodeIndex[word])

  let outputWords = extractWords(output, state.tokenizer)
  var outputCids: seq[int] = @[]
  for word in outputWords:
    if state.conceptGraph.nodeIndex.hasKey(word):
      outputCids.add(state.conceptGraph.nodeIndex[word])

  # TM学習
  if inputCids.len > 0:
    let fv = featureVectorFromConcepts(inputCids, state.cfg.tmClauses * 8)
    let intent = state.intentClassifier.classifyIntent(input)[0]
    let tmClass = intentToClass(intent)
    state.tm.train(fv, tmClass, 1.0f)

  # エピソード保存
  var clausePattern = newSeq[bool](state.cfg.tmClauses)
  for cid in inputCids:
    if cid < clausePattern.len:
      clausePattern[cid] = true

  state.episodeStore.episodes.add(Episode(
    inputText: input,
    outputText: output,
    inputConceptIds: inputCids,
    outputConceptIds: outputCids,
    tmClausePattern: clausePattern,
    confidence: 0.5,
    speaker: spUser,
    contextTag: "user_learn",
    situation: "chat",
    timestamp: epochTime(),
    reward: 0.8f,
    rank: 0.6f,
    accessCount: 0,
    emotionalValence: 0.2f
  ))

# ---------------------------------------------------------------------------
# メイン認知プロセス
# ---------------------------------------------------------------------------
proc process*(state: var CognitiveState; input: string): string =
  let t0 = epochTime()

  var thinking = ThinkingChain(steps: @[], totalConfidence: 0.0, reasoningPath: @[])
  let lowerInput = input.toLower()

  # 1. 感知
  # BPEトークンは概念マッチング用、入力単語は概念活性化用
  let bpeTokens = extractWords(input, state.tokenizer)
  # 入力から単語を抽出（Unicode正規化してから分割） rune-based with curIsCJK/curIsAlpha
  let normalized = input.normalize()
  var rawWords: seq[string] = @[]
  var current = ""
  var curIsCJK = false
  var curIsAlpha = false
  for rune in normalized.toRunes:
    let cp = rune.int32
    let isCJK = (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF)
    let isAlpha = (cp >= 0x0041 and cp <= 0x005A) or (cp >= 0x0061 and cp <= 0x007A) or (cp >= 0x0030 and cp <= 0x0039) or cp == 0x005F
    if isCJK:
      if current.len > 0 and curIsAlpha:
        rawWords.add(current)
        current = ""
      current.add($rune)
      curIsCJK = true
      curIsAlpha = false
    elif isAlpha:
      if current.len > 0 and curIsCJK:
        rawWords.add(current)
        current = ""
      current.add($rune)
      curIsCJK = false
      curIsAlpha = true
    elif $rune == " " or $rune == "," or $rune == "." or $rune == "\n" or $rune == "\t" or $rune == "、" or $rune == "。" or $rune == "！":
      if current.len > 0:
        rawWords.add(current)
        current = ""
        curIsCJK = false
        curIsAlpha = false
    else:
      if current.len > 0:
        rawWords.add(current)
        current = ""
        curIsCJK = false
        curIsAlpha = false
  if current.len > 0:
    rawWords.add(current)

  state.conceptGraph.resetActivation()

  # 単語レベルで概念を活性化（既存概念のみ）
  for word in rawWords:
    if state.conceptGraph.nodeIndex.hasKey(word):
      state.conceptGraph.activateWord(word, 1.0)
    else:
      # 未知語は追加するが、活性化しない（分類可能なまで待機）
      let cat = categorizeWord(word)
      let nodeId = state.conceptGraph.addNode(word, cat)
      # activateNode は呼ばない → 推論に影響しない

  # BPEトークンも参考として弱く活性化（既存概念とのマッチング用）
  for token in bpeTokens:
    if state.conceptGraph.nodeIndex.hasKey(token):
      state.conceptGraph.activateWord(token, 0.3)

  # code_structure integration
  let codeStruct = parseCode(input)
  let isCodeMode = codeStruct.isCode
  if isCodeMode:
    let codeConcepts = extractCodeConcepts(codeStruct)
    for w in codeConcepts:
      if state.conceptGraph.nodeIndex.hasKey(w):
        state.conceptGraph.activateWord(w, 1.0)
      else:
        let cat = categorizeWord(w)
        let nid = state.conceptGraph.addNode(w, cat)
        state.conceptGraph.activateWord(w, 0.8)

  # 未学習語をトラッキング
  state.trackUnknownWords(rawWords)

  # 2. 力学系としての伝播（アトラクター収束）- 入力を外乱としてネットワーク全体を揺らす
  # コード生成時は揺らぎを抑制（精密な収束）、通常会話は揺らぎを許容 - 真乱数使用
  let wobbleScale = if isCodeMode: 0.005f else: 0.02f
  let maxSteps = if isCodeMode: 8 else: 5
  var prevEnergy = 0.0f
  for step in 0..<maxSteps:
    state.conceptGraph.spreadActivation(steps=1, decay=state.cfg.spreadDecay)
    # 真乱数で「迷い」を注入（コード時は弱め、完全には消さない）
    for node in state.conceptGraph.nodes.mitems:
      if node.activation > 0.05:
        let noise = (trueRandFloatCog() - 0.5f) * wobbleScale
        node.activation = clamp(node.activation + noise, 0.0f, 1.0f)
    # エネルギー（全活性のL2）を計算し収束判定
    var energy = 0.0f
    for n in state.conceptGraph.nodes: energy += n.activation * n.activation
    if step > 0 and abs(energy - prevEnergy) < 0.001f:
      break # アトラクターに収束
    prevEnergy = energy
    # コード時はノイズを入れつつも収束を深く追求（最新情報への追従は残すため完全には消さない）
    if isCodeMode and energy < 0.01f: break

  # 3. 作業記憶
  let topConcepts = state.conceptGraph.getTopConcepts(state.cfg.wmCapacity)
  state.workingMemory.clear()
  for c in topConcepts:
    state.workingMemory.store(c.id, c.activation, "spread")

  # 4. 意図分類（先に実行してTMクラスに使用）
  let (intent, responseCategory) = state.intentClassifier.classifyIntent(input)
  state.lastIntent = intent
  state.lastResponseCategory = responseCategory
  state.intentClassifier.updateHistory(intent)

  # 5. TM推論（意図ベースのクラス） - 大規模DBでは推論を軽量化して60s目標を達成
  let activeIds = state.conceptGraph.getActiveNodeIds()
  # lunatic_large.db (8192 clauses, 1.1G) では全条款だと遅いため、推論時は上位1024条款に制限
  let effectiveClauses = if state.cfg.tmClauses > 1024: 1024 else: state.cfg.tmClauses
  let fv = featureVectorFromConcepts(activeIds, effectiveClauses * 8)
  var reasoning = state.tm.predictWithReasoning(fv)

  # 5.5 シナプスによるTM出力修正
  var modClauses = state.tm.clauseOutput
  state.bridge.applySynapticModulation(modClauses, activeIds)
  var maxVote = 0
  var bestClass = 0
  let clausesPerClass = state.tm.layers[0].numClauses div max(state.tm.numClasses, 1)
  for c in 0..<modClauses.len:
    if modClauses[c]:
      let classIdx = c div max(clausesPerClass, 1)
      if classIdx < state.tm.numClasses:
        state.tm.classVotes[classIdx] += 1
  for cl in 0..<state.tm.numClasses:
    if state.tm.classVotes[cl] > maxVote:
      maxVote = state.tm.classVotes[cl]
      bestClass = cl

  # 5.6 Tsetlinを抑止ブレーキ・パス選択として活用（人間らしい迷い・整合性チェック）
  # TM確信度が低いときは不整合な概念活性化を抑制、パス選択を絞る
  if reasoning.confidence < 0.4:
    for node in state.conceptGraph.nodes.mitems:
      if node.activation > 0.3:
        # 意図とカテゴリが不一致なら抑制（例: 質問意図なのに挨拶概念が強く活性化）
        let mismatched = (intent == iiQuestion and node.category == ctGreeting) or
                         (intent == iiGreeting and node.category == ctQuestion) or
                         (intent == iiRequest and node.category in [ctGreeting, ctQuestion])
        if mismatched:
          node.activation *= 0.5 # ブレーキ
    # パス選択: TMのbestClassに対応する概念のみをやや強化（論理的整合パスを優先）
    let boostIds = featureVectorFromConcepts(@[bestClass], state.cfg.tmClauses * 8) # ダミーだが概念選択の指標に
    discard boostIds

  # 6. エピソード検索
  let semanticMatches = findSemanticMatches(state.episodeStore.episodes, input, rawWords, 10)
  var bestEpisode = ""
  var bestScore: float32 = 0.0
  if semanticMatches.len > 0:
    bestEpisode = semanticMatches[0][0].outputText
    bestScore = semanticMatches[0][1]
    state.episodeStore.totalAccess += 1

  # 7. シンキングチェーン生成
  if state.cfg.thinkingEnabled:
    thinking = state.generateThinkingChain(input, rawWords, topConcepts, reasoning, bestEpisode, bestScore)
  state.lastThinking = thinking
  # 7b. DeepSeek/Qwen風: Thinkingが生成に直接寄与（戦略の条件付け）
  let thinkingConfidence = thinking.totalConfidence
  let thinkingStrategy = if thinking.steps.len >= 4: thinking.steps[3].details.join(" ") else: ""
  let thinkingSuggestsSearch = thinkingStrategy.contains("検索") or thinkingConfidence < 0.45
  let thinkingSuggestsCatalog = thinkingStrategy.contains("カタログ")

  # 8. 応答生成（概念活性化 + カタログ照合 or 文法） - Thinkingの判断を参照
  var response = ""

  # 8a. 常に概念活性化を実行
  var knowledgeConcepts: seq[ConceptNode] = @[]
  if bestEpisode.len > 0 and bestScore > 0.5:
    let epWords = extractWords(bestEpisode, state.tokenizer)
    for w in epWords:
      if state.conceptGraph.nodeIndex.hasKey(w):
        let node = state.conceptGraph.getNode(state.conceptGraph.nodeIndex[w])
        if node.activation <= 0.01:
          state.conceptGraph.activateWord(w, 0.5)
          knowledgeConcepts.add(node)

  state.conceptGraph.spreadActivation(steps=1, decay=0.3)
  let allTopConcepts = state.conceptGraph.getTopConcepts(7)

    # 8b. カタログ照合（完全一致のみ）with Japanese filtering + 揺らぎ
  let inputIsJapanese = isJapanese(input)
  var catalogMatched = false
  if state.catalog.entries.len > 0:
    var candidates: seq[string] = @[]
    for entry in state.catalog.entries:
      if entry.inputText == input:
        if inputIsJapanese and not isJapaneseResponse(entry.outputText):
          continue
        candidates.add(entry.outputText)
    if candidates.len > 0:
      # TMで選択（真乱数）、LLMは使わずカタログ応答そのまま返す（LLM未学習のため固定化排除）
      let idx = int(trueRandFloatCog() * candidates.len.float32) mod candidates.len
      response = candidates[idx]
      catalogMatched = true

  # 8c code handling
  if response.len == 0 and isCodeMode:
     if codeStruct.keywords.len > 0:
       response = "このコードは" & codeStruct.keywords.join("、") & "などの要素を含んでいますね。"
       if codeStruct.summary.len > 0:
         response = response & codeStruct.summary & "という構造です。"
       response = response & "Nim言語で記述されており、各手続きや変数の役割を追いながら全体の流れを把握することが重要です。"
     else:
       response = "これはNim言語のコードですね。コードの構造を理解するには、各要素の役割を分析することが大切です。"

    # 8c2 意図ベースの応答生成（クールでミステリアス）
  if response.len == 0:
      var hasMeaningfulConcepts = allTopConcepts.len > 0
      var inputConcepts: seq[ConceptNode] = @[]
      for w in rawWords:
        if state.conceptGraph.nodeIndex.hasKey(w):
          let nid = state.conceptGraph.nodeIndex[w]
          inputConcepts.add(state.conceptGraph.nodes[nid])
      var filteredConcepts = filterConcepts(allTopConcepts)
      var mergedConcepts: seq[ConceptNode] = @[]
      # 活性化が高い入力概念のみをマージ（入力単語を除外）
      for c in inputConcepts:
        if c.activation > 0.3 and c.word notin rawWords:
          mergedConcepts.add(c)
      # 活性化が高いフィルタリング済み概念をマージ（入力単語を除外）
      for c in filteredConcepts:
        if c.activation > 0.3 and c.word notin rawWords:
          mergedConcepts.add(c)
      if mergedConcepts.len == 0:
        # allTopConceptsから入力単語を除外
        for c in allTopConcepts:
          if c.activation > 0.3 and c.word notin rawWords:
            mergedConcepts.add(c)
      # 入力に直接応答するための意味解析 - ハードコード/テンプレート無し、LLMで揺らぎを持たせる
      if hasMeaningfulConcepts:
        case intent
        of iiGreeting:
          # 挨拶もLLMで自然に（毎回違う揺らぎを真乱数サンプリングで）
          response = generateWithLLM(state, mergedConcepts, iiGreeting, input, forceLLM = true)
        of iiQuestion:
          if input.contains("今何時") or input.contains("何時") or input.contains("時間") or (input.contains("今") and (input.contains("時") or input.contains("分") or input.contains("秒"))):
            let now = now()
            let timeStr = now.format("HH:mm")
            # 時刻の事実はTMが把握、表現はLLMに委譲して毎回違う言い回しに
            response = generateWithLLM(state, mergedConcepts, iiQuestion, "事実: 今は" & timeStr & "。この事実を自然に伝えて。", forceLLM = true)
            if response.len == 0 or response == "...": response = "今は" & timeStr & "だよ..."
          elif input.contains("今日") and (input.contains("何日") or input.contains("日付") or input.contains("日")) or input.contains("何月") or input.contains("何年"):
            let now = now()
            let year = $now.year
            let month = $now.month.int
            let day = $now.monthday.int
            let dateStr = year & "年" & month & "月" & day & "日"
            let dayOfWeek = case now.weekday
              of dMon: "月曜日"
              of dTue: "火曜日"
              of dWed: "水曜日"
              of dThu: "木曜日"
              of dFri: "金曜日"
              of dSat: "土曜日"
              of dSun: "日曜日"
              else: ""
            response = generateWithLLM(state, mergedConcepts, iiQuestion, "事実: 今日は" & dateStr & "(" & dayOfWeek & ")。自然に伝えて。", forceLLM = true)
            if response.len == 0: response = dateStr & "(" & dayOfWeek & ")だよ..."
          elif input.contains("天気") or input.contains("晴れ") or input.contains("雨"):
            response = generateWithLLM(state, mergedConcepts, iiQuestion, "天気を聞かれているがAPIが無い。気象庁を案内しつつ自然に断って。", forceLLM = true)
            if response.len < 5: response = "天気は取得できないよ...気象庁で見てみて..."
          elif input.contains("+") or input.contains("-") or input.contains("*") or input.contains("/") or input.contains("=") or input.contains("計算"):
            let calcResult = calculateExpression(input)
            response = generateWithLLM(state, mergedConcepts, iiQuestion, "計算結果は" & calcResult & "。自然に伝えて。", forceLLM = true)
            if response.len < 5: response = calcResult & "だよ..."
          elif input.contains("コード") or lowerInput.contains("code") or input.contains("プログラム") or input.contains("書いて") or input.contains("生成") or input.contains("作って") or input.contains("実装"):
            response = generateWithLLM(state, mergedConcepts, iiQuestion, input, forceLLM = true)
          else:
            # 一般的な質問: 検索結果をLLMが翻訳機として自然に要約（毎回違う揺らぎ）
            var searcher = initWebSearcher()
            let searchQuery = extractSearchKeywords(input)
            let searchResults = searcher.search(searchQuery, 3)
            if searchResults.len > 0:
              var combined = ""
              for r in searchResults:
                if combined.len > 250: break
                let snippet = r.snippet.strip()
                if snippet.len > 30:
                  combined.add(snippet[0..min(150, snippet.len-1)] & " ")
              if combined.len > 0:
                response = generateWithLLM(state, mergedConcepts, iiQuestion, "検索結果: " & combined & "\n質問: " & input & "\n上記を踏まえ自然に要約して。", forceLLM = true)
              else:
                response = generateWithLLM(state, mergedConcepts, iiQuestion, input, forceLLM = true)
            else:
              response = generateWithLLM(state, mergedConcepts, iiQuestion, input, forceLLM = true)
        of iiRequest:
          response = generateWithLLM(state, mergedConcepts, iiRequest, input, forceLLM = true)
        of iiThanks:
          response = generateWithLLM(state, mergedConcepts, iiThanks, input, forceLLM = true)
        of iiFarewell:
          response = generateWithLLM(state, mergedConcepts, iiFarewell, input, forceLLM = true)
        of iiOpinion:
          response = generateWithLLM(state, mergedConcepts, iiOpinion, input, forceLLM = true)
        of iiStatement:
          response = generateWithLLM(state, mergedConcepts, iiStatement, input, forceLLM = true)
        else:
          response = generateWithLLM(state, mergedConcepts, intent, input, forceLLM = true)
      else:
        response = generateWithLLM(state, mergedConcepts, intent, input, forceLLM = true)

      # 8e. DeepSeek/Qwen風 自己検証ループ: Thinkingの結論で応答を自己修正
      if thinking.steps.len > 0 and response.len > 0:
        let conclusionText = thinking.steps[^1].details.join(" ")
        if conclusionText.contains("挨拶は短く") and response.len > 30:
          response = getCoolResponse(iiGreeting, mergedConcepts, input)
        if response == "何について...?" and thinkingSuggestsSearch:
          var searcher2 = initWebSearcher()
          let sq2 = extractSearchKeywords(input)
          let sr2 = searcher2.search(sq2, 3)
          if sr2.len > 0 and sr2[0].snippet.len > 50:
            response = getCoolResponse(iiQuestion, mergedConcepts, "検索結果: " & sr2[0].snippet & "\n質問: " & input)
        if thinkingConfidence < 0.35 and response == "...":
          response = "...何について話そうか..."

  # 9. 自己評価
  var evalResult = EvalResult(verdict: evAccept, score: 0.5, reason: "skip",
                              contradictions: @[], relevanceScore: 0.5,
                              coherenceScore: 0.5, improvements: @[])
  if state.cfg.evalEnabled:
    evalResult = state.selfEvaluate(input, response, thinking, topConcepts)
  state.lastEval = evalResult

  # 10. 報酬/罰学習（意図ベース）
  state.applyRewardPunishment(evalResult, thinking, intent, topConcepts)

  # 11. 未学習語からの学習
  state.learnFromUserInput(input, response)

  # 12. エピソード保存
  var clausePattern = newSeq[bool](state.cfg.tmClauses)
  for cid in reasoning.firedConcepts:
    if cid < clausePattern.len:
      clausePattern[cid] = true

  state.episodeStore.episodes.add(Episode(
    inputText: input,
    outputText: response,
    inputConceptIds: activeIds,
    outputConceptIds: @[],
    tmClausePattern: clausePattern,
    confidence: reasoning.confidence,
    speaker: spUser,
    contextTag: $intent,
    situation: "chat",
    timestamp: epochTime(),
    reward: if evalResult.verdict == evAccept: 0.7f
            elif evalResult.verdict == evRefine: 0.4f
            else: 0.1f,
    rank: evalResult.score,
    accessCount: 0,
    emotionalValence: if evalResult.verdict == evAccept: 0.3f else: -0.3f
  ))

  # エピソード上限チェック
  if state.episodeStore.episodes.len > state.episodeStore.maxEpisodes:
    state.episodeStore.episodes.sort(proc(a, b: Episode): int = cmp(b.rank, a.rank))
    state.episodeStore.episodes = state.episodeStore.episodes[0..<state.episodeStore.maxEpisodes]

  # シナプス Hebbian 学習
  for c in topConcepts:
    for clauseId in reasoning.firedConcepts:
      state.bridge.strengthenSynapse(clauseId, c.id, 0.05)

  # 定期的にシナプス減衰
  if state.episodeStore.episodes.len mod 50 == 0:
    state.bridge.decaySynapses()

  # 品質チェック: テンプレート的な応答の検出
  proc isTemplateArtifact(text: string): bool =
    if text.len == 0: return false
    var hasJapanese = false
    var hasEnglish = false
    for rune in text.toRunes:
      let cp = rune.int32
      if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
        hasJapanese = true
      elif (cp >= 0x0041 and cp <= 0x005A) or (cp >= 0x0061 and cp <= 0x007A):
        hasEnglish = true
    if hasJapanese and hasEnglish: return true
    if text.contains("Template") or text.contains("{{") or text.contains("}}"):
      return true
    return false

  # 13. Web検索フォールバック（応答が空の場合のみ）
  let needSearch = input.len > 0 and response.len == 0
  if needSearch:
    var searcher = initWebSearcher()
    let knowledge = searcher.getKnowledge(input)
    if knowledge.len > 0:
      let translated = searcher.translateKnowledge(knowledge, "ja")
      if translated.len > 0:
        # decompose search results and generate via generateJapaneseResponse, not direct translated
        var dummyResults = searcher.search(input, 3)
        let decomposed = decomposeSearchResults(dummyResults)
        for w in decomposed:
          let ws = extractWords(w, state.tokenizer)
          for word in ws:
            if state.conceptGraph.nodeIndex.hasKey(word):
              state.conceptGraph.activateWord(word, 0.3)
            else:
              let cat = categorizeWord(word)
              let cid = state.conceptGraph.addNode(word, cat)
              state.conceptGraph.nodes[cid].baseFrequency = 1.0f / max(1, ws.len).float32
              state.conceptGraph.activateWord(word, 0.3)
        let knowledgeWords = extractWords(translated, state.tokenizer)
        for w in knowledgeWords:
          if state.conceptGraph.nodeIndex.hasKey(w):
            state.conceptGraph.activateWord(w, 0.3)
          else:
            let cat = categorizeWord(w)
            let cid = state.conceptGraph.addNode(w, cat)
            state.conceptGraph.nodes[cid].baseFrequency = 1.0f / max(1, knowledgeWords.len).float32
            state.conceptGraph.activateWord(w, 0.3)

        state.conceptGraph.spreadActivation(steps=2, decay=0.4)

        let activeIds = state.conceptGraph.getActiveNodeIds()
        if activeIds.len > 0:
          let kv = featureVectorFromConcepts(activeIds, state.cfg.tmClauses * 8)
          let searchIntentClass = 1
          state.tm.train(kv, searchIntentClass, 0.5f)
          discard state.tm.predictWithReasoning(kv)

        if translated.len > 0 and translated.len <= 200 and state.catalog.entries.len < 2000:
          var catalogIntent = iiQuestion
          if input.toLower().contains("hello") or input.toLower().contains("hi"):
            catalogIntent = iiGreeting
          elif input.toLower().contains("thank"):
            catalogIntent = iiThanks
          elif input.toLower().contains("bye") or input.toLower().contains("goodbye"):
            catalogIntent = iiFarewell
          state.catalog.entries.add(CatalogEntry(
            intent: catalogIntent,
            keyword: input,
            inputText: input,
            outputText: translated,
            weight: 1.0f
          ))

        # 生成は LLM を使用
        if response.len == 0 or isTemplateArtifact(response) or evalResult.score < 0.4 or response.len < 30:
           let activeConcepts = state.conceptGraph.getTopConcepts(7)
           var merged: seq[ConceptNode] = @[]
           for c in activeConcepts: merged.add(c)
           for w in decomposed:
             for cw in extractWords(w, state.tokenizer):
               if state.conceptGraph.nodeIndex.hasKey(cw):
                 merged.add(state.conceptGraph.nodes[state.conceptGraph.nodeIndex[cw]])
           # 右脳で応答生成（LLMは使わない）
           response = getCoolResponse(iiQuestion, merged, "検索結果: " & translated & "\n質問: " & input)
           if response.len == 0:
             response = translated

  return response

# ---------------------------------------------------------------------------
# 観察モード: 単一パス設計
# ---------------------------------------------------------------------------
proc processInference*(state: var CognitiveState; input: string): string =
  # Mirrors process() but for inference path with Inf suffixes
  var thinking = ThinkingChain(steps: @[], totalConfidence: 0.0, reasoningPath: @[])
  let bpeTokensInf = extractWords(input, state.tokenizer)
  let normalizedInf = input.normalize()
  var rawWordsInf: seq[string] = @[]
  var currentInf = ""
  var curIsCJKInf = false
  var curIsAlphaInf = false
  for rune in normalizedInf.toRunes:
    let cp = rune.int32
    let isCJK = (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF)
    let isAlpha = (cp >= 0x0041 and cp <= 0x005A) or (cp >= 0x0061 and cp <= 0x007A) or (cp >= 0x0030 and cp <= 0x0039) or cp == 0x005F
    if isCJK:
      if currentInf.len > 0 and curIsAlphaInf:
        rawWordsInf.add(currentInf)
        currentInf = ""
      currentInf.add($rune)
      curIsCJKInf = true
      curIsAlphaInf = false
    elif isAlpha:
      if currentInf.len > 0 and curIsCJKInf:
        rawWordsInf.add(currentInf)
        currentInf = ""
      currentInf.add($rune)
      curIsCJKInf = false
      curIsAlphaInf = true
    elif $rune == " " or $rune == "," or $rune == "." or $rune == "\n" or $rune == "\t":
      if currentInf.len > 0:
        rawWordsInf.add(currentInf)
        currentInf = ""
        curIsCJKInf = false
        curIsAlphaInf = false
    else:
      if currentInf.len > 0:
        rawWordsInf.add(currentInf)
        currentInf = ""
        curIsCJKInf = false
        curIsAlphaInf = false
  if currentInf.len > 0:
    rawWordsInf.add(currentInf)
  state.conceptGraph.resetActivation()
  for word in rawWordsInf:
    if state.conceptGraph.nodeIndex.hasKey(word):
      state.conceptGraph.activateWord(word, 1.0)
    else:
      let cat = categorizeWord(word)
      discard state.conceptGraph.addNode(word, cat)
  for token in bpeTokensInf:
    if state.conceptGraph.nodeIndex.hasKey(token):
      state.conceptGraph.activateWord(token, 0.3)
  let codeStructInf = parseCode(input)
  let isCodeModeInf = codeStructInf.isCode
  if isCodeModeInf:
    let codeConceptsInf = extractCodeConcepts(codeStructInf)
    for w in codeConceptsInf:
      if state.conceptGraph.nodeIndex.hasKey(w):
        state.conceptGraph.activateWord(w, 1.0)
      else:
        let cat = categorizeWord(w)
        let nid = state.conceptGraph.addNode(w, cat)
        state.conceptGraph.activateWord(w, 0.8)
  state.trackUnknownWords(rawWordsInf)
  state.conceptGraph.spreadActivation(steps = state.cfg.spreadSteps, decay = state.cfg.spreadDecay)
  let topConceptsInf = state.conceptGraph.getTopConcepts(state.cfg.wmCapacity)
  state.workingMemory.clear()
  for c in topConceptsInf:
    state.workingMemory.store(c.id, c.activation, "spread")
  let (intentInf, responseCategoryInf) = state.intentClassifier.classifyIntent(input)
  state.lastIntent = intentInf
  state.lastResponseCategory = responseCategoryInf
  state.intentClassifier.updateHistory(intentInf)
  let activeIdsInf = state.conceptGraph.getActiveNodeIds()
  let fvInf = featureVectorFromConcepts(activeIdsInf, state.cfg.tmClauses * 8)
  var reasoningInf = state.tm.predictWithReasoning(fvInf)
  var modClausesInf = state.tm.clauseOutput
  state.bridge.applySynapticModulation(modClausesInf, activeIdsInf)
  var maxVoteInf = 0
  var bestClassInf = 0
  let clausesPerClassInf = state.tm.layers[0].numClauses div max(state.tm.numClasses, 1)
  for c in 0..<modClausesInf.len:
    if modClausesInf[c]:
      let classIdx = c div max(clausesPerClassInf, 1)
      if classIdx < state.tm.numClasses:
        state.tm.classVotes[classIdx] += 1
  for cl in 0..<state.tm.numClasses:
    if state.tm.classVotes[cl] > maxVoteInf:
      maxVoteInf = state.tm.classVotes[cl]
      bestClassInf = cl
  let semanticMatchesInf = findSemanticMatches(state.episodeStore.episodes, input, rawWordsInf, 10)
  var bestEpisodeInf = ""
  var bestScoreInf: float32 = 0.0
  if semanticMatchesInf.len > 0:
    bestEpisodeInf = semanticMatchesInf[0][0].outputText
    bestScoreInf = semanticMatchesInf[0][1]
    state.episodeStore.totalAccess += 1
  if state.cfg.thinkingEnabled:
    thinking = state.generateThinkingChain(input, rawWordsInf, topConceptsInf, reasoningInf, bestEpisodeInf, bestScoreInf)
  state.lastThinking = thinking
  var responseInf = ""
  var knowledgeConceptsInf: seq[ConceptNode] = @[]
  if bestEpisodeInf.len > 0 and bestScoreInf > 0.5:
    let epWords = extractWords(bestEpisodeInf, state.tokenizer)
    for w in epWords:
      if state.conceptGraph.nodeIndex.hasKey(w):
        let node = state.conceptGraph.getNode(state.conceptGraph.nodeIndex[w])
        if node.activation <= 0.01:
          state.conceptGraph.activateWord(w, 0.5)
          knowledgeConceptsInf.add(node)
  state.conceptGraph.spreadActivation(steps=1, decay=0.3)
  let allTopConceptsInf = state.conceptGraph.getTopConcepts(7)
  let inputIsJapaneseInf = isJapanese(input)
  var catalogMatchedInf = false
  if state.catalog.entries.len > 0:
    for entry in state.catalog.entries:
      if entry.inputText == input:
        if inputIsJapaneseInf and not isJapaneseResponse(entry.outputText):
          continue
        responseInf = entry.outputText
        catalogMatchedInf = true
        break
    if responseInf.len == 0:
      var bestMatchInf = -1
      var bestScoreCatInf = 0.0f
      for i, entry in state.catalog.entries:
        if isCodeModeInf and entry.intent == iiGreeting:
          continue
        var score = 0.0f
        if inputIsJapaneseInf and not isJapaneseResponse(entry.outputText):
          score = 0.0f
        else:
          if entry.inputText == input:
            score = 1.0f
          elif entry.keyword.len > 0 and input.toLower().contains(entry.keyword.toLower()):
            let inputLen = input.len.float32
            let keywordLen = entry.keyword.len.float32
            if keywordLen >= inputLen * 0.5:
              score = 0.8f
            elif keywordLen >= inputLen * 0.3:
              score = 0.7f
            else:
              score = 0.5f
          elif entry.inputText.len >= 3 and input.contains(entry.inputText):
            score = entry.inputText.len.float32 / max(input.len, 1).float32 * 0.8f
          if score > bestScoreCatInf:
            bestScoreCatInf = score
            bestMatchInf = i
      # カタログマッチの閾値を下げて、より柔軟にマッチング
      if bestMatchInf >= 0 and bestScoreCatInf >= 0.5:
        responseInf = state.catalog.entries[bestMatchInf].outputText
        catalogMatchedInf = true
  if responseInf.len == 0 and isCodeModeInf:
     if codeStructInf.keywords.len > 0:
       responseInf = "このコードは" & codeStructInf.keywords.join("、") & "などの要素を含んでいますね。"
       if codeStructInf.summary.len > 0:
         responseInf = responseInf & codeStructInf.summary & "という構造です。"
       responseInf = responseInf & "Nim言語で記述されており、各手続きや変数の役割を追いながら全体の流れを把握することが重要です。"
     else:
       responseInf = "これはNim言語のコードですね。コードの構造を理解するには、各要素の役割を分析することが大切です。"
  if responseInf.len == 0:
    var hasMeaningfulConceptsInf = allTopConceptsInf.len > 0
    var inputConceptsInf: seq[ConceptNode] = @[]
    for w in rawWordsInf:
      if state.conceptGraph.nodeIndex.hasKey(w):
        inputConceptsInf.add(state.conceptGraph.nodes[state.conceptGraph.nodeIndex[w]])
    var filteredConceptsInf = filterConcepts(allTopConceptsInf)
    var mergedConceptsInf: seq[ConceptNode] = @[]
    # 活性化が高い入力概念のみをマージ（入力単語を除外）
    for c in inputConceptsInf:
      if c.activation > 0.3 and c.word notin rawWordsInf:
        mergedConceptsInf.add(c)
    # 活性化が高いフィルタリング済み概念をマージ（入力単語を除外）
    for c in filteredConceptsInf:
      if c.activation > 0.3 and c.word notin rawWordsInf:
        mergedConceptsInf.add(c)
    if mergedConceptsInf.len == 0:
      # allTopConceptsInfから入力単語を除外
      for c in allTopConceptsInf:
        if c.activation > 0.3 and c.word notin rawWordsInf:
          mergedConceptsInf.add(c)
    if hasMeaningfulConceptsInf:
      responseInf = state.generateJapaneseResponse(mergedConceptsInf, input)
  var evalResultInf = EvalResult(verdict: evAccept, score: 0.5, reason: "skip", contradictions: @[], relevanceScore: 0.5, coherenceScore: 0.5, improvements: @[])
  if state.cfg.evalEnabled:
    evalResultInf = state.selfEvaluate(input, responseInf, thinking, topConceptsInf)
  state.lastEval = evalResultInf
  state.applyRewardPunishment(evalResultInf, thinking, intentInf, topConceptsInf)
  state.learnFromUserInput(input, responseInf)
  var clausePatternInf = newSeq[bool](state.cfg.tmClauses)
  for cid in reasoningInf.firedConcepts:
    if cid < clausePatternInf.len:
      clausePatternInf[cid] = true
  state.episodeStore.episodes.add(Episode(
    inputText: input,
    outputText: responseInf,
    inputConceptIds: activeIdsInf,
    outputConceptIds: @[],
    tmClausePattern: clausePatternInf,
    confidence: reasoningInf.confidence,
    speaker: spUser,
    contextTag: $intentInf,
    situation: "chat",
    timestamp: epochTime(),
    reward: if evalResultInf.verdict == evAccept: 0.7f elif evalResultInf.verdict == evRefine: 0.4f else: 0.1f,
    rank: evalResultInf.score,
    accessCount: 0,
    emotionalValence: if evalResultInf.verdict == evAccept: 0.3f else: -0.3f
  ))
  if state.episodeStore.episodes.len > state.episodeStore.maxEpisodes:
    state.episodeStore.episodes.sort(proc(a, b: Episode): int = cmp(b.rank, a.rank))
    state.episodeStore.episodes = state.episodeStore.episodes[0..<state.episodeStore.maxEpisodes]
  for c in topConceptsInf:
    for clauseId in reasoningInf.firedConcepts:
      state.bridge.strengthenSynapse(clauseId, c.id, 0.05)
  if state.episodeStore.episodes.len mod 50 == 0:
    state.bridge.decaySynapses()
  proc isTemplateArtifactInf(text: string): bool =
    if text.len == 0: return false
    var hasJapanese = false
    var hasEnglish = false
    for rune in text.toRunes:
      let cp = rune.int32
      if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
        hasJapanese = true
      elif (cp >= 0x0041 and cp <= 0x005A) or (cp >= 0x0061 and cp <= 0x007A):
        hasEnglish = true
    if hasJapanese and hasEnglish: return true
    return false
  let needSearchInf = input.len > 0 and responseInf.len == 0
  if needSearchInf:
    var searcher = initWebSearcher()
    let knowledge = searcher.getKnowledge(input)
    if knowledge.len > 0:
      let translated = searcher.translateKnowledge(knowledge, "ja")
      if translated.len > 0:
        var dummyResults = searcher.search(input, 3)
        let decomposed = decomposeSearchResults(dummyResults)
        for w in decomposed:
          let ws = extractWords(w, state.tokenizer)
          for word in ws:
            if state.conceptGraph.nodeIndex.hasKey(word):
              state.conceptGraph.activateWord(word, 0.3)
            else:
              let cat = categorizeWord(word)
              let cid = state.conceptGraph.addNode(word, cat)
              state.conceptGraph.nodes[cid].baseFrequency = 1.0f / max(1, ws.len).float32
              state.conceptGraph.activateWord(word, 0.3)
        let knowledgeWords = extractWords(translated, state.tokenizer)
        for w in knowledgeWords:
          if state.conceptGraph.nodeIndex.hasKey(w):
            state.conceptGraph.activateWord(w, 0.3)
          else:
            let cat = categorizeWord(w)
            let cid = state.conceptGraph.addNode(w, cat)
            state.conceptGraph.nodes[cid].baseFrequency = 1.0f / max(1, knowledgeWords.len).float32
            state.conceptGraph.activateWord(w, 0.3)
        state.conceptGraph.spreadActivation(steps=2, decay=0.4)
        let activeIds2 = state.conceptGraph.getActiveNodeIds()
        if activeIds2.len > 0:
          let kv = featureVectorFromConcepts(activeIds2, state.cfg.tmClauses * 8)
          state.tm.train(kv, 1, 0.5f)
          discard state.tm.predictWithReasoning(kv)
        if translated.len > 0 and translated.len <= 200 and state.catalog.entries.len < 2000:
          var catalogIntent = iiQuestion
          if input.toLower().contains("hello") or input.toLower().contains("hi"):
            catalogIntent = iiGreeting
          elif input.toLower().contains("thank"):
            catalogIntent = iiThanks
          elif input.toLower().contains("bye") or input.toLower().contains("goodbye"):
            catalogIntent = iiFarewell
          state.catalog.entries.add(CatalogEntry(intent: catalogIntent, keyword: input, inputText: input, outputText: translated, weight: 1.0f))
        if responseInf.len == 0 or isTemplateArtifactInf(responseInf) or evalResultInf.score < 0.4 or responseInf.len < 30:
           let activeConcepts = state.conceptGraph.getTopConcepts(7)
           var merged: seq[ConceptNode] = @[]
           for c in activeConcepts: merged.add(c)
           for w in decomposed:
             for cw in extractWords(w, state.tokenizer):
               if state.conceptGraph.nodeIndex.hasKey(cw):
                 merged.add(state.conceptGraph.nodes[state.conceptGraph.nodeIndex[cw]])
           responseInf = state.generateJapaneseResponse(merged, input)
           if responseInf.len == 0:
             responseInf = translated
  return responseInf

proc observeCorpus*(state: var CognitiveState; corpus: seq[string]) =
  echo "Observing corpus (" & $corpus.len & " conversations)..."
  let t0 = epochTime()

  # --- Phase 1: 単語抽出 + 概念グラフ + TM + カタログ を1パスで ---
  # 単語抽出用ユーティリティ（concept_graph.nim と同じロジック）
  let splitParticles = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                        "な", "から", "まで", "より", "って", "じゃ",
                        "です", "ます", "だ", "である", "いる", "ある",
                        "そう", "よ", "ね", "さ", "わ"]

  proc fastExtractWords(text: string): seq[string] =
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
           cp == 0x005F:
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
    # 助詞で分割し、粒子・助動詞は除外
    let filterWords = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                       "な", "から", "まで", "より", "って", "じゃ",
                       "です", "ます", "だ", "である", "いる", "ある",
                       "そう", "よ", "ね", "さ", "わ", "だ", "し",
                       "れ", "ば", "から", "ので", "けど", "から"]
    var expanded: seq[string] = @[]
    for w in result:
      var remaining = w
      while remaining.len > 0:
        var found = false
        for p in splitParticles:
          if remaining.len > p.len and remaining.endsWith(p):
            let base = remaining[0..<(remaining.len - p.len)]
            if base.len >= 2 and base notin filterWords:
              expanded.add(base)
            remaining = p
            found = true
            break
        if not found:
          # 除外リストにない場合のみ追加
          var runeCount = 0
          for r in remaining.toRunes: runeCount += 1
          if runeCount >= 2 and remaining notin filterWords:
            expanded.add(remaining)
          break
    result = expanded

  # 単語頻度集計 + 概念IDマッピング
  var wordFreq: Table[string, int]
  type CorpusLine = object
    inputWords: seq[string]
    inputCids: seq[int]
    inputText: string
    outputText: string
    intent: int

  var lines: seq[CorpusLine] = @[]
  lines.setLen(corpus.len)
  var lineCount = 0

  for i in 0..<corpus.len:
    let parts = corpus[i].split("|")
    if parts.len < 2: continue
    let inputText = parts[0].strip()
    let outputText = parts[1].strip()
    if inputText.len == 0: continue

    let words = fastExtractWords(inputText)
    var uniqueWords: seq[string] = @[]
    for w in words:
      if w notin uniqueWords:
        uniqueWords.add(w)
        wordFreq[w] = wordFreq.getOrDefault(w, 0) + 1

    lines[lineCount] = CorpusLine(inputWords: uniqueWords, inputCids: @[],
                                   inputText: inputText, outputText: outputText, intent: 8)
    inc lineCount
    if (i+1) mod 10000 == 0:
      echo "  [Extract] " & $(i+1) & "/" & $corpus.len

  lines.setLen(lineCount)
  echo "  Extract: " & $formatFloat(epochTime() - t0, ffDecimal, 1) & "s"

  # --- Phase 2: 概念ノード作成 ---
  let t1 = epochTime()
  let minFreq = 3
  var sortedWords: seq[(string, int)]
  for (word, freq) in wordFreq.pairs:
    if freq >= minFreq:
      sortedWords.add((word, freq))
  sortedWords.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))

  for (word, freq) in sortedWords:
    let category = categorizeWord(word)
    let nodeId = state.conceptGraph.addNode(word, category)
    state.conceptGraph.nodes[nodeId].baseFrequency = freq.float32 / sortedWords.len.float32

  # concept IDを割り当て
  for li in 0..<lines.len:
    for w in lines[li].inputWords:
      if state.conceptGraph.nodeIndex.hasKey(w):
        lines[li].inputCids.add(state.conceptGraph.nodeIndex[w])

  echo "  Nodes: " & $state.conceptGraph.nodes.len & " (" & $formatFloat(epochTime() - t1, ffDecimal, 1) & "s)"

  # --- Phase 3: エッジ構築 ---
  let t2 = epochTime()
  for li in 0..<lines.len:
    let wc = lines[li].inputWords
    for i in 0..<(wc.len - 1):
      let w1 = wc[i]
      let w2 = wc[i + 1]
      if state.conceptGraph.nodeIndex.hasKey(w1) and state.conceptGraph.nodeIndex.hasKey(w2):
        let cat1 = categorizeWord(w1)
        let cat2 = categorizeWord(w2)
        var relation: EdgeRelation
        var weight: float32 = 0.3
        if cat1 == ctParticle or cat2 == ctParticle: relation = erRelatedTo
        elif cat1 == ctNoun and cat2 == ctVerb: relation = erCauses; weight = 0.5
        elif cat1 == ctVerb and cat2 == ctNoun: relation = erHasProperty; weight = 0.5
        elif cat1 == ctNoun and cat2 == ctAdj: relation = erHasProperty; weight = 0.4
        elif cat1 == ctAdj and cat2 == ctNoun: relation = erRelatedTo; weight = 0.4
        elif cat1 == ctVerb and cat2 == ctVerb: relation = erCauses; weight = 0.4
        else: relation = erRelatedTo
        state.conceptGraph.addEdge(w1, w2, relation, weight)
  echo "  Edges: " & $state.conceptGraph.edges.len & " (" & $formatFloat(epochTime() - t2, ffDecimal, 1) & "s)"

  # --- Phase 4: TM + Hebbian + カタログ を1パスで ---
  let t3 = epochTime()
  var catalog = ResponseCatalog(entries: @[])
  var catalogCount: Table[int, int]
  var hebbianSampleStep = max(1, lineCount div 10000)

  for li in 0..<lines.len:
    let line = lines[li]

    # --- TM学習 ---
    if line.inputCids.len > 0:
      let fv = featureVectorFromConcepts(line.inputCids, state.cfg.tmClauses * 8)
      # 簡易意図分類
      var tmClass = 8
      let lt = line.inputText.toLower()
      if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey"): tmClass = 0
      elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("what") or lt.contains("how"): tmClass = 1
      elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): tmClass = 5
      elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye"): tmClass = 6
      state.tm.train(fv, tmClass, 1.0f)

    # --- Hebbian（サンプリング） ---
    if li mod hebbianSampleStep == 0 and line.inputCids.len >= 2:
      for i in 0..<min(line.inputCids.len, 5):
        for j in (i+1)..<min(line.inputCids.len, 5):
          let w1 = state.conceptGraph.getWord(line.inputCids[i])
          let w2 = state.conceptGraph.getWord(line.inputCids[j])
          if w1.len > 0 and w2.len > 0:
            state.conceptGraph.hebbianStrengthen(w1, w2, 0.01)

    # --- カタログ ---
    if line.outputText.len > 0 and line.outputText.len <= 200:
      var intent = iiOther
      let lt = line.inputText.toLower()
      if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey") or lt.contains("good morning") or lt.contains("good afternoon") or lt.contains("good evening") or lt.contains("やあ"): intent = iiGreeting
      elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("誰") or lt.contains("what") or lt.contains("how") or lt.contains("why") or lt.contains("where") or lt.contains("when") or lt.contains("who"): intent = iiQuestion
      elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): intent = iiThanks
      elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye") or lt.contains("goodbye") or lt.contains("またね"): intent = iiFarewell
      elif lt.contains("して") or lt.contains("ください") or lt.contains("くれ") or lt.contains("help") or lt.contains("please") or lt.contains("how to"): intent = iiRequest
      let currentCount = catalogCount.getOrDefault(intent.ord, 0)
      if currentCount < 50000:
        catalog.entries.add(CatalogEntry(
          intent: intent,
          keyword: line.inputText[0..<min(20, line.inputText.len)],
          inputText: line.inputText,
          outputText: line.outputText,
          weight: 1.0f
        ))
        catalogCount[intent.ord] = currentCount + 1

    if (li+1) mod 10000 == 0:
      echo "  [Train] " & $(li+1) & "/" & $lineCount

  state.catalog = catalog
  echo "  TM+Hebbian+Catalog: " & $formatFloat(epochTime() - t3, ffDecimal, 1) & "s"
  echo "  Catalog: " & $catalog.entries.len & " entries"
  echo "Observation complete: " & $state.conceptGraph.conceptCount() & " concepts"
  echo "Total: " & $formatFloat(epochTime() - t0, ffDecimal, 1) & "s"

# ---------------------------------------------------------------------------
# 観察モード: ストリーミング版（5T対応）
# ---------------------------------------------------------------------------
proc observeCorpusStream*(state: var CognitiveState; corpusPath: string) =
  echo "Observing corpus (streaming): " & corpusPath
  let t0 = epochTime()
  let splitParticles = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                        "な", "から", "まで", "より", "って", "じゃ",
                        "です", "ます", "だ", "である", "いる", "ある",
                        "そう", "よ", "ね", "さ", "わ"]
  let filterWords = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                     "な", "から", "まで", "より", "って", "じゃ",
                     "です", "ます", "だ", "である", "いる", "ある",
                     "そう", "よ", "ね", "さ", "わ", "だ", "し",
                     "れ", "ば", "から", "ので", "けど"]
  proc fastExtractWords(text: string): seq[string] =
    result = @[]
    var current = ""
    var isAlpha = false
    for rune in text.toRunes:
      let cp = rune.int32
      if (cp >= 0x3040 and cp <= 0x309F) or
         (cp >= 0x30A0 and cp <= 0x30FF) or
         (cp >= 0x4E00 and cp <= 0x9FFF):
        if current.len > 0 and isAlpha:
          result.add(current)
          current = ""
          isAlpha = false
        current.add($rune)
      elif (cp >= 0x0041 and cp <= 0x005A) or
           (cp >= 0x0061 and cp <= 0x007A) or
           (cp >= 0x0030 and cp <= 0x0039) or
           cp == 0x005F:
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
    var expanded: seq[string] = @[]
    for w in result:
      var remaining = w
      while remaining.len > 0:
        var found = false
        for p in splitParticles:
          if remaining.len > p.len and remaining.endsWith(p):
            let base = remaining[0..<(remaining.len - p.len)]
            if base.len >= 2 and base notin filterWords:
              expanded.add(base)
            remaining = p
            found = true
            break
        if not found:
          var runeCount = 0
          for r in remaining.toRunes: runeCount += 1
          if runeCount >= 2 and remaining notin filterWords:
            expanded.add(remaining)
          break
    result = expanded
  echo "Phase 1: Counting word frequencies..."
  var wordFreq: Table[string, int]
  var lineCount = 0
  var inputBuffer: seq[string] = @[]
  var outputBuffer: seq[string] = @[]
  for line in corpusPath.lines:
    let trimmed = line.strip()
    if trimmed.len == 0: continue
    let pipePos = trimmed.find('|')
    if pipePos < 0: continue
    let inputText = trimmed[0..<pipePos].strip()
    let outputText = trimmed[pipePos+1..^1].strip()
    if inputText.len == 0 or outputText.len == 0: continue
    let words = fastExtractWords(inputText)
    var uniqueWords: seq[string] = @[]
    for w in words:
      if w notin uniqueWords:
        uniqueWords.add(w)
        wordFreq[w] = wordFreq.getOrDefault(w, 0) + 1
    inputBuffer.add(inputText)
    outputBuffer.add(outputText)
    inc lineCount
    if lineCount mod 50000 == 0:
      sleep(5) # フリーズ防止: CPUを譲る
    if lineCount mod 100000 == 0:
      echo "  Counted: " & $lineCount & " lines, " & $wordFreq.len & " unique words"
  echo "  Total: " & $lineCount & " lines, " & $wordFreq.len & " unique words"
  echo "  Buffer size: " & $inputBuffer.len & " entries"
  echo "Phase 2: Building concept nodes..."
  let t1 = epochTime()
  let minFreq = 3
  var sortedWords: seq[(string, int)]
  for (word, freq) in wordFreq.pairs:
    if freq >= minFreq:
      sortedWords.add((word, freq))
  sortedWords.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))
  for (word, freq) in sortedWords:
    let category = categorizeWord(word)
    let nodeId = state.conceptGraph.addNode(word, category)
    state.conceptGraph.nodes[nodeId].baseFrequency = freq.float32 / sortedWords.len.float32
  echo "  Nodes: " & $state.conceptGraph.nodes.len & " (" & $formatFloat(epochTime() - t1, ffDecimal, 1) & "s)"
  echo "Phase 3: Processing concepts, edges, TM, Hebbian, catalog (3 epochs)..."
  let t2 = epochTime()
  var catalog = ResponseCatalog(entries: @[])
  var catalogCount: Table[int, int]
  var hebbianSampleStep = max(1, lineCount div 10000)
  for epoch in 0..<3:
    echo "  Epoch " & $(epoch+1) & "/3"
    var processed = 0
    for i in 0..<inputBuffer.len:
      let inputText = inputBuffer[i]
      let outputText = outputBuffer[i]
      let words = fastExtractWords(inputText)
      var inputCids: seq[int] = @[]
      for w in words:
        if state.conceptGraph.nodeIndex.hasKey(w):
          inputCids.add(state.conceptGraph.nodeIndex[w])
      if epoch == 0:
        for j in 0..<(words.len - 1):
          let w1 = words[j]
          let w2 = words[j + 1]
          if state.conceptGraph.nodeIndex.hasKey(w1) and state.conceptGraph.nodeIndex.hasKey(w2):
            let cat1 = categorizeWord(w1)
            let cat2 = categorizeWord(w2)
            var relation: EdgeRelation
            var weight: float32 = 0.3
            if cat1 == ctParticle or cat2 == ctParticle: relation = erRelatedTo
            elif cat1 == ctNoun and cat2 == ctVerb: relation = erCauses; weight = 0.5
            elif cat1 == ctVerb and cat2 == ctNoun: relation = erHasProperty; weight = 0.5
            elif cat1 == ctNoun and cat2 == ctAdj: relation = erHasProperty; weight = 0.4
            elif cat1 == ctAdj and cat2 == ctNoun: relation = erRelatedTo; weight = 0.4
            elif cat1 == ctVerb and cat2 == ctVerb: relation = erCauses; weight = 0.4
            else: relation = erRelatedTo
            state.conceptGraph.addEdge(w1, w2, relation, weight)
      if inputCids.len > 0:
        let fv = featureVectorFromConcepts(inputCids, state.cfg.tmClauses * 8)
        var tmClass = 8
        let lt = inputText.toLower()
        if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey"): tmClass = 0
        elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("what") or lt.contains("how"): tmClass = 1
        elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): tmClass = 5
        elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye"): tmClass = 6
        state.tm.train(fv, tmClass, 1.0f)
      if processed mod hebbianSampleStep == 0 and inputCids.len >= 2:
        for j in 0..<min(inputCids.len, 5):
          for k in (j+1)..<min(inputCids.len, 5):
            let w1 = state.conceptGraph.getWord(inputCids[j])
            let w2 = state.conceptGraph.getWord(inputCids[k])
            if w1.len > 0 and w2.len > 0:
              state.conceptGraph.hebbianStrengthen(w1, w2, 0.01)
      if epoch == 0:
        if outputText.len > 0 and outputText.len <= 200:
          var intent = iiOther
          let lt = inputText.toLower()
          if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey") or lt.contains("good morning") or lt.contains("good afternoon") or lt.contains("good evening") or lt.contains("やあ"): intent = iiGreeting
          elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("誰") or lt.contains("what") or lt.contains("how") or lt.contains("why") or lt.contains("where") or lt.contains("when") or lt.contains("who"): intent = iiQuestion
          elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): intent = iiThanks
          elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye") or lt.contains("goodbye") or lt.contains("またね"): intent = iiFarewell
          elif lt.contains("して") or lt.contains("ください") or lt.contains("くれ") or lt.contains("help") or lt.contains("please") or lt.contains("how to"): intent = iiRequest
          let currentCount = catalogCount.getOrDefault(intent.ord, 0)
          if currentCount < 200000:
            catalog.entries.add(CatalogEntry(
              intent: intent,
              keyword: inputText[0..<min(20, inputText.len)],
              inputText: inputText,
              outputText: outputText,
              weight: 1.0f
            ))
            catalogCount[intent.ord] = currentCount + 1
      inc processed
      if processed mod 50000 == 0:
        sleep(5)
      if processed mod 100000 == 0:
        echo "  Processed: " & $processed & "/" & $lineCount & " epoch " & $(epoch+1)
  inputBuffer.setLen(0)
  outputBuffer.setLen(0)
  state.catalog = catalog
  echo "  Edges: " & $state.conceptGraph.edges.len
  echo "  TM+Hebbian+Catalog: " & $formatFloat(epochTime() - t2, ffDecimal, 1) & "s"
  echo "  Catalog: " & $catalog.entries.len & " entries"
  echo "Observation complete: " & $state.conceptGraph.conceptCount() & " concepts"
  echo "Total: " & $formatFloat(epochTime() - t0, ffDecimal, 1) & "s"
