import os, strutils, tables, algorithm, math, times, unicode, sequtils
import types, tokenizer, concept_graph, working_memory, tsetlin, generator, grammar
import intent_classifier, semantic_matcher, web_search

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

  var step1 = ThinkingStep(kind: tsPerception, description: "入力を分析")
  step1.details.add("入力: \"" & input & "\"")
  step1.details.add("トークン数: " & $words.len)
  step1.confidence = 1.0
  chain.steps.add(step1)

  var step2 = ThinkingStep(kind: tsActivation, description: "関連概念を活性化")
  for c in topConcepts:
    step2.details.add(c.word & " (act=" & $formatFloat(c.activation, ffDecimal, 3) & ")")
  step2.confidence = if topConcepts.len > 0: 0.8f else: 0.2f
  chain.steps.add(step2)

  var step3 = ThinkingStep(kind: tsSpreading, description: "概念ネットワークを探索")
  step3.details.add("伝播ステップ: " & $state.cfg.spreadSteps)
  step3.confidence = if topConcepts.len >= 3: 0.7f else: 0.4f
  chain.steps.add(step3)

  var step4 = ThinkingStep(kind: tsReasoning, description: "論理ルールを適用")
  chain.reasoningPath = reasoning.firedConcepts
  step4.details.add("発火条款数: " & $reasoning.firedConcepts.len)
  step4.details.add("推論確信度: " & $formatFloat(reasoning.confidence, ffDecimal, 3))
  step4.confidence = reasoning.confidence
  chain.steps.add(step4)

  var step5 = ThinkingStep(kind: tsRetrieval, description: "過去の経験を参照")
  if bestEpisode.len > 0 and bestScore > 0.3:
    step5.details.add("類似エピソード発見 (score=" & $formatFloat(bestScore, ffDecimal, 3) & ")")
    step5.confidence = bestScore
  else:
    step5.details.add("類似エピソードなし")
    step5.confidence = 0.1
  chain.steps.add(step5)

  var step6 = ThinkingStep(kind: tsSelection, description: "表現パターンを選択")
  step6.confidence = 0.5
  chain.steps.add(step6)

  var totalConf = 0.0
  for s in chain.steps:
    totalConf += s.confidence.float
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

  # 1. 感知
  # BPEトークンは概念マッチング用、入力単語は概念活性化用
  let bpeTokens = extractWords(input, state.tokenizer)
  # 入力から単語を抽出（Unicode正規化してから分割）
  let normalized = input.normalize()
  var rawWords: seq[string] = @[]
  var current = ""
  for ch in normalized:
    let c = ch.ord
    # ASCII範囲外（日本語文字）は単語の一部として扱う
    if c > 0x7F:
      current.add($ch)
    elif ch == ' ' or ch == ',' or ch == '.' or ch == '\n' or ch == '\t':
      if current.len > 0:
        rawWords.add(current)
        current = ""
    else:
      # ASCII文字（助詞や記号）も単語として扱う
      current.add($ch)
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

  # 未学習語をトラッキング
  state.trackUnknownWords(rawWords)

  # 2. 伝播
  state.conceptGraph.spreadActivation(
    steps = state.cfg.spreadSteps,
    decay = state.cfg.spreadDecay
  )

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

  # 5. TM推論（意図ベースのクラス）
  let activeIds = state.conceptGraph.getActiveNodeIds()
  let fv = featureVectorFromConcepts(activeIds, state.cfg.tmClauses * 8)
  var reasoning = state.tm.predictWithReasoning(fv)

  # 5.5 シナプスによるTM出力修正
  var modClauses = state.tm.clauseOutput
  state.bridge.applySynapticModulation(modClauses, activeIds)
  # 修正後の条款で再推論
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

  # 8. 応答生成（概念活性化 + カタログ照合 or 文法）
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

  # 8b. カタログ照合（完全一致→部分一致）
  if state.catalog.entries.len > 0:
    for entry in state.catalog.entries:
      if entry.inputText == input:
        response = entry.outputText
        break
    if response.len == 0:
      var bestMatch = -1
      var bestScoreCat = 0.0f
      for i, entry in state.catalog.entries:
        var score = 0.0f
        # 完全一致は最高スコア
        if entry.inputText == input:
          score = 1.0f
        # 入力がカタログエントリのinputTextを含む（キーワードベースの完全一致）
        elif entry.keyword.len > 0 and input.toLower().contains(entry.keyword.toLower()):
          # キーワードが入力の主要部分である場合のみスコア
          let inputLen = input.len.float32
          let keywordLen = entry.keyword.len.float32
          if keywordLen >= inputLen * 0.5:
            score = 0.8f
          elif keywordLen >= inputLen * 0.3:
            score = 0.7f
          else:
            score = 0.5f
        # 入力がエントリを含む（部分一致）
        elif entry.inputText.len >= 3 and input.contains(entry.inputText):
          score = entry.inputText.len.float32 / max(input.len, 1).float32 * 0.8f
        if score > bestScoreCat:
          bestScoreCat = score
          bestMatch = i
      if bestMatch >= 0 and bestScoreCat >= 0.7:
        response = state.catalog.entries[bestMatch].outputText

  # 8c. カタログで応答できなかった場合のみ文法生成
  if response.len == 0:
    # 意味的な概念がない場合は空を返す（不自然な応答を防止）
    # 学習済み概念（base_frequency > 0）のみ使用
    var hasMeaningfulConcepts = false
    for c in allTopConcepts:
      if c.category in [ctNoun, ctVerb, ctAdj, ctGreeting, ctQuestion] and c.baseFrequency > 0:
        hasMeaningfulConcepts = true
        break

    if hasMeaningfulConcepts:
      case intent
      of iiGreeting:
        var greetingConcepts: seq[ConceptNode] = @[]
        for c in allTopConcepts:
          if c.category == ctGreeting or c.category == ctNoun:
            greetingConcepts.add(c)
        if greetingConcepts.len == 0:
          greetingConcepts = allTopConcepts
        response = state.generator.generate(greetingConcepts, "greeting")

      of iiThanks:
        response = state.generator.generate(allTopConcepts, "greeting")

      of iiFarewell:
        response = state.generator.generate(allTopConcepts, "greeting")

      of iiQuestion:
        var topicConcepts: seq[ConceptNode] = @[]
        for c in allTopConcepts:
          if c.category != ctQuestion and c.category != ctParticle:
            topicConcepts.add(c)
        if topicConcepts.len == 0:
          topicConcepts = allTopConcepts
        response = state.generator.generate(topicConcepts, "question")

      of iiRequest:
        response = state.generator.generate(allTopConcepts, "polite")

      of iiOpinion, iiAgreement:
        response = state.generator.generate(allTopConcepts, "description")

      of iiStatement, iiOther:
        if knowledgeConcepts.len > 0:
          response = state.generator.generate(knowledgeConcepts, "description")
        else:
          response = state.generator.generate(allTopConcepts, "description")

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
    return false

  # 13. Web検索フォールバック（応答が空または低品質/テンプレート的な場合）
  # 検索結果は直接返すのではなく、認知プロセスに入力して推論に使う
  if input.len > 0 and (response.len == 0 or isTemplateArtifact(response) or evalResult.score < 0.3):
    var searcher = initWebSearcher()
    let knowledge = searcher.getKnowledge(input)
    if knowledge.len > 0:
      # 検索結果を日本語に翻訳
      let translated = searcher.translateKnowledge(knowledge, "ja")
      if translated.len > 0:
        # 翻訳結果を概念グラフに追加し、推論に使用
        let knowledgeWords = extractWords(translated, state.tokenizer)
        for w in knowledgeWords:
          if state.conceptGraph.nodeIndex.hasKey(w):
            state.conceptGraph.activateWord(w, 0.3)
          else:
            let cat = categorizeWord(w)
            let cid = state.conceptGraph.addNode(w, cat)
            state.conceptGraph.nodes[cid].baseFrequency = 1.0f / knowledgeWords.len.float32
            state.conceptGraph.activateWord(w, 0.3)

        # 新しい概念で活性化を拡散
        state.conceptGraph.spreadActivation(steps=2, decay=0.4)

        # TM推論（Web検索結果を学習）
        let activeIds = state.conceptGraph.getActiveNodeIds()
        if activeIds.len > 0:
          let kv = featureVectorFromConcepts(activeIds, state.cfg.tmClauses * 8)
          # TM学習: Web検索結果は新しい知識として強化
          let searchIntentClass = 1  # iiQuestion
          state.tm.train(kv, searchIntentClass, 0.5f)
          # TM推論で新しい概念の関連を学習
          discard state.tm.predictWithReasoning(kv)

        # 翻訳結果をカタログに追加（次回以降の推論に使用）
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

        # 応答が空の場合は翻訳結果を使用（日本語で）
        if response.len == 0 or isTemplateArtifact(response):
          response = translated

  return response

# ---------------------------------------------------------------------------
# 観察モード: 単一パス設計
# ---------------------------------------------------------------------------
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
  let minFreq = max(1, min(3, lineCount div 100))
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
      if currentCount < 5000:
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

  # 単語抽出ユーティリティ
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

  # --- Phase 1: ストリーミングで単語頻度集計 ---
  echo "Phase 1: Counting word frequencies..."
  var wordFreq: Table[string, int]
  var lineCount = 0
  var outputBuffer: seq[string] = @[]  # 出力テキストを一時保持
  var inputBuffer: seq[string] = @[]   # 入力テキストを一時保持

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

    if lineCount mod 100000 == 0:
      echo "  Counted: " & $lineCount & " lines, " & $wordFreq.len & " unique words"

  echo "  Total: " & $lineCount & " lines, " & $wordFreq.len & " unique words"
  echo "  Buffer size: " & $inputBuffer.len & " entries"

  # --- Phase 2: 高頻度単語で概念ノード作成 ---
  echo "Phase 2: Building concept nodes..."
  let t1 = epochTime()
  let minFreq = max(3, lineCount div 50000)
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

  # --- Phase 3: 2パス目（概念ID割り当て＋エッジ＋TM＋Hebbian＋カタログ） ---
  echo "Phase 3: Processing concepts, edges, TM, Hebbian, catalog..."
  let t2 = epochTime()
  var catalog = ResponseCatalog(entries: @[])
  var catalogCount: Table[int, int]
  var hebbianSampleStep = max(1, lineCount div 10000)
  var processed = 0

  for i in 0..<inputBuffer.len:
    let inputText = inputBuffer[i]
    let outputText = outputBuffer[i]

    let words = fastExtractWords(inputText)
    var inputCids: seq[int] = @[]
    for w in words:
      if state.conceptGraph.nodeIndex.hasKey(w):
        inputCids.add(state.conceptGraph.nodeIndex[w])

    # --- エッジ構築 ---
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

    # --- TM学習 ---
    if inputCids.len > 0:
      let fv = featureVectorFromConcepts(inputCids, state.cfg.tmClauses * 8)
      var tmClass = 8
      let lt = inputText.toLower()
      if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey"): tmClass = 0
      elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("what") or lt.contains("how"): tmClass = 1
      elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): tmClass = 5
      elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye"): tmClass = 6
      state.tm.train(fv, tmClass, 1.0f)

    # --- Hebbian（サンプリング） ---
    if processed mod hebbianSampleStep == 0 and inputCids.len >= 2:
      for j in 0..<min(inputCids.len, 5):
        for k in (j+1)..<min(inputCids.len, 5):
          let w1 = state.conceptGraph.getWord(inputCids[j])
          let w2 = state.conceptGraph.getWord(inputCids[k])
          if w1.len > 0 and w2.len > 0:
            state.conceptGraph.hebbianStrengthen(w1, w2, 0.01)

    # --- カタログ ---
    if outputText.len > 0 and outputText.len <= 200:
      var intent = iiOther
      let lt = inputText.toLower()
      if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey") or lt.contains("good morning") or lt.contains("good afternoon") or lt.contains("good evening") or lt.contains("やあ"): intent = iiGreeting
      elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("誰") or lt.contains("what") or lt.contains("how") or lt.contains("why") or lt.contains("where") or lt.contains("when") or lt.contains("who"): intent = iiQuestion
      elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): intent = iiThanks
      elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye") or lt.contains("goodbye") or lt.contains("またね"): intent = iiFarewell
      elif lt.contains("して") or lt.contains("ください") or lt.contains("くれ") or lt.contains("help") or lt.contains("please") or lt.contains("how to"): intent = iiRequest
      let currentCount = catalogCount.getOrDefault(intent.ord, 0)
      if currentCount < 20000:
        catalog.entries.add(CatalogEntry(
          intent: intent,
          keyword: inputText[0..<min(20, inputText.len)],
          inputText: inputText,
          outputText: outputText,
          weight: 1.0f
        ))
        catalogCount[intent.ord] = currentCount + 1

    inc processed
    if processed mod 100000 == 0:
      echo "  Processed: " & $processed & "/" & $lineCount

  # バッファ解放
  inputBuffer.setLen(0)
  outputBuffer.setLen(0)

  state.catalog = catalog
  echo "  Edges: " & $state.conceptGraph.edges.len
  echo "  TM+Hebbian+Catalog: " & $formatFloat(epochTime() - t2, ffDecimal, 1) & "s"
  echo "  Catalog: " & $catalog.entries.len & " entries"
  echo "Observation complete: " & $state.conceptGraph.conceptCount() & " concepts"
  echo "Total: " & $formatFloat(epochTime() - t0, ffDecimal, 1) & "s"
