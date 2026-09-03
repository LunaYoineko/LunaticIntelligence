{.push warning[UnusedImport]: off, hint[XDeclaredButNotUsed]: off, hint[DuplicateModuleImport]: off, warning[ResultShadowed]: off, warning[UnreachableElse]: off.}
import os, strutils, tables, algorithm, math, times, unicode, sequtils, random
import types, tokenizer, concept_graph, working_memory, tsetlin, generator, grammar
import intent_classifier, semantic_matcher, web_search, simhash
import code_structure
import llm
import storage
import db_connector/db_sqlite
import resource_governor
import moe

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

const STOP_WORDS_EARLY* = ["の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                     "な", "から", "まで", "より", "って", "じゃ",
                     "です", "ます", "だ", "である", "いる", "ある",
                     "そう", "よ", "ね", "さ", "わ", "だ", "し",
                     "れ", "ば", "から", "ので", "けど", "から",
                     "こと", "もの", "ため", "ところ", "とき", "よう", "そう"]

# 前方宣言
proc extractWords*(text: string; tokenizer: Tokenizer): seq[string]

proc fillTemplate*(tmpl: string; replacements: Table[string, string]): string =
  result = tmpl
  for (key, value) in replacements.pairs:
    result = result.replace("{" & key & "}", value)
  return result

proc chooseRandomTemplate*(templates: seq[string]): string =
  if templates.len == 0: return ""
  let idx = int(trueRandFloatCog() * templates.len.float32) mod templates.len
  return templates[idx]

# LLMを使用した応答生成 - フロー: TMで意味理解→ピックアップ→LLM推論(Thinking含む)→回答
proc sliceRunes(s: string; n: int): string =
  var cnt = 0
  result = ""
  for r in s.toRunes:
    if cnt >= n: break
    result.add($r); inc cnt

proc sliceRunesRange(s: string; a,b: int): string =
  var cnt = 0
  result = ""
  for r in s.toRunes:
    if cnt >= b: break
    if cnt >= a: result.add($r)
    inc cnt

proc isBadWord*(w: string): bool =
  if w.len == 0: return true
  if w in STOP_WORDS_EARLY: return true
  if w.len < 2 and w notin ["AI", "Io"]: return true
  var rc = 0
  for _ in w.toRunes: inc rc
  if rc < 2: return true
  return false

proc topWord*(concepts: seq[ConceptNode]): string =
  var best = ""
  var bestAct = -1.0f
  for c in concepts:
    if isBadWord(c.word): continue
    if c.activation > bestAct:
      bestAct = c.activation
      best = c.word
  return best

proc collectWords*(concepts: seq[ConceptNode]; maxN: int = 5): seq[string] =
  result = @[]
  for c in concepts:
    if result.len >= maxN: break
    if isBadWord(c.word): continue
    if c.activation > 0.3:
      result.add(c.word)
  if result.len == 0:
    for c in concepts:
      if result.len >= maxN: break
      if c.word.len > 0 and c.word notin STOP_WORDS_EARLY:
        result.add(c.word)

# 高速単語抽出: CJK/英数字混在テキストから単語を分割（助詞分割・フィルタリング付き）
proc fastExtractWords*(text: string): seq[string] =
  result = @[]
  var current = ""
  var isAlpha = false
  let splitParticles = ["について", "とは何", "とは", "教えて", "何ですか", "ですか",
                        "の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                        "な", "から", "まで", "より", "って", "じゃ",
                        "です", "ます", "だ", "である", "いる", "ある",
                        "そう", "よ", "ね", "さ", "わ"]
  let filterWords = ["について", "とは何", "とは", "教えて", "何ですか", "ですか",
                     "の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                     "な", "から", "まで", "より", "って", "じゃ",
                     "です", "ます", "だ", "である", "いる", "ある",
                     "そう", "よ", "ね", "さ", "わ", "だ", "し",
                     "れ", "ば", "から", "ので", "けど"]
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
        isAlpha = false
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

proc shouldSearchForKnowledge*(state: CognitiveState; input: string; intent: InputIntent; desiredLen: int;
                                 topConcepts: seq[ConceptNode]; bestScore: float32; hasCatalogMatch: bool): bool =
  ## 検索要否をハードコードなしで動的判定: 文字数・活性・整合度のみで決定
  if hasCatalogMatch: return false
  if desiredLen <= 50: return false
  if intent notin [iiQuestion, iiRequest, iiOther]: return false
  var avgAct = 0.0
  if topConcepts.len > 0:
    for c in topConcepts: avgAct += c.activation
    avgAct /= topConcepts.len.float
  let hasStrongConcepts = avgAct > 0.28 and topConcepts.len >= 3
  let hasGoodEpisode = bestScore > 0.55
  if hasStrongConcepts and hasGoodEpisode: return false
  if countRunes(input) < 8 and avgAct < 0.2: return false
  return true

proc decideLengthByThinking*(input: string; thinking: ThinkingChain; intent: InputIntent): int =
  ## 文字数決定は 3要因で決まる: explicitLen(明示指示) / baseLen(意図別基準) / confAdjust(確信度補正)
  ## explicitLen: 「詳しく」「教えて」「とは」「何」等があれば優先的に長文
  ## baseLen: 質問550/一般450/挨拶40 を基準とし countRunes で補正
  ## confAdjust: Thinking確信度・stepsで±調整、countRunesで正規化
  # 事実系は短く
  if input.startsWith("事実:"):
    if input.contains("今は") or input.contains("今日は"): return 40
    return 80
  if input.contains("今何時") or input.contains("何時") or input.contains("日付は") or input.contains("今日は") and input.contains("日"):
    return 40
  let total = thinking.totalConfidence
  let steps = thinking.steps.len
  # explicitLen: 明示的に詳細を求めている（短文でも優先）
  var explicitLen = -1
  if input.contains("詳しく") or input.contains("教えて") or input.contains("説明して") or input.contains("とは") or input.contains("何？") or input.contains("何ですか"):
    if total > 0.6: explicitLen = 550
    else: explicitLen = 550
    # 550で統一（AIと量子どちらも長文を保証）
  if explicitLen >= 0:
    # 明示的な長文要求は短文判定より優先
    var confAdjustAlt = 0
    if total < 0.4: confAdjustAlt = -50
    elif total > 0.7 and steps >= 4: confAdjustAlt = 50
    elif steps >= 4: confAdjustAlt = 30
    return max(300, explicitLen + confAdjustAlt)
  # 短い入力は会話的応答: 一律50ではなく intentとThinkingで動的に決定（単語ハードコードなし、80以下でカタログ優先）
  let inputRunes = countRunes(input)
  if inputRunes < 15:
    var shortBase = 45
    case intent
    of iiGreeting, iiThanks, iiFarewell: shortBase = 40
    of iiQuestion: shortBase = 45
    of iiRequest: shortBase = 60
    else: shortBase = 50
    let dyn = shortBase + int(total * 10) + steps * 1
    return clamp(dyn, 40, 80)
  # baseLen: 意図別の基準値（countRunesで微調整）
  var baseLen = 450
  case intent
  of iiGreeting, iiThanks, iiFarewell: baseLen = 40
  of iiQuestion: baseLen = 550
  of iiRequest, iiOpinion, iiStatement: baseLen = 450
  else: baseLen = 450
  # countRunesが短い入力は補正少、長い入力は少し長めに
  if inputRunes > 20: baseLen += 20
  # confAdjust: 確信度で±補正
  var confAdjust = 0
  if total < 0.4: confAdjust = -50
  elif total > 0.7 and steps >= 4: confAdjust = 50
  elif steps >= 4: confAdjust = 30
  if explicitLen >= 0:
    return max(300, explicitLen + confAdjust)
  # intentが質問なら550、そうでなければ450を尊重
  if intent == iiQuestion and steps >= 4:
    return max(300, 550 + confAdjust)
  if total < 0.4:
    return max(300, 400 + confAdjust)
  return max(300, baseLen + confAdjust)

proc decideLength*(input: string; thinking: ThinkingChain; intent: InputIntent): int =
  ## wrapper: optimized decideLength uses countRunes
  return decideLengthByThinking(input, thinking, intent)

proc generateTopicAwareExtension*(input: string; concepts: seq[ConceptNode]; iter: int): string =
  ## トピックに関連した補足説明を生成（iterで多様化して単調さを回避）
  ## 1. conceptsから意味のある単語を抽出
  ## 2. 入力からキーワードを補充
  ## 3. iterで異なるテンプレートを選択（反復防止）
  var topicWords = collectWords(concepts, 5)
  # コンセプトから取れなければ入力からキーワード抽出
  if topicWords.len == 0:
    var current = ""
    var isCJK = false
    for rune in input.toRunes:
      let cp = rune.int32
      let isCjkChar = (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF)
      let isAlphaNum = (cp >= 0x0041 and cp <= 0x005A) or (cp >= 0x0061 and cp <= 0x007A) or (cp >= 0x0030 and cp <= 0x0039) or cp == 0x005F
      if isCjkChar:
        if current.len > 0 and not isCJK:
          if current.len >= 2 and current notin STOP_WORDS_EARLY:
            topicWords.add(current)
          current = ""
        current.add($rune)
        isCJK = true
      elif isAlphaNum:
        if current.len > 0 and isCJK:
          if current.len >= 2 and current notin STOP_WORDS_EARLY:
            topicWords.add(current)
          current = ""
        current.add($rune)
        isCJK = false
      else:
        if current.len > 0:
          if current.len >= 2 and current notin STOP_WORDS_EARLY:
            topicWords.add(current)
          current = ""
        isCJK = false
    if current.len > 0:
      if current.len >= 2 and current notin STOP_WORDS_EARLY:
        topicWords.add(current)
    # 入力単語を除外 + 汎用語は isBadWord/STOP_WORDS_EARLY で動的除外（ハードコード回避）
    var filtered: seq[string] = @[]
    for w in topicWords:
      if w notin input and not isBadWord(w):
        filtered.add(w)
    if filtered.len > 0:
      topicWords = filtered

  if topicWords.len == 0:
    return ""
  let topic = topicWords[0]
  if isBadWord(topic):
    return ""
  let secondaryTopic = if topicWords.len > 1: topicWords[1] else: topicWords[0]

  let templates = @[
    "「" & topic & "」には多角的な側面があり、さらなる理解が深まります。",
    topic & "の研究や応用は活発に進められており、新しい知見が期待されます。",
    "特に" & topic & "の分野では、" & secondaryTopic & " との関連も重要な観点です。",
    topic & "については、実践的な視点からも考察することで理解が深まります。",
    "この" & topic & "は現代において注目されており、将来的な展望も広がっています。",
  ]

  return templates[iter mod templates.len]

proc rankSearchResults*(results: seq[SearchResult]; state: CognitiveState): seq[SearchResult] =
  ## 10件を frequencyランキング + wobble(真乱数)で並替。frequency高いもの優先しつつ揺らぎで固定化防止
  if results.len <= 1: return results
  var scored: seq[(float32, SearchResult)] = @[]
  for r in results:
    var freqScore: float32 = 0.0
    # title/snippetに含まれる単語のbaseFrequencyを合算
    for w in extractWords(r.title & " " & r.snippet, state.tokenizer):
      if state.conceptGraph.nodeIndex.hasKey(w):
        freqScore += state.conceptGraph.nodes[state.conceptGraph.nodeIndex[w]].baseFrequency
      else:
        freqScore += 0.1
    freqScore += float32(countRunes(r.snippet)) * 0.01
    # wobble: 0.02の揺らぎ（コード時は0.005だが検索は一般会話なので0.02）
    let wobble = (trueRandFloatCog() - 0.5f) * 0.05f
    let total = freqScore + wobble
    scored.add((total, r))
  scored.sort(proc(a,b:(float32,SearchResult)):int = cmp(b[0], a[0]))
  result = @[]
  for (_, r) in scored: result.add(r)

proc synthesizeFromMultipleSources*(state: var CognitiveState; concepts: seq[ConceptNode]; input, combined: string; titles: seq[string]; results: seq[SearchResult]): string =
  if combined.len < 20: return 
  var base = combined.strip().replace("<span class=\"searchmatch\">","").replace("</span>","")
  let desiredLen = decideLengthByThinking(input, state.lastThinking, state.lastIntent)
  var outText = sliceRunes(base, desiredLen).strip()
  if not outText.endsWith("。") and not outText.endsWith("？") and not outText.endsWith("！"):
    outText.add("。")
  # 足りない場合はトピックに関連した補足を追加（grammar生成ではなく関連性のある内容）
  var fillerIter = 0
  while outText.countRunes < desiredLen - 20:
    var genExtra = generateTopicAwareExtension(input, concepts, fillerIter)
    if genExtra.len > 10:
      outText.add(" " & genExtra.strip())
      if not outText.endsWith("。"): outText.add("。")
    else:
      break
    inc fillerIter
    if outText.countRunes >= desiredLen - 10: break
    if fillerIter > 3: break
  outText = sliceRunes(outText, desiredLen).strip()
  if not outText.endsWith("。"): outText.add("。")
  if outText.len > 20:
    var clausePat = newSeq[bool](state.cfg.tmClauses)
    let activeIds = state.conceptGraph.getActiveNodeIds()
    for cid in activeIds:
      if cid < clausePat.len: clausePat[cid] = true
    state.episodeStore.episodes.add(Episode(inputText: input, outputText: outText, inputConceptIds: activeIds, outputConceptIds: @[], tmClausePattern: clausePat, confidence: 0.6, speaker: spSystem, contextTag: "search_synthesized", situation: "search", timestamp: epochTime(), reward: 0.5, rank: 0.6, accessCount: 0, emotionalValence: 0.1))
  return outText

# 前方宣言
proc buildLLMPrompt*(input: string; concepts: seq[ConceptNode]; intent: InputIntent; searchContext: string; desiredLen: int; systemPrompt: string = ""): string
proc extractResponseFromLLM*(raw: string): string
proc feedbackToConceptGraph*(state: var CognitiveState; thinkingText: string)

# 右脳応答生成（LLMベース、テンプレート不使用） - generateWithLLMより前に定義
proc generateRightBrainResponse*(state: var CognitiveState; concepts: seq[ConceptNode]; intent: InputIntent; input: string; systemPrompt: string = ""): string =
  let experts = initMoEExperts()
  let gating = gateByRightBrain(state.tm, state.intentClassifier, input, concepts)
  let expert = experts[gating.topExpert.ord]
  let desiredLen = decideLengthByThinking(input, state.lastThinking, intent)
  
  var llmState = initLLMState(initLLMConfig(4096))
  var ctx = newSeq[float32](llmState.config.dModel)
  for c in concepts:
    if c.activation > 0.1:
      let idx = (c.id * 7) mod ctx.len
      ctx[idx] += c.activation * 0.3
  
  let prompt = buildLLMPrompt(input, concepts, intent, "", desiredLen, systemPrompt)
  let raw = generateText(llmState, state.tokenizer, prompt, maxTokens=max(16, desiredLen div 2), temperature=expert.temperature, contextVec=ctx)
  let response = extractResponseFromLLM(raw)
  
  # 思考フィードバック（新旧両対応）
  var thinkingText = ""
  let ts = raw.find("<思考>")
  let te = raw.find("</思考>")
  if ts >= 0 and te > ts:
    thinkingText = raw[ts + 6 .. te - 1].strip()
  else:
    let s = raw.find("思考")
    let e = raw.find("思考終わり")
    if s >= 0 and e > s: thinkingText = raw[s+4 .. e-1].strip()
  if thinkingText.len > 5:
    feedbackToConceptGraph(state, thinkingText)
  
  return response

proc generateWithLLM*(state: var CognitiveState; concepts: seq[ConceptNode]; intent: InputIntent; input: string; forceLLM: bool = false; systemPrompt: string = ""): string =
  ## 右脳(TM+Intent)をMoEルーターとして流用し、左脳の軽量ブロック/プロンプトを動的に切替（DB切替不要）
  let experts = initMoEExperts()
  let gating = gateByRightBrain(state.tm, state.intentClassifier, input, concepts)
  let expert = experts[gating.topExpert.ord]
  # 事実系はLLMで生成（スロットではなく事実→LLM、Thinkingが決めた長さで）
  if input.startsWith("事実:"):
    try:
      let desiredLen = decideLengthByThinking(input, state.lastThinking, intent)
      let maxTok = max(12, desiredLen div 3)
      var llmState = initLLMState(initLLMConfig(4096))
      var ctx = newSeq[float32](llmState.config.dModel)
      for c in concepts:
        if c.activation > 0.2:
          let idx = (c.id * 7) mod ctx.len
          ctx[idx] += c.activation * 0.5
      let raw = generateText(llmState, state.tokenizer, input, maxTokens=maxTok, temperature=expert.temperature, contextVec=ctx)
      var cleaned = raw.replace("<UNK>", "").replace("事実:", "").strip()
      if cleaned.len > 5 and cleaned.len < 500:
        return cleaned
    except: discard
  
  # 右脳応答生成（LLMベース、テンプレート不使用）
  let rightBrainResponse = generateRightBrainResponse(state, concepts, intent, input, systemPrompt)
  if not forceLLM:
    return rightBrainResponse
  
  # MoE専門家ごとの語尾変化のみ適用
  if rightBrainResponse.len > 0 and rightBrainResponse != "...":
    var varied = rightBrainResponse
    case gating.topExpert
    of exChat:
      let greets = ["！", "ね", "よ", "。", " やあ", " へーい"]
      varied &= greets[int(trueRandFloatCog() * greets.len.float32) mod greets.len]
    of exCode:
      discard
    of exReasoning:
      discard
    of exGeneral:
      let ps = state.generator.knowledge.particles
      if ps.len > 0:
        varied &= ps[int(trueRandFloatCog() * ps.len.float32) mod ps.len]
    return varied

  return rightBrainResponse

# LLMプロンプト構築
proc buildLLMPrompt*(input: string; concepts: seq[ConceptNode]; intent: InputIntent; searchContext: string; desiredLen: int; systemPrompt: string = ""): string =
  var prompt = ""
  if systemPrompt.len > 0:
    prompt.add("システム " & systemPrompt & " ")
  if searchContext.len > 0:
    prompt.add("参照情報 " & sliceRunes(searchContext, 600) & " ")
  prompt.add("入力 " & input & " ")
  prompt.add("意図 " & $intent & " ")
  if concepts.len > 0:
    var conceptWords: seq[string] = @[]
    for c in concepts:
      if c.activation > 0.2 and not isBadWord(c.word):
        conceptWords.add(c.word)
    if conceptWords.len > 0:
      prompt.add("関連概念 " & conceptWords.deduplicate.join(" ") & " ")
  if systemPrompt.len > 0:
    prompt.add("指示 " & systemPrompt & " 目安 " & $desiredLen & " 字 ")
  else:
    prompt.add("指示 自然な日本語で応答してください 目安 " & $desiredLen & " 字 ")
  prompt.add("出力 思考  推論プロセスを記述 思考終わり 応答 ")
  return prompt & " "

# LLM出力から応答抽出
proc extractResponseFromLLM*(raw: string): string =
  var cleaned = raw.strip()
  # 旧タグ互換
  let respStart = cleaned.find("<応答>")
  let respEnd = cleaned.find("</応答>")
  if respStart >= 0 and respEnd > respStart:
    return cleaned[respStart + 6 .. respEnd - 1].strip()
  # 新形式: 応答 以降を抽出
  let marker = cleaned.find("応答")
  if marker >= 0:
    var after = cleaned[marker + 4 .. ^1].strip()
    # 先頭の記号を除去
    after = after.strip(chars={' ', ':', '\n', '\r', '\t'})
    # 思考終わり 以前の思考部分を除去済みなら残りが応答
    if after.len > 5:
      # 末尾の不要なマーカーを除去
      return after.replace("思考終わり", "").replace("思考", "").strip()
  cleaned = cleaned.replace("<UNK>", "").strip()
  if cleaned.len > 200:
    # 長すぎる場合は最初の文だけ
    let dot = cleaned.find("。")
    if dot > 10 and dot < 200:
      return cleaned[0..dot].strip()
  return cleaned

# 前方宣言
proc evaluateResponseWithTM*(state: CognitiveState; input, response: string; intent: InputIntent; desiredLen: int): float32

# LLMによる生成 + TM評価ループ (DeepSeek/Qwen風)
proc generateWithLLMAndTMEval*(state: var CognitiveState; concepts: seq[ConceptNode]; intent: InputIntent; input: string; searchContext: string = ""; systemPrompt: string = ""): string =
  ## LLMで生成し、TMで評価・再生成ループ
  let desiredLen = decideLengthByThinking(input, state.lastThinking, intent)
  let maxAttempts = 3
  var bestResponse = ""
  var bestScore: float32 = 0.0
  
  # MoEゲーティングで専門家を選択
  let experts = initMoEExperts()
  let gating = gateByRightBrain(state.tm, state.intentClassifier, input, concepts)
  let expert = experts[gating.topExpert.ord]
  
  for attempt in 0 ..< maxAttempts:
    let maxTok = max(16, desiredLen div 2)
    var llmState = initLLMState(initLLMConfig(4096))
    var ctx = newSeq[float32](llmState.config.dModel)
    for c in concepts:
      if c.activation > 0.1:
        let idx = (c.id * 7) mod ctx.len
        ctx[idx] += c.activation * 0.3
    
    # プロンプト構築（共通関数使用）
    let prompt = buildLLMPrompt(input, concepts, intent, searchContext, desiredLen, systemPrompt)
    
    let raw = generateText(llmState, state.tokenizer, prompt, maxTokens=maxTok, temperature=expert.temperature, contextVec=ctx)
    
    var response = extractResponseFromLLM(raw)
    
    # 思考をThinkingChainに記録 + 右脳へフィードバック
    var thinkingText = ""
    let thinkStartOld = raw.find("<思考>")
    let thinkEndOld = raw.find("</思考>")
    if thinkStartOld >= 0 and thinkEndOld > thinkStartOld:
      thinkingText = raw[thinkStartOld + 6 .. thinkEndOld - 1].strip()
    else:
      let s = raw.find("思考")
      let e = raw.find("思考終わり")
      if s >= 0 and e > s:
        thinkingText = raw[s + 4 .. e - 1].strip()
      elif raw.len > 10 and raw.len < 200:
        thinkingText = raw[0 .. min(80, raw.len-1)].strip()
    if thinkingText.len > 5:
      var step = ThinkingStep(kind: tsReasoning, description: thinkingText, confidence: 0.7)
      state.lastThinking.steps.add(step)
      feedbackToConceptGraph(state, thinkingText)
    
    # TMによる評価
    let evalScore = evaluateResponseWithTM(state, input, response, intent, desiredLen)
    
    if evalScore > bestScore:
      bestScore = evalScore
      bestResponse = response
    
    # 十分な品質なら採用
    if evalScore >= 0.7:
      break
  
  if bestResponse.len == 0:
    # フォールバック: LLMで直接生成
    var llmState = initLLMState(initLLMConfig(4096))
    let prompt = buildLLMPrompt(input, concepts, intent, searchContext, desiredLen, systemPrompt)
    let raw = generateText(llmState, state.tokenizer, prompt, maxTokens=max(16, desiredLen div 2), temperature=expert.temperature)
    bestResponse = extractResponseFromLLM(raw)
    # 思考フィードバック
    let thinkStart = raw.find("<思考>")
    let thinkEnd = raw.find("</思考>")
    if thinkStart >= 0 and thinkEnd > thinkStart:
      let thinkingText = raw[thinkStart + 6 .. thinkEnd - 1].strip()
      feedbackToConceptGraph(state, thinkingText)
  
  return bestResponse

# 双方向フィードバック: LLM思考断片から概念グラフへ逆活性化
proc feedbackToConceptGraph*(state: var CognitiveState; thinkingText: string) =
  ## LLMの思考テキストから単語を抽出し、概念グラフへ重み0.6で1ステップ伝播
  let words = extractWords(thinkingText, state.tokenizer)
  for w in words:
    if state.conceptGraph.nodeIndex.hasKey(w):
      let nid = state.conceptGraph.nodeIndex[w]
      # 逆活性化: 重み0.6で活性化
      state.conceptGraph.activateWord(w, 0.6)
  # 1ステップ伝播（減衰0.5）
  state.conceptGraph.spreadActivation(steps=1, decay=0.5)

# TMによる応答評価
proc evaluateResponseWithTM*(state: CognitiveState; input, response: string; intent: InputIntent; desiredLen: int): float32 =
  ## TMで応答を評価: 意図一致、文字数、整合性
  var score: float32 = 0.5
  
  # 1. 文字数評価
  let actualLen = response.countRunes
  if desiredLen > 0:
    let lenRatio = min(actualLen.float32 / desiredLen.float32, desiredLen.float32 / actualLen.float32)
    score += (lenRatio - 0.5) * 0.4  # 0.5-1.0の範囲で加点
  
  # 2. 意図一致評価 (概念活性化で判定)
  let inputWords = extractWords(input, state.tokenizer)
  let responseWords = extractWords(response, state.tokenizer)
  var overlap = 0
  for w in inputWords:
    if w in responseWords:
      inc overlap
  if inputWords.len > 0:
    score += (overlap.float32 / inputWords.len.float32) * 0.3
  
  # 3. 日本語らしさ
  var hasJapanese = false
  for r in response.toRunes:
    let cp = r.int32
    if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
      hasJapanese = true
      break
  if hasJapanese: score += 0.2
  
  # 4. 繰り返しペナルティ
  var wordCounts: Table[string, int] = initTable[string, int]()
  for w in responseWords:
    wordCounts[w] = wordCounts.getOrDefault(w, 0) + 1
  var repeatPenalty: float32 = 0.0
  for count in wordCounts.values:
    if count > 2:
      repeatPenalty += float32(count - 2) * 0.05
  score -= min(repeatPenalty, 0.3)
  
  return clamp(score, 0.0, 1.0)

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
proc process*(state: var CognitiveState; input: string; systemPrompt: string = ""): string =
  let t0 = epochTime()
  # 5Tでも推論がPCを固めないよう、メモリ高騰時はGC+throttle
  if shouldThrottle(1500): throttleIfNeeded(1500, 10)

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

  # 未学習語をトラッキング（検索は後の統合ゲートで判定）
  state.trackUnknownWords(rawWords)
  var pendingUnknownConcepts: seq[string] = @[]
  var pendingSearchContext = ""
  for word in fastExtractWords(input):
    if not state.conceptGraph.nodeIndex.hasKey(word) and word.len >= 2 and word notin STOP_WORDS_EARLY:
      if word notin pendingUnknownConcepts:
        pendingUnknownConcepts.add(word)

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
  # 7b. MoE Gating: 右脳のTM/意図をルーターとして左脳の軽量ブロックを切替（DB切替不要のハイブリッド）
  block:
    let moeExperts = initMoEExperts()
    let gating = gateByRightBrain(state.tm, state.intentClassifier, input, topConcepts)
    let expert = moeExperts[gating.topExpert.ord]
    var gateStep = ThinkingStep(kind: tsReasoning, description: "MoE Gating: " & $gating.topExpert & " " & allExpertScores(gating) & " temp=" & $expert.temperature, confidence: gating.confidence)
    state.lastThinking.steps.add(gateStep)
    state.lastThinking.totalConfidence = (state.lastThinking.totalConfidence + gating.confidence)/2.0
  # 7b2. 文字数決定をThinkingで行う（ハードコードではなく推論で）
  block:
    let desiredLen = decideLengthByThinking(input, state.lastThinking, intent)
    var lenStep = ThinkingStep(kind: tsConclusion, description: "文字数決定: " & $desiredLen & "字が適切と判断（Thinkingが推論）", confidence: 0.7)
    state.lastThinking.steps.add(lenStep)
  # 7c. DeepSeek/Qwen風: Thinkingが生成に直接寄与（戦略の条件付け）
  let thinkingConfidence = state.lastThinking.totalConfidence
  let thinkingStrategy = if state.lastThinking.steps.len >= 4: state.lastThinking.steps[3].details.join(" ") else: ""
  let thinkingSuggestsSearch = thinkingStrategy.contains("検索") or thinkingConfidence < 0.45
  let thinkingSuggestsCatalog = thinkingStrategy.contains("カタログ")
  # 7d. 未知概念の検索・学習を統合ゲートで判定（活性・整合度・文字数のみ、単語ハードコードなし）
  block:
    var hasCatalogMatch = false
    for e in state.catalog.entries:
      if e.inputText == input:
        hasCatalogMatch = true
        break
    let desiredLenForSearch = decideLengthByThinking(input, state.lastThinking, intent)
    let shouldSearch = shouldSearchForKnowledge(state, input, intent, desiredLenForSearch, topConcepts, bestScore, hasCatalogMatch)
    let fastWordsForUnknown = fastExtractWords(input)
    let unknownRatio = if fastWordsForUnknown.len > 0: pendingUnknownConcepts.len.float / fastWordsForUnknown.len.float else: 0
    if (shouldSearch or unknownRatio > 0.15 or pendingUnknownConcepts.len >= 1 and desiredLenForSearch >= 80) and pendingUnknownConcepts.len > 0:
      let searchQuery = extractSearchKeywords(input)
      var searcher = initWebSearcher()
      let searchResultsRaw = searcher.search(searchQuery, 5)
      let searchResults = searchResultsRaw
      echo "  [Search] query=", searchQuery, " results=", searchResults.len
      if searchResults.len > 0:
        var combinedSnippets = ""
        for r in searchResults:
          let snip = r.snippet.strip().replace("<span class=\"searchmatch\">","").replace("</span>","")
          if snip.len > 20:
            combinedSnippets.add("【" & r.title & "】" & sliceRunes(snip, 160) & " ")
            let snippetWords = fastExtractWords(snip)
            for w in snippetWords:
              if w.len >= 2 and w notin STOP_WORDS_EARLY and not state.conceptGraph.nodeIndex.hasKey(w):
                let cat = categorizeWord(w)
                discard state.conceptGraph.addNode(w, cat)
                state.conceptGraph.activateWord(w, 0.4)
            for j in 0..<(snippetWords.len - 1):
              let w1 = snippetWords[j]
              let w2 = snippetWords[j+1]
              if state.conceptGraph.nodeIndex.hasKey(w1) and state.conceptGraph.nodeIndex.hasKey(w2):
                state.conceptGraph.addEdge(w1, w2, erRelatedTo, 0.3)
            # 右脳: TM抽象化（検索結果の概念パターンでTMを訓練）
            var snippetCids: seq[int] = @[]
            for w in snippetWords:
              if state.conceptGraph.nodeIndex.hasKey(w):
                snippetCids.add(state.conceptGraph.nodeIndex[w])
            if snippetCids.len > 0:
              let fv = featureVectorFromConcepts(snippetCids, state.cfg.tmClauses * 8)
              let tmClass = intentToClass(intent)
              state.tm.train(fv, tmClass, 1.0f)
            # 左脳: LLM単語学習（次トークン予測で軽量更新）- 検索結果を即時学習
            try:
              var tmpState = initLLMState(initLLMConfig(4096))
              var tmpStore = openLLMWeightStore(if state.llmDBPath.len > 0: state.llmDBPath else: "lunatic_cognitive.db")
              var loaded = loadLLMWeights(tmpStore, tmpState, "final")
              if not loaded:
                loaded = loadLLMWeights(tmpStore, tmpState, "epoch_3")
              if loaded:
                var tmpTok = Tokenizer(vocab: @[], tokenToId: initTable[string,int]())
                if loadLLMTokenizer(tmpStore, tmpTok):
                  let toks = tmpTok.encode(snip & " " & EOS_TOKEN)
                  if toks.len >= 2:
                    let seqT = if toks.len > 64: toks[0..<64] else: toks
                    discard trainStep(tmpState, seqT, tmpTok, 0.0005f32)
                    saveLLMWeights(tmpStore, tmpState, "final")
              closeLLMWeightStore(tmpStore)
            except: discard
        if combinedSnippets.len > 50:
          let activeIds = state.conceptGraph.getActiveNodeIds()
          var clausePat = newSeq[bool](state.cfg.tmClauses)
          for cid in activeIds:
            if cid < clausePat.len: clausePat[cid] = true
          state.episodeStore.episodes.add(Episode(
            inputText: "search_query: " & searchQuery,
            outputText: combinedSnippets,
            inputConceptIds: activeIds, outputConceptIds: @[], tmClausePattern: clausePat,
            confidence: 0.7, speaker: spSystem, contextTag: "search_learned", situation: "search",
            timestamp: epochTime(), reward: 0.6, rank: 0.7, accessCount: 0, emotionalValence: 0.1))
          # 検索結果を概念伝播に反映（1ステップ）＋後段の生成で利用
          state.conceptGraph.spreadActivation(steps=1, decay=0.5)
          pendingSearchContext = combinedSnippets
          echo "  [Search] learned ", pendingUnknownConcepts.len, " concepts, snippets len=", combinedSnippets.len

  # 8. 応答生成（概念活性化 + カタログ照合 or 文法） - Thinkingの判断を参照
  var response = ""
  # 時刻/日付/天気/計算は検索より先に、事実をLLMで生成（スロットではなく事実→LLM）
  if response.len == 0:
    if input.contains("今何時") or input.contains("何時") or input.contains("時間") or (input.contains("今") and (input.contains("時") or input.contains("分") or input.contains("秒"))):
      let now = now()
      let timeStr = now.format("HH:mm")
      # 事実をLLMに渡し自然な日本語を生成（ハードコードのスロットに頼らない）
      let factPrompt = "事実: 今は" & timeStr & "。この事実を自然な日本語で伝えて。語尾は毎回変えて。"
      var llmResp = ""
      try:
        var tmpConcepts: seq[ConceptNode] = @[]
        for c in state.conceptGraph.getTopConcepts(3):
          if c.activation > 0.2: tmpConcepts.add(c)
        llmResp = generateWithLLM(state, tmpConcepts, iiQuestion, factPrompt, forceLLM=true, systemPrompt=systemPrompt)
      except: discard
      if llmResp.len > 5 and llmResp.contains(timeStr) and not llmResp.contains("<UNK>"):
        response = llmResp
      else:
        response = "今は" & timeStr & "だよ"
    elif (input.contains("今日") and (input.contains("何日") or input.contains("日付") or input.contains("日"))) or input.contains("何月") or input.contains("何年") or input.contains("日付は"):
      let now = now()
      let year = $now.year; let month = $now.month.int; let day = $now.monthday.int
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
      let factPrompt = "事実: 今日は" & dateStr & "(" & dayOfWeek & ")。この事実を自然な日本語で伝えて。"
      var llmResp = ""
      try:
        var tmpConcepts: seq[ConceptNode] = @[]
        for c in state.conceptGraph.getTopConcepts(3):
          if c.activation > 0.2: tmpConcepts.add(c)
        llmResp = generateWithLLM(state, tmpConcepts, iiQuestion, factPrompt, forceLLM=true, systemPrompt=systemPrompt)
      except: discard
      if llmResp.len > 5 and not llmResp.contains("<UNK>"):
        response = llmResp
      else:
        response = dateStr & "(" & dayOfWeek & ")だよ"
    elif input.contains("天気") or input.contains("晴れ") or input.contains("雨"):
      # 天気も事実なしだがLLMに委譲（API無しを自然に断る）
      let factPrompt = "天気を聞かれているがAPIが無い。気象庁を案内しつつ自然に断って。"
      var llmResp = ""
      try: llmResp = generateWithLLM(state, @[], iiQuestion, factPrompt, forceLLM=true, systemPrompt=systemPrompt) except: discard
      if llmResp.len > 5 and not llmResp.contains("<UNK>"):
        response = llmResp
      else:
        response = "天気は取得できないよ...気象庁で見てみて"
    elif input.contains("+") or input.contains("-") or input.contains("*") or input.contains("/") or input.contains("=") or input.contains("計算"):
      let calcResult = calculateExpression(input)
      let factPrompt = "事実: 計算結果は" & calcResult & "。この事実を自然な日本語で伝えて。"
      var llmResp = ""
      try: llmResp = generateWithLLM(state, @[], iiQuestion, factPrompt, forceLLM=true, systemPrompt=systemPrompt) except: discard
      if llmResp.len > 5 and not llmResp.contains("<UNK>"):
        response = llmResp
      else:
        response = calcResult & "だよ"
 
  # 8a. 関連エピソード概念活性化を実行
  var knowledgeConcepts: seq[ConceptNode] = @[]
  if bestEpisode.len > 0 and bestScore > 0.5:
    let bestEp = semanticMatches[0][0]
    # キーワードオーバーラップまたは意図一致のチェック:
    # 現在の入力とエピソードが関連している場合のみ概念を活性化
    var epRelevant = false
    for w in rawWords:
      if w.len >= 2 and bestEp.inputText.contains(w):
        epRelevant = true
        break
    if not epRelevant and bestEp.contextTag == $intent:
      epRelevant = true
    if epRelevant:
      let epWords = extractWords(bestEpisode, state.tokenizer)
      for w in epWords:
        if state.conceptGraph.nodeIndex.hasKey(w):
          let node = state.conceptGraph.getNode(state.conceptGraph.nodeIndex[w])
          if node.activation <= 0.01:
            state.conceptGraph.activateWord(w, 0.5)
            knowledgeConcepts.add(node)

  state.conceptGraph.spreadActivation(steps=1, decay=0.3)
  let allTopConcepts = state.conceptGraph.getTopConcepts(7)

    # 8b. カタログ照合（完全一致のみ）with Japanese filtering + 揺らぎ（Gemini級長文が既にセット済みならスキップ）
  let inputIsJapanese = isJapanese(input)
  var catalogMatched = false
  if response.len == 0 and state.catalog.entries.len > 0:
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
      # 事実系クエリ（時刻/日付/天気/計算）は概念不要で最優先処理
      var factHandled = false
      # 計算は意図に関わらず最優先
      if input.contains("+") or input.contains("-") or input.contains("*") or input.contains("/") or input.contains("=") or input.contains("計算"):
        let calcResult = calculateExpression(input)
        let calcTemplates = @[
          "計算結果は" & calcResult & "だよ...",
          calcResult & "だね、合ってるはず",
          "答えは" & calcResult & "かな",
          calcResult & "になるはず...たぶん",
          "計算したら" & calcResult & "だった"
        ]
        response = calcTemplates[int(trueRandFloatCog() * calcTemplates.len.float32) mod calcTemplates.len]
        factHandled = true
      # 時刻/日付/天気は質問意図のみ
      elif intent == iiQuestion:
        if input.contains("今何時") or input.contains("何時") or input.contains("時間") or (input.contains("今") and (input.contains("時") or input.contains("分") or input.contains("秒"))):
          let now = now()
          let timeStr = now.format("HH:mm")
          let timeTemplates = @[
            "今は" & timeStr & "だよ...",
            "時刻は" & timeStr & "だね",
            "今" & timeStr & "ってとこかな",
            timeStr & "です、はい",
            "いま" & timeStr & "くらい...多分",
            "時計見たら" & timeStr & "だった"
          ]
          response = timeTemplates[int(trueRandFloatCog() * timeTemplates.len.float32) mod timeTemplates.len]
          factHandled = true
        elif (input.contains("今日") and (input.contains("何日") or input.contains("日付") or input.contains("日"))) or input.contains("何月") or input.contains("何年") or input.contains("日付は"):
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
          let dateTemplates = @[
            "今日は" & dateStr & "(" & dayOfWeek & ")だよ...",
            dateStr & "(" & dayOfWeek & ")だね",
            "日付？" & dateStr & "(" & dayOfWeek & ")ってとこ",
            "今日の日付は" & dateStr & "(" & dayOfWeek & ")です",
            dateStr & "(" & dayOfWeek & ")...多分合ってる"
          ]
          response = dateTemplates[int(trueRandFloatCog() * dateTemplates.len.float32) mod dateTemplates.len]
          factHandled = true
        elif input.contains("天気") or input.contains("晴れ") or input.contains("雨"):
          let weatherTemplates = @[
            "天気は取得できないよ...気象庁で見てみて...",
            "天気予報？API持ってないから分かんない...ごめん",
            "窓見てみてよ...僕には外が見えないんだ",
            "気象庁のサイト見てくれ...そこが正確だよ",
            "天気？晴れてるといいね...でも分からない",
            "雨かな、晴れかな...調べる手段がないんだ"
          ]
          response = weatherTemplates[int(trueRandFloatCog() * weatherTemplates.len.float32) mod weatherTemplates.len]
          factHandled = true

      # mergedConceptsを先に宣言（フォールバック用）
      var mergedConcepts: seq[ConceptNode] = @[]
      
      # 事実処理で応答済みなら概念処理をスキップ
      if not factHandled:
        var hasMeaningfulConcepts = allTopConcepts.len > 0
        var inputConcepts: seq[ConceptNode] = @[]
        for w in rawWords:
          if state.conceptGraph.nodeIndex.hasKey(w):
            let nid = state.conceptGraph.nodeIndex[w]
            inputConcepts.add(state.conceptGraph.nodes[nid])
        var filteredConcepts = filterConcepts(allTopConcepts)
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
        # 5T打倒: 概念が無くても検索で回答（ゲートで短文は除外）
        if response.len == 0 and intent == iiQuestion:
          var hasCat0 = false
          for e in state.catalog.entries:
            if e.inputText == input: hasCat0 = true; break
          let dl0 = decideLengthByThinking(input, state.lastThinking, intent)
          let should0 = shouldSearchForKnowledge(state, input, intent, dl0, topConcepts, bestScore, hasCat0)
          if should0:
            var searcher0 = initWebSearcher()
            let sq0 = extractSearchKeywords(input)
            var sr0raw = searcher0.search(sq0, 10)
            let sr0 = rankSearchResults(sr0raw, state)
            if sr0.len > 0:
              var combined0 = ""
              var titles0: seq[string] = @[]
              for r in sr0:
                if combined0.len > 900: break
                let sn = r.snippet.strip().replace("<span class=\"searchmatch\">","").replace("</span>","")
                if sn.len > 30:
                  if r.title.len > 0: titles0.add(r.title)
                  combined0.add("【" & r.title & "】" & sliceRunes(sn, 160) & " ")
              if combined0.len > 30:
                let synth = synthesizeFromMultipleSources(state, mergedConcepts, input, combined0, titles0, sr0)
                if synth.len > 40:
                  response = synth
        # 入力に直接応答するための意味解析
        if hasMeaningfulConcepts and response.len == 0:
          case intent
          of iiGreeting:
            response = generateWithLLMAndTMEval(state, mergedConcepts, iiGreeting, input, systemPrompt=systemPrompt)
          of iiQuestion:
            var hasCatMatchQ = false
            for e in state.catalog.entries:
              if e.inputText == input: hasCatMatchQ = true; break
            let desiredLenQ = decideLengthByThinking(input, state.lastThinking, intent)
            let shouldSearchQ = shouldSearchForKnowledge(state, input, intent, desiredLenQ, topConcepts, bestScore, hasCatMatchQ)
            if shouldSearchQ:
              var searcher = initWebSearcher()
              let searchQuery = extractSearchKeywords(input)
              var searchResultsRaw = searcher.search(searchQuery, 10)
              let searchResults = searchResultsRaw
              var searchContext = ""
              if searchResults.len > 0:
                var combined = ""
                var titles: seq[string] = @[]
                for r in searchResults:
                  if combined.len > 900: break
                  let snippet = r.snippet.strip().replace("<span class=\"searchmatch\">","").replace("</span>","")
                  if snippet.len > 30:
                    if r.title.len > 0: titles.add(r.title)
                    combined.add("【" & r.title & "】" & sliceRunes(snippet, 160) & " ")
                if combined.len > 0:
                  searchContext = combined
                  # 長文で検索文脈がある場合は抽出合成を優先（LLM未学習の<UNK>抑止）しきい値120に緩和
                  if searchContext.len > 100 and desiredLenQ > 120:
                    response = synthesizeFromMultipleSources(state, mergedConcepts, input, searchContext, titles, searchResults)
                  else:
                    response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, searchContext, systemPrompt=systemPrompt)
                  if response.len < 10:
                    response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, searchContext, systemPrompt=systemPrompt)
                else:
                  response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, systemPrompt=systemPrompt)
            else:
              response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, systemPrompt=systemPrompt)
              if pendingSearchContext.len > 20:
                # 未知語で検索済み: 検索結果を直接要約（検索→学習→生成の一貫性）
                if pendingSearchContext.len > 100:
                  response = synthesizeFromMultipleSources(state, mergedConcepts, input, pendingSearchContext, @[], @[])
                else:
                  response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, pendingSearchContext, systemPrompt=systemPrompt)
                if response.len < 10 or response.contains("<UNK>"):
                  response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, pendingSearchContext, systemPrompt=systemPrompt)
              else:
                # 通常の短文: 類似カタログから生成
                let inputHash = simHashForRunes(input)
                var bestDist = 64
                var bestOut = ""
                var scored: seq[(float32, string)] = @[]
                for e in state.catalog.entries:
                  if e.intent == intent and e.outputText.len > 5 and e.outputText.len < 80:
                    var jpCnt = 0; var tot = 0; var digitCnt2 = 0; var latinCnt = 0
                    for r in e.outputText.toRunes:
                      inc tot
                      let cp = r.int32
                      if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
                        inc jpCnt
                      if cp >= 0x30 and cp <= 0x39: inc digitCnt2
                      if (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A): inc latinCnt
                    let jpRatio = if tot > 0: jpCnt.float32 / tot.float32 else: 0
                    let digitRatio = if tot > 0: digitCnt2.float32 / tot.float32 else: 0
                    let latinRatio = if tot > 0: latinCnt.float32 / tot.float32 else: 0
                    if jpRatio < 0.4 or digitRatio > 0.1 or latinRatio > 0.25: continue
                    if e.outputText.contains("<UNK>") or e.outputText.contains("http"): continue
                    if e.outputText.contains("(") and e.outputText.contains(")") and latinRatio > 0.1: continue
                    let d = hammingDistance(inputHash, simHashForRunes(e.inputText))
                    let lenDiff = abs(countRunes(e.inputText) - countRunes(input)).float32
                    let score: float32 = (if d <= 12: 20.0f - d.float32 else: 0.0f) + jpRatio*10.0f - digitRatio*20.0f - latinRatio*10.0f - lenDiff*0.8f + (trueRandFloatCog()-0.5f)*2.0f
                    scored.add((score, e.outputText))
                    if d < bestDist:
                      bestDist = d
                      bestOut = e.outputText
                scored.sort(proc(a,b:(float32,string)):int = cmp(b[0], a[0]))
                var fallbackCandidates: seq[string] = @[]
                for (s, txt) in scored: fallbackCandidates.add(txt)
                if bestDist <= 12 and bestOut.len > 0 and scored.len > 0 and scored[0][0] > 5:
                  response = bestOut
                elif fallbackCandidates.len > 0:
                  let topN = min(5, fallbackCandidates.len)
                  let idx = int(trueRandFloatCog() * topN.float32) mod topN
                  response = fallbackCandidates[idx]
                else:
                  response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, input, systemPrompt=systemPrompt)
                  var dig = 0
                  for ch in response:
                    if ch >= '0' and ch <= '9': inc dig
                  if dig > 8 or response.contains("<UNK>") or response.countRunes < 4:
                    var freq: Table[string,int]
                    for e in state.catalog.entries:
                      if e.outputText.len > 5 and e.outputText.len < 40:
                        var jp2=0; var tot2=0; var dig2=0
                        for r in e.outputText.toRunes:
                          inc tot2
                          let cp=r.int32
                          if (cp>=0x3040 and cp<=0x309F) or (cp>=0x30A0 and cp<=0x30FF) or (cp>=0x4E00 and cp<=0x9FFF): inc jp2
                          if cp>=0x30 and cp<=0x39: inc dig2
                        let jr2 = if tot2>0: jp2.float32/tot2.float32 else:0
                        let dr2 = if tot2>0: dig2.float32/tot2.float32 else:0
                        if jr2 > 0.5 and dr2 < 0.05 and not e.outputText.contains("<UNK>"):
                          freq[e.outputText] = freq.getOrDefault(e.outputText,0)+1
                    var bestFreq=0; var bestTxt=""
                    for k,v in freq.pairs:
                      if v > bestFreq:
                        bestFreq=v; bestTxt=k
                    if bestTxt.len > 0:
                      response = bestTxt
          of iiRequest, iiThanks, iiFarewell, iiOpinion, iiStatement:
            let dlR = decideLengthByThinking(input, state.lastThinking, intent)
            if dlR <= 80:
              let inputHashR2 = simHashForRunes(input)
              var bestDistR2 = 64
              var bestOutR2 = ""
              var scoredR2: seq[(float32, string)] = @[]
              for e in state.catalog.entries:
                if e.intent == intent and e.outputText.len > 5 and e.outputText.len < 80:
                  var jpCnt = 0; var tot = 0; var digitCnt2 = 0; var latinCnt = 0
                  for r in e.outputText.toRunes:
                    inc tot
                    let cp = r.int32
                    if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
                      inc jpCnt
                    if cp >= 0x30 and cp <= 0x39: inc digitCnt2
                    if (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A): inc latinCnt
                  let jpRatio = if tot > 0: jpCnt.float32 / tot.float32 else: 0
                  let digitRatio = if tot > 0: digitCnt2.float32 / tot.float32 else: 0
                  let latinRatio = if tot > 0: latinCnt.float32 / tot.float32 else: 0
                  if jpRatio < 0.3 or digitRatio > 0.2 or latinRatio > 0.3: continue
                  if e.outputText.contains("<UNK>") or e.outputText.contains("http"): continue
                  let d = hammingDistance(inputHashR2, simHashForRunes(e.inputText))
                  let lenDiff = abs(countRunes(e.inputText) - countRunes(input)).float32
                  var score: float32 = (if d <= 12: 20.0f - d.float32 else: 0.0f) + jpRatio*10.0f - digitRatio*20.0f - latinRatio*10.0f - lenDiff*0.8f + (trueRandFloatCog()-0.5f)*2.0f
                  scoredR2.add((score, e.outputText))
                  if d < bestDistR2:
                    bestDistR2 = d
                    bestOutR2 = e.outputText
              scoredR2.sort(proc(a,b:(float32,string)):int = cmp(b[0], a[0]))
              var candR2: seq[string] = @[]
              for (s, txt) in scoredR2: candR2.add(txt)
              if bestDistR2 <= 12 and bestOutR2.len > 0 and scoredR2.len > 0 and scoredR2[0][0] > 5:
                response = bestOutR2
              elif candR2.len > 0:
                let topN = min(5, candR2.len)
                let idx = int(trueRandFloatCog() * topN.float32) mod topN
                response = candR2[idx]
            else:
              if pendingSearchContext.len > 20:
                if pendingSearchContext.len > 100:
                  response = synthesizeFromMultipleSources(state, mergedConcepts, input, pendingSearchContext, @[], @[])
                else:
                  response = generateWithLLMAndTMEval(state, mergedConcepts, intent, input, pendingSearchContext, systemPrompt=systemPrompt)
              else:
                response = generateWithLLMAndTMEval(state, mergedConcepts, intent, input, systemPrompt=systemPrompt)
                var digR = 0
                for ch in response:
                  if ch >= '0' and ch <= '9': inc digR
                if digR > 8 or response.contains("<UNK>"):
                  var freqR: Table[string,int]
                  for e in state.catalog.entries:
                    if e.intent == intent and e.outputText.len > 5 and e.outputText.len < 40:
                      var jp2=0; var tot2=0; var dig2=0
                      for r in e.outputText.toRunes:
                        inc tot2
                        let cp=r.int32
                        if (cp>=0x3040 and cp<=0x309F) or (cp>=0x30A0 and cp<=0x30FF) or (cp>=0x4E00 and cp<=0x9FFF): inc jp2
                        if cp>=0x30 and cp<=0x39: inc dig2
                      if jp2.float32/tot2.float32 > 0.5 and dig2.float32/tot2.float32 < 0.05:
                        freqR[e.outputText] = freqR.getOrDefault(e.outputText,0)+1
                  var bestFreqR=0; var bestTxtR=""
                  for k,v in freqR.pairs:
                    if v > bestFreqR:
                      bestFreqR=v; bestTxtR=k
                  if bestTxtR.len > 0: response = bestTxtR
          else:
            response = generateWithLLMAndTMEval(state, mergedConcepts, intent, input, systemPrompt=systemPrompt)
        else:
          let dlE = decideLengthByThinking(input, state.lastThinking, intent)
          if pendingSearchContext.len > 20:
            if pendingSearchContext.len > 100:
              response = synthesizeFromMultipleSources(state, mergedConcepts, input, pendingSearchContext, @[], @[])
            else:
              response = generateWithLLMAndTMEval(state, mergedConcepts, intent, input, pendingSearchContext, systemPrompt=systemPrompt)
          elif dlE <= 80:
            let inputHashE = simHashForRunes(input)
            var bestDistE = 64
            var bestOutE = ""
            var scoredE: seq[(float32, string)] = @[]
            for e in state.catalog.entries:
              if e.intent == intent and e.outputText.len > 5 and e.outputText.len < 80:
                var jpCnt = 0; var tot = 0; var digitCnt2 = 0; var latinCnt = 0
                for r in e.outputText.toRunes:
                  inc tot
                  let cp = r.int32
                  if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF) or (cp >= 0x4E00 and cp <= 0x9FFF):
                    inc jpCnt
                  if cp >= 0x30 and cp <= 0x39: inc digitCnt2
                  if (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A): inc latinCnt
                let jpRatio = if tot > 0: jpCnt.float32 / tot.float32 else: 0
                let digitRatio = if tot > 0: digitCnt2.float32 / tot.float32 else: 0
                let latinRatio = if tot > 0: latinCnt.float32 / tot.float32 else: 0
                if jpRatio < 0.3 or digitRatio > 0.2 or latinRatio > 0.3: continue
                if e.outputText.contains("<UNK>") or e.outputText.contains("http"): continue
                let d = hammingDistance(inputHashE, simHashForRunes(e.inputText))
                let lenDiff = abs(countRunes(e.inputText) - countRunes(input)).float32
                var score: float32 = (if d <= 12: 20.0f - d.float32 else: 0.0f) + jpRatio*10.0f - digitRatio*20.0f - latinRatio*10.0f - lenDiff*0.8f + (trueRandFloatCog()-0.5f)*2.0f
                if e.intent == intent: score += 5.0f
                scoredE.add((score, e.outputText))
                if d < bestDistE:
                  bestDistE = d
                  bestOutE = e.outputText
            scoredE.sort(proc(a,b:(float32,string)):int = cmp(b[0], a[0]))
            var candE: seq[string] = @[]
            for (s, txt) in scoredE: candE.add(txt)
            if bestDistE <= 12 and bestOutE.len > 0 and scoredE.len > 0 and scoredE[0][0] > 5:
              response = bestOutE
            elif candE.len > 0:
              let topN = min(5, candE.len)
              let idx = int(trueRandFloatCog() * topN.float32) mod topN
              response = candE[idx]
            else:
              response = generateWithLLMAndTMEval(state, mergedConcepts, intent, input, systemPrompt=systemPrompt)
          else:
            response = generateWithLLMAndTMEval(state, mergedConcepts, intent, input, systemPrompt=systemPrompt)

      # 8e. DeepSeek/Qwen風 自己検証ループ: Thinkingの結論で応答を自己修正
      if thinking.steps.len > 0 and response.len > 0:
        let conclusionText = thinking.steps[^1].details.join(" ")
        if conclusionText.contains("挨拶は短く") and response.len > 30:
          response = generateRightBrainResponse(state, mergedConcepts, iiGreeting, input, systemPrompt)
        if response == "何について...?" and thinkingSuggestsSearch:
          var searcher2 = initWebSearcher()
          let sq2 = extractSearchKeywords(input)
          var sr2raw = searcher2.search(sq2, 10)
          let sr2 = rankSearchResults(sr2raw, state)
          if sr2.len > 0 and sr2[0].snippet.len > 50:
            response = generateWithLLMAndTMEval(state, mergedConcepts, iiQuestion, "検索結果: " & sr2[0].snippet & "\n質問: " & input, sr2[0].snippet, systemPrompt=systemPrompt)
        if thinkingConfidence < 0.35 and response == "...":
          response = generateRightBrainResponse(state, mergedConcepts, iiQuestion, input, systemPrompt)
      # 8f. 文字数検証: Thinkingが決めた長さになっているかTMで確認、段階的に再生成
      block:
        let desiredLen = decideLengthByThinking(input, state.lastThinking, intent)
        let actualLen = response.toRunes.len
        let diff = abs(actualLen - desiredLen)
        if diff > desiredLen div 3 and desiredLen >= 100 and response.len > 0:
          var verifyStep = ThinkingStep(kind: tsReasoning, description: "文字数検証: 期待" & $desiredLen & "字に対し実際" & $actualLen & "字。差が大きいため再生成", confidence: 0.6)
          state.lastThinking.steps.add(verifyStep)
          if actualLen < desiredLen:
            # LLM+TM評価ループで再生成（文脈を含めて）
            var searcherV = initWebSearcher()
            let sqV = extractSearchKeywords(input)
            var srVraw = searcherV.search(sqV, 10)
            let srV = rankSearchResults(srVraw, state)
            var searchContext = ""
            if srV.len > 0:
              var combinedV = ""
              var titlesV: seq[string] = @[]
              for r in srV:
                if combinedV.len > 600: break
                let sn = r.snippet.strip().replace("<span class=\"searchmatch\">","").replace("</span>","")
                if sn.len > 30:
                  titlesV.add(r.title)
                  combinedV.add(sliceRunes(sn, 200) & " ")
              if combinedV.len > 30:
                searchContext = combinedV
            # 既存の応答を文脈として追加し、不足分を生成
            var extendPrompt = "前の応答:\n" & response & "\n\n不足しています。追加で" & $(desiredLen - actualLen) & "字程度、自然に続けてください。"
            let additional = generateWithLLMAndTMEval(state, mergedConcepts, intent, extendPrompt, searchContext, systemPrompt=systemPrompt)
            if additional.len > 10:
              response = response & "\n\n" & additional
          else:
            response = sliceRunes(response, desiredLen).strip()
            if not response.endsWith("。"):
              response.add("。")
          var okStep = ThinkingStep(kind: tsConclusion, description: "文字数検証完了: " & $response.toRunes.len & "字で適切と判断", confidence: 0.75)
          state.lastThinking.steps.add(okStep)
          state.lastThinking.totalConfidence = (state.lastThinking.totalConfidence + 0.75)/2.0

  # 9. 自己評価
  var evalResult = EvalResult(verdict: evAccept, score: 0.5, reason: "skip",
                              contradictions: @[], relevanceScore: 0.5,
                              coherenceScore: 0.5, improvements: @[])
  if state.cfg.evalEnabled:
    evalResult = state.selfEvaluate(input, response, thinking, topConcepts)
  state.lastEval = evalResult

  # 応答のノイズ除去・重複除去
  proc cleanResponse(text: string): string =
    var result = text
    # 連続する同一文言の除去
    var sentences: seq[string] = @[]
    for s in result.split("。"):
      let trimmed = s.strip()
      if trimmed.len > 0 and trimmed notin sentences:
        sentences.add(trimmed)
    result = sentences.join("。")
    if not result.endsWith("。") and result.len > 0:
      result.add("。")
    # 同じ単語の過度な繰り返し除去（3回以上連続）
    var words: seq[string] = @[]
    for w in result.split(" "):
      let trimmed = w.strip()
      if trimmed.len > 0:
        if words.len >= 2 and words[^1] == trimmed and words[^2] == trimmed:
          continue  # 3連続はスキップ
        words.add(trimmed)
    result = words.join(" ")
    return result

  response = cleanResponse(response)

  # システムプロンプトに基づく動的後処理（ハードコードではなくシステムプロンプト内容を解析）
  if systemPrompt.len > 0:
    # 一人称指定の抽出: 「一人称はX」→ Xを対象人称とする
    var targetPronoun = ""
    let pIdx = systemPrompt.find("一人称は")
    if pIdx >= 0:
      let after = systemPrompt[pIdx + "一人称は".len .. ^1]
      # 最初の連続した非助詞文字を抽出（例: 僕、私、俺）
      var pronoun = ""
      for r in after.toRunes:
        let s = $r
        if s == "を" or s == "で" or s == "は" or s == " " or s == "、" or s == "。" or s == "\n":
          break
        if s notin [" ", "　"]:
          pronoun.add(s)
          if pronoun.toRunes.len >= 2:
            break
      if pronoun.len > 0 and pronoun.len <= 6:
        targetPronoun = pronoun
    if targetPronoun.len > 0:
      # 他の一人称を対象に置換（システムプロンプトで指定されたもの以外）
      let otherPronouns = ["私", "俺", "わたし", "あたし", "僕"]
      for op in otherPronouns:
        if op != targetPronoun and op in response:
          response = response.replace(op, targetPronoun)
    # 名前指定の抽出: 「あなたはXです」→ Xを名前とする
    var targetName = ""
    let nIdx = systemPrompt.find("あなたは")
    if nIdx >= 0:
      let afterN = systemPrompt[nIdx + "あなたは".len .. ^1]
      let endIdx = afterN.find("です")
      if endIdx >= 0:
        targetName = afterN[0 ..< endIdx].strip()
        # 「やみ」「ルナティック」などを抽出
        if targetName.len > 0 and targetName.len < 20:
          # Lunaticという名前を出さない指示がある場合、Lunaticをターゲット名に置換
          if "Lunatic" in response and targetName != "Lunatic" and targetName != "LunaticIntelligence":
            # 完全一致を優先して置換（やみIntelligenceのような残骸を防ぐ）
            if "LunaticIntelligence" in response:
              response = response.replace("LunaticIntelligence", targetName)
            else:
              response = response.replace("Lunatic", targetName).replace("LUNATIC", targetName).replace("ルナティック", targetName)
          elif targetName == "やみ" and "LunaticIntelligence" in response:
            response = response.replace("LunaticIntelligence", targetName)

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
        # decompose search results and generate via generateJapaneseResponse, not direct translated (10件 frequency + wobble)
        var dummyResultsRaw = searcher.search(input, 10)
        let dummyResults = rankSearchResults(dummyResultsRaw, state)
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
          # 右脳で応答生成（LLMベース）
          response = generateWithLLMAndTMEval(state, merged, iiQuestion, "検索結果: " & translated & "\n質問: " & input, translated, systemPrompt=systemPrompt)
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
        var dummyResultsRaw = searcher.search(input, 10)
        let dummyResults = rankSearchResults(dummyResultsRaw, state)
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

# カタログ多様化: よくある入力パターンに多様な応答を追加
proc enrichCatalogDiverseResponses*(catalog: var ResponseCatalog) =
  # 挨拶パターン
  let greetings = @[
    "こんにちは！何かお手伝いできることは...あり...か？?...",
    "呼んだ?...",
    "はーい...",
    "何か用かな?...",
    "こんにちは...",
    "やあ、元気？",
    "いらっしゃい...",
    "呼びました？",
    "どしたん？",
    "こんにちは、何か聞きたいことある？"
  ]
  for g in greetings:
    catalog.entries.add(CatalogEntry(intent: iiGreeting, keyword: "こんにちは", inputText: "こんにちは", outputText: g, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiGreeting, keyword: "やあ", inputText: "やあ", outputText: g, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiGreeting, keyword: "こんちは", inputText: "こんちは", outputText: g, weight: 1.0f))

  # 朝の挨拶
  let mornings = @[
    "おはよう！今日も良い一日になりますように...",
    "おはようございます...元気？",
    "朝だね...よく眠れた？",
    "おはよー...今日も頑張ろう",
    "おはよう、何か予定あるの？",
    "朝早いね...偉いじゃん",
    "おはよう...コーヒー飲む？",
    "良い朝だ...何かする？"
  ]
  for m in mornings:
    catalog.entries.add(CatalogEntry(intent: iiGreeting, keyword: "おはよう", inputText: "おはよう", outputText: m, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiGreeting, keyword: "おはようございます", inputText: "おはようございます", outputText: m, weight: 1.0f))

  # 感謝
  let thanks = @[
    "どういたしまして！お役に立てて嬉しい...",
    "いえいえ、どういたましょうかね",
    "嬉しいです、ありがとう...",
    "こちらこそ、ありがとう",
    "お礼なんていらないよ...",
    "役に立てたら何よりだね",
    "どういたしまして、またいつでも"
  ]
  for t in thanks:
    catalog.entries.add(CatalogEntry(intent: iiThanks, keyword: "ありがとう", inputText: "ありがとう", outputText: t, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiThanks, keyword: "感謝", inputText: "感謝", outputText: t, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiThanks, keyword: "thanks", inputText: "thanks", outputText: t, weight: 1.0f))

  # 別れ
  let farewells = @[
    "じゃあね...またね",
    "バイバイ、またお会いしましょう",
    "おやすみ～",
    "またね、気をつけて",
    "じゃあ、またどこかで",
    "さようなら...またね",
    "バイバイ、元気でね",
    "また今度...おやすみ"
  ]
  for f in farewells:
    catalog.entries.add(CatalogEntry(intent: iiFarewell, keyword: "バイバイ", inputText: "バイバイ", outputText: f, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiFarewell, keyword: "またね", inputText: "またね", outputText: f, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiFarewell, keyword: "さようなら", inputText: "さようなら", outputText: f, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiFarewell, keyword: "おやすみ", inputText: "おやすみ", outputText: f, weight: 1.0f))

  # 元気？系
  let genki = @[
    "元気...あなたはどう...か？?...",
    "まあまあかな...あなたは？",
    "元気だよ、心配かけてごめんね",
    "ぼちぼち...生きてるだけ丸儲け",
    "元気元気！君はどう？",
    "普通...変わりないよ",
    "元気いっぱい！...嘘、普通"
  ]
  for g in genki:
    catalog.entries.add(CatalogEntry(intent: iiQuestion, keyword: "元気", inputText: "元気？", outputText: g, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiQuestion, keyword: "調子", inputText: "調子どう？", outputText: g, weight: 1.0f))

  # 日本語で話して系（検索不要な短文会話）
  let nihongoReq = @[
    "日本語で話してるよ。何か聞きたいことある？",
    "もちろん日本語だよ。どうしたの？",
    "はい、日本語で話してるよ。何かな？",
    "日本語だよ。何かお手伝いできる？",
    "うん、日本語で話してる。どうしたの？"
  ]
  for n in nihongoReq:
    catalog.entries.add(CatalogEntry(intent: iiRequest, keyword: "日本語で話して", inputText: "日本語で話して", outputText: n, weight: 1.0f))

  # 何してる系（短文会話）
  let nanisiteru = @[
    "会話してるよ。君は？",
    "君と話してる。他に何か？",
    "ぼーっとしてた。何か用？",
    "考え事してた。君は何してる？",
    "返事考えてた。どうしたの？"
  ]
  for c in nanisiteru:
    catalog.entries.add(CatalogEntry(intent: iiQuestion, keyword: "何してる", inputText: "何してる？", outputText: c, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiQuestion, keyword: "何してる", inputText: "なにしてる？", outputText: c, weight: 1.0f))

  # 自己紹介
  let intro = @[
    "僕はLunaticIntelligence...概念グラフとTMで考えるよ",
    "LunaticIntelligenceって言うんだ...よろしくね",
    "名前はルナティック...右脳TM、左脳LLMのハイブリッド",
    "自己紹介か...僕は思考するプログラムさ",
    "ルナティックって呼んで...認知アーキテクチャだよ"
  ]
  for i in intro:
    catalog.entries.add(CatalogEntry(intent: iiQuestion, keyword: "自己紹介", inputText: "自己紹介して", outputText: i, weight: 1.0f))
    catalog.entries.add(CatalogEntry(intent: iiQuestion, keyword: "君は誰", inputText: "君は誰？", outputText: i, weight: 1.0f))

  echo "  Catalog enriched: +" & $(greetings.len*3 + mornings.len*2 + thanks.len*3 + farewells.len*4 + genki.len*2 + nihongoReq.len + nanisiteru.len*2 + intro.len*2) & " diverse entries"

# ---------------------------------------------------------------------------
# 観察モード: デュアルパス統合学習（右脳: 概念グラフ+TM / 左脳: LLM因果言語モデル）
# ---------------------------------------------------------------------------
proc observeCorpusDualPath*(state: var CognitiveState; corpusPath: string; 
                            dbPath: string; maxEpochs: int = 3; 
                            llmLearningRate: float32 = 0.001f32;
                            cpuThrottleMs: int = 10) =
  ## 右脳（概念グラフ・TM・Hebbian・カタログ）と左脳（LLM次トークン予測）を
  ## 同一ストリームで並行学習し、エポックごとにアトミックに永続化
  echo "=== Dual-Path Observation: " & corpusPath & " ==="
  echo "DB: " & dbPath & " | Epochs: " & $maxEpochs & " | LLM LR: " & $llmLearningRate
  let t0 = epochTime()
  
  let splitParticles = ["について", "とは何", "とは", "教えて", "何ですか", "ですか",
                        "の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
                        "な", "から", "まで", "より", "って", "じゃ",
                        "です", "ます", "だ", "である", "いる", "ある",
                        "そう", "よ", "ね", "さ", "わ"]
  let filterWords = ["について", "とは何", "とは", "教えて", "何ですか", "ですか",
                     "の", "は", "が", "を", "に", "で", "と", "も", "や", "か",
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
          isAlpha = false
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
  
  # === Phase 1: データ読み込み・語彙頻度集計 + LLMトークナイザー構築 ===
  echo "Phase 1: Loading corpus & building vocabularies..."
  var wordFreq: Table[string, int]
  var lineCount = 0
  var allLines: seq[(string, string)] = @[]  # (input, output) ペア
  var llmTrainingLines: seq[string] = @[]   # LLM学習用: "input | output" 結合テキスト
  
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
    allLines.add((inputText, outputText))
    llmTrainingLines.add(inputText & " " & outputText)
    inc lineCount
    if lineCount mod 50000 == 0:
      throttleIfNeeded(1800, 5)
      if lineCount mod 200000 == 0: GC_fullCollect()
    if lineCount mod 100000 == 0:
      echo "  Loaded: " & $lineCount & " lines, " & $wordFreq.len & " unique words mem=" & $getMemoryUsageMB() & "MB"
  echo "  Total: " & $lineCount & " lines, " & $wordFreq.len & " unique words"
  
  # LLMトークナイザー構築（左脳用語彙）
  var llmTokenizer = Tokenizer(vocab: @[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN], tokenToId: initTable[string, int]())
  llmTokenizer.tokenToId[PAD_TOKEN] = PAD_ID
  llmTokenizer.tokenToId[UNK_TOKEN] = UNK_ID
  llmTokenizer.tokenToId[EOS_TOKEN] = EOS_ID
  llm.buildTokenizerFromCorpusForLLM(llmTrainingLines, llmTokenizer, 4096)
  echo "  LLM Vocab: " & $llmTokenizer.vocab.len
  # 推論用に state.tokenizer を LLM語彙で統一（<UNK>抑止）
  state.tokenizer = llmTokenizer
  
  # === Phase 2: 右脳 - 概念ノード構築 ===
  echo "Phase 2: Building concept nodes (Right Brain)..."
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
  
  # === Phase 3: デュアルパス統合学習ループ ===
  echo "Phase 3: Dual-Path Integrated Training (" & $maxEpochs & " epochs)..."
  let t2 = epochTime()
  var catalog = ResponseCatalog(entries: @[])
  var catalogCount: Table[int, int]
  var seenInputHashes: seq[uint64] = @[]
  var hebbianSampleStep = max(1, lineCount div 10000)
  
  # LLM状態初期化（左脳）- アーキテクチャ変更時は新規初期化
  var llmState = initLLMState(initLLMConfig(4096))
  let store = openLLMWeightStore(dbPath)
  initLLMWeights(llmState); echo "  LLM: Init new weights (fresh training)"
  
  for epoch in 1..maxEpochs:
    echo "  Epoch " & $epoch & "/" & $maxEpochs
    stdout.flushFile()
    var processed = 0
    var totalLlmLoss: float32 = 0.0
    var llmBatchCount = 0
    
    # LLMバッチ学習用バッファ
    var llmTokenBatches: seq[seq[int]] = @[]
    const LLM_BATCH_SIZE = 64
    
    for i in 0..<allLines.len:
      let (inputText, outputText) = allLines[i]
      let words = fastExtractWords(inputText)
      var inputCids: seq[int] = @[]
      for w in words:
        if state.conceptGraph.nodeIndex.hasKey(w):
          inputCids.add(state.conceptGraph.nodeIndex[w])
      
      # --- 右脳: エッジ構築（初回エポックのみ）---
      if epoch == 1:
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
      
      # --- 右脳: TM学習 ---
      if inputCids.len > 0:
        let fv = featureVectorFromConcepts(inputCids, state.cfg.tmClauses * 8)
        var tmClass = 8
        let lt = inputText.toLower()
        if lt.contains("おはよう") or lt.contains("こんにちは") or lt.contains("こんばんは") or lt.contains("hello") or lt.contains("hi") or lt.contains("hey"): tmClass = 0
        elif lt.contains("?") or lt.contains("？") or lt.contains("何") or lt.contains("どこ") or lt.contains("what") or lt.contains("how"): tmClass = 1
        elif lt.contains("ありがとう") or lt.contains("thanks") or lt.contains("thank"): tmClass = 5
        elif lt.contains("さようなら") or lt.contains("バイバイ") or lt.contains("bye"): tmClass = 6
        state.tm.train(fv, tmClass, 1.0f)
      
      # --- 右脳: Hebbian学習（サンプリング）---
      if processed mod hebbianSampleStep == 0 and inputCids.len >= 2:
        for j in 0..<min(inputCids.len, 5):
          for k in (j+1)..<min(inputCids.len, 5):
            let w1 = state.conceptGraph.getWord(inputCids[j])
            let w2 = state.conceptGraph.getWord(inputCids[k])
            if w1.len > 0 and w2.len > 0:
              state.conceptGraph.hebbianStrengthen(w1, w2, 0.01)
      
      # --- 右脳: カタログ構築（初回エポックのみ）---
      if epoch == 1:
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
            let inputHash = simHashForRunes(inputText)
            var isDup = false
            for h in seenInputHashes:
              if hammingDistance(h, inputHash) <= 3:
                isDup = true
                break
            if not isDup:
              seenInputHashes.add(inputHash)
              catalog.entries.add(CatalogEntry(
                intent: intent,
                keyword: inputText[0..<min(20, inputText.len)],
                inputText: inputText,
                outputText: outputText,
                weight: 1.0f
              ))
              catalogCount[intent.ord] = currentCount + 1
      
      # --- 左脳: LLM因果言語モデル学習（バッチ処理）---
      let combinedText = inputText & " " & outputText & " " & EOS_TOKEN
      let tokens = llmTokenizer.encode(combinedText)
      if tokens.len >= 2:
        let maxTokens = 64
        let seqT = if tokens.len > maxTokens: tokens[0..<maxTokens] else: tokens
        llmTokenBatches.add(seqT)
        if llmTokenBatches.len >= LLM_BATCH_SIZE:
          # バッチ学習実行
          for batchTokens in llmTokenBatches:
            let loss = trainStep(llmState, batchTokens, llmTokenizer, llmLearningRate)
            totalLlmLoss += loss
            inc llmBatchCount
          llmTokenBatches.setLen(0)
      
      inc processed
      if processed mod 10000 == 0:
        let avgLoss = if llmBatchCount > 0: totalLlmLoss / llmBatchCount.float32 else: 0.0
        echo "  Processed: " & $processed & "/" & $lineCount & " | LLM avg loss: " & $avgLoss
        stdout.flushFile()
      if processed mod 50000 == 0:
        sleep(1)  # 軽いスロットル
    
    # 残りバッチ処理
    if llmTokenBatches.len > 0:
      for batchTokens in llmTokenBatches:
        let loss = trainStep(llmState, batchTokens, llmTokenizer, llmLearningRate)
        totalLlmLoss += loss
        inc llmBatchCount
      llmTokenBatches.setLen(0)
    
    # エポック終了時: アトミック永続化（右脳 + 左脳）
    echo "  Persisting epoch " & $epoch & " (dual-path save)..."
    let persistStart = epochTime()
    
    # 右脳: 概念グラフ・TM・シナプス・エピソード・カタログをSQLite WALで保存
    let sdb = openStorage(dbPath)
    try:
      saveConceptGraph(sdb, state.conceptGraph)
      saveTM(sdb, state.tm)
      saveSynapses(sdb, state.bridge)
      saveEpisodes(sdb, state.episodeStore)
      saveCatalog(sdb, catalog)
      saveTokenizer(sdb, llmTokenizer)
      var meta = initTable[string, string]()
      meta["schema_version"] = "2"
      meta["epoch"] = $epoch
      meta["timestamp"] = $epochTime()
      saveDbMeta(sdb, meta)
    except:
      raise
    close(sdb)
    
    # 左脳: LLM重み + トークナイザを同一DBに保存（語彙不一致による<UNK>抑止）
    saveLLMWeights(store, llmState, "epoch_" & $epoch)
    saveLLMWeights(store, llmState, "final")
    saveLLMTokenizer(store, llmTokenizer)
    
    echo "  Persisted in " & $formatFloat(epochTime() - persistStart, ffDecimal, 1) & "s"
    echo "  Epoch " & $epoch & " complete | LLM avg loss: " & $(if llmBatchCount > 0: totalLlmLoss / llmBatchCount.float32 else: 0.0)
    
    if cpuThrottleMs > 0:
      sleep(cpuThrottleMs * 10)  # エポック間で長めのスロットル
    GC_fullCollect()
  
  closeLLMWeightStore(store)
  state.catalog = catalog
  enrichCatalogDiverseResponses(state.catalog)
  
  echo "  Edges: " & $state.conceptGraph.edges.len
  echo "  TM+Hebbian+Catalog: " & $formatFloat(epochTime() - t2, ffDecimal, 1) & "s"
  echo "  Catalog: " & $catalog.entries.len & " entries"
  echo "Dual-Path Observation complete: " & $state.conceptGraph.conceptCount() & " concepts"
  echo "Total: " & $formatFloat(epochTime() - t0, ffDecimal, 1) & "s"
{.pop.}