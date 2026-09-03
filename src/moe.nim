import types, intent_classifier, code_structure
import strutils, sequtils

type
  ExpertId* = enum
    exGeneral = 0
    exChat = 1
    exCode = 2
    exReasoning = 3

  MoEGating* = object
    scores*: array[4, float32]
    topExpert*: ExpertId
    confidence*: float32

  MoEExpert* = object
    id*: ExpertId
    promptPrefix*: string
    temperature*: float32
    wobbleScale*: float32
    maxTokens*: int

proc initMoEExperts*(): array[4, MoEExpert] =
  result[exGeneral.ord] = MoEExpert(id: exGeneral, promptPrefix: "一般的な質問として、検索結果を基に簡潔に答えて。", temperature: 0.72, wobbleScale: 0.02, maxTokens: 12)
  result[exChat.ord] = MoEExpert(id: exChat, promptPrefix: "親しみやすい会話として、共感的に短く答えて。", temperature: 0.85, wobbleScale: 0.03, maxTokens: 12)
  result[exCode.ord] = MoEExpert(id: exCode, promptPrefix: "正確なコード生成として、Nimの文法に厳密に従って答えて。", temperature: 0.35, wobbleScale: 0.005, maxTokens: 24)
  result[exReasoning.ord] = MoEExpert(id: exReasoning, promptPrefix: "論理的に検証し、ThinkingでQ/Aを経てから答えて。", temperature: 0.65, wobbleScale: 0.015, maxTokens: 16)

proc gateByRightBrain*(tm: HierarchicalTM; classifier: IntentClassifier; input: string; concepts: seq[ConceptNode]): MoEGating =
  ## 右脳のTMと意図分類器をGating Networkとして流用
  var scores: array[4, float32]
  let (intent, _) = classifier.classifyIntent(input)
  let isCode = isCodeInput(input)
  # 短い入力 + 強い概念なし = 会話的
  let inputRunes = countRunes(input)
  let hasStrongConcepts = concepts.anyIt(it.activation > 0.5 and it.word.len >= 2)
  let isConvQuestion = intent == iiQuestion and inputRunes < 15 and not hasStrongConcepts
  # 意図ベースの事前スコア
  case intent
  of iiGreeting, iiThanks, iiFarewell, iiOpinion, iiStatement: scores[exChat.ord] = 0.8
  of iiRequest: scores[exCode.ord] = 0.6; scores[exChat.ord] = 0.3
  of iiQuestion:
    if isConvQuestion:
      scores[exChat.ord] = 0.8  # 会話的質問はチャット専門家
    else:
      if isCode: scores[exCode.ord] = 0.7; scores[exReasoning.ord] = 0.3
      else: scores[exGeneral.ord] = 0.5; scores[exReasoning.ord] = 0.4
  else: scores[exGeneral.ord] = 0.6
  # コード検出で上書き
  if isCode:
    scores[exCode.ord] += 0.5
  # TMの確信度で重み付け（TMが迷えば reasoning を上げる）
  let tmConf {.used.} = if tm.layers.len>0 and tm.layers[0].states.len>0: 0.5 else: 0.5 # 簡易
  # 概念の活性で補正
  for c in concepts:
    if c.activation > 0.5:
      if c.category in [ctNoun, ctVerb]: scores[exGeneral.ord] += 0.1
      if c.word.contains("proc") or c.word.contains("nim") or c.word.contains("code"): scores[exCode.ord] += 0.2
  # 正規化
  var sum: float32 = 0.0
  for i in 0..<4: sum += scores[i]
  if sum > 0:
    for i in 0..<4: scores[i] /= sum
  var top = exGeneral
  var best = scores[0]
  for i in 1..<4:
    if scores[i] > best:
      best = scores[i]
      top = ExpertId(i)
  result.scores = scores
  result.topExpert = top
  result.confidence = best

proc expertForInput*(experts: array[4, MoEExpert]; gating: MoEGating): MoEExpert =
  experts[gating.topExpert.ord]

proc allExpertScores*(gating: MoEGating): string =
  result = ""
  for i in 0..<4:
    result.add($ExpertId(i) & ":" & $gating.scores[i].formatFloat(ffDecimal,2) & " ")
