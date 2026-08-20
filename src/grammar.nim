import tables, strutils, algorithm, random
import types

# ---------------------------------------------------------------------------
# Grammar: 日本語文法エンジン
# ---------------------------------------------------------------------------
# テンプレートではなく文法規則に基づいて文を生成
# 人間が言語を学習するプロセスを模倣

proc initConjugationRules(): seq[ConjugationRule] =
  # 基本的な動詞の活用規則を初期化
  # コーパスから学習する前に、基本パターンを提供
  result = @[
    # 五段動詞（基本）
    ConjugationRule(verb: "る", masu: "ります", teForm: "って", naiForm: "らない", taForm: "った", category: ctVerb),
    ConjugationRule(verb: "く", masu: "きます", teForm: "いて", naiForm: "かない", taForm: "いた", category: ctVerb),
    ConjugationRule(verb: "ぐ", masu: "ぎます", teForm: "いで", naiForm: "がない", taForm: "いだ", category: ctVerb),
    ConjugationRule(verb: "す", masu: "します", teForm: "して", naiForm: "さない", taForm: "した", category: ctVerb),
    ConjugationRule(verb: "つ", masu: "ちます", teForm: "って", naiForm: "たない", taForm: "った", category: ctVerb),
    ConjugationRule(verb: "ぬ", masu: "にます", teForm: "んで", naiForm: "ない", taForm: "んだ", category: ctVerb),
    ConjugationRule(verb: "ぶ", masu: "びます", teForm: "んで", naiForm: "ばない", taForm: "んだ", category: ctVerb),
    ConjugationRule(verb: "む", masu: "みます", teForm: "んで", naiForm: "まない", taForm: "んだ", category: ctVerb),
    ConjugationRule(verb: "う", masu: "います", teForm: "って", naiForm: "わない", taForm: "った", category: ctVerb),
    # 一段動詞
    ConjugationRule(verb: "食べる", masu: "食べます", teForm: "食べて", naiForm: "食べない", taForm: "食べた", category: ctVerb),
    ConjugationRule(verb: "見る", masu: "見ます", teForm: "見て", naiForm: "見ない", taForm: "見た", category: ctVerb),
    ConjugationRule(verb: "起きる", masu: "起きます", teForm: "起きて", naiForm: "起きない", taForm: "起きた", category: ctVerb),
    ConjugationRule(verb: "寝る", masu: "寝ます", teForm: "寝て", naiForm: "寝ない", taForm: "寝た", category: ctVerb),
    # 不規則動詞
    ConjugationRule(verb: "する", masu: "します", teForm: "して", naiForm: "しない", taForm: "した", category: ctVerb),
    ConjugationRule(verb: "来る", masu: "来ます", teForm: "来て", naiForm: "来ない", taForm: "来た", category: ctVerb),
    # 付属動詞
    ConjugationRule(verb: "ある", masu: "あります", teForm: "あって", naiForm: "ない", taForm: "あった", category: ctVerb),
    ConjugationRule(verb: "いる", masu: "います", teForm: "いて", naiForm: "いない", taForm: "いた", category: ctVerb),
  ]

proc initAdjRules(): seq[AdjRule] =
  # 基本的な形容詞の活用規則
  result = @[
    # い形容詞
    AdjRule(word: "大きい", adjType: acIAdj, positive: "大きい", negative: "大きくない", teForm: "大きくて"),
    AdjRule(word: "小さい", adjType: acIAdj, positive: "小さい", negative: "小さくない", teForm: "小さくて"),
    AdjRule(word: "高い", adjType: acIAdj, positive: "高い", negative: "高くない", teForm: "高くて"),
    AdjRule(word: "安い", adjType: acIAdj, positive: "安い", negative: "安くない", teForm: "安くて"),
    AdjRule(word: "新しい", adjType: acIAdj, positive: "新しい", negative: "新しくない", teForm: "新しくて"),
    AdjRule(word: "古い", adjType: acIAdj, positive: "古い", negative: "古くない", teForm: "古くて"),
    AdjRule(word: "良い", adjType: acIAdj, positive: "良い", negative: "良くない", teForm: "良くて"),
    AdjRule(word: "美味しい", adjType: acIAdj, positive: "美味しい", negative: "美味しくない", teForm: "美味しくて"),
    # な形容詞
    AdjRule(word: "静か", adjType: acNaAdj, positive: "静かだ", negative: "静かではない", teForm: "静かで"),
    AdjRule(word: "元気", adjType: acNaAdj, positive: "元気だ", negative: "元気ではない", teForm: "元気で"),
    AdjRule(word: "好き", adjType: acNaAdj, positive: "好きだ", negative: "好きではない", teForm: "好きで"),
    AdjRule(word: "嫌い", adjType: acNaAdj, positive: "嫌いだ", negative: "嫌いではない", teForm: "嫌いで"),
    AdjRule(word: "上手", adjType: acNaAdj, positive: "上手だ", negative: "上手ではない", teForm: "上手で"),
    AdjRule(word: "きれい", adjType: acNaAdj, positive: "きれいだ", negative: "きれいではない", teForm: "きれいで"),
  ]

proc initGrammarKnowledge*(): GrammarKnowledge =
  result.rules = @[
    # === 挨拶系 ===
    GrammarRule(
      name: "greeting_direct",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctGreeting),
      ],
      probability: 0.3, usageCount: 0, successRate: 0.8
    ),
    GrammarRule(
      name: "greeting_response",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctGreeting),
        GrammarSlot(kind: gskFixed, fixedContent: "！"),
      ],
      probability: 0.25, usageCount: 0, successRate: 0.7
    ),
    GrammarRule(
      name: "greeting_question",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctGreeting),
        GrammarSlot(kind: gskFixed, fixedContent: " "),
        GrammarSlot(kind: gskObject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "は"),
        GrammarSlot(kind: gskFixed, fixedContent: "いかがですか"),
      ],
      probability: 0.15, usageCount: 0, successRate: 0.6
    ),

    # === 質問系 ===
    GrammarRule(
      name: "question_answer",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "は"),
        GrammarSlot(kind: gskObject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "です"),
      ],
      probability: 0.25, usageCount: 0, successRate: 0.6
    ),
    GrammarRule(
      name: "question_opinion",
      slots: @[
        GrammarSlot(kind: gskFixed, fixedContent: " "),
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "について"),
        GrammarSlot(kind: gskFixed, fixedContent: " "),
        GrammarSlot(kind: gskAdjective, wordRequired: true, categoryFilter: ctAdj),
        GrammarSlot(kind: gskFixed, fixedContent: "と思います"),
      ],
      probability: 0.2, usageCount: 0, successRate: 0.5
    ),
    GrammarRule(
      name: "question_ask_back",
      slots: @[
        GrammarSlot(kind: gskFixed, fixedContent: " "),
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "について"),
        GrammarSlot(kind: gskFixed, fixedContent: "教えてください"),
      ],
      probability: 0.15, usageCount: 0, successRate: 0.5
    ),

    # === 描写系 ===
    GrammarRule(
      name: "description_is",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "は"),
        GrammarSlot(kind: gskAdjective, wordRequired: true, categoryFilter: ctAdj),
        GrammarSlot(kind: gskFixed, fixedContent: "です"),
      ],
      probability: 0.25, usageCount: 0, successRate: 0.6
    ),
    GrammarRule(
      name: "description_action",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "が"),
        GrammarSlot(kind: gskVerb, wordRequired: true, categoryFilter: ctVerb),
        GrammarSlot(kind: gskFixed, fixedContent: "ます"),
      ],
      probability: 0.2, usageCount: 0, successRate: 0.5
    ),
    GrammarRule(
      name: "description_feel",
      slots: @[
        GrammarSlot(kind: gskFixed, fixedContent: " "),
        GrammarSlot(kind: gskAdjective, wordRequired: true, categoryFilter: ctAdj),
        GrammarSlot(kind: gskFixed, fixedContent: "と思います"),
      ],
      probability: 0.15, usageCount: 0, successRate: 0.5
    ),

    # === 基本文（汎用） ===
    GrammarRule(
      name: "basic_sov",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "は"),
        GrammarSlot(kind: gskObject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "を"),
        GrammarSlot(kind: gskVerb, wordRequired: true, categoryFilter: ctVerb),
      ],
      probability: 0.15, usageCount: 0, successRate: 0.4
    ),
    GrammarRule(
      name: "basic_polite",
      slots: @[
        GrammarSlot(kind: gskSubject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "は"),
        GrammarSlot(kind: gskObject, wordRequired: true, categoryFilter: ctNoun),
        GrammarSlot(kind: gskFixed, fixedContent: "を"),
        GrammarSlot(kind: gskVerb, wordRequired: true, categoryFilter: ctVerb),
        GrammarSlot(kind: gskFixed, fixedContent: "ます"),
      ],
      probability: 0.15, usageCount: 0, successRate: 0.4
    ),
  ]

  result.verbConjugations = initConjugationRules()
  result.adjRules = initAdjRules()
  result.particles = @["は", "が", "を", "に", "で", "と", "も", "の", "から", "まで"]
  result.questionWords = @["何", "どこ", "いつ", "誰", "どう", "なぜ", "いくら", "どの"]
  result.sentencePatterns = @["SOV", "SはOをV", "SはAdj", "SにOがある"]

# ---------------------------------------------------------------------------
# 動詞活用
# ---------------------------------------------------------------------------
proc conjugateVerb*(knowledge: GrammarKnowledge; verb: string; form: VerbConjugation): string =
  # 活用規則を検索
  for rule in knowledge.verbConjugations:
    if verb == rule.verb or verb.endsWith(rule.verb):
      let prefix = if verb != rule.verb: verb[0..<(verb.len - rule.verb.len)] else: ""
      case form
      of vcMasu: return prefix & rule.masu
      of vcRu: return prefix & rule.verb
      of vcTe: return prefix & rule.teForm
      of vcNai: return prefix & rule.naiForm
      of vcTa: return prefix & rule.taForm
  # 規則が見つからない場合、そのまま返す
  return verb

# ---------------------------------------------------------------------------
# 形容詞活用
# ---------------------------------------------------------------------------
proc conjugateAdj*(knowledge: GrammarKnowledge; adj: string; positive: bool): string =
  for rule in knowledge.adjRules:
    if adj == rule.word:
      return if positive: rule.positive else: rule.negative
  # 規則が見つからない場合
  if positive: return adj
  return adj & "でない"

# ---------------------------------------------------------------------------
# 文法スロットから文を生成（活性化概念を優先使用）
# ---------------------------------------------------------------------------
proc generateFromRule*(rule: GrammarRule; knowledge: GrammarKnowledge;
                       activeConcepts: seq[ConceptNode];
                       context: string = ""): string =
  result = ""
  var usedIndices: seq[int] = @[]

  for slot in rule.slots:
    case slot.kind
    of gskFixed:
      result.add(slot.fixedContent)
    of gskSubject, gskObject:
      # 該当カテゴリの概念をランダムに選択（バリエーション用）
      var candidates: seq[(int, ConceptNode)] = @[]
      for i in 0..<activeConcepts.len:
        let c = activeConcepts[i]
        if c.category == slot.categoryFilter and i notin usedIndices:
          candidates.add((i, c))
      if candidates.len > 0:
        let idx = rand(candidates.len - 1)
        result.add(candidates[idx][1].word)
        usedIndices.add(candidates[idx][0])
      else:
        # フォルバック: 名詞を探索
        var fallbackCandidates: seq[(int, ConceptNode)] = @[]
        for i in 0..<activeConcepts.len:
          let c = activeConcepts[i]
          if c.category == ctNoun and i notin usedIndices:
            fallbackCandidates.add((i, c))
        if fallbackCandidates.len > 0:
          let idx = rand(fallbackCandidates.len - 1)
          result.add(fallbackCandidates[idx][1].word)
          usedIndices.add(fallbackCandidates[idx][0])
    of gskVerb:
      var verbCandidates: seq[(int, ConceptNode)] = @[]
      for i in 0..<activeConcepts.len:
        let c = activeConcepts[i]
        if c.category == ctVerb and i notin usedIndices:
          verbCandidates.add((i, c))
      if verbCandidates.len > 0:
        let idx = rand(verbCandidates.len - 1)
        let verb = verbCandidates[idx][1].word
        if context.contains("丁寧") or context.contains("ます"):
          result.add(conjugateVerb(knowledge, verb, vcMasu))
        elif context.contains("過去"):
          result.add(conjugateVerb(knowledge, verb, vcTa))
        elif context.contains("否定"):
          result.add(conjugateVerb(knowledge, verb, vcNai))
        else:
          result.add(conjugateVerb(knowledge, verb, vcRu))
        usedIndices.add(verbCandidates[idx][0])
      else:
        result.add("する")
    of gskAdjective:
      var adjCandidates: seq[(int, ConceptNode)] = @[]
      for i in 0..<activeConcepts.len:
        let c = activeConcepts[i]
        if c.category == ctAdj and i notin usedIndices:
          adjCandidates.add((i, c))
      if adjCandidates.len > 0:
        let idx = rand(adjCandidates.len - 1)
        result.add(conjugateAdj(knowledge, adjCandidates[idx][1].word, true))
        usedIndices.add(adjCandidates[idx][0])
      else:
        result.add("良い")
    of gskParticle:
      if knowledge.particles.len > 0:
        result.add(knowledge.particles[rand(knowledge.particles.len - 1)])
    of gskAdverb:
      result.add("とても")
    of gskCopula:
      if context.contains("丁寧"):
        result.add("です")
      else:
        result.add("だ")

# ---------------------------------------------------------------------------
# 文法規則の学習（コーパスから）
# ---------------------------------------------------------------------------
proc learnFromCorpus*(knowledge: var GrammarKnowledge; corpus: seq[string]) =
  echo "Learning grammar from corpus..."

  # 文パターンの頻度を数える
  var patternCount: Table[string, int]
  for text in corpus:
    let parts = text.split("|")
    if parts.len < 2: continue
    let output = parts[1].strip()
    if output.len == 0: continue

    # 簡易的な文構造解析
    var pattern = ""
    var hasSubject = false
    var hasObject = false
    var hasVerb = false
    var hasQuestion = false

    # パーティクルで文構造を推定
    if output.contains("は") or output.contains("が"):
      hasSubject = true
      pattern.add("S")
    if output.contains("を") or output.contains("に") or output.contains("で"):
      hasObject = true
      pattern.add("O")
    if output.contains("ます") or output.contains("る") or output.contains("す") or output.contains("く"):
      hasVerb = true
      pattern.add("V")
    if output.contains("か") or output.contains("？") or output.contains("?"):
      hasQuestion = true

    if pattern.len > 0:
      patternCount[pattern] = patternCount.getOrDefault(pattern, 0) + 1

  # 頻度の高いパターンを文法規則として追加
  var sortedPatterns: seq[(string, int)]
  for (pat, count) in patternCount.pairs:
    if count >= 3:  # 少なくとも3回出現
      sortedPatterns.add((pat, count))
  sortedPatterns.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))

  echo "  Discovered patterns: " & $sortedPatterns.len
  for (pat, count) in sortedPatterns[0..<min(10, sortedPatterns.len)]:
    echo "    " & pat & ": " & $count & " times"

# ---------------------------------------------------------------------------
# 文生成（メインエントリ）
# ---------------------------------------------------------------------------
proc generateSentence*(knowledge: var GrammarKnowledge;
                       activeConcepts: seq[ConceptNode];
                       context: string = ""): string =
  # 概念をカテゴリ別に分類
  var wordsByCategory: Table[ConceptType, seq[string]]
  for c in activeConcepts:
    if not wordsByCategory.hasKey(c.category):
      wordsByCategory[c.category] = @[]
    wordsByCategory[c.category].add(c.word)

  # コンテキストに応じて文法規則を選択
  var selectedRule: GrammarRule
  var bestScore = -1.0

  for rule in knowledge.rules:
    var score = rule.probability * rule.successRate
    # コンテキストマッチング
    if context == "greeting" and rule.name.startsWith("greeting"):
      score *= 3.0
    elif context == "question" and rule.name.startsWith("question"):
      score *= 3.0
    elif context == "description" and rule.name.startsWith("description"):
      score *= 3.0
    elif context == "polite" and rule.name.contains("polite"):
      score *= 2.0
    # スロットを満たせるかチェック
    var slotScore = 0.0
    var slotCount = 0
    for slot in rule.slots:
      if slot.wordRequired:
        slotCount += 1
        for c in activeConcepts:
          if c.category == slot.categoryFilter:
            slotScore += 1.0
            break
    if slotCount > 0:
      score *= (slotScore / slotCount.float)

    if score > bestScore:
      bestScore = score
      selectedRule = rule

  # 文を生成（活性化概念から選択）
  result = generateFromRule(selectedRule, knowledge, activeConcepts, context)

  # 文法規則の使用回数を更新
  for i in 0..<knowledge.rules.len:
    if knowledge.rules[i].name == selectedRule.name:
      knowledge.rules[i].usageCount += 1
      break

# ---------------------------------------------------------------------------
# 入力に基づく文生成
# ---------------------------------------------------------------------------
proc generateFromInput*(knowledge: var GrammarKnowledge;
                        inputConcepts: seq[ConceptNode];
                        outputConcepts: seq[ConceptNode];
                        context: string = ""): string =
  # 入力概念と出力概念を組み合わせて文を生成
  var combinedConcepts: seq[ConceptNode]
  # 出力概念を優先
  for c in outputConcepts:
    combinedConcepts.add(c)
  # 入力概念を追加（重複を避ける）
  for c in inputConcepts:
    var found = false
    for ec in combinedConcepts:
      if ec.id == c.id:
        found = true
        break
    if not found:
      combinedConcepts.add(c)

  return generateSentence(knowledge, combinedConcepts, context)

# ---------------------------------------------------------------------------
# 文の評価（文法的に正しいか）
# ---------------------------------------------------------------------------
proc evaluateSentence*(knowledge: GrammarKnowledge; sentence: string): float32 =
  result = 0.0

  # 基本的な文法チェック
  # 1. 助詞の使用
  var particleCount = 0
  for p in knowledge.particles:
    if sentence.contains(p):
      particleCount += 1
  if particleCount > 0:
    result += 0.2

  # 2. 動詞の存在
  var hasVerb = false
  for rule in knowledge.verbConjugations:
    if sentence.contains(rule.masu) or sentence.contains(rule.verb) or
       sentence.contains(rule.teForm):
      hasVerb = true
      break
  if hasVerb:
    result += 0.3

  # 3. 文末表現
  if sentence.endsWith("ます") or sentence.endsWith("る") or
     sentence.endsWith("だ") or sentence.endsWith("です"):
    result += 0.2

  # 4. 文の長さ（適切な長さか）
  let len = sentence.len
  if len >= 5 and len <= 50:
    result += 0.1

  # 5. 疑問符の整合性
  if sentence.contains("？") or sentence.contains("?"):
    if sentence.contains("か"):
      result += 0.1

  result = min(1.0, result)
