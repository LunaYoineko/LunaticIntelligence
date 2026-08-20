import strutils, tables, unicode
import types

# ---------------------------------------------------------------------------
# IntentClassifier: 入力意図の分類
# ---------------------------------------------------------------------------
# データ駆動: パターンは全てテーブルで定義
# 追加・変更はテーブルを編集するだけ

proc initDefaultPatterns(): seq[IntentPattern] =
  # 意図 → キーワード → 優先度 の対応表
  let patternDefs = [
    # (意図, キーワードリスト, 応答カテゴリ, 優先度)
    (iiGreeting, @["おはよう", "おは", "おはございます", "はろー",
                   "こんにちは", "ちわ", "ちわす",
                   "こんばんは", "ばんは",
                   "はじめまして", "初めまして",
                   "やあ", "よっす", "あんにょ",
                   "hello", "hi", "hey", "good morning", "good afternoon", "good evening"], rcGreetingResponse, 10),

    (iiThanks, @["ありがとう", "あざす", "サンキュー", "感謝",
                 "すみません", "ごめん", "ごめんなさい", "すまん",
                 "thank", "thanks", "appreciate", "sorry", "excuse me"], rcThanksResponse, 10),

    (iiFarewell, @["さようなら", "さよなら", "じゃあね", "またね",
                   "バイバイ", "また明日",
                   "bye", "goodbye", "see you", "farewell"], rcFarewellResponse, 10),

    (iiQuestion, @["何", "なに", "どこ", "いつ", "誰", "だれ",
                   "なぜ", "どう", "どの", "どんな", "いくら", "いくつ",
                   "what", "where", "when", "who", "why", "how",
                   "which", "whom", "whose"], rcQuestionAnswer, 8),

    (iiQuestion, @["か", "？", "?"], rcQuestionAnswer, 5),

    (iiRequest, @["して", "教えて", "やって", "頼む", "お願い", "～して",
                  "ほしい", "見たい", "知りたい", "行きたい",
                  "please", "help", "show", "tell", "explain",
                  "want", "need", "like to"], rcRequestCompliance, 8),

    (iiOpinion, @["いいね", "好き", "楽しい", "面白い", "かわいい", "すごい",
                  "嫌い", "つまらない", "うんざり", "イヤ",
                  "good", "bad", "like", "love", "hate", "fun",
                  "boring", "amazing", "awesome", "terrible"], rcOpinionResponse, 7),

    (iiAgreement, @["そうですね", "ほんとだ", "確かに", "まさに", "その通り",
                    "なるほど", "へえ", "マジ", "そうなんだ",
                    "yes", "yeah", "right", "exactly", "agree",
                    "absolutely", "totally", "indeed"], rcAgreementResponse, 7),
  ]

  result = @[]
  for (intent, keywords, responseCat, priority) in patternDefs:
    result.add(IntentPattern(
      keywords: keywords,
      intent: intent,
      responseCategory: responseCat,
      priority: priority
    ))

proc initIntentClassifier*(): IntentClassifier =
  result.patterns = initDefaultPatterns()
  result.intentHistory = @[]

# ---------------------------------------------------------------------------
# 意図分類: 最も一致するパターンを返す
# ---------------------------------------------------------------------------
proc classifyIntent*(classifier: IntentClassifier; input: string): (InputIntent, ResponseCategory) =
  let normalized = input.strip().toLower()

  var bestPriority = -1
  var bestIntent = iiOther
  var bestResponseCategory = rcElaboration

  for pattern in classifier.patterns:
    for keyword in pattern.keywords:
      if normalized.contains(keyword):
        if pattern.priority > bestPriority:
          bestPriority = pattern.priority
          bestIntent = pattern.intent
          bestResponseCategory = pattern.responseCategory
        break

  return (bestIntent, bestResponseCategory)

# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
proc updateHistory*(classifier: var IntentClassifier; intent: InputIntent) =
  for i in 0..<classifier.intentHistory.len:
    if classifier.intentHistory[i][0] == intent:
      classifier.intentHistory[i][1] += 1
      return
  classifier.intentHistory.add((intent, 1))

proc getIntentFrequency*(classifier: IntentClassifier; intent: InputIntent): float32 =
  var total = 0
  var count = 0
  for (i, c) in classifier.intentHistory:
    total += c
    if i == intent:
      count = c
  if total > 0:
    return count.float32 / total.float32
  return 0.0
