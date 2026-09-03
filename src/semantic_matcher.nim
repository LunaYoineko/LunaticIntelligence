import strutils, tables, unicode, algorithm
import types

# ---------------------------------------------------------------------------
# SemanticMatcher: 意味レベルのマッチング
# ---------------------------------------------------------------------------
# 語彙一致ではなく、概念ネットワークを使った意味的理解
# 「元気？」→ 健康/体調 の概念を活性化 → 健康に関連する応答を選択

type
  TopicCategory* = enum
    tcHealth,      # 健康/体調
    tcFood,        # 食べ物/料理
    tcWork,        # 仕事/活動
    tcHobby,       # 趣味/娯楽
    tcStudy,       # 学習/教育
    tcRelationship, # 人間関係
    tcEmotion,     # 感情/気持ち
    tcPlace,       # 場所/場面
    tcTime,        # 時間
    tcGeneral      # その他

  TopicKeyword = object
    keywords: seq[string]
    category: TopicCategory

# トピックキーワード辞書
let topicKeywords: seq[TopicKeyword] = @[
  TopicKeyword(keywords: @["元気", "体調", "健康", "病気", "疲れた", "眠い", "頭痛", "風邪"],
               category: tcHealth),
  TopicKeyword(keywords: @["料理", "食べ物", "食事", "好きな食べ", "レシピ", "レストラン", "美味しい"],
               category: tcFood),
  TopicKeyword(keywords: @["仕事", "働", "会社", "プロジェクト", "タスク", "目標"],
               category: tcWork),
  TopicKeyword(keywords: @["趣味", "音楽", "映画", "本", "ゲーム", "スポーツ", "旅行"],
               category: tcHobby),
  TopicKeyword(keywords: @["勉強", "学", "学校", "テスト", "日本語", "言語"],
               category: tcStudy),
  TopicKeyword(keywords: @["友達", "家族", "恋人", "人間関係", "会う", "一緒"],
               category: tcRelationship),
  TopicKeyword(keywords: @["嬉しい", "悲しい", "楽しい", "辛い", "寂しい", "怒り"],
               category: tcEmotion),
]

# ---------------------------------------------------------------------------
# トピックの分類
# ---------------------------------------------------------------------------
proc classifyTopic*(input: string): TopicCategory =
  let normalized = input.strip().toLower()

  for tk in topicKeywords:
    for kw in tk.keywords:
      if normalized.contains(kw):
        return tk.category

  return tcGeneral

# ---------------------------------------------------------------------------
# エピソードのトピック分類
# ---------------------------------------------------------------------------
proc classifyEpisodeTopic*(ep: Episode): TopicCategory =
  # 入力テキストからトピックを分類
  return classifyTopic(ep.inputText)

# ---------------------------------------------------------------------------
# トピック一致度の計算
# ---------------------------------------------------------------------------
proc topicMatchScore*(inputTopic: TopicCategory; ep: Episode): float32 =
  let epTopic = classifyEpisodeTopic(ep)

  if inputTopic == epTopic:
    return 1.0  # 完全一致

  # 類似トピックの定義
  let similarTopics: Table[TopicCategory, seq[TopicCategory]] = {
    tcHealth: @[tcEmotion, tcFood],
    tcFood: @[tcHealth, tcHobby],
    tcWork: @[tcStudy],
    tcHobby: @[tcEmotion, tcFood],
    tcStudy: @[tcWork],
    tcRelationship: @[tcEmotion],
    tcEmotion: @[tcHealth, tcRelationship],
    tcPlace: @[tcGeneral],
    tcTime: @[tcGeneral],
    tcGeneral: @[]
  }.toTable

  if similarTopics.hasKey(inputTopic):
    if epTopic in similarTopics[inputTopic]:
      return 0.5  # 類似トピック

  return 0.0  # 不一致

# ---------------------------------------------------------------------------
# 意味的にマッチするエピソードを検索
# ---------------------------------------------------------------------------
proc findSemanticMatches*(episodes: seq[Episode]; input: string;
                          words: seq[string]; maxResults: int = 5): seq[(Episode, float32)] =
  let inputTopic = classifyTopic(input)
  var candidates: seq[(Episode, float32)] = @[]

  for ep in episodes:
    var score: float32 = 0.0

    # 1. トピック一致度 (最重要)
    let topicScore = topicMatchScore(inputTopic, ep)
    score += topicScore * 3.0

    # 2. キーワード一致度
    var keywordMatch = 0
    for w in words:
      if ep.inputText.contains(w):
        keywordMatch += 1
    if words.len > 0:
      score += (keywordMatch.float32 / words.len.float32) * 1.0

    # 3. エピソードの質（ランクと報酬）
    score += ep.rank * 0.2 + ep.reward * 0.2

    # 4. 入力テキストの長さ（短い入力を優先）
    if ep.inputText.len < 30:
      score += 0.3

    if score > 0.5:  # 最低スコア以上
      candidates.add((ep, score))

  # スコアでソート
  candidates.sort(proc(a, b: (Episode, float32)): int =
    if a[1] > b[1]: return -1
    elif a[1] < b[1]: return 1
    else: return 0
  )

  # 上位N件を返す
  if candidates.len > maxResults:
    return candidates[0..<maxResults]
  return candidates
