import tables, math, unicode

# ---------------------------------------------------------------------------
# Basic math utilities
# ---------------------------------------------------------------------------
type
  Matrix* = object
    data*: seq[float32]
    rows*, cols*: int

  Vector* = seq[float32]

proc newMatrix*(rows, cols: int; fill: float32 = 0.0): Matrix =
  result = Matrix(rows: rows, cols: cols)
  result.data = newSeq[float32](rows * cols)
  if fill != 0.0:
    for i in 0..<rows * cols: result.data[i] = fill

proc `[]`*(m: Matrix, i, j: int): float32 {.inline.} =
  m.data[i * m.cols + j]

proc `[]=`*(m: var Matrix, i, j: int, v: float32) {.inline.} =
  m.data[i * m.cols + j] = v

proc len*(m: Matrix): int = m.rows

# 文字数カウント（ルーン単位、CJK対応）
proc countRunes*(s: string): int =
  result = s.toRunes.len

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
const
  PAD_TOKEN* = "<PAD>"
  UNK_TOKEN* = "<UNK>"
  EOS_TOKEN* = "<EOS>"
  PAD_ID* = 0
  UNK_ID* = 1
  EOS_ID* = 2

type
  Tokenizer* = object
    vocab*: seq[string]
    tokenToId*: Table[string, int]

# ---------------------------------------------------------------------------
# Concept Network types (semantic memory)
# ---------------------------------------------------------------------------
type
  ConceptType* = enum
    ctNoun      # 名詞: 猫、山、食べ物
    ctVerb      # 動詞: 食べる、行く、話す
    ctAdj       # 形容詞: 美しい、大きい、楽しい
    ctParticle  # 助詞: は、が、を、に
    ctAbstract  # 抽象: 愛、幸福、知恵
    ctGreeting  # 挨拶: おはよう、こんにちは
    ctQuestion  # 質問: 何、どこ、いつ
    ctEmotion   # 感情: 嬉しい、悲しい、驚き

  ConceptNode* = object
    id*: int
    word*: string            # "猫", "美しい", "こんにちは"
    category*: ConceptType
    activation*: float32     # 現在の活性化レベル (0.0-1.0)
    baseFrequency*: float32  # 基本出現頻度（長期記憶）
    lastAccessed*: float     # 最終アクセス時刻
    accessCount*: int        # アクセス回数

  EdgeRelation* = enum
    erIsA          # 猫 → 動物
    erHasProperty  # 猫 → 毛がある
    erCauses       # 雨 → 濡れる
    erRelatedTo    # 猫 → 犬
    erPartOf       # 爪 → 猫
    erTemporal     # 朝 → おはよう

  ConceptEdge* = object
    fromId*: int
    toId*: int
    relation*: EdgeRelation
    weight*: float32         # 結合強度 (0.0-1.0)
    hebbianCount*: int       # 同時発火回数
    lastCoactivated*: float  # 最終同時活性化時刻

  ConceptGraph* = object
    nodes*: seq[ConceptNode]
    edges*: seq[ConceptEdge]
    nodeIndex*: Table[string, int]  # word → nodeID
    adjacency*: Table[int, seq[int]]  # nodeID → 隣接edgeID list
    edgeSet*: Table[(int, int), int]  # (fromId, toId) → edgeId（重複チェック用）

# ---------------------------------------------------------------------------
# Working Memory types (Miller's Law: 7±2)
# ---------------------------------------------------------------------------
type
  WMItem* = object
    conceptId*: int
    activation*: float32
    timestamp*: float
    source*: string           # "input", "spread", "episode", "reasoning"

  WorkingMemory* = object
    items*: seq[WMItem]
    capacity*: int            # 7 (Miller's Law)

# ---------------------------------------------------------------------------
# Tsetlin Machine types (logical reasoning)
# ---------------------------------------------------------------------------
type
  TsetlinClause* = object
    actions*: seq[int8]
    output*: int8

  TsetlinLayer* = object
    numFeatures*: int
    numClauses*: int
    threshold*: float32
    sParam*: float32
    clauses*: seq[TsetlinClause]
    states*: seq[int8]
    pos_reward*: seq[float32]   # 条款ごとの正の報酬
    neg_reward*: seq[float32]   # 条款ごとの罰

  HierarchicalTM* = object
    layers*: seq[TsetlinLayer]
    numClasses*: int
    classVotes*: seq[int]
    confidence*: float32
    clauseOutput*: seq[bool]

  ClauseReasoning* = object
    clauseId*: int
    firedConcepts*: seq[int]  # 発火した概念ID
    confidence*: float32
    pattern*: string           # "挨拶+質問→返答テンプレート"

# ---------------------------------------------------------------------------
# Hebbian Synapse types (connection strength dynamics)
# ---------------------------------------------------------------------------
type
  Synapse* = object
    clauseId*: int            # TM条款ID
    conceptId*: int           # 概念ID
    strength*: float32        # 結合強度 (0.0-1.0)
    activationCount*: int     # 同時発火回数
    lastActivated*: float     # 最終活性化時刻
    halfLifeDays*: float32    # 半減期（日）

  SynapticBridge* = object
    synapses*: seq[Synapse]
    halfLifeDays*: float32
    conceptIndex*: Table[int, seq[int]]  # conceptId → synapse indices

  UnknownWordTracker* = object
    wordCounts*: Table[string, int]      # word →出現回数
    threshold*: int                       # 学習に必要な出現回数
    pendingWords*: seq[string]            # 学習待ち単語リスト

# ---------------------------------------------------------------------------
# Episode types (experiential memory)
# ---------------------------------------------------------------------------
type
  Speaker* = enum
    spUser = "user"
    spSystem = "system"

  Episode* = object
    inputText*: string         # 入力テキスト
    outputText*: string        # 出力テキスト
    inputConceptIds*: seq[int] # 入力の概念ID列
    outputConceptIds*: seq[int]
    tmClausePattern*: seq[bool]  # TM条款の活性化パターン
    confidence*: float32
    speaker*: Speaker
    contextTag*: string        # "greeting", "question", "answer"
    situation*: string         # "observe", "parrot", "chat"
    timestamp*: float
    reward*: float32           # 報酬 (0.0-1.0)
    rank*: float32             # 総合ランク (0.0-1.0)
    accessCount*: int
    emotionalValence*: float32 # 感情価 (-1.0〜1.0、驚き・重要度)

  EpisodeStore* = object
    episodes*: seq[Episode]
    maxEpisodes*: int
    totalAccess*: int
    phase*: int

# ---------------------------------------------------------------------------
# Grammar types (Japanese grammar-based text generation)
# ---------------------------------------------------------------------------
type
  # 文法要素の種別
  GrammarSlotKind* = enum
    gskSubject      # 主語（は/が）
    gskObject       # 目的語（を/に/で）
    gskVerb         # 動詞（活用形）
    gskAdjective    # 形容詞（い/な）
    gskParticle     # 助詞
    gskAdverb       # 副詞
    gskCopula       # 断定（だ/です）
    gskFixed        # 固定文字列

  GrammarSlot* = object
    kind*: GrammarSlotKind
    wordRequired*: bool         # このスロットに単語が必要か
    categoryFilter*: ConceptType  # 許容する概念カテゴリ
    fixedContent*: string       # gskFixedの場合の文字列

  # 文法規則（文の構造パターン）
  GrammarRule* = object
    name*: string               # 規則名（例: "基本文", "疑問文"）
    slots*: seq[GrammarSlot]    # 文の構造
    probability*: float32       # 使用確率
    usageCount*: int            # 使用回数
    successRate*: float32       # 成功率（報酬ベース）

  # 動詞活用規則
  VerbConjugation* = enum
    vcMasu,       # ます形（丁寧）
    vcRu,         # ル形（普通）
    vcTe,         # て形
    vcNai,        # ない形
    vcTa,         # た形（過去）

  ConjugationRule* = object
    verb*: string               # 辞書形
    masu*: string               # ます形
    teForm*: string             # て形
    naiForm*: string            # ない形
    taForm*: string             # た形
    category*: ConceptType      # 動詞カテゴリ

  # 形容詞活用規則
  AdjConjugation* = enum
    acIAdj,       # い形容詞
    acNaAdj,      # な形容詞

  AdjRule* = object
    word*: string               # 原形
    adjType*: AdjConjugation
    positive*: string           # 肯定形
    negative*: string           # 否定形
    teForm*: string             # て形

  # 文法知識ベース
  GrammarKnowledge* = object
    rules*: seq[GrammarRule]           # 文法規則
    verbConjugations*: seq[ConjugationRule]  # 動詞活用
    adjRules*: seq[AdjRule]            # 形容詞活用
    particles*: seq[string]            # 助詞リスト
    questionWords*: seq[string]        # 疑問詞
    sentencePatterns*: seq[string]     # 文パターン（S-O-V等）

  TextGenerator* = object
    knowledge*: GrammarKnowledge       # 文法知識
    templates*: seq[Template]          # 互換性のため保持
    particleRules*: Table[string, seq[string]]

  # テンプレート（後方互換）
  SlotType* = enum
    stConcept
    stFixed
    stParticle

  TokenSlot* = object
    slotType*: SlotType
    content*: string
    conceptCategory*: ConceptType

  Template* = object
    name*: string
    slots*: seq[TokenSlot]
    context*: string
    usageCount*: int

# ---------------------------------------------------------------------------
# Thinking & Self-Evaluation Types
# ---------------------------------------------------------------------------
type
  # 入力意図の種別
  InputIntent* = enum
    iiGreeting,    # 挨拶: おはよう, こんにちは, etc.
    iiQuestion,    # 質問: 何？, どこ？, etc.
    iiRequest,     # 依頼: ～して, ～教えて
    iiOpinion,     # 感想: ～いいね, ～好き
    iiAgreement,   # 同意: そうですね, ほんとだ
    iiThanks,      # 感謝: ありがとう, すみません
    iiFarewell,    # 離別: さようなら, またね
    iiStatement,   # 陈述: 事実や意見の伝達
    iiOther        # その他

  CatalogEntry* = object
    intent*: InputIntent
    keyword*: string
    inputText*: string
    outputText*: string
    weight*: float32

  ResponseCatalog* = object
    entries*: seq[CatalogEntry]

  # 応答カテゴリ
  ResponseCategory* = enum
    rcGreetingResponse,    # 挨拶への返答
    rcQuestionAnswer,      # 質問への回答
    rcRequestCompliance,   # 依頼への応答
    rcOpinionResponse,     # 感想への共感
    rcAgreementResponse,   # 同意への応答
    rcThanksResponse,      # 感謝への返答
    rcFarewellResponse,    # 離別への返答
    rcElaboration,         # 詳細説明
    rcParrot               # オウム返し（学習用）

  IntentPattern* = object
    keywords*: seq[string]
    intent*: InputIntent
    responseCategory*: ResponseCategory
    priority*: int

  IntentClassifier* = object
    patterns*: seq[IntentPattern]
    intentHistory*: seq[(InputIntent, int)]

  ThinkingStepKind* = enum
    tsPerception,       # 感知: 入力の理解
    tsActivation,       # 活性化: 関連概念の発見
    tsSpreading,        # 伝播: 概念ネットワークの探索
    tsReasoning,        # 推論: TM条款の発火
    tsRetrieval,        # 検索: 類似エピソードの発見
    tsSelection,        # 選択: テンプレート/表現の決定
    tsConclusion        # 結論: 最終出力の構成

  ThinkingStep* = object
    kind*: ThinkingStepKind
    description*: string      # 人間可読な説明
    details*: seq[string]     # 詳細情報
    confidence*: float32      # このステップの確信度
    timeTaken*: float         # 処理時間（ms）

  ThinkingChain* = object
    steps*: seq[ThinkingStep]
    totalConfidence*: float32
    reasoningPath*: seq[int]  # 発火したTM条款ID列

  EvalVerdict* = enum
    evAccept,       # 受容: 出力は論理的で適切
    evRefine,       # 改善: 部分的に問題あり、修正可能
    evReject        # 拒否: 出力に重大な問題あり

  EvalResult* = object
    verdict*: EvalVerdict
    score*: float32            # 0.0-1.0 (評価スコア)
    reason*: string            # 評価理由
    contradictions*: seq[string]  # 発見された矛盾
    relevanceScore*: float32   # 入力との関連度
    coherenceScore*: float32   # 論理的一貫性
    improvements*: seq[string] # 改善提案

  RewardSignal* = object
    clauseBoosts*: seq[(int, float32)]  # (clauseId, 報酬量)
    synapseBoosts*: seq[(int, int, float32)]  # (clauseId, conceptId, 報酬量)
    penaltyClauses*: seq[(int, float32)]  # (clauseId, 罰量)
    penaltySynapses*: seq[(int, int, float32)]  # (clauseId, conceptId, 罰量)

# ---------------------------------------------------------------------------
# Cognitive State (main orchestrator)
# ---------------------------------------------------------------------------
type
  CognitiveState* = object
    cfg*: CognitiveConfig
    tokenizer*: Tokenizer
    conceptGraph*: ConceptGraph
    workingMemory*: WorkingMemory
    tm*: HierarchicalTM
    bridge*: SynapticBridge
    episodeStore*: EpisodeStore
    catalog*: ResponseCatalog
    generator*: TextGenerator
    intentClassifier*: IntentClassifier  # 意図分類器
    llmDBPath*: string                  # LLM重みDBパス
    phase*: int                # 0:観察, 1:オウム返し, 2:理解, 3:生成
    lastThinking*: ThinkingChain   # 最後のシンキングチェーン
    lastEval*: EvalResult          # 最後の自己評価結果
    lastIntent*: InputIntent       # 最後の入力意図
    lastResponseCategory*: ResponseCategory  # 最後の応答カテゴリ
    totalReward*: float32          # 累積報酬
    totalPunish*: float32          # 累積罰
    unknownWords*: UnknownWordTracker  # 未学習語トラッカー

  CognitiveConfig* = object
    wmCapacity*: int           # 作業記憶容量 (default: 7)
    spreadSteps*: int          # 伝播ステップ数 (default: 3)
    spreadDecay*: float32      # 伝播減衰率 (default: 0.5)
    activationThreshold*: float32  # 活性化閾値 (default: 0.1)
    tmClauses*: int
    tmThreshold*: float32
    tmSParam*: float32
    halfLifeDays*: float32
    maxEpisodes*: int
    topKEpisodes*: int         # エピソード検索数 (default: 3)
    thinkingEnabled*: bool     # シンキング有効 (default: true)
    evalEnabled*: bool         # 自己評価有効 (default: true)
    rewardRate*: float32       # 報酬学習率 (default: 0.05)
    punishRate*: float32       # 罰学習率 (default: 0.03)
