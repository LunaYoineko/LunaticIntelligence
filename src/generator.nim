import tables, algorithm, strutils, times, random
import types, grammar

# ---------------------------------------------------------------------------
# TextGenerator: 文法ベースの文章生成
# ---------------------------------------------------------------------------
# テンプレートではなく文法規則に基づいて文を生成
# 人間が文法を学習し、語を組み合わせて文を構築するプロセスを模倣

proc initTextGenerator*(): TextGenerator =
  result.knowledge = initGrammarKnowledge()
  result.templates = @[]
  result.particleRules = initTable[string, seq[string]]()
  result.particleRules["topic"] = @["は"]
  result.particleRules["subject"] = @["が"]
  result.particleRules["object"] = @["を"]
  result.particleRules["location"] = @["に", "で"]

# ---------------------------------------------------------------------------
# 文法ベースの文章生成
# ---------------------------------------------------------------------------
proc generate*(gen: var TextGenerator;
               activeConcepts: seq[ConceptNode];
               context: string;
               episodeText: string = ""): string =
  # エピソード返答モード: そのまま返す
  if context == "parrot" and episodeText.len > 0:
    return episodeText

  # 概念が少ない場合、単純な返答
  if activeConcepts.len == 0:
    return ""

  # 文法知識を使って文を生成
  result = generateSentence(gen.knowledge, activeConcepts, context)

  # 生成が空の場合、概念から直接文を構築（ハードコードなし）
  if result.len == 0:
    # 挨拶コンテキスト: 挨拶概念のみ使用
    if context == "greeting":
      for c in activeConcepts:
        if c.category == ctGreeting:
          return c.word
      return ""

    # その他: 空を返す（呼び出し側で処理）
    return ""

  return result

# ---------------------------------------------------------------------------
# コンテキスト判定
# ---------------------------------------------------------------------------
proc detectContext*(tokens: seq[string]): string =
  for tok in tokens:
    if tok in ["おはよう", "こんにちは", "こんばんは", "はじめまして",
               "ありがとう", "すみません"]:
      return "greeting"
    if tok in ["何", "どこ", "いつ", "誰", "なぜ", "どう", "どの", "どんな", "か", "？", "?"]:
      return "question"
  return "description"

proc detectContextFromConcepts*(concepts: seq[ConceptNode]): string =
  for c in concepts:
    if c.category == ctGreeting:
      return "greeting"
    if c.category == ctQuestion:
      return "question"
  return "description"

# ---------------------------------------------------------------------------
# 文の文法評価
# ---------------------------------------------------------------------------
proc evaluateGenerated*(gen: TextGenerator; sentence: string): float32 =
  return evaluateSentence(gen.knowledge, sentence)

# ---------------------------------------------------------------------------
# 文法規則の学習
# ---------------------------------------------------------------------------
proc learnGrammar*(gen: var TextGenerator; corpus: seq[string]) =
  gen.knowledge.learnFromCorpus(corpus)
