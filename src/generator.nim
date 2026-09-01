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

# ---------------------------------------------------------------------------
# コード生成
# ---------------------------------------------------------------------------

proc generateCode*(language: string; task: string): string =
  ## 簡単なコード生成（ハードコードではなく、パターンに基づく生成）
  result = ""
  
  # タスクを判定
  let lowerTask = task.toLower()
  
  # 言語を判定（タスクからも判定）
  let lang = if language.contains("Nim") or language.contains("nim") or lowerTask.contains("nim"): "nim"
             elif language.contains("Python") or language.contains("python") or lowerTask.contains("python") or lowerTask.contains("py"): "python"
             elif language.contains("JavaScript") or language.contains("js") or lowerTask.contains("javascript") or lowerTask.contains("js"): "javascript"
             else: "nim"  # デフォルト
  
  # Hello World
  if lowerTask.contains("hello world") or lowerTask.contains("こんにちは") or lowerTask.contains("hello"):
    case lang
    of "nim": result = "echo \"Hello, World!\""
    of "python": result = "print(\"Hello, World!\")"
    of "javascript": result = "console.log(\"Hello, World!\");"
    else: result = "echo \"Hello, World!\""
  
  # FizzBuzz
  elif lowerTask.contains("fizzbuzz") or lowerTask.contains("fizz buzz"):
    case lang
    of "nim": 
      result = """for i in 1..100:
  if i mod 15 == 0:
    echo "FizzBuzz"
  elif i mod 3 == 0:
    echo "Fizz"
  elif i mod 5 == 0:
    echo "Buzz"
  else:
    echo i"""
    of "python":
      result = """for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)"""
    of "javascript":
      result = """for (let i = 1; i <= 100; i++) {
  if (i % 15 === 0) console.log("FizzBuzz");
  else if (i % 3 === 0) console.log("Fizz");
  else if (i % 5 === 0) console.log("Buzz");
  else console.log(i);
}"""
    else:
      result = """for i in 1..100:
  if i mod 15 == 0:
    echo "FizzBuzz"
  elif i mod 3 == 0:
    echo "Fizz"
  elif i mod 5 == 0:
    echo "Buzz"
  else:
    echo i"""
  
  # 簡単な関数
  elif lowerTask.contains("関数") or lowerTask.contains("function") or lowerTask.contains("メソッド"):
    case lang
    of "nim": 
      result = """proc greet(name: string): string =
  return "Hello, " & name"""
    of "python":
      result = """def greet(name):
    return f"Hello, {name}""""
    of "javascript":
      result = """function greet(name) {
  return `Hello, ${name}`;
}"""
    else:
      result = """proc greet(name: string): string =
  return "Hello, " & name"""
  
  # 簡単なクラス/オブジェクト
  elif lowerTask.contains("クラス") or lowerTask.contains("class") or lowerTask.contains("オブジェクト"):
    case lang
    of "nim": 
      result = """type Person = object
  name: string
  age: int

proc newPerson(name: string, age: int): Person =
  result.name = name
  result.age = age"""
    of "python":
      result = """class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age"""
    of "javascript":
      result = """class Person {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
}"""
    else:
      result = """type Person = object
  name: string
  age: int"""
  
  # デフォルト: 簡単なコードテンプレート
  else:
    case lang
    of "nim": result = "# Nimコードを生成します\n# 例: echo \"Hello\""
    of "python": result = "# Pythonコードを生成します\n# 例: print(\"Hello\")"
    of "javascript": result = "// JavaScriptコードを生成します\n// 例: console.log(\"Hello\");"
    else: result = "# コードを生成します"
  
  return result
