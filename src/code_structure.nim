import strutils, sequtils, unicode, tables

type
  CodeStructure* = object
    isCode*: bool
    summary*: string
    keywords*: seq[string]
    concepts*: seq[string]
    lang*: string

proc isCodeInput*(text: string): bool =
  let lower = text.toLower()
  let codeKeywords = @["proc ", "func ", "import ", "let ", "var ", "const ", "type ", "if ", "for ", "while ", "return ", "def ", "class ", "include ", "from ", "echo ", "result ", "discard ", "case ", "of ", "when ", "template ", "macro ", "iterator ", "converter ", "method ", "using ", "object ", "enum ", "tuple ", "seq[", "string", "int ", "bool", "float", "=>", "->", "{", "}", ";", "nim ", "python ", "javascript", "```"]
  var score = 0
  for kw in codeKeywords:
    if lower.contains(kw):
      inc score
  if score >= 1 and (lower.contains("proc") or lower.contains("import") or lower.contains("func") or lower.contains("let ") or lower.contains("var ") or lower.contains("```") or lower.contains("echo") or lower.contains(";")):
    return true
  if text.contains("```"):
    return true
  var hasIndentCode = false
  for line in text.splitLines():
    if line.startsWith("  ") and (line.contains("=") or line.contains("(")):
      hasIndentCode = true
  if hasIndentCode and score >= 2:
    return true
  return score >= 3

proc parseNimStructure*(text: string): CodeStructure =
  result.isCode = isCodeInput(text)
  result.summary = ""
  result.keywords = @[]
  result.concepts = @[]
  result.lang = "nim"
  if not result.isCode:
    return
  var kw: seq[string] = @[]
  let lower = text.toLower()
  for w in ["proc", "func", "import", "let", "var", "const", "type", "if", "for", "while", "return", "echo", "result", "discard", "case", "object", "enum"]:
    if lower.contains(w):
      kw.add(w)
  result.keywords = kw
  var firstLine = ""
  for line in text.splitLines():
    let t = line.strip()
    if t.len > 0:
      firstLine = t
      break
  if firstLine.len > 80:
    firstLine = firstLine[0..<80]
  result.summary = "コード解析: " & firstLine
  result.concepts = kw

proc extractCodeConcepts*(cs: CodeStructure): seq[string] =
  result = @[]
  for kw in cs.keywords:
    if kw.len >= 2:
      result.add(kw)
  if cs.summary.len > 0:
    for w in strutils.splitWhitespace(cs.summary):
      if w.len >= 2:
        result.add(w)
  result = result.deduplicate()

proc parseCode*(text: string): CodeStructure =
  result = parseNimStructure(text)
  if result.isCode and result.summary.len == 0:
    var preview = text.strip()
    if preview.len > 100:
      preview = preview[0..<100]
    result.summary = "コード: " & preview
