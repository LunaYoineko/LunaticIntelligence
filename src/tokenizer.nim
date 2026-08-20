import tables, unicode, strutils, sequtils, algorithm, heapqueue
import types

# ---------------------------------------------------------------------------
# BPE (Byte Pair Encoding) Tokenizer
# ---------------------------------------------------------------------------
# Character-levelから出発し、頻出ペアを反復的にマージして単語レベル词汇を構築

type
  Merge = object
    left: string
    right: string
    freq: int

proc buildTokenizer*(corpus: seq[string]; maxVocab: int = 4096): Tokenizer =
  # 単語ベース: コーパスから高頻度文字列を辞書として構築（BPE不要）
  result.vocab = @[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN]
  result.tokenToId = initTable[string, int]()
  result.tokenToId[PAD_TOKEN] = PAD_ID
  result.tokenToId[UNK_TOKEN] = UNK_ID
  result.tokenToId[EOS_TOKEN] = EOS_ID

  var wordFreq: Table[string, int]
  for text in corpus:
    var current = ""
    var isAlpha = false
    for rune in text.toRunes:
      let cp = rune.int32
      # CJK文字（日本語）
      if (cp >= 0x3040 and cp <= 0x309F) or
         (cp >= 0x30A0 and cp <= 0x30FF) or
         (cp >= 0x4E00 and cp <= 0x9FFF):
        if current.len > 0 and isAlpha:
          wordFreq[current] = wordFreq.getOrDefault(current, 0) + 1
          current = ""
          isAlpha = false
        current.add($rune)
      # 英語（アルファベット＋数字）
      elif (cp >= 0x0041 and cp <= 0x005A) or
           (cp >= 0x0061 and cp <= 0x007A) or
           (cp >= 0x0030 and cp <= 0x0039) or
           cp == 0x005F:  # アンダースコア
        if current.len > 0 and not isAlpha:
          wordFreq[current] = wordFreq.getOrDefault(current, 0) + 1
          current = ""
        current.add($rune)
        isAlpha = true
      else:
        if current.len > 0:
          wordFreq[current] = wordFreq.getOrDefault(current, 0) + 1
          current = ""
          isAlpha = false
    if current.len > 0:
      wordFreq[current] = wordFreq.getOrDefault(current, 0) + 1

  var sorted = toSeq(wordFreq.pairs)
  sorted.sort(proc(a, b: (string, int)): int = cmp(b[1], a[1]))

  var idx = 3
  for (word, _) in sorted:
    if idx >= maxVocab: break
    if word.len > 0 and not result.tokenToId.hasKey(word):
      result.vocab.add(word)
      result.tokenToId[word] = idx
      inc idx

  echo "Tokenizer built: " & $result.vocab.len & " vocab (from " & $maxVocab & " max)"

proc encode*(tokenizer: Tokenizer; text: string): seq[int] =
  result = @[]
  # ルーン列に変換
  var runes: seq[string] = @[]
  for rune in text.toRunes:
    let ch = $rune
    if ch == " " or ch == "\t":
      continue
    runes.add(ch)

  # トークン辞書から最長一致検索用のセットを構築
  var tokenSet = initTable[string, int]()
  for id, token in tokenizer.vocab:
    if token != PAD_TOKEN and token != UNK_TOKEN and token.len > 0:
      tokenSet[token] = id

  var pos = 0
  while pos < runes.len:
    var bestLen = 0
    var bestId = UNK_ID
    # 最大20文字まで検索（日本語のBPEトークンは通常短い）
    let maxTry = min(20, runes.len - pos)
    var candidate = ""
    for tryLen in 1..maxTry:
      candidate.add(runes[pos + tryLen - 1])
      if tokenSet.hasKey(candidate):
        bestLen = tryLen
        bestId = tokenSet[candidate]
    if bestLen > 0:
      result.add(bestId)
      pos += bestLen
    else:
      result.add(UNK_ID)
      pos += 1

proc decode*(tokenizer: Tokenizer; ids: seq[int]): string =
  result = ""
  for id in ids:
    if id >= 0 and id < tokenizer.vocab.len:
      let t = tokenizer.vocab[id]
      if t == EOS_TOKEN: break
      if t != PAD_TOKEN:
        result.add(t)
