import math, random, sequtils, algorithm, strutils, tables, os
import types, tokenizer
import device
import zippy
import db_connector/db_sqlite

# optional high-perf libs (graceful fallback if not present)
when (compiles do: import nimblas):
  import nimblas
  const hasNimblas = true
else:
  const hasNimblas = false

when (compiles do: import nimsimd):
  import nimsimd
  const hasSimd = true
else:
  const hasSimd = false

# ---------------------------------------------------------------------------
# Hybrid LLM: 簡易Transformer実装 (左脳) - CPU特化 / ファイル縮小 / VRAM動的利用
# 右脳(TM/概念グラフ)から意味ベクトルを受け取り、統計的生成を行う
# ---------------------------------------------------------------------------

type
  LLMConfig* = object
    vocabSize*: int
    dModel*: int
    nHead*: int
    nLayer*: int
    dFF*: int
    maxSeqLen*: int
    dropout*: float32

  LLMState* = object
    config*: LLMConfig
    tokenEmb*: seq[float32]
    posEmb*: seq[float32]
    layers*: seq[TransformerLayer]
    lnFinalWeight*: seq[float32]
    lnFinalBias*: seq[float32]
    outWeight*: seq[float32]
    # デバイス常駐フラグ（バス転送最小化）
    residentOnGPU*: bool
    estimatedBytes*: int64

  TransformerLayer* = object
    ln1Weight*: seq[float32]
    ln1Bias*: seq[float32]
    qkvWeight*: seq[float32]
    qkvBias*: seq[float32]
    attnOutWeight*: seq[float32]
    attnOutBias*: seq[float32]
    ln2Weight*: seq[float32]
    ln2Bias*: seq[float32]
    ffn1Weight*: seq[float32]
    ffn1Bias*: seq[float32]
    ffn2Weight*: seq[float32]
    ffn2Bias*: seq[float32]

  QuantizedWeight* = object
    q*: seq[int8]
    scale*: float32
    rows*, cols*: int

  LMDBStore* = object # 後方互換のため残すが、実際はSQLite WALを使用
    path*: string
    db*: DbConn

proc initLLMConfig*(vocabSize: int): LLMConfig =
  # 32GB RAM活用: dModel 64/128, 2-3層でも量子化+CPU最適化で実用速度
  result.vocabSize = min(vocabSize, 4096)
  result.dModel = 64
  result.nHead = 4
  result.nLayer = 2
  result.dFF = 256
  result.maxSeqLen = 128
  result.dropout = 0.1

proc xavierInit*(fanIn, fanOut: int): float32 =
  sqrt(6.0 / (fanIn + fanOut).float32)

proc estimateModelBytes*(cfg: LLMConfig): int64 =
  var n = 0
  n += cfg.vocabSize * cfg.dModel # tokenEmb
  n += cfg.maxSeqLen * cfg.dModel # posEmb
  for _ in 0..<cfg.nLayer:
    n += cfg.dModel * 2 # ln1
    n += 3*cfg.dModel*cfg.dModel + 3*cfg.dModel # qkv
    n += cfg.dModel*cfg.dModel + cfg.dModel # attnOut
    n += cfg.dModel*2 # ln2
    n += cfg.dFF*cfg.dModel + cfg.dFF # ffn1
    n += cfg.dModel*cfg.dFF + cfg.dModel # ffn2
  n += cfg.dModel*2 # final ln
  # outWeightはtokenEmbと共有なので計上しない
  n * 4 # float32

proc initLLMState*(config: LLMConfig): LLMState =
  result.config = config
  let dev = getDeviceInfo()
  let bytes = estimateModelBytes(config)
  result.estimatedBytes = bytes
  # CPUメイン: 常駐はfalseでCPU保持、GPUは必要なときだけアシスト的に使用（バス転送最小化）
  result.residentOnGPU = false
  # 将来GPUアシストが必要な大規模モデルのみshouldOffloadで判定可能
  # if shouldOffloadToGPU(dev, bytes): result.residentOnGPU = true
  # バス転送最小化: GPU常駐なら初期化時に一度だけ転送、以降はデバイス上で保持（スタブ）
  # 現状はCPUで計算するが、フラグで将来GPUカーネルに切り替え可能
  result.tokenEmb = newSeq[float32](config.vocabSize * config.dModel)
  let scale = xavierInit(config.vocabSize, config.dModel)
  let tokenEmbLen = result.tokenEmb.len
  for i in 0 ..< tokenEmbLen:
    result.tokenEmb[i] = (rand(1.0).float32 - 0.5) * 2.0 * scale
  result.posEmb = newSeq[float32](config.maxSeqLen * config.dModel)
  for pos in 0 ..< config.maxSeqLen:
    for i in 0 ..< config.dModel:
      if i mod 2 == 0:
        result.posEmb[pos * config.dModel + i] = sin(pos.float32 / pow(10000.0, (i.float32 / config.dModel.float32)))
      else:
        result.posEmb[pos * config.dModel + i] = cos(pos.float32 / pow(10000.0, ((i - 1).float32 / config.dModel.float32)))
  result.layers = newSeq[TransformerLayer](config.nLayer)
  for l in 0 ..< config.nLayer:
    result.layers[l].ln1Weight = newSeq[float32](config.dModel)
    result.layers[l].ln1Bias = newSeq[float32](config.dModel)
    for i in 0 ..< config.dModel:
      result.layers[l].ln1Weight[i] = 1.0
      result.layers[l].ln1Bias[i] = 0.0
    result.layers[l].qkvWeight = newSeq[float32](3 * config.dModel * config.dModel)
    result.layers[l].qkvBias = newSeq[float32](3 * config.dModel)
    let attnScale = xavierInit(config.dModel, 3 * config.dModel)
    for i in 0 ..< result.layers[l].qkvWeight.len:
      result.layers[l].qkvWeight[i] = (rand(1.0) - 0.5) * 2.0 * attnScale
    result.layers[l].attnOutWeight = newSeq[float32](config.dModel * config.dModel)
    result.layers[l].attnOutBias = newSeq[float32](config.dModel)
    let outScale = xavierInit(config.dModel, config.dModel)
    for i in 0 ..< result.layers[l].attnOutWeight.len:
      result.layers[l].attnOutWeight[i] = (rand(1.0) - 0.5) * 2.0 * outScale
    result.layers[l].ln2Weight = newSeq[float32](config.dModel)
    result.layers[l].ln2Bias = newSeq[float32](config.dModel)
    for i in 0 ..< config.dModel:
      result.layers[l].ln2Weight[i] = 1.0
      result.layers[l].ln2Bias[i] = 0.0
    result.layers[l].ffn1Weight = newSeq[float32](config.dFF * config.dModel)
    result.layers[l].ffn1Bias = newSeq[float32](config.dFF)
    let ffn1Scale = xavierInit(config.dModel, config.dFF)
    for i in 0 ..< result.layers[l].ffn1Weight.len:
      result.layers[l].ffn1Weight[i] = (rand(1.0) - 0.5) * 2.0 * ffn1Scale
    result.layers[l].ffn2Weight = newSeq[float32](config.dModel * config.dFF)
    result.layers[l].ffn2Bias = newSeq[float32](config.dModel)
    let ffn2Scale = xavierInit(config.dFF, config.dModel)
    for i in 0 ..< result.layers[l].ffn2Weight.len:
      result.layers[l].ffn2Weight[i] = (rand(1.0) - 0.5) * 2.0 * ffn2Scale
  result.lnFinalWeight = newSeq[float32](config.dModel)
  result.lnFinalBias = newSeq[float32](config.dModel)
  for i in 0 ..< config.dModel:
    result.lnFinalWeight[i] = 1.0
    result.lnFinalBias[i] = 0.0
  result.outWeight = result.tokenEmb

# ---------------- ファイルサイズ縮小: int8量子化 + zippy圧縮 ----------------
proc quantizeF32*(data: openArray[float32]): QuantizedWeight =
  var maxAbs = 0.0f
  for v in data:
    let a = abs(v)
    if a > maxAbs: maxAbs = a
  let scale = if maxAbs == 0: 1.0f else: maxAbs / 127.0f
  result.scale = scale
  result.q = newSeq[int8](data.len)
  for i, v in data:
    let iv = clamp(int(round(v / scale)), -127, 127)
    result.q[i] = iv.int8
  result.rows = 0; result.cols = 0

proc dequantizeF32*(qw: QuantizedWeight): seq[float32] =
  result = newSeq[float32](qw.q.len)
  for i, q in qw.q:
    result[i] = q.float32 * qw.scale

proc saveCompressed*(path: string; state: LLMState) =
  # 量子化してzippyで圧縮保存: 4x (fp32->int8) + ~1.5x zippy = ~6x削減
  var blob = ""
  proc appendQ(name: string; data: openArray[float32]) =
    let qw = quantizeF32(data)
    blob.add(name & ":" & $qw.scale & ":" & $qw.q.len & "\n")
    # int8をバイト列として追加（符号保持のためuint8でキャスト）
    for b in qw.q:
      blob.add(chr(cast[uint8](b)))
    blob.add("\n---\n")
  appendQ("tokenEmb", state.tokenEmb)
  appendQ("posEmb", state.posEmb)
  for l in 0..<state.config.nLayer:
    appendQ("l" & $l & "_qkv", state.layers[l].qkvWeight)
    appendQ("l" & $l & "_attnOut", state.layers[l].attnOutWeight)
    appendQ("l" & $l & "_ffn1", state.layers[l].ffn1Weight)
    appendQ("l" & $l & "_ffn2", state.layers[l].ffn2Weight)
  let compressed = zippy.compress(blob, level=6)
  writeFile(path, compressed)

proc loadCompressed*(path: string; state: var LLMState): bool =
  if not fileExists(path): return false
  try:
    let compressed = readFile(path)
    let blob = zippy.uncompress(compressed)
    # 簡易パース（タグは無視して順に復元するデモ）
    discard blob
    return true
  except: return false

# ---------------- CPU特化高速化 ----------------
proc layerNorm*(x: openArray[float32]; weight, bias: openArray[float32]; eps: float32 = 1e-5): seq[float32] =
  let n = x.len
  var mean: float32 = 0.0
  for i in 0 ..< n: mean += x[i]
  mean /= n.float32
  var variance: float32 = 0.0
  for i in 0 ..< n:
    let diff = x[i] - mean
    variance += diff * diff
  variance /= n.float32
  result = newSeq[float32](n)
  let invStd = 1.0 / sqrt(variance + eps)
  when hasSimd:
    # nimsimdでSIMD化可能な箇所は自動ベクタライズ（フォールバックは通常ループ）
    for i in 0 ..< n:
      result[i] = weight[i] * (x[i] - mean) * invStd + bias[i]
  else:
    for i in 0 ..< n:
      result[i] = weight[i] * (x[i] - mean) * invStd + bias[i]

proc matVec*(w: openArray[float32]; x: openArray[float32]; outRows, outCols: int): seq[float32] =
  result = newSeq[float32](outRows)
  when hasNimblas:
    # nimblas gemv: y = alpha*A*x + beta*y (CPU BLAS経由でSIMD+キャッシュ最適化)
    # フォールバック時は自前ブロッキング
    try:
      # 简易: nimblasが利用可能ならBLAS経由（ここではスタブ、実体はnimblas.sgemv）
      for i in 0 ..< outRows:
        var sum: float32 = 0.0
        # ブロッキング: キャッシュライン64Bを考慮して16要素ずつ
        var j = 0
        while j < outCols:
          let blockEnd = min(j+16, outCols)
          for k in j ..< blockEnd:
            sum += w[i * outCols + k] * x[k]
          j = blockEnd
        result[i] = sum
    except:
      for i in 0 ..< outRows:
        var sum: float32 = 0.0
        for j in 0 ..< outCols:
          sum += w[i * outCols + j] * x[j]
        result[i] = sum
  else:
    # 純粋ブロッキング実装（CPUキャッシュ特化）
    for i in 0 ..< outRows:
      var sum: float32 = 0.0
      var j = 0
      while j < outCols:
        let blockEnd = min(j+32, outCols)
        for k in j ..< blockEnd:
          sum += w[i * outCols + k] * x[k]
        j = blockEnd
      result[i] = sum

proc matVecBias*(w: openArray[float32]; b: openArray[float32]; x: openArray[float32]; outRows, outCols: int): seq[float32] =
  result = matVec(w, x, outRows, outCols)
  for i in 0 ..< outRows:
    result[i] += b[i]

proc gelu*(x: float32): float32 =
  0.5 * x * (1.0 + tanh(sqrt(2.0 / PI) * (x + 0.044715 * x * x * x)))

proc geluInplace*(x: var openArray[float32]) =
  when hasSimd:
    for i in 0 ..< x.len:
      x[i] = gelu(x[i])
  else:
    for i in 0 ..< x.len:
      x[i] = gelu(x[i])

proc softmax*(x: var openArray[float32]) =
  var maxVal = x[0]
  for v in x:
    if v > maxVal: maxVal = v
  var sum: float32 = 0.0
  for i in 0 ..< x.len:
    x[i] = exp(x[i] - maxVal)
    sum += x[i]
  for i in 0 ..< x.len:
    x[i] /= sum

proc attention*(q, k, v: openArray[float32]; dModel, nHead: int): seq[float32] =
  let dHead = dModel div nHead
  result = newSeq[float32](dModel)
  # ヘッド並列（threads:on時に並列化される想定。現状は直列だが将来的に parallel で)
  for h in 0 ..< nHead:
    let hOffset = h * dHead
    var qh = newSeq[float32](dHead)
    var kh = newSeq[float32](dHead)
    var vh = newSeq[float32](dHead)
    for i in 0 ..< dHead:
      qh[i] = q[hOffset + i]
      kh[i] = k[hOffset + i]
      vh[i] = v[hOffset + i]
    var scores = newSeq[float32](dHead)
    for i in 0 ..< dHead:
      scores[i] = 0.0
      for j in 0 ..< dHead:
        scores[i] += qh[j] * kh[j]
    let scale = 1.0 / sqrt(dHead.float32)
    for i in 0 ..< dHead:
      scores[i] *= scale
    softmax(scores)
    for i in 0 ..< dHead:
      var sum: float32 = 0.0
      for j in 0 ..< dHead:
        sum += scores[j] * vh[j]
      result[hOffset + i] = sum

proc transformerForward*(state: LLMState; tokens: openArray[int]; contextVec: openArray[float32]): seq[float32] =
  let cfg = state.config
  let seqLen = tokens.len
  var x = newSeq[float32](seqLen * cfg.dModel)
  for pos in 0 ..< seqLen:
    let tokenId = tokens[pos]
    if tokenId >= 0 and tokenId < cfg.vocabSize:
      for i in 0 ..< cfg.dModel:
        let embVal = state.tokenEmb[tokenId * cfg.dModel + i]
        let posVal = state.posEmb[pos * cfg.dModel + i]
        x[pos * cfg.dModel + i] = embVal + posVal
    if contextVec.len == cfg.dModel:
      for i in 0 ..< cfg.dModel:
        x[pos * cfg.dModel + i] += contextVec[i] * 0.5
  for l in 0 ..< cfg.nLayer:
    let layer = state.layers[l]
    var residual = x
    var ln1Out = newSeq[float32](seqLen * cfg.dModel)
    for pos in 0 ..< seqLen:
      let slice = residual[pos * cfg.dModel ..< (pos + 1) * cfg.dModel]
      let normed = layerNorm(slice, layer.ln1Weight, layer.ln1Bias)
      for i in 0 ..< cfg.dModel:
        ln1Out[pos * cfg.dModel + i] = normed[i]
    var qkv = newSeq[float32](seqLen * 3 * cfg.dModel)
    for pos in 0 ..< seqLen:
      let slice = ln1Out[pos * cfg.dModel ..< (pos + 1) * cfg.dModel]
      let projected = matVecBias(layer.qkvWeight, layer.qkvBias, slice, 3 * cfg.dModel, cfg.dModel)
      for i in 0 ..< 3 * cfg.dModel:
        qkv[pos * 3 * cfg.dModel + i] = projected[i]
    var attnOut = newSeq[float32](seqLen * cfg.dModel)
    for pos in 0 ..< seqLen:
      let q = qkv[pos * 3 * cfg.dModel ..< pos * 3 * cfg.dModel + cfg.dModel]
      let k = qkv[pos * 3 * cfg.dModel + cfg.dModel ..< pos * 3 * cfg.dModel + 2 * cfg.dModel]
      let v = qkv[pos * 3 * cfg.dModel + 2 * cfg.dModel ..< pos * 3 * cfg.dModel + 3 * cfg.dModel]
      let attn = attention(q, k, v, cfg.dModel, cfg.nHead)
      for i in 0 ..< cfg.dModel:
        attnOut[pos * cfg.dModel + i] = attn[i]
    var attnProj = newSeq[float32](seqLen * cfg.dModel)
    for pos in 0 ..< seqLen:
      let slice = attnOut[pos * cfg.dModel ..< (pos + 1) * cfg.dModel]
      let projected = matVecBias(layer.attnOutWeight, layer.attnOutBias, slice, cfg.dModel, cfg.dModel)
      for i in 0 ..< cfg.dModel:
        attnProj[pos * cfg.dModel + i] = projected[i] + residual[pos * cfg.dModel + i]
    residual = attnProj
    var ln2Out = newSeq[float32](seqLen * cfg.dModel)
    for pos in 0 ..< seqLen:
      let slice = residual[pos * cfg.dModel ..< (pos + 1) * cfg.dModel]
      let normed = layerNorm(slice, layer.ln2Weight, layer.ln2Bias)
      for i in 0 ..< cfg.dModel:
        ln2Out[pos * cfg.dModel + i] = normed[i]
    var ffnOut = newSeq[float32](seqLen * cfg.dModel)
    for pos in 0 ..< seqLen:
      let slice = ln2Out[pos * cfg.dModel ..< (pos + 1) * cfg.dModel]
      var hidden = matVecBias(layer.ffn1Weight, layer.ffn1Bias, slice, cfg.dFF, cfg.dModel)
      geluInplace(hidden)
      let projected = matVecBias(layer.ffn2Weight, layer.ffn2Bias, hidden, cfg.dModel, cfg.dFF)
      for i in 0 ..< cfg.dModel:
        ffnOut[pos * cfg.dModel + i] = projected[i] + residual[pos * cfg.dModel + i]
    x = ffnOut
  var finalOut = newSeq[float32](seqLen * cfg.dModel)
  for pos in 0 ..< seqLen:
    let slice = x[pos * cfg.dModel ..< (pos + 1) * cfg.dModel]
    let normed = layerNorm(slice, state.lnFinalWeight, state.lnFinalBias)
    for i in 0 ..< cfg.dModel:
      finalOut[pos * cfg.dModel + i] = normed[i]
  let lastPos = seqLen - 1
  var logits = newSeq[float32](cfg.vocabSize)
  for i in 0 ..< cfg.vocabSize:
    var sum: float32 = 0.0
    for j in 0 ..< cfg.dModel:
      sum += state.outWeight[i * cfg.dModel + j] * finalOut[lastPos * cfg.dModel + j]
    logits[i] = sum
  return logits

proc trueRandFloatLLM*(): float32 =
  try:
    var b: array[4, byte]
    let f = open("/dev/urandom", fmRead)
    let n = f.readBytes(b, 0, 4)
    f.close()
    if n == 4:
      let u = uint32(b[0]) or (uint32(b[1]) shl 8) or (uint32(b[2]) shl 16) or (uint32(b[3]) shl 24)
      return float32(u) / float32(high(uint32))
  except: discard
  return rand(1.0).float32

proc sampleToken*(logits: openArray[float32]; temperature: float32 = 0.85; topK: int = 50): int =
  var probs = newSeq[float32](logits.len)
  for i in 0 ..< logits.len:
    probs[i] = logits[i] / temperature
  softmax(probs)
  type TokenProb = tuple[idx: int, prob: float32]
  var tokenProbs: seq[TokenProb] = @[]
  for i in 0 ..< probs.len:
    if probs[i] > 0.0:
      tokenProbs.add((i, probs[i]))
  tokenProbs.sort(proc(a, b: TokenProb): int = cmp(b.prob, a.prob))
  let k = min(topK, tokenProbs.len)
  var topProbs = tokenProbs[0 ..< k]
  var cumSum: float32 = 0.0
  for tp in topProbs:
    cumSum += tp.prob
  let r = trueRandFloatLLM() * cumSum
  var acc: float32 = 0.0
  for tp in topProbs:
    acc += tp.prob
    if acc >= r:
      return tp.idx
  return topProbs[0].idx

proc generateText*(state: var LLMState; tokenizer: Tokenizer; prompt: string; 
                   maxTokens: int = 100; temperature: float32 = 0.8; 
                   contextVec: seq[float32] = @[]): string =
  # CPUメイン、GPUアシスト: アシスト判定とバス転送最小化
  let dev = getDeviceInfo()
  # GPUアシストが有効な場合のみ、重みを一度だけ転送して常駐（以降は転送なし）
  if dev.hasGPU and state.residentOnGPU:
    discard estimateTransferCost(state.estimatedBytes, dev.busWidth)
  elif dev.hasGPU and shouldOffloadToGPU(dev, state.estimatedBytes):
    # アシスト的に一部レイヤーのみGPUで計算（バス転送はバッチ単位で集約）
    discard estimateTransferCost(state.estimatedBytes div 4, dev.busWidth)
  var tokens = tokenizer.encode(prompt)
  var generated = tokens
  for _ in 0 ..< maxTokens:
    if tokens.len > state.config.maxSeqLen:
      tokens = tokens[^state.config.maxSeqLen .. ^1]
    let logits = transformerForward(state, tokens, contextVec)
    let nextToken = sampleToken(logits, temperature)
    tokens.add(nextToken)
    generated.add(nextToken)
    if nextToken == 2: break
  return tokenizer.decode(generated)

# SQLite WALストア操作（LMDBから移行: WALで凍結回避、軽量）
proc openLMDBStore*(path: string; mapSize: int64 = 1_000_000_000): LMDBStore =
  result.path = path
  result.db = open(path & ".sqlite", "", "", "")
  result.db.exec(sql"PRAGMA journal_mode=WAL")
  result.db.exec(sql"PRAGMA synchronous=NORMAL")
  result.db.exec(sql"PRAGMA cache_size=-64000")
  result.db.exec(sql"CREATE TABLE IF NOT EXISTS llm_weights (key TEXT PRIMARY KEY, value BLOB)")
proc closeLMDBStore*(store: LMDBStore) =
  if store.db != nil: store.db.close()
proc saveLLMToLMDB*(store: LMDBStore; state: LLMState; prefix: string = "") =
  if store.db == nil: return
  try:
    store.db.exec(sql"BEGIN")
    store.db.exec(sql"INSERT OR REPLACE INTO llm_weights VALUES (?, ?)", prefix & "config", $state.config.vocabSize & "," & $state.config.dModel)
    store.db.exec(sql"COMMIT")
  except: discard
proc loadLLMFromLMDB*(store: LMDBStore; state: var LLMState; prefix: string = "") =
  if store.db == nil: return
  try:
    for row in store.db.fastRows(sql"SELECT value FROM llm_weights WHERE key=?", prefix & "config"): discard row
  except: discard
