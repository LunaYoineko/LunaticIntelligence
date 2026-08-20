import random, math, sequtils, strutils, tables
import types

proc initTsetlinLayer*(numFeatures, numClauses: int;
                       threshold, sParam: float32): TsetlinLayer =
  result.numFeatures = numFeatures
  result.numClauses = numClauses
  result.threshold = threshold
  result.sParam = sParam
  result.clauses = newSeq[TsetlinClause](numClauses)
  result.states = newSeq[int8](numClauses * numFeatures * 2)
  result.pos_reward = newSeq[float32](numClauses)
  result.neg_reward = newSeq[float32](numClauses)
  for i in 0..<numClauses:
    result.clauses[i].actions = newSeq[int8](numFeatures * 2)
    for j in 0..<(numFeatures * 2):
      let val = rand(1)
      result.clauses[i].actions[j] = if val == 0: -1'i8 else: 1'i8
      result.states[i * numFeatures * 2 + j] = if val == 0: 0'i8 else: 1'i8

proc initHierarchicalTM*(layerConfigs: seq[tuple[numFeatures, numClauses: int;
                           threshold, sParam: float32]],
                         numClasses: int): HierarchicalTM =
  result.numClasses = numClasses
  result.classVotes = newSeq[int](numClasses)
  result.confidence = 0.0f
  result.clauseOutput = @[]
  result.layers = @[]
  for cfg in layerConfigs:
    result.layers.add(initTsetlinLayer(cfg.numFeatures, cfg.numClauses,
                                       cfg.threshold, cfg.sParam))

proc featureVectorFromConcepts*(conceptIds: seq[int]; maxFeatures: int): seq[int8] =
  result = newSeq[int8](maxFeatures)
  for cid in conceptIds:
    # ブロック符号: 各概念IDに対して16ビットのブロックを活性化
    let blockStart = (cid * 16) mod maxFeatures
    for bit in 0..<16:
      let idx = (blockStart + bit) mod maxFeatures
      result[idx] = 1'i8

proc clauseOutput*(layer: var TsetlinLayer; features: seq[int8]): seq[bool] =
  result = newSeq[bool](layer.numClauses)
  let fLen = min(features.len, layer.numFeatures)
  for c in 0..<layer.numClauses:
    var posSum = 0
    var includedCount = 0
    for f in 0..<fLen:
      let stateIdx = c * layer.numFeatures * 2 + f * 2
      let polarityIdx = stateIdx + 1
      let included = layer.states[polarityIdx] == 1
      let polarity = layer.clauses[c].actions[f * 2 + 1]
      if included:
        includedCount += 1
        if polarity == 1:
          posSum += features[f].int
        else:
          posSum += (1 - features[f]).int
    if includedCount > 0:
      let matchRatio = posSum.float32 / includedCount.float32
      result[c] = matchRatio >= layer.threshold
    else:
      result[c] = false

proc predict*(tm: var HierarchicalTM; features: seq[int8]): (int, float32, seq[bool]) =
  tm.classVotes = newSeq[int](tm.numClasses)
  var allClauses: seq[bool] = @[]
  for li in 0..<tm.layers.len:
    let co = tm.layers[li].clauseOutput(features)
    allClauses.add(co)
    var layerVotes = newSeq[int](tm.numClasses)
    let clausesPerClass = tm.layers[li].numClauses div max(tm.numClasses, 1)
    for c in 0..<tm.layers[li].numClauses:
      let classIdx = c div max(clausesPerClass, 1)
      if classIdx < tm.numClasses and co[c]:
        layerVotes[classIdx] += 1
    for cl in 0..<tm.numClasses:
      tm.classVotes[cl] += layerVotes[cl]
  var maxVote = 0
  var bestClass = 0
  var totalVotes = 0
  for cl in 0..<tm.numClasses:
    totalVotes += abs(tm.classVotes[cl])
    if tm.classVotes[cl] > maxVote:
      maxVote = tm.classVotes[cl]
      bestClass = cl
  tm.confidence = if totalVotes > 0: maxVote.float32 / totalVotes.float32 else: 0.0f
  tm.clauseOutput = allClauses
  result = (bestClass, tm.confidence, allClauses)

proc train*(tm: var HierarchicalTM; features: seq[int8]; targetClass: int;
            reward: float32 = 1.0f) =
  ## 高速学習: 条款出力をスキップして直接状態を更新
  for li in 0..<tm.layers.len:
    let clausesPerClass = tm.layers[li].numClauses div max(tm.numClasses, 1)
    let fLen = min(features.len, tm.layers[li].numFeatures)
    # 特徴量のインデックスを事前計算
    var activeFeatures: seq[int] = @[]
    for f in 0..<fLen:
      if features[f] == 1:
        activeFeatures.add(f)

    for c in 0..<tm.layers[li].numClauses:
      let classIdx = c div max(clausesPerClass, 1)
      let isTarget = classIdx == targetClass
      for f in activeFeatures:
        let stateIdx = c * tm.layers[li].numFeatures * 2 + f * 2
        let polarityIdx = stateIdx + 1
        # Type-I: ターゲットクラスの条款を強化
        if isTarget:
          if tm.layers[li].states[polarityIdx] < 1:
            tm.layers[li].states[polarityIdx] += 1
        # Type-II: 非ターゲットクラスの条款を抑制
        elif not isTarget:
          if tm.layers[li].states[stateIdx] > -1:
            tm.layers[li].states[stateIdx] -= 1

proc predictWithReasoning*(tm: var HierarchicalTM; features: seq[int8]): ClauseReasoning =
  let (_, conf, clausePattern) = tm.predict(features)
  result.clauseId = 0
  result.confidence = conf
  result.firedConcepts = @[]
  result.pattern = ""
  var firedCount = 0
  for i, fired in clausePattern:
    if fired:
      result.firedConcepts.add(i)
      firedCount += 1
  result.pattern = $firedCount & " clauses fired (confidence: " &
                   formatFloat(conf, ffDecimal, 3) & ")"

proc type1Feedback*(tm: var HierarchicalTM; clauseId: int;
                    amount: float32 = 0.1) =
  for li in 0..<tm.layers.len:
    if clauseId < tm.layers[li].numClauses:
      for f in 0..<tm.layers[li].numFeatures:
        let stateIdx = clauseId * tm.layers[li].numFeatures * 2 + f * 2
        if stateIdx + 1 < tm.layers[li].states.len:
          if tm.layers[li].states[stateIdx + 1] < 1:
            tm.layers[li].states[stateIdx + 1] += 1

proc type2Feedback*(tm: var HierarchicalTM; clauseId: int;
                    amount: float32 = 0.1) =
  for li in 0..<tm.layers.len:
    if clauseId < tm.layers[li].numClauses:
      for f in 0..<tm.layers[li].numFeatures:
        let stateIdx = clauseId * tm.layers[li].numFeatures * 2 + f * 2
        if stateIdx < tm.layers[li].states.len:
          if tm.layers[li].states[stateIdx] > -1:
            tm.layers[li].states[stateIdx] -= 1
