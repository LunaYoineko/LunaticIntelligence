import times, algorithm, strutils
import types

# ---------------------------------------------------------------------------
# WorkingMemory: 作業記憶（Miller's Law: 7±2）
# ---------------------------------------------------------------------------
# 人間の短期記憶を模倣。限られた容量の中で最も関連性の高い概念を保持。

proc initWorkingMemory*(capacity: int = 7): WorkingMemory =
  result.items = @[]
  result.capacity = capacity

# ---------------------------------------------------------------------------
# アイテム追加
# ---------------------------------------------------------------------------
proc store*(wm: var WorkingMemory; conceptId: int; activation: float32;
            source: string = "input") =
  # 既存アイテムの更新チェック
  for i in 0..<wm.items.len:
    if wm.items[i].conceptId == conceptId:
      wm.items[i].activation = max(wm.items[i].activation, activation)
      wm.items[i].timestamp = epochTime()
      return

  # 容量超過時は最古のアイテムを忘却
  if wm.items.len >= wm.capacity:
    wm.items.sort(proc(a, b: WMItem): int =
      # 活性化が低い順、次に古い順
      let actCmp = cmp(b.activation, a.activation)
      if actCmp != 0: return actCmp
      cmp(a.timestamp, b.timestamp)
    )
    wm.items.delete(wm.items.len - 1)

  wm.items.add(WMItem(
    conceptId: conceptId,
    activation: activation,
    timestamp: epochTime(),
    source: source
  ))

proc storeBatch*(wm: var WorkingMemory; conceptIds: seq[int];
                 activations: seq[float32]; source: string = "spread") =
  for i in 0..<conceptIds.len:
    let act = if i < activations.len: activations[i] else: 0.5
    wm.store(conceptIds[i], act, source)

# ---------------------------------------------------------------------------
# 取得
# ---------------------------------------------------------------------------
proc getItems*(wm: WorkingMemory): seq[WMItem] =
  result = wm.items
  result.sort(proc(a, b: WMItem): int = cmp(b.activation, a.activation))

proc getTopConceptIds*(wm: WorkingMemory; topK: int = 7): seq[int] =
  result = @[]
  var sorted_items = wm.items
  sorted_items.sort(proc(a, b: WMItem): int = cmp(b.activation, a.activation))
  for i in 0..<min(topK, sorted_items.len):
    result.add(sorted_items[i].conceptId)

proc getTopConceptIdsWithActivation*(wm: WorkingMemory;
                                     topK: int = 7): seq[(int, float32)] =
  result = @[]
  var sorted_items = wm.items
  sorted_items.sort(proc(a, b: WMItem): int = cmp(b.activation, a.activation))
  for i in 0..<min(topK, sorted_items.len):
    result.add((sorted_items[i].conceptId, sorted_items[i].activation))

# ---------------------------------------------------------------------------
# フィードバック（使用されたアイテムの強化）
# ---------------------------------------------------------------------------
proc reinforce*(wm: var WorkingMemory; conceptId: int; amount: float32 = 0.1) =
  for i in 0..<wm.items.len:
    if wm.items[i].conceptId == conceptId:
      wm.items[i].activation = min(1.0, wm.items[i].activation + amount)
      return

# ---------------------------------------------------------------------------
# リセット
# ---------------------------------------------------------------------------
proc clear*(wm: var WorkingMemory) =
  wm.items = @[]

proc decayAll*(wm: var WorkingMemory; rate: float32 = 0.1) =
  for i in 0..<wm.items.len:
    wm.items[i].activation = max(0.0, wm.items[i].activation - rate)

# ---------------------------------------------------------------------------
# デバッグ出力
# ---------------------------------------------------------------------------
proc dump*(wm: WorkingMemory) =
  echo "Working Memory (" & $wm.items.len & "/" & $wm.capacity & "):"
  for item in wm.getItems():
    echo "  concept=" & $item.conceptId &
         " act=" & $formatFloat(item.activation, ffDecimal, 3) &
         " src=" & item.source
