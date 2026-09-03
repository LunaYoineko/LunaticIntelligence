import os, times, strutils

const
  DefaultMaxMemoryMB* = 1800  # この環境(2GB)で動作保証
  DefaultCpuThrottleMs* = 5
  InferenceCacheMB* = 64
  TrainingCacheMB* = 64

proc getMemoryUsageMB*(): int =
  try:
    let data = readFile("/proc/self/status")
    for line in data.splitLines():
      if line.startsWith("VmRSS:"):
        let parts = line.splitWhitespace()
        if parts.len >= 2:
          return parseInt(parts[1]) div 1024
  except: discard
  return 0

proc shouldThrottle*(maxMB: int = DefaultMaxMemoryMB): bool =
  getMemoryUsageMB() > maxMB

proc throttleCpu*(ms: int = DefaultCpuThrottleMs) =
  if ms > 0: sleep(ms)

proc throttleIfNeeded*(maxMB: int = DefaultMaxMemoryMB; ms: int = DefaultCpuThrottleMs) =
  if shouldThrottle(maxMB):
    GC_fullCollect()
    throttleCpu(ms * 2)
  elif ms > 0:
    # 5TでもPCが固まらないよう定期的に譲る
    if (epochTime()*1000).int mod 1000 < 10:
      throttleCpu(1)

proc setMemoryLimit*(maxMB: int) =
  # SQLiteキャッシュを制限、GCを積極的に
  putEnv("NIM_GC", "orc")
  GC_fullCollect()

proc printResourceStats*(label: string) =
  let mem = getMemoryUsageMB()
  echo label & " mem=" & $mem & "MB"
