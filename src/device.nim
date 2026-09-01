import os, strutils, osproc, tables

type
  ComputeDevice* = enum
    cdCPU, cdGPU

  DeviceInfo* = object
    device*: ComputeDevice
    totalVRAM*: int64    # bytes, 0 if CPU-only
    freeVRAM*: int64     # bytes
    usableVRAM*: int64   # free - reserve
    hasGPU*: bool
    busWidth*: int       # for transfer cost estimation

proc parseMemValue(s: string): int64 =
  let t = s.strip()
  if t.endsWith("MiB"):
    return parseInt(t[0..^4].strip()) * 1024 * 1024
  if t.endsWith("MB"):
    return parseInt(t[0..^3].strip()) * 1000 * 1000
  if t.endsWith("GiB"):
    return parseInt(t[0..^4].strip()) * 1024 * 1024 * 1024
  try: return parseInt(t) except: return 0

proc detectNvidiaVRAM*(): tuple[total, free: int64] =
  try:
    let (outp, code) = execCmdEx("nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null")
    if code == 0 and outp.strip().len > 0:
      let parts = outp.strip().split(",")
      if parts.len >= 2:
        let total = parseInt(parts[0].strip()) * 1024 * 1024
        let free = parseInt(parts[1].strip()) * 1024 * 1024
        return (total.int64, free.int64)
  except: discard
  return (0,0)

proc detectRocmVRAM*(): tuple[total, free: int64] =
  try:
    let (outp, code) = execCmdEx("rocm-smi --showmeminfo vram 2>/dev/null | grep -i free")
    if code == 0: discard outp
  except: discard
  return (0,0)

proc detectCpuMem*(): int64 =
  try:
    for line in "/proc/meminfo".lines:
      if line.startsWith("MemAvailable:"):
        let parts = line.splitWhitespace()
        if parts.len >= 2:
          return parseInt(parts[1]) * 1024
  except: discard
  return 0

proc getDeviceInfo*(): DeviceInfo =
  let (nTotal, nFree) = detectNvidiaVRAM()
  if nTotal > 0:
    result.hasGPU = true
    result.totalVRAM = nTotal
    result.freeVRAM = nFree
    result.usableVRAM = max(0, nFree - 512*1024*1024) # 512MB reserve
    result.busWidth = 16 # PCIe 4.0 x16 approx
    if result.usableVRAM > 1_000_000_000:
      result.device = cdGPU
    else:
      result.device = cdCPU
    return
  let (rTotal, rFree) = detectRocmVRAM()
  if rTotal > 0:
    result.hasGPU = true
    result.totalVRAM = rTotal
    result.freeVRAM = rFree
    result.usableVRAM = max(0, rFree - 512*1024*1024)
    result.device = cdGPU
    return
  result.hasGPU = false
  result.totalVRAM = 0
  result.freeVRAM = detectCpuMem()
  result.usableVRAM = result.freeVRAM
  result.device = cdCPU
  result.busWidth = 0

proc shouldOffloadToGPU*(info: DeviceInfo; modelBytes: int64): bool =
  # CPUメイン、GPUはアシスト: 原則CPUで実行、GPUは大規模モデルやバッチで補助的に使用
  if not info.hasGPU: return false
  # アシスト閾値: モデルが100MB以上かつ usableの50%以上ある場合のみ補助的にオフロード
  # 小規模モデル（数MB）はCPUが十分高速でバス転送のオーバーヘッドが無駄
  if modelBytes < 100*1024*1024:
    return false
  if modelBytes < (info.usableVRAM * 5 div 10):
    return true
  return false

proc estimateTransferCost*(bytes: int64; busWidth: int): float =
  if busWidth == 0: return 0.0
  # PCIe帯域 ~32GB/s を想定、バッチ転送コスト
  float(bytes) / (32.0 * 1024 * 1024 * 1024)

proc printDeviceInfo*(info: DeviceInfo) =
  echo "Device: ", $info.device, " hasGPU=", info.hasGPU,
       " totalVRAM=", info.totalVRAM div (1024*1024), "MB",
       " free=", info.freeVRAM div (1024*1024), "MB",
       " usable=", info.usableVRAM div (1024*1024), "MB"
