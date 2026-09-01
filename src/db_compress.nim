import os, osproc, strutils, times

const CompressorBin* = "/home/luna/CatelliteCompressor/CatelliteCompressor"

proc getCompressedSize*(dbPath: string): int64 =
  if not fileExists(dbPath):
    return 0
  let compressed = dbPath & ".catcmp"
  if fileExists(compressed):
    return getFileSize(compressed)
  return getFileSize(dbPath)

proc compressDB*(dbPath: string): string =
  if not fileExists(dbPath):
    return ""
  let outPath = dbPath & ".catcmp"
  let size = getFileSize(dbPath)
  try:
    if size > 50 * 1024 * 1024:
      discard startProcess(CompressorBin, args=["c", dbPath, outPath], options={poUsePath})
      return outPath
    else:
      let (output, code) = execCmdEx(CompressorBin & " c " & quoteShell(dbPath) & " " & quoteShell(outPath))
      if code == 0 and fileExists(outPath):
        return outPath
      return ""
  except CatchableError:
    return ""

proc decompressDB*(compressedPath: string; outputPath: string = ""): string =
  if not fileExists(compressedPath):
    return ""
  let outPath = if outputPath.len > 0: outputPath else: compressedPath.replace(".catcmp", "")
  try:
    let (output, code) = execCmdEx(CompressorBin & " d " & quoteShell(compressedPath) & " " & quoteShell(outPath))
    if code == 0 and fileExists(outPath):
      return outPath
    return ""
  except CatchableError:
    return ""
