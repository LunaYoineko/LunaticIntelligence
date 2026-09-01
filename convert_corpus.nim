import strutils, sequtils, tables, os

proc toYamiStyle(response: string): string =
  ## 応答をYami風に変換
  var result = response
  
  # 丁寧語を削除/変換
  result = result.replace("です。", "...")
  result = result.replace("ます。", "...")
  result = result.replace("である。", "...")
  result = result.replace("である", "...")
  result = result.replace("ます", "...")
  result = result.replace("です", "...")
  result = result.replace("でしょうか", "...?")
  result = result.replace("ですか", "?...")
  result = result.replace("ですか", "?...")
  result = result.replace("ますか", "?...")
  result = result.replace("でしょう", "...")
  result = result.replace("だろう", "...")
  result = result.replace("ですね", "ね...")
  result = result.replace("ですね", "ね...")
  result = result.replace("ですよ", "よ...")
  result = result.replace("ですよね", "ね...")
  result = result.replace("ますね", "ね...")
  result = result.replace("ますよ", "よ...")
  
  # 説明調を短く
  result = result.replace("について説明します。", "について...")
  result = result.replace("についてお答えします。", "について...")
  result = result.replace("以下のようになります。", "...")
  result = result.replace("というものです。", "...")
  result = result.replace("というものです", "...")
  result = result.replace("とされています。", "...")
  result = result.replace("とされています", "...")
  result = result.replace("と言われています。", "...")
  result = result.replace("と言われています", "...")
  result = result.replace("という意味です。", "...")
  result = result.replace("という意味です", "...")
  result = result.replace("を意味します。", "...")
  result = result.replace("を意味します", "...")
  result = result.replace("を指します。", "...")
  result = result.replace("を指します", "...")
  result = result.replace("と呼びます。", "...")
  result = result.replace("と呼びます", "...")
  result = result.replace("と呼ばれています。", "...")
  result = result.replace("と呼ばれています", "...")
  
  # 長い文章を短く
  let patterns = [
    ("について教えて", "について..."),
    ("とは何", "とは..."),
    ("とは", "とは..."),
    ("について", "について..."),
    ("を教えて", "..."),
    ("を知りたい", "..."),
    ("を解説", "..."),
  ]
  
  for (frm, to) in patterns:
    result = result.replace(frm, to)
  
  # 語尾を...に
  if result.endsWith("。"):
    result = result[0..^2] & "..."
  if result.endsWith("."):
    result = result[0..^1] & "..."
  if result.endsWith("!") or result.endsWith("！"):
    result = result[0..^1] & "..."
  if result.endsWith("?") or result.endsWith("？"):
    result = result[0..^1] & "?..."
  
  # 空の場合
  if result.len == 0 or result.strip() == "":
    result = "..."
  
  # 既に...で終わってる場合はそのまま
  if not result.endsWith("..."):
    result = result & "..."
  
  return result

proc main() =
  let inputPath = "corpus_download/large_corpus.txt"
  let outputPath = "corpus_download/large_corpus_yami.txt"
  
  if not fileExists(inputPath):
    echo "入力ファイルが見つかりません: " & inputPath
    quit(1)
  
  var count = 0
  var outFile = open(outputPath, fmWrite)
  
  for line in inputPath.lines:
    count += 1
    let parts = line.split("|", 1)
    if parts.len >= 2:
      let input = parts[0].strip()
      let response = parts[1].strip()
      let yamiResponse = toYamiStyle(response)
      outFile.writeLine(input & "|" & yamiResponse)
    
    if count mod 100000 == 0:
      echo "処理済み: " & $count & " 行"
  
  outFile.close()
  echo "完了: " & $count & " 行処理しました"
  echo "出力: " & outputPath

main()