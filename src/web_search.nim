import httpclient, strutils, json, os, sequtils, uri, unicode, tables

type
  SearchResult* = object
    title*: string
    snippet*: string
    url*: string

  WebSearcher* = object
    enabled*: bool
    cacheDir*: string
    cache*: seq[(string, seq[SearchResult])]
    translationCache*: Table[string, string]

proc initWebSearcher*(cacheDir: string = "/tmp/luna_cache"): WebSearcher =
  result.enabled = true
  result.cacheDir = cacheDir
  result.cache = @[]
  result.translationCache = initTable[string, string]()
  if not dirExists(cacheDir):
    createDir(cacheDir)

proc search*(ws: var WebSearcher; query: string; maxResults: int = 3): seq[SearchResult] =
  result = @[]

  # キャッシュ確認
  for (q, cached) in ws.cache:
    if q == query:
      return cached[0..<min(maxResults, cached.len)]

  # クエリ正規化: 日本語の助詞・質問詞を除去してキーワードだけにする
  var cleanQuery = query
  # 質問パターン除去
  let removePatterns = ["とは何", "是什么", "って何", "は何か", "について教えて",
                        "について知りたい", "についての情報",
                        "とは", "って", "ですか", "でしょうか", "はどう",
                        "の特徴", "の歴史", "の意味", "の仕組み", "の種類",
                        "はいい", "は良い", "を教えて", "を解説",
                        "か？", "か", "？", "?", "！", "!", "。", "、",
                        "について", "の話", "を知りたい", "を_search"]
  for p in removePatterns:
    cleanQuery = cleanQuery.replace(p, "")

  # 末尾の助詞除去（「は」「が」「を」「に」「で」「と」「も」）
  let trailingParticles = ["から", "まで", "より", "って", "じゃ", "けど"]
  while cleanQuery.len > 0:
    var removed = false
    for p in trailingParticles:
      if cleanQuery.endsWith(p) and cleanQuery.len > p.len:
        cleanQuery = cleanQuery[0..<(cleanQuery.len - p.len)]
        removed = true
        break
    if removed: continue
    let last = cleanQuery[^1]
    if last in "はがをにでとものや":
      cleanQuery = cleanQuery[0..^2]
    else:
      break
  cleanQuery = cleanQuery.strip()

  if cleanQuery.len == 0:
    cleanQuery = query

  # DuckDuckGo Instant Answer API + Wikipedia API fallback
  let client = newHttpClient(timeout = 10000)
  try:
    # 1. Wikipedia検索API（より関連性の高い結果）
    let wikiSearchUrl = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" & encodeUrl(cleanQuery) & "&format=json&srprop=snippet&srlimit=3"
    let wikiSearchResponse = client.getContent(wikiSearchUrl)
    let wikiSearchData = parseJson(wikiSearchResponse)

    if wikiSearchData.hasKey("query") and wikiSearchData["query"].hasKey("search"):
      let searchResults = wikiSearchData["query"]["search"]
      for searchResult in searchResults:
        if result.len >= maxResults: break
        if searchResult.hasKey("title"):
          let wikiTitle = searchResult["title"].getStr()
          let wikiTitleUrl = wikiTitle.replace(" ", "_")
          let wikiUrl = "https://en.wikipedia.org/api/rest_v1/page/summary/" & encodeUrl(wikiTitleUrl)
          let wikiResponse = client.getContent(wikiUrl)
          let wikiData = parseJson(wikiResponse)
          if wikiData.hasKey("extract"):
            let extract = wikiData["extract"].getStr()
            if extract.len > 0:
              var articleUrl = ""
              if wikiData.hasKey("content_urls") and wikiData["content_urls"].hasKey("desktop"):
                articleUrl = wikiData["content_urls"]["desktop"]["page"].getStr()
              result.add(SearchResult(
                title: wikiTitle,
                snippet: extract,
                url: articleUrl
              ))

    # 2. Wikipediaが空の場合、DuckDuckGo Instant Answer API
    if result.len == 0:
      var url = "https://api.duckduckgo.com/?q=" & encodeUrl(cleanQuery) & "&format=json&no_html=1&skip_disambig=1"
      var response = client.getContent(url)
      var data = parseJson(response)

      # Abstract（要約）
      if data.hasKey("Abstract") and data["Abstract"].getStr().len > 0:
        result.add(SearchResult(
          title: data["AbstractSource"].getStr(),
          snippet: data["Abstract"].getStr(),
          url: data["AbstractURL"].getStr()
        ))

      # Related Topics
      if data.hasKey("RelatedTopics"):
        for topic in data["RelatedTopics"]:
          if result.len >= maxResults: break
          if topic.hasKey("Text"):
            let text = topic["Text"].getStr()
            if text.len > 0 and text.len < 300:
              var topicUrl = ""
              if topic.hasKey("FirstURL"):
                topicUrl = topic["FirstURL"].getStr()
              result.add(SearchResult(
                title: query,
                snippet: text,
                url: topicUrl
              ))

  except CatchableError as e:
    # ネットワークエラー時は空を返す
    echo "Web search error: ", e.msg

  # キャッシュ保存
  if result.len > 0:
    ws.cache.add((query, result))
    if ws.cache.len > 100:
      ws.cache = ws.cache[90..^1]

proc getKnowledge*(ws: var WebSearcher; query: string): string =
  let results = ws.search(query, 3)
  if results.len == 0:
    return ""

  var knowledge = ""
  for r in results:
    if r.snippet.len > 0:
      knowledge.add(r.snippet & " ")

  return knowledge.strip()

proc isReachable*(): bool =
  let client = newHttpClient(timeout = 3000)
  try:
    let response = client.getContent("https://duckduckgo.com")
    return response.len > 0
  except CatchableError:
    return false

# ---------------------------------------------------------------------------
# 翻訳機能
# ---------------------------------------------------------------------------
proc detectLanguage*(text: string): string =
  var hasJapanese = false
  var hasLatin = false
  var hasCJK = false
  for rune in text.toRunes:
    let cp = rune.int32
    if (cp >= 0x3040 and cp <= 0x309F) or (cp >= 0x30A0 and cp <= 0x30FF):
      hasJapanese = true
    elif (cp >= 0x4E00 and cp <= 0x9FFF):
      hasCJK = true
    elif (cp >= 0x0041 and cp <= 0x005A) or (cp >= 0x0061 and cp <= 0x007A):
      hasLatin = true
  if hasJapanese: return "ja"
  if hasCJK: return "zh"
  if hasLatin: return "en"
  return "unknown"

proc translate*(ws: var WebSearcher; text: string; targetLang: string = "ja"): string =
  # キャッシュ確認
  let cacheKey = text & "->" & targetLang
  if ws.translationCache.hasKey(cacheKey):
    return ws.translationCache[cacheKey]

  # 長いテキストは最初の500文字のみ翻訳
  let textToTranslate = if text.len > 500: text[0..<500] else: text

  let client = newHttpClient(timeout = 10000)
  try:
    # MyMemory翻訳API（APIキー不要、無料枠あり）
    let sourceLang = detectLanguage(textToTranslate)
    let url = "https://api.mymemory.translated.net/get?q=" & encodeUrl(textToTranslate) & "&langpair=" & sourceLang & "|" & targetLang
    let response = client.getContent(url)
    let data = parseJson(response)

    if data.hasKey("responseData") and data["responseData"].hasKey("translatedText"):
      let translated = data["responseData"]["translatedText"].getStr()
      if translated.len > 0 and translated.toLower() != textToTranslate.toLower():
        ws.translationCache[cacheKey] = translated
        return translated

  except CatchableError:
    discard

  return ""

proc translateKnowledge*(ws: var WebSearcher; knowledge: string; targetLang: string = "ja"): string =
  let lang = detectLanguage(knowledge)
  if lang == targetLang:
    return knowledge
  return ws.translate(knowledge, targetLang)
