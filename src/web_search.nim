import httpclient, strutils, json, os, sequtils, uri, unicode, tables, times, asyncdispatch

# ASCII文字のみか判定
proc isAsciiOnly*(s: string): bool =
  for c in s:
    if c.ord >= 128:
      return false
  return true

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

proc getWithRetry*(client: HttpClient; url: string; maxRetries: int = 3): string =
  var retries = 0
  while retries <= maxRetries:
    try:
      client.headers = newHttpHeaders([("User-Agent", "LunaticIntelligence/1.0")])
      result = client.getContent(url)
      return result
    except HttpRequestError as e:
      if "429" in e.msg and retries < maxRetries:
        let waitMs = 1000 * (1 shl retries)
        sleep(waitMs)
        inc retries
        continue
      raise
    except CatchableError:
      if retries < maxRetries:
        sleep(500)
        inc retries
        continue
      raise

proc initWebSearcher*(cacheDir: string = "/tmp/lunatic_cache"): WebSearcher =
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
  # 質問パターン除去（ASCIIのみにならないよう注意）
  let removePatterns = ["とは何", "是什么", "って何", "は何か", "について教えて",
                        "について知りたい", "についての情報",
                        "って", "ですか", "でしょうか", "はどう",
                        "の特徴", "の歴史", "の意味", "の仕組み", "の種類",
                        "はいい", "は良い", "を教えて", "を解説",
                        "か？", "か", "？", "?", "！", "!", "。", "、",
                        "について", "の話", "を知りたい", "を_search"]
  for p in removePatterns:
    # "とは" は末尾にある場合は除去しない（"Pythonとは" -> "Python" になってしまうため）
    if p == "とは" and cleanQuery.endsWith("とは"):
      continue
    cleanQuery = cleanQuery.replace(p, "")

  # 末尾の助詞除去（「は」「が」「を」「に」「で」「と」「も」）
  # ただし、「とは」「というのは」などの疑問パターンの一部は除去しない
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
    # 「とは」「というのは」の「は」「と」は除去しない
    if cleanQuery.endsWith("とは") or cleanQuery.endsWith("というのは"):
      break
    if last in "はがをにでとものや":
      cleanQuery = cleanQuery[0..^2]
    else:
      break
  cleanQuery = cleanQuery.strip()

  if cleanQuery.len == 0:
    cleanQuery = query

  # DuckDuckGo Instant Answer API + Wikipedia API fallback
  let client = newHttpClient(timeout = 10000)
  client.headers = newHttpHeaders([("User-Agent", "LunaticIntelligence/1.0")])
  try:
    # 1. Wikipedia検索API（より関連性の高い結果） - 日本語Wikipediaを優先
    let wikiSearchUrl = "https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch=" & encodeUrl(cleanQuery) & "&format=json&srprop=snippet&srlimit=3&srinfo=totalhits&srnamespace=0&srwhat=text"
    let wikiSearchResponse = client.getWithRetry(wikiSearchUrl)
    let wikiSearchData = parseJson(wikiSearchResponse)

    if wikiSearchData.hasKey("query") and wikiSearchData["query"].hasKey("search"):
      let searchResults = wikiSearchData["query"]["search"]
      for searchResult in searchResults:
        if result.len >= maxResults: break
        if searchResult.hasKey("title"):
          let wikiTitle = searchResult["title"].getStr()
          # 検索結果のスニペットを優先使用（日本語の場合）
          let searchSnippet = if searchResult.hasKey("snippet"): searchResult["snippet"].getStr() else: ""
          # 日本語のスニペットがある場合はそれを優先使用
          if searchSnippet.len > 0 and not isAsciiOnly(searchSnippet):
            let wikiTitle = searchResult["title"].getStr()
            result.add(SearchResult(
              title: wikiTitle,
              snippet: searchSnippet,
              url: ""
            ))

    # 2. Wikipediaが空の場合、DuckDuckGo Instant Answer API（日本語対応）
    if result.len == 0:
      var url = "https://api.duckduckgo.com/?q=" & encodeUrl(cleanQuery) & "&format=json&no_html=1&skip_disambig=1&kl=jp-jp"
      var response = client.getWithRetry(url)
      var data = parseJson(response)

      # Abstract（要約）
      if data.hasKey("Abstract") and data["Abstract"].getStr().len > 0:
        let snippet = data["Abstract"].getStr()
        if not isAsciiOnly(snippet):
          result.add(SearchResult(
            title: data["AbstractSource"].getStr(),
            snippet: snippet,
            url: data["AbstractURL"].getStr()
          ))

      # Related Topics
      if data.hasKey("RelatedTopics"):
        for topic in data["RelatedTopics"]:
          if result.len >= maxResults: break
          if topic.hasKey("Text"):
            let text = topic["Text"].getStr()
            if text.len > 0 and text.len < 300 and not isAsciiOnly(text):
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

proc searchParallel*(ws: var WebSearcher; query: string; maxResults: int = 3): seq[SearchResult] =
  ## 並列検索: WikipediaとDuckDuckGoを同時に取得
  result = @[]
  # キャッシュ確認
  for (q, cached) in ws.cache:
    if q == query:
      return cached[0..<min(maxResults, cached.len)]

  # 並列取得のための非同期タスク
  proc fetchWiki(): Future[seq[SearchResult]] {.async.} =
    var wikiRes: seq[SearchResult] = @[]
    let client = newAsyncHttpClient()
    client.headers = newHttpHeaders({"User-Agent": "LunaticIntelligence/1.0"})
    try:
      var cleanQuery = query
      for p in ["とは何", "とは", "について教えて", "か？", "か", "？", "?", "！", "!", "。", "、"]:
        cleanQuery = cleanQuery.replace(p, "")
      cleanQuery = cleanQuery.strip()
      if cleanQuery.len == 0: cleanQuery = query
      let wikiSearchUrl = "https://ja.wikipedia.org/w/api.php?action=query&list=search&srsearch=" & encodeUrl(cleanQuery) & "&format=json&srprop=snippet&srlimit=3"
      let resp = await client.getContent(wikiSearchUrl)
      let data = parseJson(resp)
      if data.hasKey("query") and data["query"].hasKey("search"):
        for sr in data["query"]["search"]:
          if wikiRes.len >= maxResults: break
          if sr.hasKey("title"):
            let title = sr["title"].getStr()
            let url = "https://ja.wikipedia.org/api/rest_v1/page/summary/" & encodeUrl(title.replace(" ", "_"))
            try:
              let sumResp = await client.getContent(url)
              let sumData = parseJson(sumResp)
              if sumData.hasKey("extract"):
                let extract = sumData["extract"].getStr()
                # 日本語の記事のみを採用
                if extract.len > 0 and not isAsciiOnly(title):
                  wikiRes.add(SearchResult(title: title, snippet: extract, url: ""))
            except: discard
    except CatchableError as e:
      echo "Wiki parallel error: ", e.msg
    client.close()
    return wikiRes

  proc fetchDDG(): Future[seq[SearchResult]] {.async.} =
    var ddgRes: seq[SearchResult] = @[]
    let client = newAsyncHttpClient()
    client.headers = newHttpHeaders({"User-Agent": "LunaticIntelligence/1.0"})
    try:
      var cleanQuery = query
      for p in ["とは何", "とは", "について教えて", "か？", "か", "？", "?", "！", "!", "。", "、"]:
        cleanQuery = cleanQuery.replace(p, "")
      cleanQuery = cleanQuery.strip()
      if cleanQuery.len == 0: cleanQuery = query
      let url = "https://api.duckduckgo.com/?q=" & encodeUrl(cleanQuery) & "&format=json&no_html=1&skip_disambig=1&kl=jp-jp"
      let resp = await client.getContent(url)
      let data = parseJson(resp)
      if data.hasKey("Abstract") and data["Abstract"].getStr().len > 0:
        let snippet = data["Abstract"].getStr()
        if not isAsciiOnly(snippet):
          ddgRes.add(SearchResult(title: data["AbstractSource"].getStr(), snippet: snippet, url: data["AbstractURL"].getStr()))
      if data.hasKey("RelatedTopics"):
        for topic in data["RelatedTopics"]:
          if ddgRes.len >= maxResults: break
          if topic.hasKey("Text") and topic["Text"].getStr().len > 0:
            ddgRes.add(SearchResult(title: query, snippet: topic["Text"].getStr(), url: ""))
    except CatchableError as e:
      echo "DDG parallel error: ", e.msg
    client.close()
    return ddgRes

  # 並列実行
  let wikiFut = fetchWiki()
  let ddgFut = fetchDDG()
  # どちらかが返れば即時、両方待つ
  try:
    let wikiRes = waitFor wikiFut
    let ddgRes = waitFor ddgFut
    # Wikiを優先、マージ
    result.add(wikiRes)
    for r in ddgRes:
      if result.len >= maxResults: break
      # 重複除外
      var dup = false
      for ex in result:
        if ex.snippet == r.snippet: dup = true; break
      if not dup:
        result.add(r)
  except CatchableError as e:
    echo "Parallel search error: ", e.msg
    # フォールバックは逐次
    result = ws.search(query, maxResults)

  if result.len > 0:
    ws.cache.add((query, result))
    if ws.cache.len > 100:
      ws.cache = ws.cache[90..^1]

proc getKnowledge*(ws: var WebSearcher; query: string): string =
  let results = ws.searchParallel(query, 3)
  if results.len == 0:
    # フォールバックで逐次も試す
    let res2 = ws.search(query, 3)
    if res2.len == 0: return ""
    var knowledge = ""
    for r in res2:
      if r.snippet.len > 0: knowledge.add(r.snippet & " ")
    return knowledge.strip()

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
  client.headers = newHttpHeaders([("User-Agent", "LunaticIntelligence/1.0")])
  try:
    # MyMemory翻訳API（APIキー不要、無料枠あり）
    let sourceLang = detectLanguage(textToTranslate)
    let url = "https://api.mymemory.translated.net/get?q=" & encodeUrl(textToTranslate) & "&langpair=" & sourceLang & "|" & targetLang
    let response = client.getWithRetry(url)
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

# 天気情報取得
proc getWeather*(ws: var WebSearcher; location: string = "Tokyo"): string =
  ## OpenWeatherMap APIを使用して天気情報を取得
  ## APIキーは環境変数 OPENWEATHER_API_KEY から取得
  let apiKey = getEnv("OPENWEATHER_API_KEY", "")
  if apiKey == "":
    return "天気情報を取得するには OPENWEATHER_API_KEY 環境変数が必要です"
  
  try:
    let client = newHttpClient(timeout = 10000)
    let url = "https://api.openweathermap.org/data/2.5/weather?q=" & encodeUrl(location) & "&appid=" & apiKey & "&lang=ja"
    let response = client.getWithRetry(url)
    let data = parseJson(response)
    
    if data.hasKey("weather") and data["weather"].len > 0:
      let weather = data["weather"][0]
      let main = data["main"]
      let desc = weather["description"].getStr()
      let temp = main["temp"].getFloat() - 273.15  # Kelvin to Celsius
      let humidity = main["humidity"].getInt()
      return "現在の" & location & "の天気は " & desc & "、気温は " & $formatFloat(temp, ffDecimal, 1) & "°C、湿度は " & $humidity & "% です。"
    else:
      return "天気情報を取得できませんでした"
  except CatchableError:
    return "天気情報の取得中にエラーが発生しました"

# 簡単な計算
proc calculateExpression*(expr: string): string =
  ## 簡単な数式を評価（安全性のために制限付き）
  try:
    # 危険な文字を除外
    var safeExpr = ""
    for c in expr:
      if c in "0123456789+-*/(). ":
        safeExpr.add(c)
    
    # eval は危険なので、簡単なパーサーを使用
    # ここでは単純な四則演算のみサポート
    var result = 0.0
    var currentNum = 0.0
    var op = '+'
    var i = 0
    while i < safeExpr.len:
      if safeExpr[i] in '0'..'9':
        var numStr = ""
        while i < safeExpr.len and safeExpr[i] in '0'..'9':
          numStr.add(safeExpr[i])
          inc(i)
        currentNum = parseFloat(numStr)
        case op:
        of '+': result += currentNum
        of '-': result -= currentNum
        of '*': result *= currentNum
        of '/': 
           if currentNum != 0.0:
            result /= currentNum
           else:
            return "0で割ることはできません"
        else: discard
      elif safeExpr[i] in {'+', '-', '*', '/'}:
        op = safeExpr[i]
        inc(i)
      else:
        inc(i)
    return "結果: " & $formatFloat(result, ffDecimal, 2)
  except CatchableError:
    return "計算できませんでした"
