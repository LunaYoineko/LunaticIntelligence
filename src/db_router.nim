import os, strutils, tables
import types, storage, intent_classifier, code_structure

type
  DbRole* = enum
    drChat = "chat"
    drNim = "nim"
    drCode = "code"
    drGeneral = "general"
    drCustom = "custom"

  DbMeta* = object
    role*: string
    description*: string
    tags*: seq[string]
    concepts*: int
    catalogSize*: int
    createdAt*: string
    sourceCorpus*: string

  DbRouter* = object
    basePath*: string
    baseDir*: string
    roleToPath*: Table[DbRole, string]
    customPathToMeta*: Table[string, DbMeta]
    roleToStorage*: Table[DbRole, StorageDB]
    loadedRoles*: Table[DbRole, bool]
    defaultRole*: DbRole
    lastSelectedRole*: DbRole

proc dbPathForRole*(baseDir: string; role: DbRole): string =
  case role
  of drChat: baseDir / "lunatic_chat.db"
  of drNim: baseDir / "lunatic_nim.db"
  of drCode: baseDir / "lunatic_code.db"
  of drGeneral: baseDir / "lunatic_general.db"
  of drCustom: baseDir / "lunatic_custom.db"

proc parseRoleFromMeta*(meta: Table[string,string]): DbRole =
  let r = meta.getOrDefault("role", "general").toLower()
  case r
  of "chat": drChat
  of "nim": drNim
  of "code": drCode
  of "general": drGeneral
  else: drCustom

proc scanDbFolder*(router: var DbRouter; dir: string) =
  if not dirExists(dir): return
  for kind, path in walkDir(dir):
    if kind == pcDir:
      # knowledgeフォルダを再帰的にスキャン（極限までDBを知識として蓄積）
      if path.endsWith("knowledge") or path.contains("knowledge"):
        router.scanDbFolder(path)
      continue
    if kind != pcFile: continue
    if not (path.endsWith(".db") or path.endsWith(".sqlite")): continue
    if path.endsWith("-shm") or path.endsWith("-wal") or path.endsWith("-journal"): continue
    try:
      var sdb = openStorage(path)
      let metaTbl = sdb.loadDbMeta()
      sdb.close()
      let roleStr = metaTbl.getOrDefault("role", "")
      let desc = metaTbl.getOrDefault("description", "")
      let tagsStr = metaTbl.getOrDefault("tags", "")
      let conceptsStr = metaTbl.getOrDefault("concepts", "0")
      let catalogStr = metaTbl.getOrDefault("catalog", "0")
      let role = if roleStr.len>0: parseRoleFromMeta(metaTbl) else: drCustom
      if role == drCustom or not router.roleToPath.hasKey(role) or not fileExists(router.roleToPath[role]):
        let m = DbMeta(role: $role, description: desc, tags: tagsStr.split(","), concepts: parseInt(conceptsStr), catalogSize: parseInt(catalogStr), createdAt: metaTbl.getOrDefault("createdAt",""), sourceCorpus: metaTbl.getOrDefault("sourceCorpus",""))
        router.customPathToMeta[path] = m
        if role != drCustom and not fileExists(router.roleToPath.getOrDefault(role,"")):
          router.roleToPath[role] = path
      else:
        let m = DbMeta(role: $role, description: desc, tags: tagsStr.split(","), concepts: parseInt(conceptsStr), catalogSize: parseInt(catalogStr), createdAt: metaTbl.getOrDefault("createdAt",""), sourceCorpus: metaTbl.getOrDefault("sourceCorpus",""))
        router.customPathToMeta[path] = m
    except CatchableError: discard

proc initRouter*(basePath: string): DbRouter =
  result.basePath = basePath
  result.roleToPath = initTable[DbRole, string]()
  result.customPathToMeta = initTable[string, DbMeta]()
  result.roleToStorage = initTable[DbRole, StorageDB]()
  result.loadedRoles = initTable[DbRole, bool]()
  result.defaultRole = drGeneral
  result.lastSelectedRole = drGeneral
  let baseDir = if dirExists(basePath): basePath else: parentDir(basePath)
  let effectiveDir = if baseDir.len == 0: "." else: baseDir
  result.baseDir = effectiveDir
  for role in [drChat, drNim, drCode, drGeneral, drCustom]:
    let p = if role == drCustom: "" else: dbPathForRole(effectiveDir, role)
    result.roleToPath[role] = p
    result.loadedRoles[role] = false
  if fileExists(basePath):
    result.roleToPath[drGeneral] = basePath
  # フォルダ内全DBをスキャンしてメタデータ保持（入れるだけで使われる）
  result.scanDbFolder(effectiveDir)
  # knowledgeフォルダは5T知識の集積地として常にスキャン
  let knowledgeDir = effectiveDir / "knowledge"
  if dirExists(knowledgeDir) and knowledgeDir != effectiveDir:
    result.scanDbFolder(knowledgeDir)
  # カレントのknowledgeも
  if dirExists("knowledge"):
    result.scanDbFolder("knowledge")

proc ensureLoaded*(router: var DbRouter; role: DbRole) =
  if router.loadedRoles.getOrDefault(role, false): return
  let path = router.roleToPath.getOrDefault(role, "")
  if path.len == 0 or not fileExists(path): return
  try:
    let sdb = openStorage(path)
    router.roleToStorage[role] = sdb
    router.loadedRoles[role] = true
  except CatchableError:
    discard

proc selectRoleForInput*(input: string; classifier: var IntentClassifier): DbRole =
  let lower = input.toLower()
  if isCodeInput(input):
    if lower.contains("nim") or lower.contains("proc ") or lower.contains("import ") or lower.contains("echo "):
      return drNim
    return drCode
  let (intent, _) = classifier.classifyIntent(input)
  if countRunes(input) < 10:
    return drChat
  case intent
  of iiGreeting, iiThanks, iiFarewell, iiOpinion, iiStatement:
    return drChat
  of iiRequest:
    if lower.contains("nim") or lower.contains("コード") or lower.contains("プログラム"):
      return drNim
    return drChat
  of iiQuestion:
    if lower.contains("nim") or lower.contains("コード"):
      return drNim
    return drGeneral
  else:
    return drGeneral

proc selectDbByMeta*(router: var DbRouter; input: string; classifier: var IntentClassifier): tuple[role: DbRole, path: string, meta: DbMeta] =
  # メタデータのtags/descriptionでスコアリングして最適DBを選択（LIが判断）
  # ハードコードせずタグ/説明/ロールで動的に判断
  let lower = input.toLower()
  var bestScore = -1
  var bestPath = ""
  var bestMeta = DbMeta()
  let originalRole = selectRoleForInput(input, classifier)
  var bestRoleForReturn = originalRole
  for path, meta in router.customPathToMeta.pairs:
    if not fileExists(path): continue
    var score = 0
    for tag in meta.tags:
      let t = tag.strip().toLower()
      if t.len>0 and lower.contains(t): score += 10
    if meta.description.toLower().contains(lower.split(" ")[0]): score += 3
    if meta.role.toLower() == $originalRole: score += 5
    if score > bestScore:
      bestScore = score
      bestPath = path
      bestMeta = meta
      var tmp = initTable[string,string]()
      tmp["role"] = meta.role
      bestRoleForReturn = parseRoleFromMeta(tmp)
  if bestScore >= 5 and bestPath.len>0:
    return (bestRoleForReturn, bestPath, bestMeta)
  # フォールバックはロールベース（元の意図を尊重）
  return (originalRole, router.roleToPath.getOrDefault(originalRole,""), DbMeta())

proc getStorageForInput*(router: var DbRouter; input: string; classifier: var IntentClassifier): tuple[role: DbRole, path: string, exists: bool] =
  # まずメタデータで判断
  let metaSel = router.selectDbByMeta(input, classifier)
  if metaSel.path.len>0 and fileExists(metaSel.path):
    router.lastSelectedRole = metaSel.role
    router.ensureLoaded(metaSel.role)
    # customの場合はそのパスを直接使う
    if metaSel.role == drCustom:
      return (drCustom, metaSel.path, true)
    return (metaSel.role, metaSel.path, true)
  let role = selectRoleForInput(input, classifier)
  router.lastSelectedRole = role
  router.ensureLoaded(role)
  let p = router.roleToPath.getOrDefault(role, "")
  let ex = fileExists(p)
  if not ex:
    let gpath = router.roleToPath.getOrDefault(drGeneral, "")
    if fileExists(gpath):
      return (drGeneral, gpath, true)
    # customフォルダから任意のDBをフォールバック
    for path in router.customPathToMeta.keys:
      if fileExists(path): return (drCustom, path, true)
    return (role, p, false)
  return (role, p, ex)

proc getAllAvailableRoles*(router: DbRouter): seq[DbRole] =
  for role, path in router.roleToPath.pairs:
    if fileExists(path):
      result.add(role)
  for path in router.customPathToMeta.keys:
    var already = false
    for _, rp in router.roleToPath.pairs:
      if rp == path: already = true
    if fileExists(path) and not already:
      if drCustom notin result: result.add(drCustom)

proc closeAll*(router: var DbRouter) =
  for role, sdb in router.roleToStorage.pairs:
    try: sdb.close() except: discard
  router.roleToStorage.clear()
  router.loadedRoles.clear()
