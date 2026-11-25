## The LLM half of a champion policy.
##
## Transport ladder: Coworld sidecar (`COWORLD_LLM_ENDPOINT`), else Bedrock
## (`AWS_ENDPOINT_URL_BEDROCK_RUNTIME` / `AWS_BEARER_TOKEN_BEDROCK`), else
## `ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY_URI`.
##
## Decisions in this game are SIMULTANEOUS, so every prompt seat's request
## goes out in the same batch. The batch has a 12 s deadline and the sim
## never waits on it: hunters keep executing their previous plan while it is
## in flight, so the LLM contributes 0 s to the critical path.
##
## Missing credentials, auth failures, or transport/network failures end the
## episode with an explicit failure rather than substituting a scripted bot or
## fabricating normal results. Malformed or illegal model replies are retried
## once and then recorded as a missed turn with no synthesized action.

import std/[algorithm, json, monotimes, options, os, sequtils, strutils, times, unicode]
import curly
import bitworld/cogame_runtime
import ./sim_types
import ./sim
import ./prompts
import ./[decision_trajectory, native_evidence]

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"

  ## Rate discipline. Six seats per 15 s planning turn is 24 req/min against
  ## the sidecar's 30 req/min-per-episode cap; this rolling budget is the
  ## hard backstop that keeps a retry storm underneath it.
  RateBudgetRequests* = 28
  RateBudgetSeconds* = 60.0

  LegalIntents* = ["hunt", "assist", "forage", "rest", "regroup", "flee"]
  LegalSides* = ["N", "S", "E", "W", "any"]

  RetryHint* = "Your last reply was rejected. Reply with ONE JSON object " &
    "(optionally inside a ```json ... ``` fence) with a valid intent and a target from LEGAL TARGETS."

  ResponseContract = """Reply with a single JSON object (optionally inside a ```json ... ``` fence) and NOTHING else.
Schema:
{"intent":"hunt|assist|forage|rest|regroup|flee","target":"<one id from LEGAL TARGETS, or none>",
 "side":"N|S|E|W|any","with":["<ally alias>",...],"say":"<=120 chars","note":"<=200 chars"}
Example:
{"intent":"hunt","target":"rabbit@14,12","side":"any","with":[],"say":"Taking rabbit","note":"solo"}
"side" defaults to "any", "with" defaults to [], and "say"/"note" default to "" if omitted.
"say" is broadcast to spectators. "note" is private and is handed back to you next turn."""

type
  FallbackCause* = enum
    fcNone = ""
    fcTimeout = "timeout"
    fcTransport = "transport"
    fcAuth = "auth_failed"
    fcParse = "parse"
    fcIllegalTarget = "illegal_target"
    fcInvalidAction = "invalid_action"
    fcRateBudget = "rate_budget"
    fcNoCredentials = "no_credentials"

  Plan* = object
    turn*: int
    intent*: string
    target*: string
    side*: string
    partners*: seq[string]
    say*: string
    note*: string
    src*: string          ## "llm" or "rejected:<cause>"

  PlanRequest* = object
    slot*: int
    system*: string
    user*: string
    policy*: string

  PlanReply* = object
    slot*: int
    plan*: Plan
    ok*: bool
    cause*: FallbackCause
    error*: string
    attempts*: seq[DecisionAttempt]

  LlmTransport* = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    curl: Curly
    transport*: LlmTransport
    apiKey: string
    sidecarEndpoint: string
    bedrockEndpoint: string
    bedrockModels: seq[string]
    bedrockModel: int
    bedrockToken: string
    model*: string
    maxOutputTokens*: int
    timeoutSeconds*: int
    disabled*: bool
    requestTimes: seq[float]   ## monotonic seconds, for the rolling budget
    totalRequests*: int

  ChError* = object of CatchableError

# ---------------------------------------------------------------------------
# Credentials and transport
# ---------------------------------------------------------------------------

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    let path = pathFromCogameUri(uri, "ANTHROPIC_API_KEY_URI")
    if path.len > 0 and fileExists(path):
      result = readFile(path).strip()
  except CatchableError as error:
    echo "cooperative-hunting llm: failed to fetch ANTHROPIC_API_KEY_URI: ",
      error.msg
    result = ""

proc bedrockModelIds(): seq[string] =
  ## Haiku leads: hosted Bedrock capacity is shared account-wide and the
  ## sonnet profiles run out of daily tokens first. `us.anthropic.
  ## claude-sonnet-4-6` is deliberately NOT a candidate -- it times out on
  ## every sidecar call and one throttle cascades into scripted fallbacks
  ## (raid round 2, 2026-08-23).
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return @[pinned]
  @[
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
  ]

proc tryNextBedrockModel(client: LlmClient, why: string): bool =
  if client.transport != ltBedrock or
      client.bedrockModel + 1 >= client.bedrockModels.len:
    return false
  client.bedrockModel.inc
  echo "cooperative-hunting llm: ",
    client.bedrockModels[client.bedrockModel - 1],
    " unusable (", why, "); falling back to ",
    client.bedrockModels[client.bedrockModel]
  true

proc bedrockUrl(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" &
    client.bedrockModels[client.bedrockModel] & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.planTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION", getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModels = bedrockModelIds()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "cooperative-hunting llm: bedrock transport, model ",
      result.bedrockModels[result.bedrockModel]
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "cooperative-hunting llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "cooperative-hunting llm: no LLM credentials configured"

proc hasLlmCredentials*(client: LlmClient): bool =
  not client.isNil and not client.disabled and client.transport != ltNone

proc isTransportFailure*(cause: FallbackCause): bool =
  cause in {fcNoCredentials, fcAuth, fcTransport, fcTimeout, fcRateBudget}

proc newDisabledLlmClient*(): LlmClient =
  ## The offline client used by the tests and by any episode with no
  ## credentials. Makes zero network calls, by construction.
  LlmClient(transport: ltNone, disabled: true, maxOutputTokens: 900,
    timeoutSeconds: 12, model: "claude-haiku-4-5")

proc monoSeconds(): float =
  float(getMonoTime().ticks) / 1_000_000_000.0

proc budgetAvailable*(client: LlmClient, wanted: int): bool =
  ## The hard rolling budget: at most `RateBudgetRequests` in any
  ## `RateBudgetSeconds` window, retries included.
  let now = monoSeconds()
  var live: seq[float] = @[]
  for t in client.requestTimes:
    if now - t < RateBudgetSeconds:
      live.add(t)
  client.requestTimes = live
  live.len + wanted <= RateBudgetRequests

proc noteRequests(client: LlmClient, count: int) =
  let now = monoSeconds()
  for _ in 0 ..< count:
    client.requestTimes.add(now)
  client.totalRequests += count

# ---------------------------------------------------------------------------
# Observation: composed from EXACTLY the per-seat visibility predicate, so a
# prompt seat can never see further than a scripted seat.
# ---------------------------------------------------------------------------

proc targetIdFor*(kind: PreyKind, x, y: int): string =
  preyLabel(kind) & "@" & $x & "," & $y

proc targetIdForItem*(item: Item): string =
  (case item.kind
   of itIron: "iron"
   of itGold: "gold"
   of itFood: "food" & $item.level) & "@" & $item.tileX & "," & $item.tileY

proc targetIdForForager*(player: Player): string =
  player.alias & "@" & $player.tileX & "," & $player.tileY

proc legalTargets*(sim: SimServer, seatIndex: int): seq[string] =
  ## Precomputed by the SAME predicate the validator applies: shipping the
  ## legal set in the observation is what actually fixes formal-output
  ## fallbacks (escrow 2026-08-23); prompt drills alone only halve them.
  if seatIndex < 0 or seatIndex >= sim.players.len:
    return @["none"]
  for i in 0 ..< sim.prey.len:
    if sim.preyVisibleToSeat(seatIndex, i):
      result.add(targetIdFor(sim.prey[i].kind, sim.prey[i].tileX,
        sim.prey[i].tileY))
  for i in 0 ..< sim.items.len:
    if sim.itemVisibleToSeat(seatIndex, i):
      result.add(targetIdForItem(sim.items[i]))
  if sim.isPredatorPrey():
    if sim.players[seatIndex].role == roleHunter:
      for i in 0 ..< sim.players.len:
        if i == seatIndex or sim.players[i].role != roleForager:
          continue
        if sim.visibleToSeat(seatIndex, i):
          result.add(targetIdForForager(sim.players[i]))
    else:
      let (cameraX, cameraY) = playerCamera(sim.players[seatIndex])
      for b in sim.berries:
        if b.regrow == 0 and tileInViewport(cameraX, cameraY, b.tileX, b.tileY):
          result.add("berries@" & $b.tileX & "," & $b.tileY)
          if result.len >= 12:
            break
  if result.len > 12:
    result.setLen(12)
  result.add("none")

proc sidesTakenText(sim: SimServer, stamps: array[4, SideStamp]): string =
  let sides = sim.occupiedSides(stamps)
  var taken: seq[string] = @[]
  for side in 0 ..< 4:
    if sides[side]:
      taken.add(SideNames[side])
  if taken.len == 0: "-" else: taken.join("")

proc coalitionText(kind: PreyKind): string =
  case kind
  of Rabbit: "needs 1"
  of Boar: "needs 2 (perpendicular sides)"
  of Stag: "needs 2 (opposite sides)"
  of Moose: "needs 3 (any sides)"
  of Elephant: "needs 4 (all sides)"

proc observationFor*(
  sim: SimServer,
  seatIndex: int,
  turn, turns: int,
  lastPlan: Plan,
  note: string,
  recent: seq[string],
  prompt: string
): string =
  ## Deterministic, bounded, and built from the per-seat visibility set.
  if seatIndex < 0 or seatIndex >= sim.players.len:
    return ""
  let me = sim.players[seatIndex]
  let roundNo = sim.roundIndex + 1
  result.add("TURN " & $turn & "/" & $turns & "  ROUND " & $roundNo & "/" &
    $sim.config.rounds & "  TICK " & $sim.tickCount & "/" &
    $sim.config.ticksPerRound & "  VARIANT " & sim.config.variant & "\n")
  result.add("YOU " & me.alias & " at (" & $me.tileX & "," & $me.tileY &
    ") facing " & (case me.facing
                   of FaceUp: "N"
                   of FaceDown: "S"
                   of FaceLeft: "W"
                   of FaceRight: "E") &
    " energy " & $me.energy & "/" & $MaxEnergy & " score " & $me.score &
    " level " & $me.level & " role " &
    (if me.role == roleHunter: "hunter" else: "forager") & "\n")

  result.add("PARTY VISIBLE (<=5 lines)\n")
  var partyLines = 0
  for i in 0 ..< sim.players.len:
    if i == seatIndex or partyLines >= 5:
      continue
    if not sim.visibleToSeat(seatIndex, i):
      continue
    result.add("  " & sim.players[i].alias & " (" & $sim.players[i].tileX &
      "," & $sim.players[i].tileY & ") d=" &
      $chebyshevDistance(me.tileX, me.tileY, sim.players[i].tileX,
        sim.players[i].tileY) & "\n")
    inc partyLines
  if partyLines == 0:
    result.add("  (nobody in sight)\n")

  result.add("ANIMALS VISIBLE (<=12 lines, nearest first)\n")
  var lines: seq[tuple[dist: int, text: string]] = @[]
  for i in 0 ..< sim.prey.len:
    if not sim.preyVisibleToSeat(seatIndex, i):
      continue
    let d = chebyshevDistance(me.tileX, me.tileY, sim.prey[i].tileX,
      sim.prey[i].tileY)
    lines.add((d, "  " & targetIdFor(sim.prey[i].kind, sim.prey[i].tileX,
      sim.prey[i].tileY) & " " & coalitionText(sim.prey[i].kind) &
      " sides taken: " & sim.sidesTakenText(sim.prey[i].sideSeen) &
      " d=" & $d & " worth " & $rewardsFor(sim.prey[i].kind).score))
  for i in 0 ..< sim.items.len:
    if not sim.itemVisibleToSeat(seatIndex, i):
      continue
    let d = chebyshevDistance(me.tileX, me.tileY, sim.items[i].tileX,
      sim.items[i].tileY)
    let need =
      case sim.items[i].kind
      of itIron: "needs 1"
      of itGold: "needs 2 (any sides, within 3 ticks)"
      of itFood: "needs adjacent levels summing to " & $sim.items[i].level
    let worth =
      case sim.items[i].kind
      of itIron: IronScoreReward
      of itGold: GoldScoreReward
      of itFood: LbfScorePerLevel * sim.items[i].level
    lines.add((d, "  " & targetIdForItem(sim.items[i]) & " " & need &
      " sides taken: " & sim.sidesTakenText(sim.items[i].sideSeen) &
      " d=" & $d & " worth " & $worth))
  if sim.isPredatorPrey():
    if me.role == roleHunter:
      for i in 0 ..< sim.players.len:
        if i == seatIndex or sim.players[i].role != roleForager:
          continue
        if not sim.visibleToSeat(seatIndex, i):
          continue
        let d = chebyshevDistance(me.tileX, me.tileY, sim.players[i].tileX,
          sim.players[i].tileY)
        lines.add((d, "  " & targetIdForForager(sim.players[i]) &
          " forager needs 2 (opposite sides) d=" & $d & " worth " &
          $TagScoreReward))
    else:
      let (cameraX, cameraY) = playerCamera(me)
      for b in sim.berries:
        if b.regrow != 0 or not tileInViewport(cameraX, cameraY, b.tileX,
            b.tileY):
          continue
        let d = chebyshevDistance(me.tileX, me.tileY, b.tileX, b.tileY)
        lines.add((d, "  berries@" & $b.tileX & "," & $b.tileY &
          " stand on it d=" & $d & " worth " & $BerryScoreReward))
  lines.sort(proc (a, b: tuple[dist: int, text: string]): int =
    cmp(a.dist, b.dist))
  if lines.len > 12:
    lines.setLen(12)
  if lines.len == 0:
    result.add("  (nothing in sight)\n")
  else:
    for line in lines:
      result.add(line.text & "\n")

  result.add("LEGAL TARGETS: " & legalTargets(sim, seatIndex).join(", ") & "\n")

  result.add("BLOCKED TILES NEAR YOU (<=40):")
  var blocked = 0
  for dy in -3 .. 3:
    for dx in -3 .. 3:
      if blocked >= 40:
        break
      let
        tx = me.tileX + dx
        ty = me.tileY + dy
      if sim.tileIsBlocked(tx, ty):
        result.add(" (" & $tx & "," & $ty & ")")
        inc blocked
  if blocked == 0:
    result.add(" none")
  result.add("\n")

  if lastPlan.intent.len > 0:
    result.add("LAST PLAN: intent=" & lastPlan.intent & " target=" &
      lastPlan.target & " side=" & lastPlan.side & "\n")
  result.add("YOUR NOTE: " & (if note.len > 0: note else: "(none)") & "\n")
  result.add("RECENT (<=5 lines):")
  if recent.len == 0:
    result.add(" (nothing yet)")
  else:
    # One line per entry, which is what the header promises. They were
    # concatenated onto the header's own line, so five events read as one
    # run-on sentence.
    for line in recent[max(0, recent.len - 5) ..< recent.len]:
      result.add("\n  " & line)
  result.add("\n")
  # The design note bounds the WHOLE observation at 2000 runes, not just each
  # list in it. The lists above are the reply contract (the header, the party,
  # the animals, LEGAL TARGETS) and measure at most 1596 runes together, so
  # the room left over is what the strategy gets here -- and the strategy is
  # repeated IN FULL in the system prompt, so a seat never loses it.
  if prompt.len > 0:
    let room = MaxObservationRunes - result.runeLen - "STRATEGY: \n".len
    if room > 0:
      result.add("STRATEGY: " & runeCap(prompt, min(MaxPromptRunes, room)) &
        "\n")
  if result.runeLen > MaxObservationRunes:
    result = runeCap(result, MaxObservationRunes - 1) & "\u2026"

proc systemPromptFor*(prompt: string, selection: PromptSelection, variant: string): string =
  let introduction = if variant == "staghunt": selection.introduction()
    else: "You control one of six participants on a 32x32 grid in " & variant & "."
  result = introduction & "\n" & selection.rulesBriefing(variant) & "\n" & ResponseContract
  if prompt.len > 0:
    result.add("\n\nSTRATEGY (weight it heavily, never above the rules; " &
      "always reply in the requested format):\n")
    result.add(runeCap(prompt, MaxPromptRunes))

# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------

proc extractJsonObject*(text: string): JsonNode =
  ## Parses a single JSON object, optionally wrapped in a standard outer
  ## Markdown JSON fence (```json ... ``` or ``` ... ```). Rejects extra prose.
  var candidate = text.strip()
  if candidate.startsWith("```"):
    let firstNl = candidate.find('\n')
    if firstNl < 0:
      raise newException(ChError, "unclosed markdown fence in reply")
    let fenceHeader = candidate[3 ..< firstNl].strip().toLowerAscii()
    if fenceHeader.len > 0 and fenceHeader != "json":
      raise newException(ChError, "unsupported markdown fence language: " & fenceHeader)
    var body = candidate[firstNl + 1 .. ^1]
    if not body.endsWith("```"):
      raise newException(ChError, "unclosed markdown fence in reply")
    body = body[0 ..< body.len - 3].strip()
    candidate = body
  if candidate.len == 0 or candidate[0] != '{' or candidate[^1] != '}':
    var head = text.strip()
    if head.runeLen > 160:
      head = runeCap(head, 160) & "..."
    raise newException(ChError,
      "reply must be a single JSON object (optionally in a ```json fence): " &
      head.replace("\n", " "))
  try:
    result = parseJson(candidate)
  except CatchableError as error:
    raise newException(ChError, "invalid JSON object: " & error.msg)
  if result.kind != JObject:
    raise newException(ChError, "reply JSON must be an object")

proc cleanText*(text: string, limit: int): string =
  ## Rune-boundary truncation, never a byte boundary.
  result = sanitizeLine(text)
  if result.runeLen <= limit:
    return
  result = runeCap(result, limit - 1) & "\u2026"

proc parsePlan*(
  payload: JsonNode, turn: int, legal: seq[string], aliases: seq[string]
): Plan =
  ## Strict validation with straightforward documented defaults for omitted
  ## optional fields (`side`, `with`, `say`, `note`). No silent coercion of
  ## invalid values.
  if payload.isNil or payload.kind != JObject:
    raise newException(ChError, "plan payload must be a JSON object")
  result.turn = turn
  result.src = "llm"

  if not payload.hasKey("intent") or payload["intent"].kind != JString:
    raise newException(ChError, "invalid action: missing or non-string intent")
  let intent = payload["intent"].getStr().strip()
  if intent notin LegalIntents:
    raise newException(ChError, "invalid action: unknown intent: " &
      runeCap(intent, MaxIntentChars))
  result.intent = intent

  if not payload.hasKey("target") or payload["target"].kind != JString:
    raise newException(ChError, "target not in LEGAL TARGETS: missing or non-string target")
  let target = payload["target"].getStr().strip()
  if target.len == 0 or target notin legal:
    raise newException(ChError, "target not in LEGAL TARGETS: " &
      runeCap(target, MaxTargetChars))
  result.target = target

  if payload.hasKey("side"):
    if payload["side"].kind != JString:
      raise newException(ChError, "invalid action: side must be a string")
    let side = payload["side"].getStr().strip()
    if side notin LegalSides:
      raise newException(ChError, "invalid action: unknown side: " &
        runeCap(side, MaxSideChars))
    result.side = side
  else:
    result.side = "any"

  if payload.hasKey("with"):
    let withNode = payload["with"]
    if withNode.kind != JArray:
      raise newException(ChError, "invalid action: with must be an array")
    if withNode.len > MaxWithItems:
      raise newException(ChError, "invalid action: with exceeds max items (" &
        $MaxWithItems & ")")
    for entry in withNode:
      if entry.kind != JString:
        raise newException(ChError, "invalid action: with entries must be strings")
      let name = entry.getStr().strip()
      if name notin aliases:
        raise newException(ChError, "invalid action: unknown ally in with: " &
          runeCap(name, MaxWithChars))
      if name notin result.partners:
        result.partners.add(name)

  if payload.hasKey("say"):
    if payload["say"].kind != JString:
      raise newException(ChError, "invalid action: say must be a string")
    result.say = cleanText(payload["say"].getStr(), MaxSayRunes)

  if payload.hasKey("note"):
    if payload["note"].kind != JString:
      raise newException(ChError, "invalid action: note must be a string")
    result.note = cleanText(payload["note"].getStr(), MaxNoteRunes)

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  headers["content-type"] = "application/json"
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## Only the Claude 5 / Opus tiers accept an effort setting; Haiku 4.5
    ## rejects the whole request with a 400 if it is present, so it is never
    ## sent (playbook Phase 1).
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc textOf(
  client: LlmClient, response: Response, error, url: string
): string =
  if error.len > 0:
    raise newException(ChError, "llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = response.body[0 .. min(response.body.high, 400)]
    if "Model access is denied" in response.body and
        client.tryNextBedrockModel("no model access"):
      raise newException(ChError, "bedrock model access denied: " & detail)
    client.disabled = true
    raise newException(ChError,
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    let detail = response.body[0 .. min(response.body.high, 300)]
    discard client.tryNextBedrockModel("throttled")
    raise newException(ChError, "llm throttled (429): " & detail)
  if response.code < 200 or response.code >= 300:
    raise newException(ChError, "anthropic error " & $response.code & ": " &
      response.body[0 .. min(response.body.high, 300)])
  let payload = parseJson(response.body)
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(ChError, "anthropic refusal")
  let content = payload{"content"}
  if not content.isNil and content.kind == JArray:
    for contentBlock in content:
      if contentBlock{"type"}.getStr() == "text":
        result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(ChError, "reply cut off at max_tokens before any " &
      "JSON: " & result[0 .. min(result.high, 160)].replace("\n", " "))

proc planActionJson*(plan: Plan): JsonNode =
  %*{"intent": plan.intent, "target": plan.target, "side": plan.side,
    "with": plan.partners, "say": plan.say, "note": plan.note}

proc runPlanBatch*(
  client: LlmClient,
  requests: seq[PlanRequest],
  legal: seq[seq[string]],
  aliases: seq[string],
  turn: int
): seq[PlanReply] =
  ## ONE parallel batch per planning turn, then at most ONE retry for the
  ## seats that failed. Never raises: every seat comes back with either a
  ## plan or a fallback cause.
  result = newSeq[PlanReply](requests.len)
  for i, request in requests:
    result[i] = PlanReply(slot: request.slot, ok: false, cause: fcNone)

  if requests.len == 0:
    return
  if client.isNil or client.disabled:
    for i in 0 ..< result.len:
      result[i].cause = fcNoCredentials
      result[i].error = "no LLM credentials configured"
    return
  if not client.budgetAvailable(requests.len):
    for i in 0 ..< result.len:
      result[i].cause = fcRateBudget
      result[i].error = "rate budget exceeded"
    return

  var open = toSeq(0 ..< requests.len)
  for attempt in 0 .. 1:
    if open.len == 0 or client.disabled:
      break
    if attempt > 0 and not client.budgetAvailable(open.len):
      for index in open:
        result[index].cause = fcRateBudget
      return
    var batch: RequestBatch
    var bodies: seq[string]
    for index in open:
      var user = requests[index].user
      if attempt > 0:
        user.add("\n" & RetryHint)
      let request = client.requestFor(requests[index].system, user, requests[index].slot)
      batch.post(request.url, request.headers, request.body, $index)
      bodies.add(request.body)
    client.noteRequests(open.len)
    let started = getMonoTime()
    let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
    let latency = int((getMonoTime() - started).inMilliseconds)
    var stillOpen: seq[int] = @[]
    for position, index in open:
      let capturing = getEnv(CogameSaveTrajectoryUriEnv).len > 0
      var evidence: DecisionAttempt
      if capturing:
        evidence = modelAttempt($turn & "-" & $requests[index].slot & "-" & $attempt,
          requests[index].policy,
          (if client.transport == ltBedrock: client.bedrockModels[client.bedrockModel] else: client.model),
          batch[position].url, bodies[position],
          responses[position].response, responses[position].error, latency)
      try:
        let text = client.textOf(responses[position].response,
          responses[position].error, batch[position].url)
        if capturing: evidence.response = %text
        result[index].plan = parsePlan(extractJsonObject(text), turn,
          legal[index], aliases)
        result[index].ok = true
        result[index].cause = fcNone
      except CatchableError as error:
        let message = runeCap(error.msg, MaxErrorRunes)
        if capturing: evidence.rejectionReason = some(message)
        result[index].error = message
        result[index].cause =
          if "LEGAL TARGETS" in message: fcIllegalTarget
          elif "invalid action:" in message: fcInvalidAction
          elif "llm auth failed" in message or "model access denied" in message: fcAuth
          elif "timed out" in message or "timeout" in message: fcTimeout
          elif "llm transport:" in message or "llm throttled" in message or
               "anthropic error " in message: fcTransport
          else: fcParse
        echo "cooperative-hunting llm: slot ", requests[index].slot,
          " attempt ", attempt, " failed: ", message
        stillOpen.add(index)
      if capturing: result[index].attempts.add(evidence)
    open = stillOpen
