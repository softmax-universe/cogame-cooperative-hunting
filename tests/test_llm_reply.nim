import cooperative_hunting/prompts
## Reply handling: tolerant extraction, the enum coercions, the rune caps,
## the single retry on an illegal target, and -- the load-bearing one -- the
## no-credentials path making ZERO network calls.

import std/[json, strutils, unicode]
import cooperative_hunting/[sim, llm]

var failures = 0
proc check(label: string, condition: bool) =
  if not condition:
    echo "FAIL: ", label
    inc failures
  else:
    echo "ok: ", label

const Legal = @["stag@13,17", "rabbit@18,21", "none"]
const Aliases = @["Cog-A", "Cog-B", "Cog-C", "Cog-D", "Cog-E", "Cog-F"]

proc parseText(text: string): Plan =
  parsePlan(extractJsonObject(text), 7, Legal, Aliases)

block cleanJson:
  let plan = parseText("""{"intent":"hunt","target":"stag@13,17","side":"S",
    "with":["Cog-A"],"say":"north side","note":"pair with A"}""")
  check("a clean reply parses", plan.target == "stag@13,17")
  check("the intent survives", plan.intent == "hunt")
  check("the side survives", plan.side == "S")
  check("the ally list survives", plan.partners == @["Cog-A"])
  check("say survives", plan.say == "north side")
  check("note survives", plan.note == "pair with A")
  check("src is llm", plan.src == "llm")

block markdownFence:
  let plan = parseText(
    "```json\n{\"intent\":\"flee\",\"target\":\"none\"}\n```")
  check("standard outer markdown json fence is accepted", plan.intent == "flee")
  let bareFence = parseText(
    "```\n{\"intent\":\"rest\",\"target\":\"none\"}\n```")
  check("standard outer bare markdown fence is accepted", bareFence.intent == "rest")

block proseRejected:
  var trailingRaised = false
  try:
    discard parseText(
      """{"intent":"assist","target":"none","side":"N"} -- I will help Cog-A.""")
  except CatchableError:
    trailingRaised = true
  check("trailing prose outside JSON is rejected", trailingRaised)

  var leadingRaised = false
  try:
    discard parseText(
      "Here is my plan:\n```json\n{\"intent\":\"flee\",\"target\":\"none\"}\n```")
  except CatchableError:
    leadingRaised = true
  check("leading prose before markdown fence is rejected", leadingRaised)

block nestedBraces:
  let plan = parseText(
    """{"intent":"hunt","target":"none","note":"a } inside a string"}""")
  check("a brace inside a string does not end the object",
    plan.note == "a } inside a string")

block optionalDefaultsAndRequiredFields:
  let plan = parseText("""{"intent":"rest","target":"none"}""")
  check("an omitted side defaults to any", plan.side == "any")
  check("an omitted with defaults to empty", plan.partners.len == 0)
  check("an omitted say defaults to empty", plan.say.len == 0)
  check("an omitted note defaults to empty", plan.note.len == 0)

  var missingIntent = false
  try:
    discard parseText("""{"target":"none"}""")
  except CatchableError:
    missingIntent = true
  check("a missing intent is rejected", missingIntent)

  var missingTarget = false
  try:
    discard parseText("""{"intent":"hunt"}""")
  except CatchableError:
    missingTarget = true
  check("a missing target is rejected", missingTarget)

block invalidEnumsRejected:
  var badIntent = false
  try:
    discard parseText("""{"intent":"ponder","target":"none","side":"any"}""")
  except CatchableError:
    badIntent = true
  check("an unknown intent is rejected (no silent coercion)", badIntent)

  var badSide = false
  try:
    discard parseText("""{"intent":"hunt","target":"none","side":"northwest"}""")
  except CatchableError:
    badSide = true
  check("an unknown side is rejected (no silent coercion)", badSide)

block allyValidation:
  var badAlly = false
  try:
    discard parseText(
      """{"intent":"hunt","target":"none","with":["Cog-A","nobody"]}""")
  except CatchableError:
    badAlly = true
  check("non-alias entries in with are rejected", badAlly)

  var tooManyAllies = false
  try:
    discard parseText(
      """{"intent":"hunt","target":"none","with":["Cog-A","Cog-B","Cog-C","Cog-D"]}""")
  except CatchableError:
    tooManyAllies = true
  check("more than three allies in with is rejected", tooManyAllies)

block illegalTarget:
  var raised = false
  try:
    discard parseText("""{"intent":"hunt","target":"dragon@1,1"}""")
  except CatchableError as error:
    raised = "LEGAL TARGETS" in error.msg
  check("a target outside LEGAL TARGETS is illegal", raised)

block retryContract:
  ## An unusable reply is retried EXACTLY once, with a hint that names the
  ## JSON contract and LEGAL TARGETS; after that the turn is missed.
  check("the retry hint demands ONE JSON object",
    "ONE JSON object" in RetryHint)
  check("the retry hint names LEGAL TARGETS",
    "LEGAL TARGETS" in RetryHint)
  check("every failure cause exists",
    (block:
      var seen: seq[string] = @[]
      for cause in FallbackCause:
        seen.add($cause)
      "timeout" in seen and "transport" in seen and "auth_failed" in seen and
        "parse" in seen and "illegal_target" in seen and
        "invalid_action" in seen and "rate_budget" in seen and
        "no_credentials" in seen))
  check("transport failures are classified separately from reply rejections",
    isTransportFailure(fcNoCredentials) and isTransportFailure(fcAuth) and
    isTransportFailure(fcTransport) and isTransportFailure(fcTimeout) and
    not isTransportFailure(fcParse) and not isTransportFailure(fcIllegalTarget) and
    not isTransportFailure(fcInvalidAction))

block unparseable:
  var raised = false
  try:
    discard parseText("I would like to hunt the stag, please.")
  except CatchableError:
    raised = true
  check("a reply with no balanced object is rejected", raised)

block runeTruncation:
  ## Over-long free text is cut on a RUNE boundary and marked, never on a
  ## byte boundary.
  let longSay = repeat("\u{1F98C}", 400)
  let longNote = repeat("\u00e9", 500)
  let plan = parseText($(%*{
    "intent": "hunt", "target": "none", "say": longSay, "note": longNote
  }))
  check("say is capped at 120 runes", plan.say.runeLen <= MaxSayRunes)
  check("note is capped at 200 runes", plan.note.runeLen <= MaxNoteRunes)
  check("the truncated say is still valid UTF-8",
    validateUtf8(plan.say) == -1)
  check("the truncated note is still valid UTF-8",
    validateUtf8(plan.note) == -1)
  check("the truncation is marked", plan.say.endsWith("\u2026"))

block everyCapIsARuneCap:
  ## Over-long invalid enum/target values are rejected with valid UTF-8 errors.
  for payload in [
    $(%*{"intent": repeat("\u{1F98C}", 40), "target": "none"}),
    $(%*{"intent": "hunt", "side": repeat("\u{1F98C}", 8), "target": "none"}),
    $(%*{"intent": "hunt", "with": [repeat("\u00e9", 60)], "target": "none"}),
    $(%*{"intent": "hunt", "target": repeat("\u{1F98C}", 200)})
  ]:
    var raised = false
    var message = ""
    try:
      discard parseText(payload)
    except CatchableError as error:
      raised = true
      message = error.msg
    check("over-long invalid field is rejected with valid UTF-8",
      raised and validateUtf8(message) == -1)

block controlCharacters:
  let plan = parseText($(%*{
    "intent": "hunt", "target": "none", "say": "line one\nline two\ttabbed"
  }))
  check("control characters are collapsed to spaces",
    '\n' notin plan.say and '\t' notin plan.say)

block noCredentialsMakesNoCalls:
  ## The load-bearing fallback: with no credentials the client marks itself
  ## disabled at startup and makes NO network calls at all, so offline
  ## certification still completes.
  let client = newDisabledLlmClient()
  check("a client with no credentials is disabled", client.disabled)
  var requests: seq[PlanRequest] = @[]
  var legal: seq[seq[string]] = @[]
  for slot in 0 ..< 6:
    requests.add(PlanRequest(slot: slot, system: "s", user: "u"))
    legal.add(Legal)
  let replies = client.runPlanBatch(requests, legal, Aliases, 1)
  check("every seat comes back", replies.len == 6)
  check("no request was issued", client.totalRequests == 0)
  check("every seat reports the no_credentials cause",
    (block:
      var ok = true
      for reply in replies:
        if reply.ok or reply.cause != fcNoCredentials: ok = false
      ok))

block rateBudget:
  ## The hard rolling budget is 28 requests per 60 s, retries included.
  let client = newDisabledLlmClient()
  check("a fresh budget admits a full six-seat turn",
    client.budgetAvailable(6))
  check("the budget refuses more than 28 at once",
    not client.budgetAvailable(RateBudgetRequests + 1))

block observationIsBounded:
  ## The observation is deterministic and bounded; every list has a cap.
  var config = defaultGameConfig()
  config.numAgents = 6
  var sim = initSim(config)
  for slot in 0 ..< 6:
    discard sim.addPlayer("p" & $slot, aliasForSlot(slot), slot)
  sim.ensureStats(6)
  for _ in 0 ..< 400:
    sim.step(newSeq[InputState](6))
  let prompt = repeat("strategy ", 400)
  let observation = sim.observationFor(0, 7, 25, Plan(), "a note", @[], prompt)
  check("the observation is deterministic",
    observation == sim.observationFor(0, 7, 25, Plan(), "a note", @[], prompt))
  check("the observation names the seat's own alias",
    sim.players[0].alias in observation)
  check("the observation ships LEGAL TARGETS",
    "LEGAL TARGETS:" in observation)
  check("the observation always offers `none` as a legal target",
    "none" in sim.legalTargets(0))
  check("the strategy block is rune-capped",
    observation.runeLen < 4000)
  check("the whole observation is inside its 2000-rune bound",
    observation.runeLen <= MaxObservationRunes)
  check("the observation still carries the reply contract after the cap",
    "LEGAL TARGETS:" in observation and "BLOCKED TILES" in observation and
    "YOUR NOTE:" in observation)
  let systemPrompt = systemPromptFor(prompt, PromptSelection(), "staghunt")
  check("the system prompt carries a concrete example",
    "Example:" in systemPrompt)
  check("the system prompt carries the schema",
    "\"intent\"" in systemPrompt)

if failures > 0:
  quit("test_llm_reply: " & $failures & " failures", 1)
echo "test_llm_reply: all checks passed"
