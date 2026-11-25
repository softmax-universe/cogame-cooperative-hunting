## Public GamePlayerFailure schema checks.

import std/[json, os, strutils, unicode]
import cooperative_hunting
import cooperative_hunting/sim_types

var failures = 0
proc check(label: string, condition: bool) =
  if not condition:
    echo "FAIL: ", label
    inc failures
  else:
    echo "ok: ", label

let dir = getTempDir() / "ch-player-failure-test"
removeDir(dir)
createDir(dir)
let path = dir / "player_failure.json"

block playerErrorEnvelope:
  putEnv(CogamePlayerFailureUriEnv, "file://" & path)
  writePlayerFailure("player_error", 2, "Player returned a malformed plan")
  check("player_failure.json written", fileExists(path))
  let payload = parseJson(readFile(path))
  check("exact three keys", payload.kind == JObject and payload.len == 3)
  check("kind is player_error", payload["kind"].getStr() == "player_error")
  check("failed_policy_index is actual slot",
    payload["failed_policy_index"].getInt() == 2)
  check("message matches",
    payload["message"].getStr() == "Player returned a malformed plan")

block inconclusiveEnvelopeAndBounds:
  removeFile(path)
  putEnv(CogamePlayerFailureUriEnv, "file://" & path)
  let huge = repeat("\u{1F98C}", 400) & "\nsecret"
  writePlayerFailure("unknown_kind", -3, huge)
  let payload = parseJson(readFile(path))
  check("exact three keys on fallback kind", payload.len == 3)
  check("non-player_error maps to episode_inconclusive",
    payload["kind"].getStr() == "episode_inconclusive")
  check("failed_policy_index is nonnegative",
    payload["failed_policy_index"].getInt() == 0)
  let msg = payload["message"].getStr()
  check("message is rune-bounded and valid UTF-8",
    msg.runeLen > 0 and msg.runeLen <= MaxErrorRunes and
    validateUtf8(msg) == -1 and '\n' notin msg)

delEnv(CogamePlayerFailureUriEnv)
removeDir(dir)

if failures > 0:
  quit("test_player_failure: " & $failures & " failures", 1)
echo "test_player_failure: all checks passed"
