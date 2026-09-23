# LLM Agent — Plan & Findings

Goal: give an LLM enough perception, memory, and actions to autonomously play
Ultima VII: The Black Gate. Validate the capability surface by driving the game
directly (as a strong model) to find bugs, then compare against the Ollama model
(gemma) from a save.

## Current test objective
Complete the Trinsic murder → gate-password chain by hand to shake out bugs,
then let Ollama attempt the same from a save.

## The password dependency chain (verified)
1. Investigate the murder scene (stables): body, blood trail ("tracks lead out
   the back"), chest with gold + medallion + scroll, Gargoyle jewelry.
2. Gather the villain-description clue from witnesses. The report's villain
   answer options are DYNAMIC: "hook" only appears once the clue is gathered.
   Likely sources: the gate guard (Johnson), Spark. (Christopher = the victim,
   the dead blacksmith; Spark is his son and knows about the key/chest.)
3. Report to Finnigan (the Mayor): talk → "report" → "continue?" Yes →
   Q1 "what was in the chest?" = **all of these** (verified; medallion-only
   rejected) → Q2 "suspect?" = Yes → Q3 "what does the villain look like?" =
   **hook** (only available after the clue; scar/pegleg/eyepatch are decoys,
   pegleg verified WRONG).
4. Finnigan gives the town-gate password (gated behind a satisfactory report:
   "I will give thee the password when thou hast given me a report").

## Capability surface — status
Working (validated by direct play):
- Perception: 21-field observation (player stats, semantic ASCII grid, nearby
  NPCs w/ status+condition, objects w/ owned flag, doors, gump_contents,
  ambient speech, conversation state, number prompts).
- Movement: goto (+ engine single-step fallback when A* budget exceeded),
  wall-aware move, flood-fill wedge escape, gump-unblock.
- Actions: move, goto, talk, answer, key, search (redirects to look when nothing
  searchable), pickup (radius 6, no-retry on repeated fails), take, close,
  equip, inventory, feed, heal, combat, wait, save, annotate, look, recall,
  add_topic (inline), new_quest/resolve_quest (inline).
- Memory: quests (LLM-owned, priority-sorted, auto-resolve talk/examine quests,
  dedup), per-NPC dialogue trees (transcript + topics offered/asked/unasked),
  LLM-authored topic notebook, places map, hints (+ file channel), observations,
  episodic summary, tool-call stats. All persist in agent_memory.json.
- Off-screen NPC navigation: goto <name> routes to remembered last_pos.
- Sleeping-NPC awareness: condition:"sleeping" + guard blocks futile talks.
- Context budget: num_ctx 16384, squeeze tiers, dialogue folding; parse-fail ~0
  after action-first reply format + num_predict + JSON-grammar retry.

## Findings / bugs (this session)
- FIXED: search misuse (87% fail) → redirect to look when nothing searchable.
- FIXED: pickup retry loop (79% fail) → track fails, stop retrying, explain.
- FIXED: quests never resolved → auto-resolve talk/examine quests + dedup.
- FIXED: companion over-talk (Iolo x140) → split party from nearby + guard.
- FIXED: dialogue tree empty for goto-reached NPCs → infer partner from nearest.
- FIXED: sleeping NPCs → condition field + guard.
- NON-BUG CONTENT GATE: villain answer must be gathered in-game ("hook").

## OPEN GAP (next to implement)
(none currently blocking — see Resolved below)

## Resolved gaps
- TIME-OF-DAY + WAIT: observe() now reports hour/minute/day/time_of_day/is_night.
  New "wait_until" action advances the clock to a target hour AND calls
  gwin->schedule_npcs() so NPCs actually transition schedules (wake up).
  Verified: at hour 0 Johnson/witnesses asleep; wait_until 9 -> morning, Johnson
  awake and reachable. This unblocks the "hook" clue chain which needs awake
  witnesses. NOTE: setting the clock alone does NOT wake NPCs - must call
  schedule_npcs (learned this the hard way).

## Process notes
- Launch driver detached via tools/llm_agent/launch_driver.bat (returns
  immediately; grandchild console owns stdout/err so the shell doesn't hang).
- Save the game (save action) before killing Exult, else progress resets
  (--nomenu resumes gamedat only if IDENTITY exists and it wasn't overwritten).
- Re-apply audio config after Exult restarts (effects/speech enabled revert).
- Conversations need ~2-3s to open; don't rush space-presses.
- Exult occasionally exits on its own between long idle periods; check + relaunch.

## Session progress (driving by hand)
- Morning reached via wait_until; witnesses awake.
- Petre (stablehand): footprints "lead out the back way... tracks of the
  murderer"; Spark = Christopher's son (Christopher = the murdered blacksmith,
  a.k.a. the victim/Gilberto).
- Johnson (gate guard): has the "Hook" topic; "A man with a hook" - the clue
  that (should) unlock "hook" as the report villain-description answer.
- Report chain confirmed: chest = "all of these" (verified), suspect = Yes,
  villain = need "hook" (pegleg rejected). password gated behind a satisfactory
  report.

## Known friction (not a code bug)
- WANDERING NPCs: Finnigan/Johnson/Spark roam widely; goto-by-name only resolves
  VISIBLE npcs, and goto-to-last_pos often "no path" through the stable's
  internal walls. Reaching a specific wandering NPC costs many turns. talk()
  DOES work from a few tiles away without a walking path (finds visible npc),
  which helps. Possible future aid: a "path exists?" check or letting goto route
  to a remembered npc even when off-screen (driver already maps name->last_pos).
