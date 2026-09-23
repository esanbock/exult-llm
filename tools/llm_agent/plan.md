# LLM Agent — Plan & Findings

## GOAL
Get a **locally-hosted LLM (Ollama, e.g. gpt-oss:20b on the RTX 5090)** to
**progress through Ultima VII: The Black Gate on its own** — perceiving, deciding,
and acting through the in-engine agent bridge with no human steering. Success is
the model making real, self-directed narrative progress (investigating, talking,
looting, resolving quests, advancing the plot), which validates that a general
agent framework (fair perception + durable memory + a real action set) lets a
capable model play. The framework must be model-agnostic; the differentiator
should be the model's planning, not hand-holding.

## GUIDING PRINCIPLES
1. **Make the LLM self-sufficient — do NOT steer it or edit its memory to move
   it along.** Give it facts and a good action surface; let it decide. The
   engine/driver INFORMS (surfaces state, flags contradictions); it does not
   puppeteer. Overriding an action is a last resort, only to break a genuine
   infinite stall, never to choose content for the model.
2. **General capabilities, not puzzle-specific hints.** Teach RPG player wisdom
   (explore, exhaust dialogue, keep useful items, level up) and expose real
   capabilities; never encode the solution to a specific puzzle.
3. **Fair, human-like perception.** Screen-radius grid, last-seen NPC positions,
   landmark memory — approximate what a human sees, without cheating (no live
   off-screen NPC positions, no oracle answers).
4. **Consequences, not prohibitions.** Explain outcomes (e.g. theft may anger
   guards) and let the model choose; don't hard-ban actions.
5. **The LLM owns its memory.** Quests, topics, and the plot summary are the
   model's to author and revise. We surface durable facts (already-searched,
   quests, places) so it can correct itself; we never rewrite its narrative.
6. **Only document capabilities that exist.** No prompt references to tools we
   haven't built (e.g. no spellcasting until a cast action exists).
7. **The map is for WALKABILITY + ROUTES; identity lives in the objects list.**
   Every map glyph must earn its place.

## Original validation approach
Drive the game directly (as a strong operator) to shake out capability bugs,
then let the local model attempt the same from a save — comparing where the
model gets stuck to distinguish framework gaps from model-planning limits.


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

## Real gaps found + fixed (methodical-search session)
- FLICKERING NPC PERCEPTION (the big one): observe() used gwin->get_nearby_npcs
  (proximity manager) which returned an INCONSISTENT NPC set turn-to-turn (8
  npcs incl 40+ tiles away, then 1, then 0 at the same spot). Made systematic
  NPC-finding nearly impossible. FIXED: find_nearby_actors within 24 tiles,
  nearest-first. Verified stable across 5 consecutive observations.
- POOR GOTO PATHFINDING through walls: goto no-op'd/"no path" when the exact
  target was unreachable. FIXED: on A* failure, try A* to intermediate waypoints
  along the line to the target (85/70/55/40/25/15%) with lateral nudges; go to
  first reachable (partial:true). Verified: avatar now crosses the walled town
  ~10 tiles/goto where it previously stuck. (Fair: mirrors a human seeing the
  2D map and walking to the nearest reachable spot.)

## Witness chain mapped (for the report villain clue)
- Petre: footprints "lead out the back way"; Spark is Christopher's son.
- Johnson (gate guard): has "Hook" topic ("a man with a hook").
- Spark: KEY WITNESS. "key" = "the key to Father's chest" (I hold the key ->
  there's an evidence CHEST to open). nightmare = "a big red-faced man". what he
  saw = "a man and a wingless gargoyle running... toward the dock". => next
  investigation site is the DOCK. The "hook" report answer likely unlocks from a
  dock witness / confronting the dock lead.
- STILL could not find Finnigan outdoors across a full stable-perception sweep
  (9 NPCs censused, no Finnigan) - he is inside a building interior. Reaching
  him needs entering the right building.

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

## Future capability
- SPELLCASTING: no cast action exists yet. Ultima VII has a spellbook + reagents. Add a 'cast' action (select spell, consume reagents) so the RPG-wisdom 'use magic' guidance becomes real. Removed the spell prompt line for now to avoid pointing the agent at a non-existent tool.


## Session progress — local-model run + framework hardening
Focus shifted to actually running the **local model (gpt-oss:20b)** and fixing
what blocked its self-sufficient play. Changes (all on branch `llm-agent`):

- **Model comparison.** gemma4 = locally coherent but weak strategic planning
  (looped on the empty body, 0 quests resolved). gpt-oss:20b = substantially
  better (worked the report chain, resolved ~20 quests, recovered from
  setbacks). Confirms the framework is model-agnostic; planning is the
  differentiator.
- **gpt-oss empty-reply fix.** As a reasoning model, gpt-oss emits a separate
  `thinking` channel; on dead-ends it over-thought and consumed the old 768
  `num_predict` budget before emitting JSON → empty reply → parse-fail. FIX:
  num_predict 768→2048. (Do NOT combine `think:false` with `format:json` for
  gpt-oss — it triggers a token-repeat abort / HTTP 500.) Parse-fails → 0.
- **Tool-error fixes.** `look` was 100% error (a guard emitted a look action
  AFTER the look handler already ran → fell through to the engine as "unknown
  action"; now the guard produces the description directly). `open` errored
  repeatedly on the town gate (a portcullis, not a door); now, after a repeat,
  the agent is told the gate needs the password, not `open`.
- **search now actually LOOTS.** Previously `search` only opened the gump,
  leaving gold/bread/torch sitting inside; now it transfers takeable contents
  into inventory and reports `looted`/`empty`/`count`.
- **Lootable body vs non-lootable corpse.** Engine distinguishes via
  `as_container()`: `body:true` (lootable, e.g. the slain gargoyle) vs
  `corpse:true` (a corpse with nothing to take, e.g. the ritual-murder victim).
  Surfaced as `corpse_not_lootable`; searching a bare corpse is short-circuited
  to an examine so the model stops looping on it.
- **Map redesign (walkability-first).** Dropped identity-only glyphs
  (tree/wall/furniture/sign/obstacle → all just `#`), split body into `b`
  (lootable) / `x` (corpse), added `E` (exit/route: gate, stairs, ladder). 14
  glyphs, all navigational. Fixed a latent bug: `~` water was in the walkable
  flood-fill set. Legend is a one-time prompt block; the map is 1 char/tile, so
  richer glyphs cost ~0 recurring — the tradeoff is signal, not space.
- **Coordinate model documented once.** One consistent frame: absolute (tx,ty),
  per-object relative (dx,dy) so object abs = (tx+dx, ty+dy), and an @-centered
  map. `goto` takes absolute tiles. No coordinate range on the map (@ is the
  anchor).

## Self-sufficiency correction (important)
Mid-session we (wrongly) added a regex that rewrote the LLM's plot summary to
scrub a stale "search the body" objective, and hand-edited live memory once.
That violated Principle 1/5 and over-shrank the summary. **Reverted.** Now:
- We never edit the LLM's narrative. `mark_searched_empty` only records facts.
- The emptied-body handling is **inform-first**: surface the fact ("that spot is
  already empty; rewrite your plot_summary if it still lists it — your call"),
  let the model decide; override to explore ONLY after 4 persistent repeats,
  purely to prevent an infinite stall.
- The periodic plot nudge is **advisory + contradiction-aware**: if the summary
  frames a searched-empty body as a goal, tell the model to rewrite ITS summary.

## Persistence note
`story_so_far` (episodic_summary) persists across restarts (saved in and loaded
from agent_memory.json, same durable tier as quests/NPCs/topics). A CLEAN RESET
must clear agent_memory.json together with gamedat, or a stale summary leaks
into a fresh game.

## Correct launch (all three flags matter)
`Exult.exe --bg --nomenu --llmagent` from the repo root (working dir must resolve
`./blackgate`): `--bg` selects Black Gate, `--nomenu` skips the Journey-Onward
menu (auto-loads the latest save), **`--llmagent` starts the TCP bridge on
127.0.0.1:45999** (this flag was the cause of a "bridge won't connect" detour).

## NEXT
- Keep observing gpt-oss playing self-sufficiently; watch whether it reaches
  Finnigan and completes the report → password → leave-Trinsic chain on its own.
- Only add framework capability where the model is blocked by a genuine gap
  (not by its own planning). Candidate future capability: spellcasting (cast
  action + reagents) — still not implemented.
- If testing self-sufficiency from scratch, do a full clean reset.



## Long observation run (gpt-oss:20b, self-sufficiency hardening)
Ran a multi-hour autonomous session with ~15-min check-ins, fixing driver
issues live. Key work and findings:

- MONITORING: added monitor.py (non-intrusive; reads agent_memory.json +
  driver_log.txt) that snapshots turns/parse-fails/distinct-positions/quests and
  flags likely loops. Used it as the check-in tool.

- LOOP CAUSES were mostly MEMORY SALIENCE, not mechanics:
  * Guard-generated filler ("looked around", waits) FLOODED recent_actions,
    burying the meaningful outcomes ("search bag -> EMPTY") that break loops.
    Fix: record_action dedupes consecutive duplicates + collapses filler, so
    real outcomes stay visible. This was the single biggest anti-loop win - the
    agent had literally been unable to see what it just did.
  * already_searched_empty is now surfaced IMPERATIVELY
    (already_searched_empty_DO_NOT_RETURN + an explicit note).
  * The agent CONFLATES 'search X' with goto: it writes "search bag for key" but
    emits a goto. Added an inform-only nudge when it gotos while reasoning
    'search' next to a container ("arriving isn't searching; emit search").
  * HALLUCINATED GOALS: the agent's own plot summary invented a "gate key" /
    "key in the bag" (leaving Trinsic needs the PASSWORD, not a key). Per the
    self-sufficiency principle we do NOT edit its summary; we surface turn
    number + same_goal_streak + progress_note + a review nudge so it can notice
    and drop the dead lead itself.

- CAPABILITIES ADDED THIS RUN:
  * search now LOOTS container/body contents (was only opening the gump).
  * lootable body ('b') vs non-lootable corpse ('x') via as_container().
  * read action: read signs/plaques (a human double-clicks them). Signs are
    modal (usecode Get_click); display_runes stashes the rune-translated text
    into a global, and read pre-injects a click to dismiss the modal, returning
    the text. Verified: "THe / honorable / hound", no hang.

- MAP FIDELITY: grid now matches the human-visible tile window (get_width/height
  / c_tilesize) as a landscape rectangle (e.g. 39x25), not a 25x25 square;
  glyphs trimmed to navigational-only (b/x/n/*/E/~/=/+//./#). Driver grid math
  derives center from the actual grid size.

- GUI: combined reasoning+action into one Turn log (last 12 turns); labelled
  stats grid; finished-quests list; dedicated tool-calls table (tool|calls|ok|
  err|err%, red error rows); "Show context" button (full prompt in a window).
  tool_stats now RESET per run (were cumulative across runs).

- COORDINATE MODEL documented once (absolute tx/ty, relative dx/dy, @-centered
  map, goto takes absolute tiles) + a 'turn' counter for time-progression
  awareness.

## Honest model-behavior assessment (gpt-oss:20b)
The framework informs clearly (searched-empty, streaks, conflation, review
prompts) WITHOUT steering or editing memory. gpt-oss reaches NPCs (talked to
Finnigan), reviews its quest log, maintains a plot summary, resolves ~50 quests,
and explores 100+ distinct tiles per run. Its remaining weaknesses are PLANNING,
not framework gaps: it (a) hallucinates sub-goals into its own summary and
clings to them, (b) conflates 'search' with 'goto', and (c) re-opens an
already-open door instead of walking through. These persist despite clear
in-context signals - i.e. they are model-planning limits. This validates the
thesis: the framework is model-agnostic and sufficient; a stronger planner
should progress further on the same surface. Next candidate model tests on the
5090: qwen3.6, deepseek-r1:32b.
