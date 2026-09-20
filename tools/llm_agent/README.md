# Exult LLM Agent

Let a local LLM (via [Ollama](https://ollama.com)) observe and play
*Ultima VII: The Black Gate* through Exult.

## How it works

```
  Ollama (LLM)  <--HTTP-->  driver.py  <--TCP JSON lines-->  Exult (--llmagent)
```

* Exult is built with the `USE_LLM_AGENT` define. When launched with
  `--llmagent` it opens a TCP server on `127.0.0.1:45999`.
* The server speaks newline-delimited JSON:
  * `{"cmd":"observe"}` -> a JSON snapshot of the game state.
  * `{"cmd":"act","action":{...}}` -> executes an action, returns a result.
  * `{"cmd":"ping"}` -> `{"ok":true,"pong":true}`.
* Actions are injected into the engine as synthetic SDL input events, so the
  agent drives the game exactly like a human at the keyboard.

### Observation shape

```json
{
  "world_loaded": true,
  "player": {"name":"Avatar","tx":1234,"ty":2345,"tz":0,
             "hp":30,"str":20,"dex":20,"int":18,"mana":10,"food":30,"dead":false},
  "in_combat": false,
  "moving": false,
  "in_dungeon": false,
  "nearby": [{"name":"Iolo","tx":1236,"ty":2345,"dx":2,"dy":0,"dead":false}],
  "conversation_active": false,
  "answers": []
}
```

### Action types

| Action | JSON |
| --- | --- |
| Move | `{"type":"move","dir":"n|s|e|w|ne|nw|se|sw","speed":200}` |
| Stop | `{"type":"stop"}` |
| Key press | `{"type":"key","key":"space"}` (space, escape, a-z, 0-9) |
| Answer (index) | `{"type":"answer","index":0}` |
| Answer (text) | `{"type":"answer","text":"bye"}` |
| Wait | `{"type":"wait"}` |

## Running

1. **Build Exult** (Windows / VS2026), which defines `USE_LLM_AGENT`:
   ```powershell
   $env:VCPKG_ROOT="C:\vcpkg"
   & "C:\Program Files\Microsoft Visual Studio\18\Professional\MSBuild\Current\Bin\MSBuild.exe" `
       msvcstuff\vs2019\Exult.sln /t:Exult /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v145 /m
   ```

2. **Install game data** so Exult can run (see the repo `README.windows`):
   copy the `STATIC` folder from your Ultima VII install into a `blackgate`
   sub-folder next to `Exult.exe`.

3. **Launch Exult with the bridge** and load/start a game past the main menu:
   ```powershell
   .\Exult.exe --llmagent
   ```

4. **Verify the bridge** (no LLM needed):
   ```powershell
   python tools\llm_agent\test_bridge.py
   ```

5. **Let the LLM play** (Ollama must be running, e.g. `ollama run llama3.1`):
   ```powershell
   python tools\llm_agent\driver.py --model llama3.1 --steps 50
   ```
   Or drive it without an LLM using a scripted policy:
   ```powershell
   python tools\llm_agent\driver.py --dry-run --steps 20
   ```

## Files

* `exult_client.py` - client for the Exult TCP bridge.
* `ollama_client.py` - minimal Ollama HTTP client (stdlib only).
* `driver.py` - observe -> think -> act loop.
* `test_bridge.py` - standalone bridge verification (no Ollama).

## Notes

* The server binds to loopback only (`127.0.0.1`) and accepts one client.
* The protocol has no authentication; it is intended for local experimentation.
* `answer` actions require `conversation_active` to be true.
