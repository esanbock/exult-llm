"""ollama_client.py - Minimal Ollama HTTP client (no third-party deps).

Talks to a local Ollama server (default http://127.0.0.1:11434) using the
/api/chat endpoint.  Uses only the Python standard library so the driver has
no pip requirements.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        num_ctx: int = 8192,
        allow_think: bool = False,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # When True, do NOT disable a reasoning model's thinking channel (useful
        # for TROUBLESHOOTING - the thinking shows WHY the model chose an action;
        # captured to raw_comms.log). Off by default for stability/speed (qwen's
        # verbose thinking can starve the JSON reply).
        self.allow_think = allow_think
        self.num_ctx = num_ctx

    def chat(
        self,
        system: str,
        user: str,
        *,
        force_json: bool = True,
        temperature: float = 0.2,
    ) -> str:
        """Send a system+user message, return the assistant's text content."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if force_json:
            # Ask Ollama to constrain output to a JSON object.
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("message", {}).get("content", "")

    def chat_ex(
        self,
        system: str,
        user: str,
        *,
        force_json: bool = True,
        temperature: float = 0.35,
    ) -> dict:
        """Like chat() but returns {content, prompt_tokens, response_tokens,
        total_tokens} using Ollama's own token counts. If the model was given a
        num_ctx, we also include it so callers can compute % of context used.

        Retries if the model returns an empty completion. Ollama's format=json
        grammar can occasionally yield a blank message for a given prompt state;
        the retries bump temperature AND drop the JSON grammar (plain text, from
        which the caller extracts the JSON object), which reliably breaks the
        empty-output state."""
        result = self._chat_once(system, user, force_json, temperature)
        if not (result.get("content") or "").strip():
            # Retry 1: higher temperature, still JSON-constrained.
            result = self._chat_once(system, user, force_json,
                                     min(1.0, temperature + 0.3))
            result["retried"] = True
        if not (result.get("content") or "").strip() and force_json:
            # Retry 2: drop the JSON grammar entirely (plain text). The caller's
            # parse_reply extracts the {...} from free text, so this still works
            # and escapes the degenerate empty-under-json-grammar state.
            result = self._chat_once(system, user, False,
                                     min(1.0, temperature + 0.5))
            result["retried"] = True
            result["dropped_json"] = True
        return result

    def _chat_once(self, system: str, user: str, force_json: bool,
                   temperature: float) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Reasoning models emit a separate chain-of-thought that consumes
            # response tokens BEFORE the JSON, which can starve the reply and
            # yield an empty completion. Two known-good strategies by model:
            #  - qwen3.x: DISABLE thinking (think=false). It works cleanly WITH
            #    format=json, removes ~6k chars of thinking/turn, and is faster.
            #  - gpt-oss: do NOT disable thinking with format=json (that combo
            #    triggers a repeat-loop abort); instead give num_predict headroom
            #    so thinking + JSON both fit.
            "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                        "num_predict": 4096 if self.allow_think else 2048},
        }
        _ml = (self.model or "").lower()
        # Thinking models emit a long chain-of-thought that consumes response
        # tokens (slow, and can starve the JSON). Unless the caller explicitly
        # asked for thinking (--think), turn it OFF so we get a fast, clean JSON
        # action. gpt-oss is the known EXCEPTION (think=false + format=json
        # triggers an empty-reply/repeat-loop abort), so leave it alone.
        _is_gptoss = "gpt-oss" in _ml or "gpt_oss" in _ml
        if not self.allow_think and not _is_gptoss:
            payload["think"] = False
        if force_json:
            payload["format"] = "json"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            # Surface Ollama's actual rejection reason (a 400 body explains WHY,
            # e.g. context length). Without this the driver's generic OSError
            # handler treats it as an engine-socket drop and spins forever.
            try:
                _b = he.read().decode("utf-8")[:300]
            except Exception:
                _b = "(no body)"
            raise RuntimeError(f"ollama {he.code}: {_b}") from he
        pt = int(body.get("prompt_eval_count", 0) or 0)
        rt = int(body.get("eval_count", 0) or 0)
        return {
            "content": body.get("message", {}).get("content", ""),
            "thinking": body.get("message", {}).get("thinking", "") or "",
            "prompt_tokens": pt,
            "response_tokens": rt,
            "total_tokens": pt + rt,
        }

    def context_size(self) -> int:
        """Query the model's context window (num_ctx) via /api/show. Returns 0
        if unknown."""
        try:
            data = json.dumps({"model": self.model}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/show", data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            # Look for context length in model_info (key varies by arch).
            info = body.get("model_info", {}) or {}
            for k, v in info.items():
                if k.endswith(".context_length") and isinstance(v, int):
                    return v
            params = body.get("parameters", "") or ""
            for line in params.splitlines():
                if line.strip().startswith("num_ctx"):
                    try:
                        return int(line.split()[-1])
                    except ValueError:
                        pass
        except Exception:
            pass
        return 0

    def is_up(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5.0):
                return True
        except Exception:
            return False
