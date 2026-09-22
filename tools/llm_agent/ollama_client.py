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
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # IMPORTANT: Ollama defaults num_ctx to 2048, which would silently
        # TRUNCATE our multi-thousand-token prompt (dropping tool/policy text so
        # the model never sees it). Set it large enough to hold the whole prompt.
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
        temperature: float = 0.2,
    ) -> dict:
        """Like chat() but returns {content, prompt_tokens, response_tokens,
        total_tokens} using Ollama's own token counts. If the model was given a
        num_ctx, we also include it so callers can compute % of context used.

        Retries ONCE if the model returns an empty completion (Ollama sometimes
        yields a blank message, especially with format=json); the retry nudges
        temperature up slightly to break the degenerate generation."""
        result = self._chat_once(system, user, force_json, temperature)
        if not (result.get("content") or "").strip():
            # One retry with a small temperature bump.
            result = self._chat_once(system, user, force_json,
                                     min(1.0, temperature + 0.3))
            result["retried"] = True
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
            "options": {"temperature": temperature, "num_ctx": self.num_ctx,
                        "num_predict": 300},
        }
        if force_json:
            payload["format"] = "json"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        pt = int(body.get("prompt_eval_count", 0) or 0)
        rt = int(body.get("eval_count", 0) or 0)
        return {
            "content": body.get("message", {}).get("content", ""),
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
