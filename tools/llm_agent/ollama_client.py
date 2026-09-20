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
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

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
            "options": {"temperature": temperature},
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

    def is_up(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5.0):
                return True
        except Exception:
            return False
