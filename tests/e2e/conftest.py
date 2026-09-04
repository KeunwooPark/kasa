from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _chunk(delta: dict[str, object], finish_reason: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl-e2e",
        "model": "e2e-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse(*chunks: dict[str, object]) -> bytes:
    events = [*(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks), "data: [DONE]\n\n"]
    return "".join(events).encode()


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    """A deterministic provider at the real HTTP boundary used by the CLI."""

    server: FakeOpenAIServer

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)

        messages = request["messages"]
        last_user = next(
            message["content"] for message in reversed(messages) if message["role"] == "user"
        )
        has_tool_result = any(message["role"] == "tool" for message in messages)

        if last_user == "Hold this turn.":
            self.server.request_started.set()
            if not self.server.release_request.wait(timeout=10):
                raise TimeoutError("test did not release the held provider request")
            if self.server.fail_held_request:
                body = b'{"error":{"message":"planned E2E failure"}}'
                self.send_response(503)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

        if last_user == "Use the clock tool." and not has_tool_result:
            body = _sse(
                _chunk(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_e2e_clock",
                                "type": "function",
                                "function": {"name": "current_time", "arguments": "{}"},
                            }
                        ],
                    }
                ),
                _chunk({}, "tool_calls"),
                {"model": "e2e-model", "choices": [], "usage": self._usage()},
            )
        else:
            answer = "Clock checked." if has_tool_result else f"E2E reply: {last_user}"
            body = _sse(
                _chunk({"role": "assistant", "content": answer}),
                _chunk({}, "stop"),
                {"model": "e2e-model", "choices": [], "usage": self._usage()},
            )

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _usage() -> dict[str, int]:
        return {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}

    def log_message(self, format: str, *args: object) -> None:
        # A failing assertion shows captured requests; HTTP access logs are noise.
        return


class FakeOpenAIServer(ThreadingHTTPServer):
    requests: list[dict[str, object]]
    request_started: threading.Event
    release_request: threading.Event
    fail_held_request: bool


@dataclass(frozen=True)
class KasaRig:
    config: Path
    database: Path
    env: dict[str, str]
    server: FakeOpenAIServer

    def command(
        self,
        *args: str,
        input: str | None = None,
        env: dict[str, str] | None = None,
        config: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "kasa.cli", *args, "--config", str(config or self.config)],
            input=input,
            text=True,
            capture_output=True,
            env=env or self.env,
            timeout=15,
            check=False,
        )

    def run(self, script: str) -> subprocess.CompletedProcess[str]:
        return self.command(
            "run",
            input=script,
        )


@pytest.fixture
def kasa_rig(tmp_path: Path) -> Iterator[KasaRig]:
    server = FakeOpenAIServer(("127.0.0.1", 0), FakeOpenAIHandler)
    server.requests = []
    server.request_started = threading.Event()
    server.release_request = threading.Event()
    server.fail_held_request = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    database = tmp_path / "kasa.db"
    config = tmp_path / "config.toml"
    host, port = server.server_address
    config.write_text(
        f'[store]\npath = "{database}"\n\n'
        '[llm.chat]\nkind = "openai"\nmodel = "e2e-model"\n'
        f'base_url = "http://{host}:{port}/v1"\nkey_env = "KASA_E2E_API_KEY"\n'
    )
    env = os.environ.copy()
    env["KASA_E2E_API_KEY"] = "not-a-real-secret"

    try:
        yield KasaRig(config=config, database=database, env=env, server=server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
