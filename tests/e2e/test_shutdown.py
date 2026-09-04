from __future__ import annotations

import asyncio
import json
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import WSMsgType, web

from tests.e2e.conftest import KasaRig


def eventually[T](read: Callable[[], T], accept: Callable[[T], bool], *, timeout: float = 10) -> T:
    deadline = time.monotonic() + timeout
    last: T
    while True:
        last = read()
        if accept(last):
            return last
        if time.monotonic() >= deadline:
            raise AssertionError(f"condition was not met before deadline; last value: {last!r}")
        time.sleep(0.02)


@dataclass
class FakeSlack:
    loop: asyncio.AbstractEventLoop
    runner: web.AppRunner
    thread: threading.Thread
    api_url: str
    connected: threading.Event = field(default_factory=threading.Event)
    acknowledgements: list[str] = field(default_factory=list)
    posts: list[dict[str, str]] = field(default_factory=list)
    _socket: web.WebSocketResponse | None = None

    def event(self, *, event_id: str, text: str) -> None:
        payload = {
            "envelope_id": f"envelope-{event_id}",
            "type": "events_api",
            "accepts_response_payload": False,
            "payload": {
                "api_app_id": "A_E2E",
                "team_id": "T_E2E",
                "type": "event_callback",
                "event_id": event_id,
                "event_time": 1,
                "authorizations": [
                    {
                        "enterprise_id": None,
                        "team_id": "T_E2E",
                        "user_id": "U_BOT",
                        "is_bot": True,
                        "is_enterprise_install": False,
                    }
                ],
                "is_ext_shared_channel": False,
                "event": {
                    "type": "message",
                    "user": "U_USER",
                    "text": text,
                    "channel": "D_E2E",
                    "ts": f"1.{event_id[-3:]}",
                    "event_ts": f"1.{event_id[-3:]}",
                },
            },
        }

        async def send() -> None:
            assert self._socket is not None
            await self._socket.send_json(payload)

        self._submit(send()).result(timeout=5)

    def close(self) -> None:
        if self._socket is not None:
            self._submit(self._socket.close()).result(timeout=5)
        self._submit(self.runner.cleanup()).result(timeout=5)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    def _submit(self, coro: Any) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)


@pytest.fixture
def fake_slack() -> Iterator[FakeSlack]:
    loop = asyncio.new_event_loop()
    ready: Future[FakeSlack] = Future()

    async def start() -> None:
        app = web.Application()
        runner = web.AppRunner(app)
        slack_ref: dict[str, FakeSlack] = {}

        async def api(request: web.Request) -> web.Response:
            method = request.match_info["method"]
            if method == "auth.test":
                return web.json_response({"ok": True, "user_id": "U_BOT", "team_id": "T_E2E"})
            if method == "apps.connections.open":
                slack = slack_ref["server"]
                socket_url = slack.api_url.replace("http", "ws", 1) + "socket"
                return web.json_response({"ok": True, "url": socket_url})
            if method == "users.info":
                return web.json_response(
                    {"ok": True, "user": {"id": "U_USER", "profile": {"display_name": "QA User"}}}
                )
            if method == "chat.postMessage":
                payload = await request.json()
                slack_ref["server"].posts.append({str(k): str(v) for k, v in payload.items()})
                return web.json_response({"ok": True, "ts": "2.000"})
            return web.json_response({"ok": False, "error": f"unexpected_method:{method}"})

        async def socket(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            slack = slack_ref["server"]
            slack._socket = ws
            slack.connected.set()
            async for message in ws:
                if message.type is WSMsgType.TEXT:
                    ack = json.loads(message.data)
                    if envelope := ack.get("envelope_id"):
                        slack.acknowledgements.append(str(envelope))
            return ws

        app.router.add_post("/api/{method}", api)
        app.router.add_get("/api/socket", socket)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        slack = FakeSlack(
            loop=loop,
            runner=runner,
            thread=threading.current_thread(),
            api_url=f"http://127.0.0.1:{port}/api/",
        )
        slack_ref["server"] = slack
        ready.set_result(slack)

    def serve() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(start())
        loop.run_forever()
        loop.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    slack = ready.result(timeout=5)
    try:
        yield slack
    finally:
        slack.close()


def daemon(rig: KasaRig, slack: FakeSlack) -> subprocess.Popen[str]:
    if "[slack]" not in rig.config.read_text():
        with rig.config.open("a") as config:
            config.write(
                '\n[slack]\napp_token_env = "KASA_E2E_SLACK_APP"\n'
                'bot_token_env = "KASA_E2E_SLACK_BOT"\nstream = false\n'
                f'api_url = "{slack.api_url}"\n'
            )
    env = rig.env | {
        "KASA_E2E_SLACK_APP": "xapp-e2e",
        "KASA_E2E_SLACK_BOT": "xoxb-e2e",
    }
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "kasa.cli",
            "run",
            "--slack",
            "--config",
            str(rig.config),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def state(rig: KasaRig, event_id: str) -> str | None:
    if not rig.database.exists():
        return None
    connection = sqlite3.connect(rig.database)
    try:
        with connection:
            row = connection.execute(
                "SELECT state FROM inbox ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        connection.close()
    return str(row[0]) if row else None


def stop(process: subprocess.Popen[str]) -> tuple[str, str]:
    process.send_signal(signal.SIGTERM)
    return process.communicate(timeout=10)


def wait_for_provider(process: subprocess.Popen[str], rig: KasaRig) -> None:
    if rig.server.request_started.wait(timeout=5):
        return
    stdout, stderr = stop(process)
    raise AssertionError(f"provider was not called; daemon output:\n{stdout}\n{stderr}")


def test_sigterm_stops_an_idle_daemon_cleanly(kasa_rig: KasaRig, fake_slack: FakeSlack) -> None:
    process = daemon(kasa_rig, fake_slack)
    assert fake_slack.connected.wait(timeout=5)

    stdout, stderr = stop(process)

    assert process.returncode == 0, (stdout, stderr)
    assert process.poll() == 0


def test_sigterm_drains_an_active_turn_and_keeps_it_done(
    kasa_rig: KasaRig, fake_slack: FakeSlack
) -> None:
    process = daemon(kasa_rig, fake_slack)
    assert fake_slack.connected.wait(timeout=5)
    fake_slack.event(event_id="Ev101", text="Hold this turn.")
    wait_for_provider(process, kasa_rig)

    process.send_signal(signal.SIGTERM)
    kasa_rig.server.release_request.set()
    stdout, stderr = process.communicate(timeout=10)

    assert process.returncode == 0, (stdout, stderr)
    assert eventually(lambda: state(kasa_rig, "Ev101"), lambda value: value == "done") == "done"
    assert [post["text"] for post in fake_slack.posts] == ["E2E reply: Hold this turn."]


def test_failed_work_is_recoverable_after_a_sigterm_restart(
    kasa_rig: KasaRig, fake_slack: FakeSlack
) -> None:
    first = daemon(kasa_rig, fake_slack)
    assert fake_slack.connected.wait(timeout=5)
    fake_slack.event(event_id="Ev102", text="Hold this turn.")
    wait_for_provider(first, kasa_rig)

    first.send_signal(signal.SIGTERM)
    kasa_rig.server.fail_held_request = True
    kasa_rig.server.release_request.set()
    stdout, stderr = first.communicate(timeout=10)

    assert first.returncode == 0, (stdout, stderr)
    assert state(kasa_rig, "Ev102") == "pending"

    kasa_rig.server.fail_held_request = False
    fake_slack.connected.clear()
    second = daemon(kasa_rig, fake_slack)
    assert fake_slack.connected.wait(timeout=5)
    eventually(lambda: state(kasa_rig, "Ev102"), lambda value: value == "done")
    stdout, stderr = stop(second)

    assert second.returncode == 0, (stdout, stderr)
    assert state(kasa_rig, "Ev102") == "done"
