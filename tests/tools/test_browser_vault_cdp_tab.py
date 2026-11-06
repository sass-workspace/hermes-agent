"""browser_vault on a browser_cdp tab when no browser_exec session exists (Iris fork).

Bots run with the ``browser-use`` toolset disabled, so their tasks have no browser_exec session and no
supervisor; they drive their own Chrome with browser_cdp (the profile's CDP endpoint). The vault then
works on exactly one tab: the one this task last drove through browser_cdp, else the only web page.
It never searches the other tabs for the saved origin, refuses when the tab is ambiguous, keeps the
origin binding and never returns the secret. With a browser_exec session the old path runs unchanged.

The browser is a fake CDP endpoint (``_cdp_call``): tabs with a URL and inspected controls, recording
every fill script as a write. Secrets are dummy values in a temporary local vault.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from unittest.mock import patch

import pytest

from agent.vault_backends.local import LocalLoginBackend
from agent.vault_store import VaultStore, normalize_origin

_ENDPOINT = "ws://127.0.0.1:9222/devtools/browser/fake"
_PASSWORD = "dummy-password-cdp-7"
_LOGIN_CONTROLS = [
    {"autocomplete": "username", "formIndex": 0, "index": 0, "label": "", "name": "user", "type": "text"},
    {"autocomplete": "current-password", "formIndex": 0, "index": 1, "label": "", "name": "pw", "type": "password"},
]
_OTP_CONTROLS = [{"autocomplete": "one-time-code", "formIndex": 0, "index": 0, "label": "Security code",
                  "name": "code", "type": "text"}]


class _Tab:
    def __init__(self, url: str, controls=()):
        self.url, self.controls, self.writes = url, list(controls), []


class _FakeChrome:
    """A browser behind one CDP endpoint. ``on_eval`` runs before every Runtime.evaluate."""

    def __init__(self, **tabs: _Tab):
        self.tabs = dict(tabs)
        self.calls: list[tuple[str, str | None]] = []
        self.on_eval = None
        self.raise_on_fill: Exception | None = None
        self.loggers: list = []
        self.calls_from_browser_cdp = False

    async def call(self, ws_url, method, params, target_id, timeout, ws_logger=None):
        assert ws_url == _ENDPOINT
        self.calls.append((method, target_id))
        if method in ("Target.getTargets", "Runtime.evaluate") and not self.calls_from_browser_cdp:
            self.loggers.append(ws_logger)
        if method == "Target.getTargets":
            return {"targetInfos": [{"targetId": tid, "type": "page", "url": tab.url} for tid, tab in self.tabs.items()]
                    + [{"targetId": "sw", "type": "service_worker", "url": "https://bank.example/sw.js"}]}
        if method == "Target.createTarget":
            tid = f"T{len(self.tabs) + 1}"
            self.tabs[tid] = _Tab(params.get("url") or "about:blank")
            return {"targetId": tid}
        if method == "Target.closeTarget":
            self.tabs.pop(params["targetId"], None)
            return {"success": True}
        if method == "Runtime.evaluate":
            if self.on_eval:
                self.on_eval()
            tab = self.tabs[target_id]
            expr = params["expression"]
            if expr.strip() in ("window.location.href", "location.href"):
                return {"result": {"type": "string", "value": tab.url}}
            if "expectedOrigin" in expr:  # the fill script (build_fill_js)
                if self.raise_on_fill:
                    raise self.raise_on_fill
                tab.writes.append(expr)
                return {"result": {"type": "string", "value": json.dumps({"filled": 1})}}
            return {"result": {"type": "string", "value": json.dumps(tab.controls)}}  # input inspection
        return {}

    def evaluated_targets(self):
        return [t for m, t in self.calls if m == "Runtime.evaluate"]

    def writes(self):
        return {tid: tab.writes for tid, tab in self.tabs.items() if tab.writes}


class _RecordingBackend(LocalLoginBackend):
    def __init__(self):
        self.resolved: list[str] = []

    def resolve_password(self, handle):
        self.resolved.append("password")
        return super().resolve_password(handle)


@pytest.fixture()
def vault(tmp_path):
    from agent import redact
    from agent.vault_backends import unlock as unlock_mod
    from tools import browser_cdp_tool

    store = VaultStore(base_dir=tmp_path / "vault")
    backend = _RecordingBackend()
    browser_cdp_tool._LAST_TARGETS.clear()
    try:
        with patch("agent.vault_store.get_vault_store", return_value=store), \
             patch("agent.vault_backends.backend_for_handle", return_value=backend), \
             patch("agent.vault_backends.unlock.can_prompt_here", return_value=False), \
             patch("tools.browser_session_cdp.get_cdp_override", return_value=_ENDPOINT):
            yield store, backend
    finally:
        unlock_mod.set_code_prompt_callback(None)
        redact.clear_vault_redaction_values()
        browser_cdp_tool._LAST_TARGETS.clear()


def _chrome(fake: _FakeChrome):
    return patch("tools.browser_cdp_tool._cdp_call", side_effect=fake.call)


def _use(tab_id: str, task_id: str = "t"):
    """The agent drives a tab with browser_cdp (the real tool, so the last-used record is real)."""
    from tools.browser_cdp_tool import browser_cdp
    with patch("tools.browser_cdp_tool._resolve_cdp_endpoint", return_value=_ENDPOINT):
        out = json.loads(browser_cdp("Runtime.evaluate", {"expression": "location.href"}, target_id=tab_id,
                                     task_id=task_id))
    assert out.get("success"), out


def _login(store, origin="https://bank.example", password=_PASSWORD):
    return store.add_item("login", "Bank", {"identifier_type": "email", "identifier": "bot@example.com",
                                            "password": password}, origin=origin)


def _fill(handle, task_id="t"):
    from tools import browser_vault_tool
    raw = browser_vault_tool._handle_vault({"action": "fill", "handle": handle}, task_id=task_id)
    return json.loads(raw), raw


class TestTheTab:
    def test_the_origin_is_read_from_the_only_web_page(self, vault):
        from tools import browser_vault_tool

        fake = _FakeChrome(T1=_Tab("about:blank"), T2=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            origin = browser_vault_tool._pin_cdp_tab(browser_vault_tool._current_page_origin)("t")
        assert origin == "https://bank.example"
        assert fake.evaluated_targets() == ["T2"]

    def test_the_tab_browser_cdp_used_last_wins_over_other_tabs(self, vault):
        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS),
                           T2=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            _use("T2")
            out, raw = _fill(meta.id)
        assert out["success"] is True and out["origin"] == "https://bank.example" and out["filled_fields"] == 1
        assert list(fake.writes()) == ["T2"]
        assert set(fake.evaluated_targets()) == {"T2"}

    def test_a_tab_opened_with_create_target_is_the_last_used_one(self, vault):
        from tools.browser_cdp_tool import browser_cdp, last_used_target

        fake = _FakeChrome(T1=_Tab("https://news.example/"))
        with _chrome(fake), patch("tools.browser_cdp_tool._resolve_cdp_endpoint", return_value=_ENDPOINT):
            json.loads(browser_cdp("Target.createTarget", {"url": "https://bank.example/login"}, task_id="t"))
            assert last_used_target("t", _ENDPOINT) == "T2"
            assert last_used_target("other-task", _ENDPOINT) is None
            assert last_used_target("t", "ws://127.0.0.1:9333/devtools/browser/other") is None

    def test_a_closed_last_used_tab_is_refused_not_replaced(self, vault):
        """The task's tab closed and one web page is left: that page was never chosen, so no fill."""
        from tools.browser_cdp_tool import browser_cdp

        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS),
                           T2=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        with _chrome(fake), patch("tools.browser_cdp_tool._resolve_cdp_endpoint", return_value=_ENDPOINT):
            _use("T1")
            json.loads(browser_cdp("Target.closeTarget", {"targetId": "T1"}, task_id="t"))
            out, _ = _fill(meta.id)
        assert out["error_type"] == "tab_ambiguous" and "closed" in out["error"]
        assert backend.resolved == [] and fake.writes() == {}

    def test_a_record_from_another_browser_is_refused(self, vault):
        """Chrome restarted (new endpoint): the old record names no tab here, and nothing replaces it."""
        from tools import browser_cdp_tool

        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        browser_cdp_tool._note_used_target("t", "ws://127.0.0.1:9222/devtools/browser/before-restart",
                                           "Runtime.evaluate", {}, "T1", {})
        with _chrome(fake):
            out, _ = _fill(meta.id)
        assert out["error_type"] == "tab_ambiguous" and fake.writes() == {} and backend.resolved == []

    def test_a_supervisor_appearing_mid_action_doesnt_move_the_fill(self, vault):
        """The action started on the pinned CDP tab; a supervisor started meanwhile isn't used."""
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        store, _ = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        other = TestBrowserExecPathUnchanged._Supervisor("https://bank.example/login", _LOGIN_CONTROLS)
        seen = {"n": 0}

        def appears(task_id):
            seen["n"] += 1
            return None if seen["n"] == 1 else other
        with _chrome(fake), patch.object(SUPERVISOR_REGISTRY, "get", side_effect=appears):
            out, _ = _fill(meta.id)
        assert out["success"] is True and list(fake.writes()) == ["T1"] and other.writes == []

    def test_the_tab_is_picked_once_per_action(self, vault):
        """browser_cdp in a parallel call moving the record mid-fill doesn't move the fill."""
        from tools import browser_cdp_tool

        store, _ = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS),
                           T2=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            _use("T1")
            fake.on_eval = lambda: browser_cdp_tool._note_used_target("t", _ENDPOINT, "Runtime.evaluate", {}, "T2", {})
            out, _ = _fill(meta.id)
        assert out["success"] is True
        assert list(fake.writes()) == ["T1"]
        assert set(fake.evaluated_targets()) == {"T1"}
        assert [m for m, _ in fake.calls].count("Target.getTargets") == 1


    def test_a_refusal_holds_even_if_a_supervisor_appears_mid_action(self, vault):
        """The pick refused (the task's tab closed); a supervisor started meanwhile doesn't get the fill."""
        from tools import browser_cdp_tool
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T2=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        browser_cdp_tool._note_used_target("t", _ENDPOINT, "Runtime.evaluate", {}, "T1-closed", {})
        other = TestBrowserExecPathUnchanged._Supervisor("https://bank.example/login", _LOGIN_CONTROLS)
        seen = {"n": 0}

        def appears(task_id):
            seen["n"] += 1
            return None if seen["n"] == 1 else other
        with _chrome(fake), patch.object(SUPERVISOR_REGISTRY, "get", side_effect=appears):
            out, _ = _fill(meta.id)
        assert out["error_type"] == "tab_ambiguous" and "closed" in out["error"]
        assert other.writes == [] and fake.writes() == {} and backend.resolved == []


class TestOriginBinding:
    def test_a_fill_on_the_saved_origin_writes_into_that_tab(self, vault):
        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            out, raw = _fill(meta.id)
        assert out["success"] is True and backend.resolved == ["password"]
        [write] = fake.tabs["T1"].writes
        assert '"https://bank.example"' in write  # the in-page assert names the saved origin

    def test_the_last_used_tab_on_another_origin_is_refused_without_searching_other_tabs(self, vault):
        """A tab on the saved origin is open, but the agent works in another one: refused, nothing resolved."""
        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS),
                           T2=_Tab("https://evil.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            _use("T2")
            out, _ = _fill(meta.id)
        assert out["success"] is False and out["error_type"] == "origin_mismatch"
        assert backend.resolved == [] and fake.writes() == {}
        assert "T1" not in fake.evaluated_targets()

    def test_the_only_web_page_on_another_origin_is_refused(self, vault):
        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://evil.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            out, _ = _fill(meta.id)
        assert out["error_type"] == "origin_mismatch" and backend.resolved == [] and fake.writes() == {}


class TestAmbiguity:
    def test_several_tabs_and_none_used_is_refused(self, vault):
        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS),
                           T2=_Tab("https://evil.example/", _LOGIN_CONTROLS))
        with _chrome(fake):
            out, _ = _fill(meta.id)
        assert out["success"] is False and out["error_type"] == "tab_ambiguous"
        assert "2 tabs" in out["error"] and "target_id" in out["error"]
        assert backend.resolved == [] and fake.writes() == {} and fake.evaluated_targets() == []

    def test_a_used_tab_from_another_task_doesnt_count(self, vault):
        store, backend = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS),
                           T2=_Tab("https://evil.example/", _LOGIN_CONTROLS))
        with _chrome(fake):
            _use("T1", task_id="someone-else")
            out, _ = _fill(meta.id, task_id="t")
        assert out["error_type"] == "tab_ambiguous" and fake.writes() == {}

    def test_no_web_page_is_refused(self, vault):
        store, _ = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("about:blank"), T2=_Tab("chrome://newtab/"))
        with _chrome(fake):
            out, _ = _fill(meta.id)
        assert out["error_type"] == "tab_ambiguous" and "No web page" in out["error"]

    def test_save_login_and_enter_code_without_a_handle_refuse_the_same_way(self, vault):
        from tools import browser_vault_tool

        fake = _FakeChrome(T1=_Tab("https://bank.example/login"), T2=_Tab("https://evil.example/"))
        with _chrome(fake):
            saved = json.loads(browser_vault_tool._handle_vault({"action": "save_login"}, task_id="t"))
            code = json.loads(browser_vault_tool._handle_vault({"action": "enter_code"}, task_id="t"))
        assert saved["error_type"] == code["error_type"] == "tab_ambiguous"


class TestNoSecretInOutput:
    def test_the_password_is_in_the_write_only(self, vault, caplog):
        store, _ = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        caplog.set_level(logging.DEBUG)
        with _chrome(fake):
            out, raw = _fill(meta.id)
        assert out["success"] is True
        assert _PASSWORD in fake.tabs["T1"].writes[0]
        assert _PASSWORD not in raw and _PASSWORD not in caplog.text

    def test_a_failing_write_returns_fixed_text(self, vault, caplog):
        """The CDP error may quote the script, and the script carries the password."""
        store, _ = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        fake.raise_on_fill = RuntimeError(f"CDP error: could not evaluate {_PASSWORD!r}")
        caplog.set_level(logging.DEBUG)
        with _chrome(fake):
            out, raw = _fill(meta.id)
        assert out["success"] is False and "could not run" in out["error"]
        assert _PASSWORD not in raw and _PASSWORD not in caplog.text

    def test_vault_frames_go_through_a_disabled_logger(self, vault):
        from tools import browser_vault_tool

        store, _ = vault
        meta = _login(store)
        fake = _FakeChrome(T1=_Tab("https://bank.example/login", _LOGIN_CONTROLS))
        with _chrome(fake):
            _fill(meta.id)
        assert fake.loggers and all(lg is browser_vault_tool._WS_LOGGER for lg in fake.loggers)
        assert browser_vault_tool._WS_LOGGER.disabled and not browser_vault_tool._WS_LOGGER.propagate

    @pytest.mark.skipif(shutil.which("node") is None, reason="needs node to run the fill script")
    def test_page_code_cannot_read_the_fill_script_through_caller(self, tmp_path):
        """A page that replaced JSON.stringify / querySelectorAll reads ``fn.caller``: the fill script is
        strict and its refusal calls nothing, so the source with the value never reaches page code."""
        from agent.vault_login_classifier import build_fill_js

        js = build_fill_js([{"index": 1, "token": "current-password", "value": _PASSWORD}],
                           expected_origin="https://bank.example", nonce="n")
        (tmp_path / "fill.js").write_text(js)
        (tmp_path / "page.js").write_text(
            "const src = require('fs').readFileSync(process.argv[2], 'utf8'); let leaked = '';\n"
            "const spy = function () { try { leaked += String(spy.caller || ''); } catch (e) {} return '{}'; };\n"
            "globalThis.JSON = { stringify: spy, parse: JSON.parse };\n"
            "globalThis.window = { location: { origin: process.argv[3] } };\n"
            "const qsa = function () { try { leaked += String(qsa.caller || ''); } catch (e) {} return []; };\n"
            "globalThis.document = { querySelectorAll: qsa };\n"
            "const out = eval(src); process.stdout.write(JSON.parse(out) && leaked);\n")
        for origin in ("https://evil.example", "https://bank.example"):
            done = subprocess.run(["node", str(tmp_path / "page.js"), str(tmp_path / "fill.js"), origin],
                                  capture_output=True, text=True, timeout=30)
            assert done.returncode == 0, done.stderr[-300:]
            assert _PASSWORD not in done.stdout

    def test_a_page_exception_returns_fixed_text(self, vault):
        from tools import browser_vault_tool

        async def throws(ws_url, method, params, target_id, timeout, ws_logger=None):
            return {"result": {"type": "object"},
                    "exceptionDetails": {"text": "Uncaught", "exception": {"description": _PASSWORD}}}
        tab = browser_vault_tool._CdpTab(_ENDPOINT, "T1")
        with patch("tools.browser_cdp_tool._cdp_call", side_effect=throws):
            res = tab.evaluate_runtime("anything")
        assert res == {"ok": False, "error": browser_vault_tool._TAB_FAILED}


class TestMailCode:
    def test_a_mailed_code_goes_into_the_last_used_tab(self, vault, monkeypatch, tmp_path):
        from tools import browser_vault_tool

        store, _ = vault
        meta = store.add_item("login", "Gulp", {"identifier_type": "email", "identifier": "bot@example.com",
                                                "password": "pw"}, origin="https://www.gulp.de")
        script = tmp_path / "mail-code"
        script.write_text("#!/bin/sh\necho \"$@\" > " + str(tmp_path / "args") + "\necho 482913\n")
        script.chmod(0o700)
        monkeypatch.setenv("HERMES_VAULT_MAIL_CODE_COMMAND", str(script))
        fake = _FakeChrome(T1=_Tab("https://www.gulp.de/verify", _OTP_CONTROLS),
                           T2=_Tab("https://news.example/", _OTP_CONTROLS))
        with _chrome(fake):
            _use("T1")
            raw = browser_vault_tool._handle_vault({"action": "enter_code", "handle": meta.id}, task_id="t")
        out = json.loads(raw)
        assert out["success"] is True and out["source"] == "mail" and out["origin"] == "https://www.gulp.de"
        [write] = fake.tabs["T1"].writes
        assert '"value": "482913"' in write and "482913" not in raw
        assert fake.tabs["T2"].writes == []
        args = (tmp_path / "args").read_text().split()
        assert args[:4] == ["--origin", "https://www.gulp.de", "--to", "bot@example.com"]

    def test_no_mail_code_is_fetched_for_a_tab_on_another_origin(self, vault, monkeypatch, tmp_path):
        from tools import browser_vault_tool

        store, _ = vault
        meta = store.add_item("login", "Gulp", {"identifier_type": "email", "identifier": "bot@example.com",
                                                "password": "pw"}, origin="https://www.gulp.de")
        script = tmp_path / "mail-code"
        script.write_text("#!/bin/sh\ntouch " + str(tmp_path / "ran") + "\necho 482913\n")
        script.chmod(0o700)
        monkeypatch.setenv("HERMES_VAULT_MAIL_CODE_COMMAND", str(script))
        fake = _FakeChrome(T1=_Tab("https://www.gulp.de/verify", _OTP_CONTROLS),
                           T2=_Tab("https://evil.example/verify", _OTP_CONTROLS))
        with _chrome(fake):
            _use("T2")
            out = json.loads(browser_vault_tool._handle_vault({"action": "enter_code", "handle": meta.id}, task_id="t"))
        assert out["error_type"] == "origin_mismatch"
        assert not (tmp_path / "ran").exists() and fake.writes() == {}


class TestBrowserExecPathUnchanged:
    class _Supervisor:
        """A browser_exec supervisor: it may focus the saved origin's tab, as upstream does."""

        def __init__(self, href, controls):
            self.href, self.controls, self.writes, self.focused = href, controls, [], []

        def focus_page(self, origin, accept=None):
            self.focused.append(origin)
            return {"ok": bool(origin) and normalize_origin(self.href) == origin, "url": self.href}

        def evaluate_runtime(self, expression):
            if expression.strip() == "window.location.href":
                return {"ok": True, "result": self.href}
            if "expectedOrigin" in expression:
                self.writes.append(expression)
                return {"ok": True, "result": json.dumps({"filled": 1})}
            return {"ok": True, "result": json.dumps(self.controls)}

    def test_a_running_supervisor_is_used_and_cdp_is_never_touched(self, vault):
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        store, _ = vault
        meta = _login(store)
        sup = self._Supervisor("https://bank.example/login", _LOGIN_CONTROLS)
        fake = _FakeChrome(T1=_Tab("https://evil.example/"), T2=_Tab("https://other.example/"))
        with _chrome(fake), patch.object(SUPERVISOR_REGISTRY, "get", return_value=sup):
            out, _ = _fill(meta.id)
        assert out["success"] is True and len(sup.writes) == 1 and sup.focused == ["https://bank.example"]
        assert fake.calls == []

    def test_a_browser_exec_session_attaches_its_own_endpoint(self, vault):
        from tools.browser_supervisor import SUPERVISOR_REGISTRY

        store, _ = vault
        meta = _login(store)
        sup = self._Supervisor("https://bank.example/login", _LOGIN_CONTROLS)
        fake = _FakeChrome(T1=_Tab("https://evil.example/"))
        session = {"cdp_url": "ws://127.0.0.1:9555/devtools/browser/harness"}
        with _chrome(fake), \
             patch("tools.browser_session.session_for_task", return_value=session), \
             patch.object(SUPERVISOR_REGISTRY, "get_or_start", return_value=sup) as start:
            out, _ = _fill(meta.id)
        assert out["success"] is True and len(sup.writes) == 1
        assert start.call_args.kwargs["cdp_url"] == session["cdp_url"]
        assert fake.calls == []

    def test_without_any_cdp_endpoint_the_old_refusal_stays(self, vault):
        store, _ = vault
        meta = _login(store)
        with patch("tools.browser_session_cdp.get_cdp_override", return_value=""):
            out, _ = _fill(meta.id)
        assert out["success"] is False
        assert out["error"].startswith("Could not determine the current page origin")
        assert re.search("tab_ambiguous", json.dumps(out)) is None
