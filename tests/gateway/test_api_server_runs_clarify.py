"""Questions with choices on /v1/runs: clarify.request, waiting_for_input, POST /v1/runs/{id}/clarify."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_runs
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from toolsets import TOOLSETS

AUTH = {"Authorization": "Bearer sk-secret"}


def _asking_agent(question="Which offer?", choices=("Pilot", "Audit")):
    agent = MagicMock()

    def _run(user_message=None, conversation_history=None, task_id=None):
        answer = agent.clarify_callback(question, list(choices))
        return {"final_response": f"answer={answer}"}

    agent.run_conversation.side_effect = _run
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    return agent


def _app():
    adapter = _make_adapter(api_key="sk-secret")
    app = _create_runs_app(adapter)
    app.router.add_post("/v1/runs/{run_id}/clarify", adapter._handle_run_clarify)
    return adapter, app


async def _wait_for(cli, run_id, status, attempts=100):
    for _ in range(attempts):
        data = await (await cli.get(f"/v1/runs/{run_id}", headers=AUTH)).json()
        if data.get("status") == status:
            return data
        await asyncio.sleep(0.05)
    raise AssertionError(f"run never reached {status}: {data}")


def test_clarify_stays_off_for_api_clients_that_dont_ask_for_it():
    assert "clarify" not in TOOLSETS["hermes-api-server"]["tools"]


GO = {"input": "go", "clarify": True}


@pytest.mark.asyncio
async def test_a_question_waits_and_only_its_own_id_answers_it():
    adapter, app = _app()
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            create.side_effect = [_asking_agent(), _asking_agent(question="Other run?")]
            first = (await (await cli.post("/v1/runs", json=GO, headers=AUTH)).json())["run_id"]
            other = (await (await cli.post("/v1/runs", json=GO, headers=AUTH)).json())["run_id"]
            waiting = await _wait_for(cli, first, "waiting_for_input")
            asked = waiting["clarify"]
            assert asked["question"] == "Which offer?"
            assert asked["choices"] == ["Pilot", "Audit"]
            other_id = (await _wait_for(cli, other, "waiting_for_input"))["clarify"]["clarify_id"]

            # Another run's question id is refused here, and a made-up one too.
            for clarify_id in (other_id, "nope"):
                resp = await cli.post(f"/v1/runs/{first}/clarify", json={"clarify_id": clarify_id, "answer": "x"},
                                      headers=AUTH)
                assert resp.status == 409
            bad = await cli.post(f"/v1/runs/{first}/clarify", json={"clarify_id": asked["clarify_id"]}, headers=AUTH)
            assert bad.status == 400
            for body in ([], None, "x"):
                odd = await cli.post(f"/v1/runs/{first}/clarify", json=body, headers=AUTH)
                assert odd.status == 400, body
            unauth = await cli.post(f"/v1/runs/{first}/clarify", json={"clarify_id": asked["clarify_id"], "answer": "x"})
            assert unauth.status == 401

            ok = await cli.post(f"/v1/runs/{first}/clarify",
                                json={"clarify_id": asked["clarify_id"], "answer": "Audit"}, headers=AUTH)
            assert ok.status == 200
            done = await _wait_for(cli, first, "completed")
            assert done["output"] == "answer=Audit"
            again = await cli.post(f"/v1/runs/{first}/clarify",
                                   json={"clarify_id": asked["clarify_id"], "answer": "Pilot"}, headers=AUTH)
            assert again.status == 409, "answered once"

            await cli.post(f"/v1/runs/{other}/clarify", json={"clarify_id": other_id, "answer": "Pilot"}, headers=AUTH)
            await _wait_for(cli, other, "completed")


@pytest.mark.asyncio
async def test_no_answer_is_never_a_choice():
    adapter, app = _app()
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create, \
                patch("tools.clarify_gateway.get_clarify_timeout", return_value=0.3):
            create.return_value = _asking_agent()
            run_id = (await (await cli.post("/v1/runs", json=GO, headers=AUTH)).json())["run_id"]
            done = await _wait_for(cli, run_id, "completed")
            assert done["output"] == f"answer={api_server_runs.CLARIFY_NO_ANSWER}"
            assert "clarify" not in done


@pytest.mark.asyncio
async def test_a_run_that_didnt_ask_for_questions_gets_no_clarify():
    adapter, app = _app()
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
            agent.clarify_callback = None
            create.return_value = agent
            run_id = (await (await cli.post("/v1/runs", json={"input": "go"}, headers=AUTH)).json())["run_id"]
            await _wait_for(cli, run_id, "completed")
            assert agent.clarify_callback is None
            assert create.call_args.kwargs.get("allow_clarify") is False


@pytest.mark.asyncio
async def test_stop_releases_a_waiting_question():
    adapter, app = _app()
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent") as create:
            agent = _asking_agent()
            agent.interrupt = MagicMock()
            create.return_value = agent
            run_id = (await (await cli.post("/v1/runs", json=GO, headers=AUTH)).json())["run_id"]
            asked = (await _wait_for(cli, run_id, "waiting_for_input"))["clarify"]
            stop = await cli.post(f"/v1/runs/{run_id}/stop", headers=AUTH)
            assert stop.status == 200
            for _ in range(100):
                if create.return_value.run_conversation.call_count and not api_server_runs._clarify_pending(adapter).get(run_id):
                    break
                await asyncio.sleep(0.05)
            assert not api_server_runs._clarify_pending(adapter).get(run_id), "the wait was released"
            late = await cli.post(f"/v1/runs/{run_id}/clarify", json={"clarify_id": asked["clarify_id"], "answer": "x"},
                                  headers=AUTH)
            assert late.status in (404, 409)
