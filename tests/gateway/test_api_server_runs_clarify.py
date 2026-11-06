"""Downstream (Iris) clarify contract on /v1/runs, carried onto upstream's session-authority clarify.

The Iris Mac app and the Iris bot host's task runner opt in with ``{"clarify": true}``, read the
question as ``clarify_id`` (SSE ``clarify.request`` and ``GET /v1/runs/{id}``: status
``waiting_for_input`` plus a ``clarify`` object) and answer ``{clarify_id, answer}``. Upstream's own
shape (``prompt_id``, ``pending_controls``, ``{request_id, execution_generation, answer}``) keeps
working unchanged beside it.
"""
import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server as _api_server
from gateway.platforms.api_server_runs import _RunLaunch, _execute_run, _legacy_clarify_body, _with_clarify_id
from gateway.session_api_turn import admit_api_turn
from tests.waiting import async_wait_until


def _turn_runner(owner, ref):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(_native_slack_task_cards=False, _voice_ack_guild=[None], progress_mode='off')
    turn._approval_owner = (owner, ref.session_id, owner.db.get_session(ref.session_id)['runtime_generation'])
    return turn


def _launch(api, admitted, run_id):
    queue = asyncio.Queue()
    api._run_streams[run_id] = queue
    api._run_statuses[run_id] = {}
    return _RunLaunch(api, run_id, queue, admitted[1].session_id, None, False, 'hello', [], False,
                      agent_kwargs={}, request_profile=None, admission=admitted), queue


def test_clarify_events_carry_the_clarify_id_alias():
    prompt = {'kind': 'clarify', 'prompt_id': 'c1', 'question': 'Which?', 'choices': ['A']}
    assert _with_clarify_id(prompt)['clarify_id'] == 'c1'
    assert _with_clarify_id(prompt)['prompt_id'] == 'c1'
    approval = {'kind': 'approval', 'prompt_id': 'a1'}
    assert 'clarify_id' not in _with_clarify_id(approval)


def test_upstream_answer_bodies_pass_through_untouched(api):
    body = {'request_id': 'c1', 'execution_generation': 3, 'answer': 'A'}
    assert _legacy_clarify_body(api, 'run_x', body) is body
    odd = {'clarify_id': 'c1', 'answer': 'A', 'extra': 1}
    assert _legacy_clarify_body(api, 'run_x', odd) is odd, "only the exact old shape is mapped"


def test_the_clarify_opt_in_is_an_admission_setting(api, owner):
    _, _, row = admit_api_turn(api, session_id='clarify-optin', request_id='run_optin', user_message='hi',
                               conversation_history=[], clarify=True)
    assert row['payload']['api_turn_v1']['settings']['clarify'] is True
    _, _, plain = admit_api_turn(api, session_id='clarify-plain', request_id='run_plain', user_message='hi',
                                 conversation_history=[])
    assert not plain['payload']['api_turn_v1']['settings'].get('clarify')


@pytest.mark.asyncio
async def test_iris_client_sees_waiting_for_input_and_answers_by_clarify_id(api, owner):
    from gateway.platforms.api_server_authority_runs import run_projection
    from tools import clarify_gateway
    admitted = admit_api_turn(api, session_id='iris-clarify', request_id='run_iris', user_message='hello',
                              conversation_history=[], clarify=True)
    _, ref, _row = admitted
    launch, queue = _launch(api, admitted, 'run_iris')
    api._run_owners['run_iris'] = api._run_idempotency_scope(
        SimpleNamespace(headers={}, path='/v1/runs/run_iris/clarify', method='POST'))

    async def handle(event):
        from gateway.session_results import execution_result
        turn = _turn_runner(owner, ref)
        entry = clarify_gateway.register('clarify-iris', 'api', 'Pilot or audit?', ['Pilot', 'Audit'])
        owner.register_clarify(ref.session_id, turn._approval_owner[2], entry)
        await asyncio.to_thread(entry.event.wait, 10)
        execution_result.get()['result'] = {'final_response': entry.response or 'unanswered'}
        return entry.response
    owner.runner._handle_message = handle
    run = asyncio.ensure_future(_execute_run(api, launch, _api_server=_api_server))
    await async_wait_until(lambda: (run_projection(api, 'run_iris') or {}).get('clarify'),
                           timeout=10, what='the clarify question projected on the run')
    projected = run_projection(api, 'run_iris')
    assert projected['status'] == 'waiting_for_input'
    assert projected['clarify'] == {'clarify_id': 'clarify-iris', 'question': 'Pilot or audit?',
                                    'choices': ['Pilot', 'Audit'], 'multi_select': False}
    app = web.Application()
    app.router.add_post('/v1/runs/{run_id}/clarify', api._handle_run_clarify)
    async with TestClient(TestServer(app)) as client:
        wrong = await client.post('/v1/runs/run_iris/clarify', json={'clarify_id': 'nope', 'answer': 'x'})
        assert wrong.status == 409, await wrong.text()
        answered = await client.post('/v1/runs/run_iris/clarify',
                                     json={'clarify_id': 'clarify-iris', 'answer': 'Audit'})
        assert answered.status == 200, await answered.text()
        await asyncio.wait_for(run, timeout=10)
    assert api._run_statuses['run_iris']['output'] == 'Audit'
    await asyncio.sleep(0)
    events = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None:
            events.append(item)
    asked = next(e for e in events if e['event'] == 'clarify.request')
    assert asked['clarify_id'] == 'clarify-iris' and asked['choices'] == ['Pilot', 'Audit']
    assert 'clarify' not in (run_projection(api, 'run_iris') or {})


def test_only_an_opted_in_api_turn_gains_the_clarify_toolset():
    from gateway.session_api_turn import api_clarify_toolsets
    base = ['hermes-api-server']
    assert api_clarify_toolsets({'settings': {'clarify': True}}, base) == ['clarify', 'hermes-api-server']
    assert api_clarify_toolsets({'settings': {'clarify': None}}, base) is base
    assert api_clarify_toolsets(None, base) is base, "not an API turn"
    assert api_clarify_toolsets({'settings': {'clarify': True}}, base, ['clarify']) is base, "profile disabled it"
    assert api_clarify_toolsets({'settings': {'clarify': True}}, None) is None, "no resolved toolsets"


def test_a_pending_approval_reads_and_answers_in_the_fork_shape(api, owner):
    """GET /v1/runs/{id} carries the pending approval under ``approval`` with its ``request_id``, and
    ``{choice}`` / ``{choice, request_id}`` map onto upstream's exact answer body."""
    from gateway.platforms.api_server_authority_runs import run_projection
    from gateway.platforms.api_server_runs import _legacy_approval_body
    from hermes_state_runtime import claim_session_input
    _, ref, _ = admit_api_turn(api, session_id='iris-approval', request_id='run_appr', user_message='hello',
                               conversation_history=[])
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=ref.session_id)
    live = owner.sessions[ref.session_id]
    owner.register_approval(ref.session_id, row['generation'], live.route,
                            {'request_id': 'approve-me', 'command': 'rm -rf build', 'description': 'delete'})
    live.controls.remote_responders['approve-me'] = lambda *answer: None
    projected = run_projection(api, 'run_appr')
    assert projected['status'] == 'waiting_for_approval'
    assert projected['approval']['request_id'] == 'approve-me'
    assert projected['approval']['command'] == 'rm -rf build'
    mapped = _legacy_approval_body(api, 'run_appr', {'choice': 'approve', 'request_id': 'approve-me'})
    assert mapped == {'request_id': 'approve-me', 'execution_generation': row['generation'], 'choice': 'once'}
    assert _legacy_approval_body(api, 'run_appr', {'choice': 'deny', 'request_id': 'approve-me'})['choice'] == 'deny'
    # Never guessed: a delayed id-less answer to one approval must not approve the next.
    assert _legacy_approval_body(api, 'run_appr', {'choice': 'once'}) == {'choice': 'once'}
    # The mapping never prunes pending controls (it reads no projection).
    assert [p['prompt_id'] for p in live.controls.snapshot(ref.session_id, row['generation'])] == ['approve-me']
    exact = {'request_id': 'approve-me', 'execution_generation': row['generation'], 'choice': 'once'}
    assert _legacy_approval_body(api, 'run_appr', exact) is exact
    assert _legacy_approval_body(api, 'run_unknown', {'choice': 'once', 'request_id': 'x'}) == {'choice': 'once', 'request_id': 'x'}


def test_approval_events_carry_the_request_id_alias():
    event = _with_clarify_id({'kind': 'approval', 'prompt_id': 'a1', 'command': 'x'})
    assert event['request_id'] == 'a1' and event['prompt_id'] == 'a1'
