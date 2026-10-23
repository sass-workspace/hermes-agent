"""Compatibility seams for extracted RoomLink dispatch handling."""

import json
from unittest.mock import MagicMock

import pytest

from gateway.platforms import api_server








@pytest.mark.asyncio
async def test_non_room_run_body_passes_through_unchanged():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="")
    request = object()
    body = {"input": "ordinary run"}

    normalized, error = await adapter._normalize_room_dispatch(request, body)

    assert normalized is body
    assert error is None
    adapter._room_grant_token.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_room_dispatch_rejects_extra_fields_before_grant_verification():
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="room-grant")
    request = object()
    body = {
        "input": "room prompt",
        "hosted_room_dispatch": {},
        "unexpected": True,
    }

    normalized, error = await adapter._normalize_room_dispatch(request, body)

    assert normalized is body
    assert error.status == 400
    assert json.loads(error.text)["error"]["code"] == "invalid_room_dispatch"


@pytest.mark.asyncio
@pytest.mark.parametrize("smuggled", [
    {"hosted_room_dispatch": {}},
    {"_room_execution_policy": {"enabled_toolsets": ["bot_room", "terminal"], "approval_mode": "off"}},
])
async def test_room_keys_without_a_grant_are_refused(smuggled):
    """Without a room grant the room keys would reach _create_agent unverified and pick the
    run's toolsets and approval mode; only the verified path may produce them."""
    adapter = api_server.APIServerAdapter.__new__(api_server.APIServerAdapter)
    adapter._room_grant_token = MagicMock(return_value="")
    body = {"input": "ordinary run", **smuggled}

    _, error = await adapter._normalize_room_dispatch(object(), body)

    assert error.status == 403
    assert json.loads(error.text)["error"]["code"] == "room_grant_required"
