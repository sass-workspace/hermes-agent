"""Tests for hermes-api-server toolset and API server tool availability."""

from unittest.mock import MagicMock, patch


class TestApiServerPlatformConfig:

    def test_default_api_server_includes_terminal_toolset(self):
        """Regression #49622: desktop-only read_terminal is registered into the
        'terminal' toolset (ships in-repo), so resolve_toolset('terminal') grows
        to include it after discovery. read_terminal is NOT in the
        hermes-api-server composite, so the old all-tools subset test dropped
        'terminal' entirely. Its static membership (terminal, process) IS in the
        composite, so it must stay enabled."""
        from tools.registry import discover_builtin_tools
        from hermes_cli.tools_config import _get_platform_tools
        discover_builtin_tools()
        assert "terminal" in _get_platform_tools({}, "api_server")



class TestApiServerContextFiles:
    def _agent_kwargs(self, config):
        from gateway.platforms.api_server import APIServerAdapter
        from gateway.config import PlatformConfig

        adapter = APIServerAdapter(PlatformConfig())
        with patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True), \
             patch("gateway.run._resolve_runtime_agent_kwargs") as mock_kwargs, \
             patch("gateway.run._resolve_gateway_model") as mock_model, \
             patch("gateway.run._load_gateway_config") as mock_config, \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_kwargs.return_value = {"api_key": "test-key", "base_url": None,
                                        "provider": None, "api_mode": None,
                                        "command": None, "args": []}
            mock_model.return_value = "test/model"
            mock_config.return_value = config
            mock_agent_cls.return_value = MagicMock()
            adapter._create_agent()
            return mock_agent_cls.call_args.kwargs

    def test_skip_context_files_from_gateway_platform_config(self):
        kwargs = self._agent_kwargs(
            {"gateway": {"platforms": {"api_server": {"skip_context_files": True}}}})
        assert kwargs["skip_context_files"] is True
        assert kwargs["load_soul_identity"] is True

    def test_context_files_load_by_default(self):
        assert self._agent_kwargs({})["skip_context_files"] is False

    def test_list_shaped_platforms_means_no_override(self):
        kwargs = self._agent_kwargs({"gateway": {"platforms": ["api_server", "slack"]}})
        assert kwargs["skip_context_files"] is False
