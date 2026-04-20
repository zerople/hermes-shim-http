from __future__ import annotations

import os
from pathlib import Path

from hermes_shim_http import hermes_mcp


def test_none_tools_yields_none():
    with hermes_mcp.request_scoped_mcp_config(tools=None) as path:
        assert path is None


def test_same_tools_returns_same_config_path():
    tools = [{"type": "function", "function": {"name": "foo", "description": "", "parameters": {}}}]
    with hermes_mcp.request_scoped_mcp_config(tools=tools) as path_a:
        assert path_a is not None
        assert os.path.isfile(path_a)
    with hermes_mcp.request_scoped_mcp_config(tools=tools) as path_b:
        assert path_b == path_a
        assert os.path.isfile(path_b)


def test_different_tools_returns_different_config_paths():
    tools_a = [{"type": "function", "function": {"name": "foo", "description": "", "parameters": {}}}]
    tools_b = [{"type": "function", "function": {"name": "bar", "description": "", "parameters": {}}}]
    with hermes_mcp.request_scoped_mcp_config(tools=tools_a) as path_a:
        pass
    with hermes_mcp.request_scoped_mcp_config(tools=tools_b) as path_b:
        pass
    assert path_a != path_b


def test_cached_config_survives_multiple_opens():
    tools = [{"type": "function", "function": {"name": "baz", "description": "", "parameters": {}}}]
    with hermes_mcp.request_scoped_mcp_config(tools=tools) as path_a:
        assert os.path.isfile(path_a)
    assert os.path.isfile(path_a)  # survives context exit — cached
    with hermes_mcp.request_scoped_mcp_config(tools=tools) as path_b:
        assert path_a == path_b
        assert os.path.isfile(path_b)


def test_key_insensitive_to_dict_ordering():
    tool_fwd = {"type": "function", "function": {"name": "x", "description": "", "parameters": {"a": 1, "b": 2}}}
    tool_rev = {"function": {"parameters": {"b": 2, "a": 1}, "description": "", "name": "x"}, "type": "function"}
    with hermes_mcp.request_scoped_mcp_config(tools=[tool_fwd]) as path_a:
        pass
    with hermes_mcp.request_scoped_mcp_config(tools=[tool_rev]) as path_b:
        pass
    assert path_a == path_b
