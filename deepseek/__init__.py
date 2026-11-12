"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

from .agent import (
    AgentResult,
    DeepSeekAgent,
    ToolExecution,
    build_tools_instruction,
    extract_tool_calls,
    is_promising_action_without_call,
    load_instructions_md,
)
from .auth import Session, get_session, login, logout
from .client import DeepSeekClient, Reply
from .pow import DeepSeekPow
from .tools import (
    AGENT_TOOLS,
    DEFAULT_TIMEOUT,
    DEFAULT_TOOLS,
    TERMINAL_TOOLS,
    execute_cmd,
    execute_command,
    execute_powershell,
    execute_tool,
    list_directory,
    read_file,
    register_tool,
    search_in_files,
    verify_path,
    write_file,
)

__all__ = [
    "Session",
    "get_session",
    "login",
    "logout",
    "DeepSeekClient",
    "Reply",
    "DeepSeekPow",
    "DeepSeekAgent",
    "AgentResult",
    "ToolExecution",
    "build_tools_instruction",
    "extract_tool_calls",
    "is_promising_action_without_call",
    "load_instructions_md",
    "execute_command",
    "execute_powershell",
    "execute_cmd",
    "read_file",
    "write_file",
    "list_directory",
    "search_in_files",
    "verify_path",
    "execute_tool",
    "register_tool",
    "AGENT_TOOLS",
    "TERMINAL_TOOLS",
    "DEFAULT_TOOLS",
]
