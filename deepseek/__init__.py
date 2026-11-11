"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

from .agent import (
    AgentResult,
    DeepSeekAgent,
    ToolExecution,
    build_tools_instruction,
    extract_tool_calls,
)
from .auth import Session, get_session, login
from .client import DeepSeekClient, Reply
from .pow import DeepSeekPow
from .tools import (
    DEFAULT_TIMEOUT,
    DEFAULT_TOOLS,
    TERMINAL_TOOLS,
    execute_cmd,
    execute_command,
    execute_powershell,
    execute_tool,
    register_tool,
)

__all__ = [
    "Session",
    "get_session",
    "login",
    "DeepSeekClient",
    "Reply",
    "DeepSeekPow",
    "DeepSeekAgent",
    "AgentResult",
    "ToolExecution",
    "build_tools_instruction",
    "extract_tool_calls",
    "execute_command",
    "execute_powershell",
    "execute_cmd",
    "execute_tool",
    "register_tool",
    "TERMINAL_TOOLS",
    "DEFAULT_TOOLS",
]
