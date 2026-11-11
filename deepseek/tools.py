"""Terminal and shell execution tools for DeepSeek.

Provides tools to execute PowerShell, CMD, and general terminal commands,
formatted for OpenAI-compatible function/tool calling.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Union

# Default timeout in seconds for terminal commands
DEFAULT_TIMEOUT = 60


def execute_powershell(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a command or script in Windows PowerShell.

    Args:
        command: The PowerShell command string to execute.
        timeout: Maximum seconds to wait before terminating.
        cwd: Optional working directory for the command.

    Returns:
        Dict containing exit_code, stdout, stderr, and success status.
    """
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "shell": "powershell",
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "shell": "powershell",
            "command": command,
        }
    except Exception as exc:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "shell": "powershell",
            "command": command,
        }


def execute_cmd(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a command in Windows Command Prompt (CMD).

    Args:
        command: The CMD command string to execute.
        timeout: Maximum seconds to wait before terminating.
        cwd: Optional working directory for the command.

    Returns:
        Dict containing exit_code, stdout, stderr, and success status.
    """
    cmd = ["cmd.exe", "/c", command]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "shell": "cmd",
            "command": command,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "shell": "cmd",
            "command": command,
        }
    except Exception as exc:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "shell": "cmd",
            "command": command,
        }


def execute_command(
    command: str,
    shell: str = "powershell",
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a command in PowerShell or CMD.

    Args:
        command: The command string to execute.
        shell: 'powershell' (default) or 'cmd'.
        timeout: Maximum seconds to wait.
        cwd: Working directory.
    """
    shell_lower = (shell or "powershell").lower().strip()
    if shell_lower in ("cmd", "cmd.exe"):
        return execute_cmd(command=command, timeout=timeout, cwd=cwd)
    return execute_powershell(command=command, timeout=timeout, cwd=cwd)


# OpenAI-compatible tool specifications
TERMINAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a terminal command in PowerShell or CMD and return its stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command line string to execute.",
                    },
                    "shell": {
                        "type": "string",
                        "enum": ["powershell", "cmd"],
                        "default": "powershell",
                        "description": "The shell to execute in: 'powershell' (default) or 'cmd'.",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_powershell",
            "description": "Execute a Windows PowerShell command or script and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The PowerShell command or script to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_cmd",
            "description": "Execute a Windows Command Prompt (CMD) command and return stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The CMD command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "description": "Timeout in seconds.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

DEFAULT_TOOLS = TERMINAL_TOOLS

# Internal mapping of tool names to callable functions
_TOOL_HANDLERS: Dict[str, Callable] = {
    "execute_command": execute_command,
    "execute_powershell": execute_powershell,
    "execute_cmd": execute_cmd,
}


def execute_tool(name: str, arguments: Union[Dict[str, Any], str]) -> str:
    """Execute a tool by name with arguments and return a formatted JSON string."""
    if isinstance(arguments, str):
        try:
            args_dict = json.loads(arguments)
        except json.JSONDecodeError:
            args_dict = {"command": arguments}
    elif isinstance(arguments, dict):
        args_dict = arguments
    else:
        args_dict = {}

    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        return json.dumps(
            {"error": f"Tool '{name}' not found. Available: {list(_TOOL_HANDLERS.keys())}"},
            ensure_ascii=False,
        )

    try:
        result = handler(**args_dict)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as exc:
        return json.dumps({"error": f"Execution failed: {exc}"}, ensure_ascii=False)


def register_tool(name: str, handler: Callable, schema: Optional[Dict[str, Any]] = None):
    """Register a custom tool handler."""
    _TOOL_HANDLERS[name] = handler
    if schema:
        TERMINAL_TOOLS.append(schema)
