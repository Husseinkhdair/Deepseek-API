"""AI Agent Tools for DeepSeek.

Provides tools for:
- Terminal & Shell execution (PowerShell / CMD)
- File system access (reading, writing, inspecting files)
- Directory inspection and search
- Result and path verification
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

DEFAULT_TIMEOUT = 25


# ---------------------------------------------------------------------------
# 1. Terminal / Shell Tools
# ---------------------------------------------------------------------------

def _kill_proc_tree(pid: int):
    """Cleanly terminate process and all child processes on Windows."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/pid", str(pid), "/f", "/t"], capture_output=True)
    except Exception:
        pass


def execute_powershell(
    command: str,
    timeout: int = DEFAULT_TIMEOUT,
    cwd: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a command or script in Windows PowerShell."""
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
    ]
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": (stdout or "").strip(),
            "stderr": (stderr or "").strip(),
            "shell": "powershell",
            "command": command,
        }
    except subprocess.TimeoutExpired:
        if proc:
            _kill_proc_tree(proc.pid)
            try:
                proc.kill()
            except Exception:
                pass
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                f"⚠️ انتهت مهلة تشغيل الأمر بعد {timeout} ثانية (Timeout).\n"
                "الأمر استغرق وقتاً طويلاً جداً (مثل البحث في كامل C:\\ أو انتظار إدخال).\n"
                "يرجى تحديد مسار محدد ومباشر (مثل C:\\AllProject) أو استخدام أدوات قراءة الملفات search_in_files / read_file بدلاً من المسح الشامل."
            ),
            "shell": "powershell",
            "command": command,
        }
    except Exception as exc:
        if proc:
            _kill_proc_tree(proc.pid)
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
    """Execute a command in Windows Command Prompt (CMD)."""
    cmd = ["cmd.exe", "/c", command]
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": (stdout or "").strip(),
            "stderr": (stderr or "").strip(),
            "shell": "cmd",
            "command": command,
        }
    except subprocess.TimeoutExpired:
        if proc:
            _kill_proc_tree(proc.pid)
            try:
                proc.kill()
            except Exception:
                pass
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"⚠️ انتهت مهلة تشغيل الأمر بعد {timeout} ثانية (Timeout). يرجى استخدام أمر أسرع ومسار محدد.",
            "shell": "cmd",
            "command": command,
        }
    except Exception as exc:
        if proc:
            _kill_proc_tree(proc.pid)
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
    """Execute a command in PowerShell or CMD."""
    shell_lower = (shell or "powershell").lower().strip()
    if shell_lower in ("cmd", "cmd.exe"):
        return execute_cmd(command=command, timeout=timeout, cwd=cwd)
    return execute_powershell(command=command, timeout=timeout, cwd=cwd)


# ---------------------------------------------------------------------------
# 2. File System Tools (Read / Write / Inspect)
# ---------------------------------------------------------------------------

def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Read the contents of a text file, with optional line range.

    Args:
        path: Path to the file.
        start_line: Optional 1-based start line.
        end_line: Optional 1-based end line (inclusive).
        encoding: Text encoding (default: utf-8).
    """
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "error": f"File does not exist: {path}"}
        if not p.is_file():
            return {"success": False, "error": f"Path is not a file: {path}"}

        try:
            content = p.read_text(encoding=encoding, errors="replace")
        except Exception:
            content = p.read_text(encoding="latin1", errors="replace")

        lines = content.splitlines()
        total_lines = len(lines)

        s = max(1, start_line) if start_line is not None else 1
        e = min(total_lines, end_line) if end_line is not None else total_lines

        selected_lines = lines[s - 1 : e]
        # Format with line numbers for clarity
        numbered = [f"{i + s}: {line}" for i, line in enumerate(selected_lines)]

        return {
            "success": True,
            "path": str(p),
            "total_lines": total_lines,
            "displayed_lines": f"{s}-{e}",
            "content": "\n".join(numbered),
        }
    except Exception as exc:
        return {"success": False, "error": f"Error reading file {path}: {exc}"}


def write_file(
    path: str,
    content: str,
    overwrite: bool = True,
    append: bool = False,
) -> Dict[str, Any]:
    """Write, overwrite, or append text content to a file, creating directories if needed.

    Args:
        path: Destination file path.
        content: Text content to write or append.
        overwrite: Allow overwriting existing file when append=False (default: True).
        append: If True, append content to the end of the file instead of overwriting.
    """
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        if append and p.exists():
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
            full_text = p.read_text(encoding="utf-8", errors="replace")
            return {
                "success": True,
                "path": str(p),
                "mode": "append",
                "chunk_bytes": len(content.encode("utf-8")),
                "total_file_lines": len(full_text.splitlines()),
                "message": f"Successfully appended chunk to {p.name}",
            }

        if p.exists() and not overwrite and not append:
            return {"success": False, "error": f"File already exists and overwrite=False: {path}"}

        p.write_text(content, encoding="utf-8")

        return {
            "success": True,
            "path": str(p),
            "mode": "write",
            "bytes_written": len(content.encode("utf-8")),
            "total_lines": len(content.splitlines()),
            "message": f"Successfully wrote {p.name}",
        }
    except Exception as exc:
        return {"success": False, "error": f"Error writing file {path}: {exc}"}


def append_file(
    path: str,
    content: str,
) -> Dict[str, Any]:
    """Append a chunk of text content to the end of an existing (or new) file."""
    return write_file(path=path, content=content, append=True)


def list_directory(
    path: str = ".",
    recursive: bool = False,
    max_items: int = 100,
) -> Dict[str, Any]:
    """List files and folders in a directory with sizes and last modified timestamps."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return {"success": False, "error": f"Directory not found: {path}"}
        if not p.is_dir():
            return {"success": False, "error": f"Path is not a directory: {path}"}

        entries = []
        iterator = p.rglob("*") if recursive else p.iterdir()

        for count, item in enumerate(iterator):
            if count >= max_items:
                break
            try:
                stat = item.stat()
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size_bytes": stat.st_size if item.is_file() else 0,
                    "relative_path": str(item.relative_to(p)),
                })
            except Exception:
                continue

        return {
            "success": True,
            "path": str(p),
            "total_items_found": len(entries),
            "items": entries,
        }
    except Exception as exc:
        return {"success": False, "error": f"Error listing directory {path}: {exc}"}


def search_in_files(
    directory: str = ".",
    query: str = "",
    extension: Optional[str] = None,
    max_results: int = 50,
) -> Dict[str, Any]:
    """Search for a text pattern or word across files in a directory."""
    if not query:
        return {"success": False, "error": "Search query cannot be empty."}

    try:
        root = Path(directory).expanduser().resolve()
        matches = []

        for p in root.rglob("*"):
            if len(matches) >= max_results:
                break
            if not p.is_file():
                continue
            if extension and not p.name.endswith(extension):
                continue
            # Skip binary / large / hidden dirs
            if any(part.startswith(".") for part in p.parts):
                continue
            if p.stat().st_size > 1_000_000:  # Skip files > 1MB
                continue

            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if query.lower() in line.lower():
                        matches.append({
                            "file": str(p.relative_to(root)),
                            "line": lineno,
                            "match": line.strip()[:150],
                        })
                        if len(matches) >= max_results:
                            break
            except Exception:
                continue

        return {
            "success": True,
            "query": query,
            "matches_count": len(matches),
            "matches": matches,
        }
    except Exception as exc:
        return {"success": False, "error": f"Search failed: {exc}"}


def verify_path(path: str) -> Dict[str, Any]:
    """Verify whether a file or directory exists and inspect its status."""
    try:
        p = Path(path).expanduser().resolve()
        exists = p.exists()
        if not exists:
            return {"exists": False, "path": str(p)}

        is_file = p.is_file()
        stat = p.stat()
        return {
            "exists": True,
            "path": str(p),
            "type": "file" if is_file else "directory",
            "size_bytes": stat.st_size if is_file else 0,
            "readable": os.access(p, os.R_OK),
            "writable": os.access(p, os.W_OK),
        }
    except Exception as exc:
        return {"exists": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool Specifications (OpenAI-compatible)
# ---------------------------------------------------------------------------

AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file with line numbers. Use this to inspect code, documents, and configuration files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file to read."},
                    "start_line": {"type": "integer", "description": "Optional start line number (1-based)."},
                    "end_line": {"type": "integer", "description": "Optional end line number (inclusive)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write, overwrite, or append content to a file on disk. For large files (>80 lines), write the first chunk with append=false and subsequent chunks with append=true or append_file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "Text content to write or append."},
                    "overwrite": {"type": "boolean", "default": True, "description": "Whether to overwrite if file exists."},
                    "append": {"type": "boolean", "default": False, "description": "If true, append content to the end of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_file",
            "description": "Append a chunk of text or code to the end of a file. Use this to write large files in multiple parts without hitting token limits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path."},
                    "content": {"type": "string", "description": "Text chunk to append to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a directory with file sizes and relative paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": ".", "description": "Directory path to explore."},
                    "recursive": {"type": "boolean", "default": False, "description": "Whether to list recursively."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Search for a specific keyword, code pattern, or text across files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": ".", "description": "Directory root to search in."},
                    "query": {"type": "string", "description": "The text pattern to search for."},
                    "extension": {"type": "string", "description": "Optional file extension filter (e.g. '.py', '.txt')."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_path",
            "description": "Verify whether a file or directory exists and retrieve its metadata (size, readability, type).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to verify."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_powershell",
            "description": "Execute a Windows PowerShell command or script and return its stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The PowerShell command or script to execute."},
                    "timeout": {"type": "integer", "default": 60, "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_cmd",
            "description": "Execute a Windows Command Prompt (CMD) command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The CMD command to execute."},
                    "timeout": {"type": "integer", "default": 60, "description": "Timeout in seconds."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a terminal command in PowerShell or CMD.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command string to run."},
                    "shell": {"type": "string", "enum": ["powershell", "cmd"], "default": "powershell"},
                    "timeout": {"type": "integer", "default": 60},
                },
                "required": ["command"],
            },
        },
    },
]

TERMINAL_TOOLS = AGENT_TOOLS
DEFAULT_TOOLS = AGENT_TOOLS

# Internal registry mapping tool names to functions
_TOOL_HANDLERS: Dict[str, Callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "append_file": append_file,
    "list_directory": list_directory,
    "search_in_files": search_in_files,
    "verify_path": verify_path,
    "execute_powershell": execute_powershell,
    "execute_cmd": execute_cmd,
    "execute_command": execute_command,
}


def execute_tool(name: str, arguments: Union[Dict[str, Any], str]) -> str:
    """Execute any tool by name with arguments and return a formatted JSON string."""
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
        return json.dumps({"error": f"Execution of {name} failed: {exc}"}, ensure_ascii=False)


def register_tool(name: str, handler: Callable, schema: Optional[Dict[str, Any]] = None):
    """Register a custom tool handler."""
    _TOOL_HANDLERS[name] = handler
    if schema:
        AGENT_TOOLS.append(schema)
