"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable, List, Optional

from deepseek.agent import build_tools_instruction, extract_tool_calls
from deepseek.client import RESPONSE, THINK

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage], tools: Optional[List[dict]] = None) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    If tools are provided, system tool instructions are prepended.
    Handles user, assistant (including tool_calls), system, and tool roles.
    """
    lines: List[str] = []

    if tools:
        lines.append(build_tools_instruction(tools))

    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        content_text = _text_of(m.content)

        if m.role == "assistant" and getattr(m, "tool_calls", None):
            calls_xml = []
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name")
                fn_args = fn.get("arguments", "{}")
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        pass
                calls_xml.append(
                    f'<tool_call>\n{json.dumps({"name": fn_name, "arguments": fn_args}, ensure_ascii=False)}\n</tool_call>'
                )
            call_block = "\n".join(calls_xml)
            if content_text:
                lines.append(f"{label}: {content_text}\n{call_block}")
            else:
                lines.append(f"{label}: {call_block}")
        elif m.role in ("tool", "function"):
            tool_name = getattr(m, "name", None) or getattr(m, "tool_call_id", None) or "tool"
            lines.append(f"Tool Output ({tool_name}):\n{content_text}")
        else:
            lines.append(f"{label}: {content_text}")

    # If it's a lone user message without tools, send verbatim
    if len(messages) == 1 and messages[0].role == "user" and not tools:
        return _text_of(messages[0].content)

    if not lines:
        return ""

    if not lines[-1].endswith("Assistant:"):
        lines.append("Assistant:")
    return "\n\n".join(lines)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(
    model: str,
    content: str,
    prompt: str,
    conversation_id: str = None,
    reasoning: str = "",
) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    Extracts any `<tool_call>` blocks and outputs them in standard OpenAI
    `tool_calls` format with `finish_reason: "tool_calls"`.
    """
    pt, ct = _est_tokens(prompt), _est_tokens(reasoning + content)
    clean_text, tool_calls = extract_tool_calls(content)

    message: dict = {"role": "assistant"}

    if tool_calls:
        formatted_calls = []
        for tc in tool_calls:
            args = tc.get("arguments", {})
            args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
            formatted_calls.append({
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": args_str,
                },
            })
        message["tool_calls"] = formatted_calls
        message["content"] = clean_text if clean_text else None
        finish_reason = "tool_calls"
    else:
        message["content"] = content
        finish_reason = "stop"

    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(model: str, stream, has_tools: bool = False) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion."""
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role
    yield frame({"role": "assistant", "content": ""})

    accumulated_content = []
    for kind, d in stream.events():
        if not d:
            continue
        if kind == RESPONSE:
            accumulated_content.append(d)
            if not has_tools:
                yield frame({"content": d})
        elif kind == THINK:
            yield frame({"reasoning_content": d})

    full_text = "".join(accumulated_content)
    conversation_id = getattr(stream, "conversation_id", None)

    if has_tools:
        clean_text, tool_calls = extract_tool_calls(full_text)
        if tool_calls:
            if clean_text:
                yield frame({"content": clean_text})
            formatted_calls = []
            for i, tc in enumerate(tool_calls):
                args = tc.get("arguments", {})
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, (dict, list)) else str(args)
                formatted_calls.append({
                    "index": i,
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": args_str,
                    },
                })
            yield frame({"tool_calls": formatted_calls})
            yield frame({}, finish="tool_calls", extra={"conversation_id": conversation_id})
            yield "data: [DONE]\n\n"
            return
        else:
            if full_text:
                yield frame({"content": full_text})

    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"


def stream_error(message: str, err_type: str = "server_error") -> Iterable[str]:
    """SSE lines reporting a failure after the stream has started."""
    err = {"error": {"message": message, "type": err_type}}
    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
