"""Autonomous Tool-Calling Agent for DeepSeek.

Allows DeepSeek to execute commands in terminal, PowerShell, CMD, or custom Python
functions in an iterative reasoning loop until the goal is accomplished.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .client import DeepSeekClient, Reply
from .tools import TERMINAL_TOOLS, execute_tool

TOOL_CALL_REGEX = re.compile(
    r"<tool_call>\s*(?:```(?:json)?\s*)?({[\s\S]*?})(?:\s*```)?\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
UNCLOSED_TOOL_CALL_REGEX = re.compile(
    r"<tool_call>\s*(?:```(?:json)?\s*)?({[\s\S]*?})(?:\s*```)?\s*$",
    re.DOTALL | re.IGNORECASE,
)


def extract_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract tool calls from text, supporting XML <tool_call>, DeepSeek internal DSML, and JSON blocks.

    Returns:
        (cleaned_text, list_of_tool_calls)
    """
    tool_calls = []

    # 1. Standard XML <tool_call>
    matches = list(TOOL_CALL_REGEX.finditer(text))
    if not matches:
        matches = list(UNCLOSED_TOOL_CALL_REGEX.finditer(text))

    if matches:
        for m in matches:
            raw_json = m.group(1).strip()
            try:
                data = json.loads(raw_json)
                name = data.get("name")
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "name": name,
                    "arguments": args,
                    "raw": raw_json,
                })
            except Exception:
                continue

    # 2. DeepSeek native DSML format (e.g. <DSML | | calls | | > <"DSML | | invoke name="..." ...>)
    if not tool_calls and "DSML" in text:
        dsml_invoke_pattern = re.compile(
            r'<"?DSML\s*\|\s*\|\s*invoke\s+name="([^"]+)"[^>]*>([\s\S]*?)(?:<\/?\s*"?DSML\s*\|\s*\|\s*invoke|\Z)',
            re.IGNORECASE,
        )
        for m in dsml_invoke_pattern.finditer(text):
            # Ensure the invoke block was properly closed, otherwise it was truncated mid-stream
            full_match = m.group(0)
            if not re.search(r'DSML\s*\|\s*\|\s*invoke\s*\|\s*\/?>|<\/?\s*"?DSML\s*\|\s*\|\s*invoke', full_match[10:], re.IGNORECASE):
                continue

            tool_name = m.group(1).strip()
            body = m.group(2)
            args = {}
            param_match = re.search(
                r'parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)(?:<\/\s*\|\s*\|\s*DSML|<\s*DSML|\Z)',
                body,
                re.IGNORECASE,
            )
            if param_match:
                p_name = param_match.group(1)
                raw_val = param_match.group(2).strip()
                raw_val = re.sub(r'</?\s*\|.*$', '', raw_val).strip()
                try:
                    parsed = json.loads(raw_val)
                    if isinstance(parsed, dict):
                        args = parsed
                    else:
                        args = {p_name: parsed}
                except Exception:
                    # If parameter is "arguments", it MUST be valid JSON; if JSON failed, it was truncated
                    if p_name.lower() == "arguments":
                        continue
                    args = {p_name: raw_val}

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "name": tool_name,
                "arguments": args,
                "raw": m.group(0),
            })

    # 3. Fallback: Direct JSON code block containing tool calling signature
    if not tool_calls and "```json" in text:
        json_blocks = re.findall(r"```json\s*({[\s\S]*?})\s*```", text, re.IGNORECASE)
        for jb in json_blocks:
            try:
                data = json.loads(jb.strip())
                if isinstance(data, dict) and "name" in data and ("arguments" in data or "command" in data):
                    name = data.get("name")
                    args = data.get("arguments") or {k: v for k, v in data.items() if k != "name"}
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "name": name,
                        "arguments": args,
                        "raw": jb.strip(),
                    })
            except Exception:
                continue

    # Clean text from all XML tags and DSML artifacts (including truncated unclosed blocks)
    clean_text = TOOL_CALL_REGEX.sub("", text)
    clean_text = UNCLOSED_TOOL_CALL_REGEX.sub("", clean_text)
    clean_text = re.sub(r"<tool_call>[\s\S]*$", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"</?tool_call>", "", clean_text, flags=re.IGNORECASE)
    # Clean DSML blocks (both closed and truncated)
    clean_text = re.sub(
        r'<\s*"?\s*DSML\s*\|\s*\|\s*calls[\s\S]*?(?:<\s*"?\s*DSML\s*\|\s*\|\s*calls\s*\|\s*\/?>|\Z)',
        "",
        clean_text,
        flags=re.IGNORECASE,
    )
    clean_text = re.sub(r'<\s*"?\s*DSML[\s\S]*$', "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"DSML\s*\|[\s\S]*$", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"</?\s*\|\s*\|\s*DSML[\s\S]*?>?", "", clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()

    return clean_text, tool_calls


def is_truncated_tool_call(text: str) -> bool:
    """Detect if the model started emitting a <tool_call>, DSML, or JSON tool call that got cut off by token limits."""
    if not text:
        return False
    _, calls = extract_tool_calls(text)
    if calls:
        return False

    lower = text.lower()
    if "<tool_call>" in lower:
        return True
    if "dsml" in text and ("invoke" in lower or "calls" in lower or "parameter" in lower):
        return True
    # Check if an unclosed ```json block at the end looks like a tool call
    if text.count("```") % 2 == 1 and "```json" in lower:
        last_fence = lower.rfind("```json")
        tail = lower[last_fence:]
        if '"name"' in tail and ('"arguments"' in tail or '"path"' in tail or '"command"' in tail):
            return True
    return False


def is_truncated_response(text: str) -> bool:
    """Detect if a long normal text/code response was cut off mid-generation due to output token limits."""
    if not text:
        return False
    trimmed = text.rstrip()
    if not trimmed:
        return False

    # 1. Unclosed markdown code fence (odd number of ```)
    if trimmed.count("```") % 2 == 1:
        return True

    # 2. Long response (>2000 chars) ending abruptly mid-sentence or mid-expression
    if len(trimmed) > 2000:
        # Characters that naturally conclude a complete response
        natural_endings = (
            ".", "!", "?", "؟", ":", ";", "؛", "`", '"', "'",
            ")", "]", "}", ">", "*", "-", "—", "/",
        )
        # Characters that strongly indicate truncation at the end of a long message
        truncation_endings = (",", "،", "(", "[", "{", "=", "+", "\\", "|", "&")
        if trimmed.endswith(truncation_endings):
            return True
        if not trimmed.endswith(natural_endings):
            # Ended on a bare letter or digit after a huge output
            return True

    return False


def stitch_continuation(prev_text: str, next_text: str) -> str:
    """Seamlessly merge a continuation chunk onto a previously truncated response."""
    if not prev_text:
        return next_text
    if not next_text:
        return prev_text

    # If prev_text ended inside an open code block (odd number of ```)
    # and the model started next_text by re-opening ```lang\n, strip the redundant opening fence
    if prev_text.count("```") % 2 == 1:
        stripped_next = next_text.lstrip()
        fence_match = re.match(r"^```[a-zA-Z0-9_-]*\r?\n", stripped_next)
        if fence_match:
            next_text = stripped_next[fence_match.end():]

    # Avoid double-spacing or missing newline when joining
    if prev_text.endswith("\n") or next_text.startswith("\n"):
        return prev_text + next_text
    return prev_text + "\n" + next_text


def is_promising_action_without_call(text: str) -> bool:
    """Check if the agent stopped prematurely promising an action without emitting a tool call."""
    if not text:
        return False

    trimmed = text.strip()
    lower = trimmed.lower()

    # Suspicious sentence or block endings indicating intent to follow up
    suspicious_endings = (":", "...", "،", "->", "=>", "التالي:", "الآتي:", "خطوة:")
    if any(trimmed.endswith(end) for end in suspicious_endings):
        return True

    # Common action promise cues in Arabic & English
    cues = [
        "سأقوم", "سأبدأ", "سأفحص", "سأتحقق", "دعني أتحقق", "دعني أفحص",
        "سأقرأ", "سأقوم بقراءة", "سأعدل", "سأقوم بتعديل", "سأكتب", "سأنشئ",
        "سأشغل", "سأنفذ", "سأقوم بتشغيل", "سأقوم بتنفيذ", "الخطوة التالية",
        "الخطوة القادمة", "سأحاول", "سأجرب", "دعني أقرأ", "سأعيد المحاولة",
        "سأصحح", "سأصلح", "سأستعرض", "سأستخرج", "سأبحث", "دعنا",
        "i will", "let me", "i'll", "going to", "next step", "next, i will",
        "now i will", "i need to", "we need to"
    ]

    tail = lower[-300:] if len(lower) > 300 else lower
    for cue in cues:
        if cue in tail:
            return True

    if len(trimmed) < 300:
        for cue in cues:
            if cue in lower:
                return True

    return False


def load_instructions_md(custom_path: Optional[str] = None) -> str:
    """Load the natural markdown developer instructions."""
    from pathlib import Path
    candidates = [
        custom_path,
        Path("instructions.md"),
        Path(__file__).resolve().parent / "instructions.md",
        Path(r"C:\AllProject\AI\instructions.md"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            try:
                return Path(c).read_text(encoding="utf-8").strip()
            except Exception:
                pass
    return ""


def build_tools_instruction(tools: List[Dict[str, Any]]) -> str:
    """Format tools schema into system instructions that DeepSeek understands."""
    md = load_instructions_md()
    if md:
        return md
    tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
    return (
        "You have access to the following tools to assist with tasks:\n"
        f"```json\n{tools_json}\n```\n\n"
        "To invoke a tool, you MUST reply with a tool call formatted strictly in XML:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {"<param>": "<value>"}}\n'
        "</tool_call>\n\n"
        "You may provide a short explanation or plan before calling the tool. "
        "After the tool executes, its result will be provided to you so you can continue. "
        "When no more tool calls are needed, respond normally with the final answer."
    )


@dataclass
class ToolExecution:
    name: str
    arguments: Dict[str, Any]
    result: str


@dataclass
class AgentResult:
    text: str
    conversation_id: str
    tool_executions: List[ToolExecution] = field(default_factory=list)
    reasoning: str = ""

    def __str__(self) -> str:
        return self.text


class DeepSeekAgent:
    """Agent that runs an autonomous tool-calling loop using DeepSeek."""

    def __init__(
        self,
        client: Optional[DeepSeekClient] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 15,
        verbose: bool = True,
    ):
        self.client = client or DeepSeekClient()
        self.tools = tools if tools is not None else TERMINAL_TOOLS
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.verbose = verbose

    def run(
        self,
        user_prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> AgentResult:
        """Run the user request through the agent loop.

        Args:
            user_prompt: The user query or task.
            conversation_id: Optional ID to resume an existing conversation.
            model: Model name ('default' or 'expert').
            thinking: Enable DeepThink reasoning.
            search: Enable web search.

        Returns:
            AgentResult: Contains final text, conversation ID, and executed tools history.
        """
        tool_instructions = build_tools_instruction(self.tools)
        base_prompt = ""
        if self.system_prompt:
            base_prompt += f"{self.system_prompt}\n\n"
        if not conversation_id:
            base_prompt += f"{tool_instructions}\n\n---\n\n{user_prompt}"
        else:
            base_prompt += user_prompt

        current_prompt = base_prompt
        current_cid = conversation_id
        all_executions: List[ToolExecution] = []
        final_text = ""
        final_reasoning = ""

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"[Agent] Step {step}/{self.max_steps}...")

            reply = self.client.chat(
                current_prompt,
                conversation_id=current_cid,
                model=model if step == 1 and not current_cid else None,
                thinking=thinking,
                search=search,
            )
            current_cid = reply.conversation_id
            if reply.reasoning:
                final_reasoning = reply.reasoning

            clean_text, calls = extract_tool_calls(reply.text)

            if not calls:
                # If agent stopped with an action promise without emitting a tool call, nudge to continue
                if step < self.max_steps and is_promising_action_without_call(reply.text):
                    if self.verbose:
                        print("[Agent] Detected action promise without tool call, prompting continuation...")
                    current_prompt = (
                        "استمر فوراً ونفذ هذه الخطوة بالأدوات المناسبة (مثل execute_powershell أو read_file أو write_file أو list_directory) "
                        "عبر <tool_call> مباشرة في ردك. لا تتوقف حتى تنجز كامل الهدف المطلوب وتتحقق من صحة النتيجة 100%."
                    )
                    continue

                # No more tools requested - task completed!
                final_text = clean_text or reply.text
                if self.verbose:
                    print("[Agent] Completed task.")
                break

            # Process each requested tool call
            results_feedback = []
            for call in calls:
                tool_name = call["name"]
                tool_args = call["arguments"]
                if self.verbose:
                    print(f"[Agent] 🔧 Tool Call: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

                tool_output = execute_tool(tool_name, tool_args)
                if self.verbose:
                    preview = tool_output[:200] + ("..." if len(tool_output) > 200 else "")
                    print(f"[Agent] 📋 Output: {preview}")

                execution = ToolExecution(name=tool_name, arguments=tool_args, result=tool_output)
                all_executions.append(execution)
                results_feedback.append(
                    f"هذه نتيجة تنفيذ الأداة {tool_name}:\n```\n{tool_output}\n```\n"
                    "يرجى مواصلة العمل والتحقق من النتيجة بالأدوات حتى تكتمل المهمة بالكامل، ولا تتوقف حتى تتأكد من صحتها 100%."
                )

            # Feed tool execution output back to DeepSeek naturally
            feedback_str = "\n\n".join(results_feedback)
            current_prompt = feedback_str
        else:
            final_text = clean_text or "Max steps reached without final reply."

        return AgentResult(
            text=final_text,
            conversation_id=current_cid or "",
            tool_executions=all_executions,
            reasoning=final_reasoning,
        )
