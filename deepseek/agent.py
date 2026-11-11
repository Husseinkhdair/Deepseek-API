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

TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*({[\s\S]*?})\s*</tool_call>", re.DOTALL)


def extract_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract <tool_call> JSON blocks from text.

    Returns:
        (cleaned_text, list_of_tool_calls)
    """
    matches = list(TOOL_CALL_REGEX.finditer(text))
    if not matches:
        return text, []

    tool_calls = []
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

    clean_text = TOOL_CALL_REGEX.sub("", text).strip()
    return clean_text, tool_calls


def build_tools_instruction(tools: List[Dict[str, Any]]) -> str:
    """Format tools schema into system instructions that DeepSeek understands."""
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
        max_steps: int = 10,
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
        base_prompt += f"{tool_instructions}\n\nUser: {user_prompt}\nAssistant:"

        current_prompt = base_prompt
        current_cid = conversation_id
        all_executions: List[ToolExecution] = []
        final_text = ""
        final_reasoning = ""

        for step in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"[Agent] Step {step}...")

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
                results_feedback.append(f"Tool Output ({tool_name}):\n{tool_output}")

            # Feed tool execution output back to DeepSeek
            feedback_str = "\n\n".join(results_feedback)
            current_prompt = f"{clean_text}\n\n{feedback_str}\n\nAssistant:".strip()
        else:
            final_text = clean_text or "Max steps reached without final reply."

        return AgentResult(
            text=final_text,
            conversation_id=current_cid or "",
            tool_executions=all_executions,
            reasoning=final_reasoning,
        )
