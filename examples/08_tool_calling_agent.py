"""Example 8 — Autonomous Tool-Calling Agent with Terminal / Shell access.

DeepSeek decides when to execute terminal commands (PowerShell / CMD), receives
the command output, and reasons to accomplish the user's task.

Run it from the project root:

    python examples/08_tool_calling_agent.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from deepseek import DeepSeekAgent, DeepSeekClient

client = DeepSeekClient()

# Initialize the agent. By default, it has access to:
# - execute_command (PowerShell or CMD)
# - execute_powershell
# - execute_cmd
agent = DeepSeekAgent(client=client, verbose=True)

query = "استخدم powershell لمعرفة اسم الجهاز (Computer Name) وإصدار نظام التشغيل، ثم لخص النتيجة بالعربية"
print(f"User Task: {query}\n" + "=" * 50)

result = agent.run(query)

print("\n" + "=" * 50)
print("Final Answer from DeepSeek:\n")
print(result.text)

print(f"\nTotal tool executions: {len(result.tool_executions)}")
for i, exec_info in enumerate(result.tool_executions, 1):
    print(f"  {i}. Tool: {exec_info.name} | Args: {exec_info.arguments}")

client.close()
