"""Example 9 — OpenAI-compatible Tool Calling via HTTP API.

Demonstrates sending a request with `tools` to the local DeepSeek server
and receiving a standard OpenAI `tool_calls` response.

Prerequisites:
    Start the server in another terminal first:
        python app.py

Run this script:
    python examples/09_server_tool_calling.py
"""

import json
import httpx
from deepseek.tools import TERMINAL_TOOLS

SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"

payload = {
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user",
            "content": "Use powershell to check the current directory path.",
        }
    ],
    "tools": TERMINAL_TOOLS,
}

print("Sending request to server with tools...")
try:
    response = httpx.post(SERVER_URL, json=payload, timeout=60.0)
    data = response.json()
    print("Response Status:", response.status_code)

    choice = data.get("choices", [{}])[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message", {})

    print(f"Finish Reason: {finish_reason}")
    if message.get("tool_calls"):
        print("\n🔧 DeepSeek requested tool call(s):")
        for tc in message["tool_calls"]:
            fn = tc["function"]
            print(f"  - Function: {fn['name']}")
            print(f"  - Arguments: {fn['arguments']}")
    elif message.get("content"):
        print("\nAssistant Content:", message["content"])

except httpx.ConnectError:
    print("\nCould not connect to server. Please start the server first with:")
    print("    python app.py")
