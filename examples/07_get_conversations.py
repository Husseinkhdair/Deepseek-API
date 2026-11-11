"""Example 7 — Fetch existing conversations and their messages.

Run it from the project root:

    python examples/07_get_conversations.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from deepseek import DeepSeekClient

client = DeepSeekClient()

# 1. Fetch conversations list
conversations = client.get_conversations(count=20)
print(f"Total conversations found: {len(conversations)}\n")

for conv in conversations:
    print(f"📌 [{conv.get('title')}]")
    print(f"   ID: {conv.get('id')}")
    print(f"   Model: {conv.get('model_type')}")
    print(f"   Updated: {conv.get('updated_at')}")

    # 2. Fetch messages for this conversation (optional)
    messages = client.get_messages(conv['id'])
    print(f"   Messages count: {len(messages)}")

    # Print the last message or preview
    if messages:
        last_msg = messages[-1]
        role = last_msg.get("role")
        fragments = last_msg.get("fragments") or []
        content = " ".join(f.get("content", "") for f in fragments if f.get("type") in ("RESPONSE", "REQUEST"))
        preview = (content[:80] + "...") if len(content) > 80 else content
        print(f"   Last ({role}): {preview}")
    print("-" * 50)

# 3. Resume / continue a specific conversation by its ID:
if conversations:
    target_id = conversations[0]["id"]
    print(f"\nContinuing conversation {target_id}...")
    reply = client.chat("Hello! What were we talking about?", conversation_id=target_id)
    print("DeepSeek:", reply.text)

client.close()
