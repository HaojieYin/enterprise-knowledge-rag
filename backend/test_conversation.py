"""测试对话历史持久化：会话 CRUD + 多轮对话存库"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import httpx

BASE = "http://127.0.0.1:8000"
# trust_env=False：访问 localhost 不走系统代理；timeout 拉长（RAG 要调多次 LLM，较慢）
client = httpx.Client(timeout=120, trust_env=False)


def cleanup():
    """清理所有标题为「测试会话」的残留会话"""
    for c in client.get(f"{BASE}/api/conversations").json():
        if c["title"] == "测试会话":
            client.delete(f"{BASE}/api/conversations/{c['id']}")


cleanup()  # 先清掉上次残留

# 1. 第一轮提问（不传 conversation_id，自动新建会话）
r = client.post(f"{BASE}/api/rag", json={"question": "员工每年有多少天年假？"})
d = r.json()
cid = d["conversation_id"]
print(f"【1】第一轮（自动新建会话 {cid[:8]}...）")
print(f"  回答: {d['answer']}")

# 2. 第二轮追问（传 conversation_id，复用同一会话）
r = client.post(f"{BASE}/api/rag", json={"question": "那请假超过几天需要总监审批？", "conversation_id": cid})
d2 = r.json()
print(f"\n【2】第二轮（追问，复用会话）")
print(f"  回答: {d2['answer']}")
print(f"  改写后: {d2.get('rewritten_query')}")
print(f"  会话ID一致: {d2['conversation_id'] == cid}")

# 3. 查历史（应 4 条：2 问 2 答）
msgs = client.get(f"{BASE}/api/conversations/{cid}/messages").json()
print(f"\n【3】历史消息（共 {len(msgs)} 条）:")
for m in msgs:
    print(f"  [{m['role']}] {m['content'][:45]}")

# 4. 会话列表
convs = client.get(f"{BASE}/api/conversations").json()
print(f"\n【4】会话列表（共 {len(convs)} 个）:")
for c in convs:
    print(f"  - {c['title']}  ({c['id'][:8]}...)")

# 5. 删除会话
r = client.delete(f"{BASE}/api/conversations/{cid}")
print(f"\n【5】删除会话 -> {r.json()}")

# 6. 验证删除后列表
convs2 = client.get(f"{BASE}/api/conversations").json()
print(f"【6】删除后列表（共 {len(convs2)} 个）")

client.close()
