"""测试对话历史持久化：会话 CRUD + 多轮对话 + sources 存库（需登录）"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import httpx

BASE = "http://127.0.0.1:8000"
# trust_env=False：访问 localhost 不走系统代理；timeout 拉长（RAG 要调多次 LLM，较慢）
client = httpx.Client(timeout=120, trust_env=False)


def register_or_login(username: str, password: str) -> str:
    """注册；用户名已存在就改登录。返回 token"""
    r = client.post(f"{BASE}/api/auth/register", json={"username": username, "password": password})
    if r.status_code == 400:
        r = client.post(f"{BASE}/api/auth/login", json={"username": username, "password": password})
    return r.json().get("token")


token = register_or_login("tester", "123456")
headers = {"Authorization": f"Bearer {token}"}

# 1. 第一轮提问（不传 conversation_id，自动新建会话）
r = client.post(f"{BASE}/api/rag", json={"question": "员工每年有多少天年假？"}, headers=headers)
d = r.json()
cid = d["conversation_id"]
print(f"【1】第一轮（自动新建会话 {cid[:8]}...）")
print(f"  回答: {d['answer']}")

# 2. 第二轮追问（传 conversation_id，复用同一会话）
r = client.post(
    f"{BASE}/api/rag",
    json={"question": "那请假超过几天需要总监审批？", "conversation_id": cid},
    headers=headers,
)
d2 = r.json()
print(f"\n【2】第二轮（追问，复用会话）")
print(f"  回答: {d2['answer']}")
print(f"  改写后: {d2.get('rewritten_query')}")
print(f"  会话ID一致: {d2['conversation_id'] == cid}")

# 3. 查历史（应 4 条：2 问 2 答，assistant 带 sources）
msgs = client.get(f"{BASE}/api/conversations/{cid}/messages", headers=headers).json()
print(f"\n【3】历史消息（共 {len(msgs)} 条）:")
for m in msgs:
    print(f"  [{m['role']}] sources={len(m['sources'])}  {m['content'][:40]}")

# 4. 会话列表
convs = client.get(f"{BASE}/api/conversations", headers=headers).json()
print(f"\n【4】会话列表（共 {len(convs)} 个）")

# 5. 删除会话
client.delete(f"{BASE}/api/conversations/{cid}", headers=headers)
convs2 = client.get(f"{BASE}/api/conversations", headers=headers).json()
print(f"【5】删除后列表（共 {len(convs2)} 个）")

client.close()
