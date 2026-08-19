"""测试用户登录（JWT）+ 多用户会话隔离"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import httpx

BASE = "http://127.0.0.1:8000"
client = httpx.Client(timeout=120, trust_env=False)


def register_or_login(username: str, password: str) -> str:
    """注册；如果用户名已存在就改登录。返回 token"""
    r = client.post(f"{BASE}/api/auth/register", json={"username": username, "password": password})
    if r.status_code == 400:
        r = client.post(f"{BASE}/api/auth/login", json={"username": username, "password": password})
    return r.json().get("token")


# 1. 未登录访问会话列表 -> 应 401
r = client.get(f"{BASE}/api/conversations")
print("【1】未登录访问会话列表 ->", r.status_code, "(应为 401)")

# 2. 注册/登录 alice
token_a = register_or_login("alice", "123456")
headers_a = {"Authorization": f"Bearer {token_a}"}
print("【2】alice token 前 12 位:", token_a[:12], "...")

# 3. alice 提问（带 token，自动建会话）
r = client.post(f"{BASE}/api/rag", json={"question": "员工每年有多少天年假？"}, headers=headers_a)
d = r.json()
print("【3】alice 提问 ->", d["answer"])
cid = d["conversation_id"]

# 4. alice 会话列表
r = client.get(f"{BASE}/api/conversations", headers=headers_a)
convs_a = r.json()
print("【4】alice 会话列表 ->", len(convs_a), "个:", [c["title"] for c in convs_a])

# 5. 注册/登录 bob，看他的会话列表（应 0，多用户隔离）
token_b = register_or_login("bob", "123456")
headers_b = {"Authorization": f"Bearer {token_b}"}
r = client.get(f"{BASE}/api/conversations", headers=headers_b)
convs_b = r.json()
print("【5】bob 会话列表 ->", len(convs_b), "个 (应为 0，看不到 alice 的)")

# 6. bob 尝试访问 alice 的会话 -> 应 404（越权访问被拦截）
r = client.get(f"{BASE}/api/conversations/{cid}/messages", headers=headers_b)
print("【6】bob 访问 alice 的会话 ->", r.status_code, "(应为 404)")

# 7. alice 登录接口验证
r = client.post(f"{BASE}/api/auth/login", json={"username": "alice", "password": "123456"})
print("【7】alice 登录 ->", r.status_code, r.json().get("username"))

# 8. 错误密码 -> 应 401
r = client.post(f"{BASE}/api/auth/login", json={"username": "alice", "password": "wrong"})
print("【8】错误密码登录 ->", r.status_code, "(应为 401)")

# 清理：删掉 alice 的测试会话
client.delete(f"{BASE}/api/conversations/{cid}", headers=headers_a)
client.close()
print("\n✅ 测试完成")
