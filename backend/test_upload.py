"""测试脚本：验证前端页面、健康检查、文档上传接口"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import httpx

BASE_URL = "http://127.0.0.1:8000"

# 1. 测试前端页面
r = httpx.get(f"{BASE_URL}/")
print(f"【前端页面】状态码: {r.status_code}, 内容类型: {r.headers.get('content-type')}")

# 2. 测试健康检查
r = httpx.get(f"{BASE_URL}/health")
print(f"【健康检查】{r.json()}")

# 3. 测试文档上传（上传一份「培训制度」文档）
data_dir = Path(__file__).resolve().parent / "data"
file_path = data_dir / "training_policy.txt"
with open(file_path, "rb") as f:
    r = httpx.post(
        f"{BASE_URL}/api/documents/upload",
        files={"file": ("training_policy.txt", f, "text/plain")},
        timeout=60,
    )
print(f"\n【文档上传】状态码: {r.status_code}")
print(f"返回: {r.json()}")

# 4. 验证新上传的文档能被检索到
r = httpx.post(
    f"{BASE_URL}/api/rag",
    json={"question": "新员工入职需要参加几天培训？"},
    timeout=60,
)
print(f"\n【RAG 验证新文档】状态码: {r.status_code}")
result = r.json()
print(f"回答: {result['answer']}")
print(f"来源: {result['sources']}")
