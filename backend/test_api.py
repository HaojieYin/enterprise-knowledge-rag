"""测试脚本：用代码模拟前端，测试后端所有接口"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

BASE_URL = "http://127.0.0.1:8000"

# 1. 首页
r = httpx.get(f"{BASE_URL}/")
print("【首页】", r.json(), "\n")

# 2. 纯大模型问答（无 RAG，用于对比）
r = httpx.post(f"{BASE_URL}/api/chat", json={"question": "你好，介绍一下自己"}, timeout=60)
print("【纯大模型问答】", r.json(), "\n")

# 3. RAG 问答（检索 + 生成）
r = httpx.post(
    f"{BASE_URL}/api/rag",
    json={"question": "报销超过5000元需要谁审批？"},
    timeout=60,
)
print("【RAG 问答】")
print("状态码:", r.status_code)
print("返回:", r.json())
