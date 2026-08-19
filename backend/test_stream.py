"""测试流式输出：逐行打印后端返回的 NDJSON 流"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

print("正在流式请求 /api/rag/stream ...\n")
with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/api/rag/stream",
    json={"question": "报销超过5000元需要谁审批？"},
    timeout=60,
) as resp:
    for line in resp.iter_lines():
        if line:
            print(line)
