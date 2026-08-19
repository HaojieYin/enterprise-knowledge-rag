"""测试引用高亮：验证回答里带引用标记 [n]，且 sources 返回带编号的来源"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

resp = httpx.post(
    "http://127.0.0.1:8000/api/rag",
    json={"question": "报销超过5000元需要谁审批？"},
    timeout=60,
).json()

print("回答：")
print(resp["answer"])
print()
print("引用来源：")
for s in resp["sources"]:
    print(f"  [{s['id']}] {s['source']}: {s['snippet']}")
