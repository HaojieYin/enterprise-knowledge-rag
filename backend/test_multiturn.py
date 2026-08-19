"""测试脚本：验证多轮对话（问题改写 + 历史管理）"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

BASE_URL = "http://127.0.0.1:8000"


def chat(question, history=None):
    payload = {"question": question}
    if history:
        payload["history"] = history
    return httpx.post(f"{BASE_URL}/api/rag", json=payload, timeout=60).json()


# 第一轮：正常提问
print("【第一轮】问：报销超过5000元需要谁审批？")
data1 = chat("报销超过5000元需要谁审批？")
print(f"回答：{data1['answer']}\n")

# 第二轮：指代追问「那请假超过3天呢？」
history = [
    {"role": "user", "content": "报销超过5000元需要谁审批？"},
    {"role": "assistant", "content": data1["answer"]},
]
print("【第二轮】问：那请假超过3天呢？")
data2 = chat("那请假超过3天呢？", history=history)
print(f"改写为：{data2.get('rewritten_query')}")
print(f"回答：{data2['answer']}\n")

# 第三轮：连续追问「那晋升的条件呢？」
history2 = history + [
    {"role": "user", "content": "那请假超过3天呢？"},
    {"role": "assistant", "content": data2["answer"]},
]
print("【第三轮】问：那晋升的条件呢？")
data3 = chat("那晋升的条件呢？", history=history2)
print(f"改写为：{data3.get('rewritten_query')}")
print(f"回答：{data3['answer']}")
