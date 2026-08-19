"""测试 Agent 查询路由：问候类问题不检索，知识类问题检索"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

BASE_URL = "http://127.0.0.1:8000"


def chat(question):
    return httpx.post(f"{BASE_URL}/api/rag", json={"question": question}, timeout=60).json()


# 场景 1：简单问候，应该不检索
print("【场景1】问：你好")
data = chat("你好")
print(f"  searched={data['searched']}（预期 False）")
print(f"  回答：{data['answer']}")
print(f"  来源数：{len(data['sources'])}（预期 0）\n")

# 场景 2：感谢，应该不检索
print("【场景2】问：谢谢你的帮助")
data = chat("谢谢你的帮助")
print(f"  searched={data['searched']}（预期 False）")
print(f"  回答：{data['answer']}")
print(f"  来源数：{len(data['sources'])}（预期 0）\n")

# 场景 3：知识类问题，应该检索
print("【场景3】问：报销超过5000元需要谁审批？")
data = chat("报销超过5000元需要谁审批？")
print(f"  searched={data['searched']}（预期 True）")
print(f"  回答：{data['answer']}")
print(f"  来源数：{len(data['sources'])}（预期 > 0）")
