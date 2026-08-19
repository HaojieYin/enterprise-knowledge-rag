"""测试脚本：验证完整 RAG 问答（检索 + 生成）"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app.services.rag import ask

questions = [
    "报销超过5000元需要谁审批？",
    "公司密码多久需要修改一次？",
    "晋升需要满足什么条件？",
    "公司年会是什么时候举办？",  # 故意问一个资料里没有的问题，验证会不会编造
]

for q in questions:
    print(f"问题：{q}")
    result = ask(q, k=4)
    print(f"回答：{result['answer']}")
    print(f"来源：{result['sources']}")
    print("-" * 50)
