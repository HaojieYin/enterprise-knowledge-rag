"""单独测试重排 API：验证返回格式是否正确"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from app.services.rerank import rerank

query = "公司晋升需要满足哪些条件？"
documents = [
    "员工报销超过5000元需财务总监审批。",
    "员工请假超过3天需部门总监审批。",
    "员工晋升需满足本岗位工作满1年，且最近一次绩效评级为B+以上。",
    "公司为员工缴纳五险一金。",
    "员工每月有两天带薪病假。",
]

print("query:", query)
print("documents 数量:", len(documents))
print()

indices = rerank(query, documents, top_n=3)

print("重排后最相关的前 3 个（下标）:", indices)
print()
print("重排后的顺序：")
for i in indices:
    print(f"  [{i}] {documents[i]}")
