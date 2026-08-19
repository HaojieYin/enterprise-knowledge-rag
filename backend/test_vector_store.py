"""测试脚本：验证「文档 → 向量化 → 存储 → 语义检索」全流程"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.services.document import process_document
from app.services.vector_store import add_documents, reset_vector_store, search

DATA_DIR = Path(__file__).resolve().parent / "data"

# 0. 先清空旧数据，避免重复
print("【0】清空旧向量库...")
reset_vector_store()

# 1. 加载并切分文档（chunk_size 调小，让每个「章节」成为独立的一块）
print("【1】加载文档并切分...")
all_chunks = []
for file_name in ["employee_handbook.txt", "it_policy.md"]:
    chunks = process_document(DATA_DIR / file_name, chunk_size=150, chunk_overlap=30)
    all_chunks.extend(chunks)
print(f"    共切分出 {len(all_chunks)} 个文本块\n")

# 2. 向量化并写入向量库（这里会调用硅基流动的 Embedding 接口）
print("【2】向量化并写入向量库...")
add_documents(all_chunks)
print("    写入完成！\n")

# 3. 语义检索测试
print("【3】语义检索测试（看看能不能找到正确的内容）：\n")
questions = [
    "报销超过5000元需要谁审批？",
    "公司密码多久需要修改一次？",
    "晋升需要满足什么条件？",
]
for q in questions:
    print(f"问题：{q}")
    results = search(q, k=2)
    for i, doc in enumerate(results, 1):
        print(f"  命中{i} [来源:{doc.metadata['source']}] {doc.page_content[:50]}...")
    print()
