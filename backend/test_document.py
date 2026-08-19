"""测试脚本：验证文档加载与切分"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from app.services.document import process_document

# 样本文件放在 backend/data 目录下
DATA_DIR = Path(__file__).resolve().parent / "data"

for file_name in ["employee_handbook.txt", "it_policy.md"]:
    chunks = process_document(
        DATA_DIR / file_name,
        chunk_size=150,   # 调小一点，让短文档也能演示出「切分」效果
        chunk_overlap=30,
    )
    print(f"========== {file_name} 切分出 {len(chunks)} 块 ==========\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- 第 {i} 块 (来源: {chunk.metadata['source']}) ---")
        print(chunk.page_content)
        print()
    print()
