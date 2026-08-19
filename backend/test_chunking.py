"""验证「结构感知切分」：对比旧（纯字符）与新（按标题 + 标题继承）的切分效果"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.document import process_document, split_by_structure

DATA_DIR = Path(__file__).resolve().parent / "data"


def old_split(text: str, chunk_size: int = 150) -> list[str]:
    """旧的切分方式：纯按字符 + 中文标点边界"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_text(text)


def show(title, text):
    print(title)
    print("-" * 46)
    print(text)
    print()


file_name = "employee_handbook.txt"
text = (DATA_DIR / file_name).read_text(encoding="utf-8")

# 1. 先看「结构切分」识别出的章节边界
print("=" * 60)
print("第一步：按标题识别章节（split_by_structure）")
print("=" * 60)
for header, body in split_by_structure(text):
    body_preview = body.strip().split("\n")[0][:20] if body.strip() else "(空)"
    print(f"  章节标题: {header or '（无标题开头）'}")
    print(f"  正文首行: {body_preview}")
print()

# 2. 对比旧的纯字符切分
print("=" * 60)
print("旧切分（纯字符 chunk_size=150）：注意标题和正文被拆开、块丢了章节信息")
print("=" * 60)
for i, chunk in enumerate(old_split(text), 1):
    show(f"[旧] 块 {i}", chunk)

# 3. 新的结构切分 + 标题继承
print("=" * 60)
print("新切分（按标题 + 标题继承 chunk_size=150）：每块都带章节标题")
print("=" * 60)
for chunk in process_document(DATA_DIR / file_name, chunk_size=150, chunk_overlap=20):
    show(f"[新] 块（章节: {chunk.metadata['section'] or '无'}）", chunk.page_content)
