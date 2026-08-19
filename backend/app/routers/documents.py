"""文档管理接口：上传文档并索引到向量库"""
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import DATA_DIR
from app.services.document import process_document
from app.services.vector_store import add_documents

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档：保存 -> 切分 -> 向量化 -> 写入向量库

    前端用 FormData 上传文件，字段名叫 file
    """
    # 1. 安全处理文件名（只取文件名部分，去掉可能夹带的路径）
    filename = Path(file.filename).name
    file_path = DATA_DIR / filename

    # 2. 保存上传的文件到 data 目录
    content = await file.read()
    file_path.write_bytes(content)

    # 3. 切分 + 写入向量库
    try:
        chunks = process_document(file_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    add_documents(chunks)

    return {
        "filename": filename,
        "chunks": len(chunks),
        "message": f"已成功索引 {len(chunks)} 个文本块",
    }
