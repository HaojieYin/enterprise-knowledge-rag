"""FastAPI 应用入口：创建应用实例，挂载各个接口和前端页面"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT
from app.routers import chat, conversations, documents, rag
from app.services import db

# 创建 FastAPI 应用实例
app = FastAPI(
    title="企业知识库 RAG 系统",
    description="基于 LangChain + DeepSeek 的企业知识库问答系统",
    version="0.2.0",
)

# 挂载各个后端接口
app.include_router(chat.router)
app.include_router(rag.router)
app.include_router(documents.router)
app.include_router(conversations.router)

# 启动时初始化数据库（创建会话表、消息表）
db.init_db()


@app.get("/health")
def health():
    """健康检查接口"""
    return {"status": "ok"}


# 最后挂载前端静态文件（访问 / 就能看到网页）
app.mount(
    "/",
    StaticFiles(directory=str(PROJECT_ROOT / "frontend"), html=True),
    name="frontend",
)
