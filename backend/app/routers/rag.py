"""RAG 问答接口：检索 + 生成（支持多轮对话）"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.rag import ask, ask_stream

router = APIRouter(prefix="/api", tags=["rag"])


class ChatMessage(BaseModel):
    """一条对话消息"""
    role: str  # "user" 或 "assistant"
    content: str


class RagRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []  # 对话历史（多轮对话时传入，首轮可不传）


class SourceInfo(BaseModel):
    """一条引用来源：编号 + 文件名 + 原文片段"""
    id: int
    source: str
    snippet: str


class RagResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    rewritten_query: str | None = None  # 发生了问题改写时返回改写后的问题
    searched: bool = True  # 是否检索了知识库（Agent 查询路由的结果）


@router.post("/rag", response_model=RagResponse)
def rag(req: RagRequest):
    """RAG 问答接口：POST /api/rag"""
    history = [m.model_dump() for m in req.history]
    result = ask(req.question, history=history)
    return RagResponse(
        answer=result["answer"],
        sources=result["sources"],
        rewritten_query=result["rewritten_query"],
        searched=result["searched"],
    )


@router.post("/rag/stream")
def rag_stream(req: RagRequest):
    """RAG 问答接口（流式）：POST /api/rag/stream，返回 NDJSON 流"""
    history = [m.model_dump() for m in req.history]

    def generate():
        for item in ask_stream(req.question, history=history):
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
