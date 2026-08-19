"""RAG 问答接口：检索 + 生成（对话历史持久化到 SQLite）"""
import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.routers.auth import get_current_user
from app.services import db
from app.services.rag import ask, ask_stream

router = APIRouter(prefix="/api", tags=["rag"])


class RagRequest(BaseModel):
    question: str
    conversation_id: str | None = None  # 会话 ID；为空则自动新建会话


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
    conversation_id: str  # 本轮对话所属的会话 ID


def _resolve_conversation(conversation_id: str | None, question: str, user_id: str) -> str:
    """确定会话：有且属于当前用户则复用，否则新建（标题取问题前 20 字）"""
    if conversation_id and db.conversation_exists(conversation_id, user_id):
        return conversation_id
    title = question[:20] + ("..." if len(question) > 20 else "")
    return db.create_conversation(title, user_id)


@router.post("/rag", response_model=RagResponse)
def rag(req: RagRequest, user: dict = Depends(get_current_user)):
    """RAG 问答接口：POST /api/rag（一次性返回完整回答）"""
    conversation_id = _resolve_conversation(req.conversation_id, req.question, user["id"])
    history = db.get_messages(conversation_id)

    result = ask(req.question, history=history)

    # 持久化：把这一轮问答写进数据库（含检索来源，切回历史时能还原引用卡片）
    db.add_message(conversation_id, "user", req.question)
    db.add_message(conversation_id, "assistant", result["answer"], sources=result["sources"])

    return RagResponse(
        answer=result["answer"],
        sources=result["sources"],
        rewritten_query=result["rewritten_query"],
        searched=result["searched"],
        conversation_id=conversation_id,
    )


@router.post("/rag/stream")
def rag_stream(req: RagRequest, user: dict = Depends(get_current_user)):
    """RAG 问答接口（流式）：POST /api/rag/stream，返回 NDJSON 流"""
    conversation_id = _resolve_conversation(req.conversation_id, req.question, user["id"])
    history = db.get_messages(conversation_id)

    def generate():
        answer_text = ""
        sources = []
        for item in ask_stream(req.question, history=history):
            if item["type"] == "meta":
                item["conversation_id"] = conversation_id  # 把会话 ID 返回给前端
                sources = item["sources"]  # 记下来源，流结束后一起写库
            elif item["type"] == "delta":
                answer_text += item["data"]
            yield json.dumps(item, ensure_ascii=False) + "\n"
        # 流结束后，把这一轮问答写进数据库（含来源，实现持久化）
        db.add_message(conversation_id, "user", req.question)
        db.add_message(conversation_id, "assistant", answer_text, sources=sources)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
