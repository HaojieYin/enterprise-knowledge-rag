"""RAG 问答接口：检索 + 生成（对话历史持久化到 SQLite + 用户长期画像）"""
import json
import threading

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.routers.auth import get_current_user
from app.services import db
from app.services.rag import ask, ask_stream, extract_profile

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


def _update_profile_if_needed(user_id: str, question: str, answer: str, profile: str) -> None:
    """这轮聊完后，让大模型判断是否透露了新的个人信息，有就更新画像。

    用 try/except 包住：画像提取失败（比如 LLM 超时）绝不能影响正常回答，
    静默跳过即可——长期记忆是「锦上添花」，不是「生死攸关」。
    """
    try:
        new_profile = extract_profile(question, answer, profile)
        if new_profile:
            db.update_profile(user_id, new_profile)
    except Exception:
        pass


@router.post("/rag", response_model=RagResponse)
def rag(req: RagRequest, user: dict = Depends(get_current_user)):
    """RAG 问答接口：POST /api/rag（一次性返回完整回答）"""
    conversation_id = _resolve_conversation(req.conversation_id, req.question, user["id"])
    history = db.get_messages(conversation_id)
    profile = user.get("profile") or ""

    result = ask(req.question, history=history, profile=profile)

    # 持久化：把这一轮问答写进数据库（含检索来源，切回历史时能还原引用卡片）
    db.add_message(conversation_id, "user", req.question)
    db.add_message(conversation_id, "assistant", result["answer"], sources=result["sources"])

    # 长期记忆：聊完后从这轮对话提取新信息，更新画像（同步，等它做完再返回）
    _update_profile_if_needed(user["id"], req.question, result["answer"], profile)

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
    profile = user.get("profile") or ""

    def generate():
        answer_text = ""
        sources = []
        for item in ask_stream(req.question, history=history, profile=profile):
            if item["type"] == "meta":
                item["conversation_id"] = conversation_id  # 把会话 ID 返回给前端
                sources = item["sources"]  # 记下来源，流结束后一起写库
            elif item["type"] == "delta":
                answer_text += item["data"]
            yield json.dumps(item, ensure_ascii=False) + "\n"
        # 流结束后，把这一轮问答写进数据库（含来源，实现持久化）
        db.add_message(conversation_id, "user", req.question)
        db.add_message(conversation_id, "assistant", answer_text, sources=sources)
        # 长期记忆：流已经返回给用户了，画像提取放后台线程做，不阻塞用户体验
        threading.Thread(
            target=_update_profile_if_needed,
            args=(user["id"], req.question, answer_text, profile),
            daemon=True,
        ).start()

    return StreamingResponse(generate(), media_type="application/x-ndjson")
