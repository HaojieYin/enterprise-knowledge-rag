"""问答接口：接收用户问题，调用大模型返回答案"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.services.llm import get_llm

# 创建路由对象，prefix 表示所有接口路径都以 /api 开头
router = APIRouter(prefix="/api", tags=["chat"])


# 请求体：前端发来的 JSON 长这样 -> { "question": "..." }
class ChatRequest(BaseModel):
    question: str


# 响应体：我们返回给前端的 JSON 长这样 -> { "answer": "..." }
class ChatResponse(BaseModel):
    answer: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """问答接口：POST /api/chat

    流程：收到问题 -> 调用大模型 -> 返回答案
    """
    llm = get_llm()
    answer = llm.invoke(req.question)
    return ChatResponse(answer=answer.content)
