"""会话管理接口：新建、列出、查看、删除对话"""
from fastapi import APIRouter
from pydantic import BaseModel

from app.services import db

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = "新对话"


@router.post("")
def create_conversation(req: ConversationCreate):
    """新建会话：POST /api/conversations"""
    conv_id = db.create_conversation(req.title)
    return {"id": conv_id, "title": req.title}


@router.get("")
def list_conversations():
    """会话列表：GET /api/conversations（最新在前）"""
    return db.list_conversations()


@router.get("/{conversation_id}/messages")
def get_messages(conversation_id: str):
    """查看某个会话的历史消息：GET /api/conversations/{id}/messages"""
    return db.get_messages(conversation_id)


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str):
    """删除会话：DELETE /api/conversations/{id}"""
    db.delete_conversation(conversation_id)
    return {"status": "ok"}
