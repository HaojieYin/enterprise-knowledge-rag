"""会话管理接口：新建、列出、查看、删除对话（按登录用户隔离）"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.routers.auth import get_current_user
from app.services import db

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class ConversationCreate(BaseModel):
    title: str = "新对话"


@router.post("")
def create_conversation(req: ConversationCreate, user: dict = Depends(get_current_user)):
    """新建会话：POST /api/conversations"""
    conv_id = db.create_conversation(req.title, user["id"])
    return {"id": conv_id, "title": req.title}


@router.get("")
def list_conversations(user: dict = Depends(get_current_user)):
    """会话列表：GET /api/conversations（只返回当前用户的，最新在前）"""
    return db.list_conversations(user["id"])


@router.get("/{conversation_id}/messages")
def get_messages(conversation_id: str, user: dict = Depends(get_current_user)):
    """查看某个会话的历史消息：GET /api/conversations/{id}/messages"""
    if not db.conversation_exists(conversation_id, user["id"]):
        raise HTTPException(status_code=404, detail="会话不存在")
    return db.get_messages(conversation_id)


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict = Depends(get_current_user)):
    """删除会话：DELETE /api/conversations/{id}"""
    if not db.conversation_exists(conversation_id, user["id"]):
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete_conversation(conversation_id, user["id"])
    return {"status": "ok"}
