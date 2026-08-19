"""数据库服务：用 SQLite 持久化对话（会话 + 消息）

比喻：数据库 = 笔记本，会话 = 一页，消息 = 一页上的一行。
之前对话历史存在前端内存（草稿纸），刷新就丢；现在存进 SQLite（笔记本），
刷新不丢，还能同时开多个会话。
"""
import sqlite3
import uuid
from datetime import datetime

from app.config import DB_PATH


def _connect() -> sqlite3.Connection:
    """建立数据库连接（row_factory 让查询结果可以用列名访问）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库：创建会话表和消息表（不存在才创建）"""
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def create_conversation(title: str) -> str:
    """新建一个会话，返回会话 ID"""
    conv_id = uuid.uuid4().hex
    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (id, title, created_at) VALUES (?, ?, ?)",
        (conv_id, title, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return conv_id


def conversation_exists(conversation_id: str) -> bool:
    """判断某个会话是否存在"""
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    conn.close()
    return row is not None


def list_conversations() -> list[dict]:
    """列出所有会话（最新在前）"""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, created_at FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_messages(conversation_id: str) -> list[dict]:
    """获取某个会话的历史消息（按时间正序，返回 role/content，正好喂给 RAG）"""
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_message(conversation_id: str, role: str, content: str) -> None:
    """给某个会话追加一条消息"""
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def delete_conversation(conversation_id: str) -> None:
    """删除会话及其所有消息"""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()
