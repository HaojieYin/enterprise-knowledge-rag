"""数据库服务：用 SQLite 持久化用户 + 对话（会话 + 消息）

比喻：数据库 = 笔记本，用户 = 一个人，会话 = 一页，消息 = 一页上的一行。
之前对话历史存在前端内存（草稿纸），刷新就丢；现在存进 SQLite（笔记本），
刷新不丢，还能同时开多个会话。现在又加了用户表，每个用户只看得到自己的会话。
"""
import json
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
    """初始化数据库：创建用户表、会话表、消息表（不存在才创建）"""
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            profile TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
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
            sources TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    # 迁移：旧版本建的表缺列，补上（老数据不受影响）
    user_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "profile" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN profile TEXT DEFAULT ''")
    conv_cols = [r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()]
    if "user_id" not in conv_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT")
    msg_cols = [r["name"] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if "sources" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
    conn.commit()
    conn.close()


# ============ 用户 ============

def create_user(username: str, password_hash: str, salt: str, profile: str = "") -> str:
    """新建用户，返回用户 ID（密码的哈希和盐由认证模块算好再传进来）"""
    user_id = uuid.uuid4().hex
    conn = _connect()
    conn.execute(
        "INSERT INTO users (id, username, password_hash, salt, profile, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, username, password_hash, salt, profile, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return user_id


def get_user_by_username(username: str) -> dict | None:
    """按用户名查用户（登录时用），找不到返回 None"""
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    """按用户 ID 查用户（校验 token 时用），找不到返回 None"""
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_profile(user_id: str, profile: str) -> None:
    """更新用户的长期画像（AI 记住的个人信息）"""
    conn = _connect()
    conn.execute("UPDATE users SET profile = ? WHERE id = ?", (profile, user_id))
    conn.commit()
    conn.close()


# ============ 会话 ============

def create_conversation(title: str, user_id: str) -> str:
    """新建一个会话（属于某个用户），返回会话 ID"""
    conv_id = uuid.uuid4().hex
    conn = _connect()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
        (conv_id, user_id, title, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return conv_id


def conversation_exists(conversation_id: str, user_id: str) -> bool:
    """判断某个会话是否存在、且属于该用户"""
    conn = _connect()
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()
    conn.close()
    return row is not None


def list_conversations(user_id: str) -> list[dict]:
    """列出某个用户的所有会话（最新在前）"""
    conn = _connect()
    rows = conn.execute(
        "SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_messages(conversation_id: str) -> list[dict]:
    """获取某个会话的历史消息（按时间正序，返回 role/content/sources，正好喂给 RAG）"""
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content, sources FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        result.append(
            {
                "role": r["role"],
                "content": r["content"],
                # sources 存的是 JSON 字符串，读出来还原成列表（没有就空列表）
                "sources": json.loads(r["sources"]) if r["sources"] else [],
            }
        )
    return result


def add_message(
    conversation_id: str, role: str, content: str, sources: list | None = None
) -> None:
    """给某个会话追加一条消息（sources 是检索来源列表，可选）"""
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            conversation_id,
            role,
            content,
            json.dumps(sources, ensure_ascii=False) if sources else None,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def delete_conversation(conversation_id: str, user_id: str) -> None:
    """删除某个用户的会话及其所有消息"""
    conn = _connect()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute(
        "DELETE FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    )
    conn.commit()
    conn.close()
