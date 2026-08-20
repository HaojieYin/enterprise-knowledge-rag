"""用户认证：注册、登录、JWT 签发与校验

JWT（JSON Web Token）是无状态的登录凭证：登录成功后，后端把用户信息
打包成一串带签名的 token 发给前端，之后每次请求前端带上它，后端验签
就能确认「你是谁」，不用每次查数据库存 session。

三个要点：
1. 密码不能明文存 → 加盐哈希（PBKDF2，比 md5/sha1 抗暴力破解）
2. token 用密钥签名 → 别人改不了 token 内容，也伪造不了
3. token 有过期时间 → 过期要重新登录
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from app.config import JWT_EXPIRE_MINUTES, JWT_SECRET
from app.services import db

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ============ 密码哈希 ============

def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """加盐哈希密码，返回 (salt, hash)。

    盐是每次注册随机生成的，这样两个用户即使密码相同，存下来的哈希也不同，
    攻击者没法用「彩虹表」批量破解。
    """
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), 100_000
    ).hex()
    return salt, digest


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """验证密码：用同一个盐重新哈希，比对是否一致"""
    _, digest = hash_password(password, salt)
    return digest == expected_hash


# ============ JWT ============

def create_token(user_id: str, username: str) -> str:
    """签发 JWT：把用户信息 + 过期时间打包，用密钥签名"""
    payload = {
        "sub": user_id,  # sub = subject，标准字段，放用户 ID
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_token(token: str) -> dict:
    """校验并解析 JWT：签名不对或过期会抛异常"""
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


# ============ 认证依赖（FastAPI 的 Depends） ============

def get_current_user(authorization: str | None = Header(None)) -> dict:
    """从请求头的 Authorization: Bearer <token> 里解析当前用户。

    这是一个「依赖」，路由函数把它放进参数里，FastAPI 会自动调用它，
    校验不通过就返回 401，通过就把 user 传进路由。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization[7:]  # 去掉 "Bearer " 前缀
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的登录凭证")
    user = db.get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    # 只把必要字段传给路由，不带密码哈希
    return {
        "id": user["id"],
        "username": user["username"],
        "profile": user.get("profile") or "",
    }


# ============ 接口 ============

class AuthRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
def register(req: AuthRequest):
    """注册：POST /api/auth/register，成功后直接返回 token（自动登录）"""
    username = req.username.strip()
    if not username or not req.password:
        raise HTTPException(status_code=400, detail="用户名和密码不能为空")
    if db.get_user_by_username(username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    salt, pw_hash = hash_password(req.password)
    user_id = db.create_user(username, pw_hash, salt)
    token = create_token(user_id, username)
    return {"token": token, "username": username}


@router.post("/login")
def login(req: AuthRequest):
    """登录：POST /api/auth/login，校验密码后返回 token"""
    user = db.get_user_by_username(req.username.strip())
    if not user or not verify_password(req.password, user["salt"], user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = create_token(user["id"], user["username"])
    return {"token": token, "username": user["username"]}


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    """当前登录用户信息：GET /api/auth/me（用于前端刷新页面时确认登录态）"""
    return user
