"""大模型服务：负责创建并返回 DeepSeek 大模型实例"""
from langchain_openai import ChatOpenAI

from app.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_llm() -> ChatOpenAI:
    """创建并返回一个指向 DeepSeek 的大模型实例

    关键点：base_url 指向 DeepSeek（而不是 OpenAI），
    因为 DeepSeek 的接口是「OpenAI 兼容」格式。
    """
    return ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        model=DEEPSEEK_MODEL,
        temperature=0.7,  # 0~1，越大回答越有创造性，越小越稳定
    )
