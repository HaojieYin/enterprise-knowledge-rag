"""向量化服务：把文本转成向量（Embedding）"""
from langchain_openai import OpenAIEmbeddings

from app.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    EMBEDDING_MODEL,
)


def get_embeddings() -> OpenAIEmbeddings:
    """返回指向硅基流动的 Embedding 模型实例

    和 DeepSeek 一样，硅基流动的接口也是 OpenAI 兼容格式，
    所以用 OpenAIEmbeddings，只是把 base_url 指向硅基流动。
    """
    return OpenAIEmbeddings(
        api_key=SILICONFLOW_API_KEY,
        base_url=SILICONFLOW_BASE_URL,
        model=EMBEDDING_MODEL,
    )
