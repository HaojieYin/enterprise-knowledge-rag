"""重排服务：用更精细的模型，对粗筛结果重新排序"""
import httpx

from app.config import RERANK_MODEL, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL


def rerank(query: str, documents: list[str], top_n: int = 4) -> list[int]:
    """调用重排模型，返回最相关文档的下标（已按相关度从高到低排序）

    和向量检索不同，重排模型会「逐字比较」query 和每篇文档，
    算出的相关度更精确，但更慢，所以只用于对少量候选做精排。
    """
    resp = httpx.post(
        f"{SILICONFLOW_BASE_URL}/rerank",
        headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}"},
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    # results 已按相关度降序排列，取出对应的原始下标
    return [item["index"] for item in results]
