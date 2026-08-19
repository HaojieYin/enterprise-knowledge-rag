"""混合检索：向量检索（语义）+ 关键词检索（BM25），用 RRF 融合"""
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.vector_store import get_all_documents, get_vector_store

# RRF 融合常数：越大，排名靠后的文档影响越小（业界常用 60）
RRF_K = 60


def _bm25_search(query: str, k: int) -> list[Document]:
    """关键词检索：BM25，用 jieba 分词适配中文"""
    docs = get_all_documents()
    retriever = BM25Retriever.from_documents(docs, preprocess_func=jieba.lcut)
    retriever.k = k
    return retriever.invoke(query)


def hybrid_search(query: str, k: int = 20) -> list[Document]:
    """混合召回：向量检索 + BM25 关键词检索，用 RRF 融合，返回 top-k

    为什么两种都要：
      - 向量检索：懂语义，但可能漏掉精确关键词（如编号、专有名词）
      - BM25 关键词检索：精确匹配强，但不懂同义词
    RRF（倒数排名融合）把两者的排序合并，取长补短。
    """
    vector_docs = get_vector_store().similarity_search(query, k=k)
    bm25_docs = _bm25_search(query, k=k)

    # RRF：每个文档得分 = Σ 1 / (RRF_K + rank)，rank 从 1 开始
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for ranked_list in (vector_docs, bm25_docs):
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)

    # 按融合得分降序，取前 k 个
    sorted_docs = [doc_map[key] for key in sorted(scores, key=scores.get, reverse=True)]
    return sorted_docs[:k]
