"""向量数据库服务：把文档块存进 Chroma，并支持语义检索"""
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_DIR
from app.services.embedding import get_embeddings

# 向量库的「集合」名字（一个知识库对应一个集合）
COLLECTION_NAME = "enterprise_kb"

# 全量文档缓存：给混合检索的 BM25 建索引用（首次访问时从 Chroma 读取）
_documents_cache: list[Document] | None = None


def get_vector_store() -> Chroma:
    """创建/打开向量库（Chroma 会自动持久化到本地磁盘）"""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


def _invalidate_cache() -> None:
    """文档变更后，让全量文档缓存失效（下次访问时重新从 Chroma 读取）"""
    global _documents_cache
    _documents_cache = None


def add_documents(documents: list[Document]) -> None:
    """把文档块写入向量库（内部会先向量化，再存储）"""
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
    _invalidate_cache()


def search(query: str, k: int = 4) -> list[Document]:
    """语义检索（纯向量）：返回与问题最相关的 k 个文档块"""
    vector_store = get_vector_store()
    return vector_store.similarity_search(query, k=k)


def get_all_documents() -> list[Document]:
    """读取向量库中的全部文档块（用于构建 BM25 关键词索引）"""
    global _documents_cache
    if _documents_cache is None:
        store = get_vector_store()
        data = store.get()
        _documents_cache = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
    return _documents_cache


def reset_vector_store() -> None:
    """清空向量库（重新索引前调用，避免数据重复）"""
    vector_store = get_vector_store()
    vector_store.delete_collection()
    _invalidate_cache()
