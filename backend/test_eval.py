"""效果评估：对比三种检索方案的正确率

方案 A：纯向量检索（基线）
方案 B：混合检索（向量 + BM25）
方案 C：混合检索 + 重排（完整方案）

评估方式：关键词匹配（粗略但够用，用来展示各方案的相对提升）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import DATA_DIR
from app.services.document import process_document
from app.services.hybrid import hybrid_search
from app.services.llm import get_llm
from app.services.rag import SYSTEM_PROMPT, build_context
from app.services.rerank import rerank
from app.services.vector_store import add_documents, reset_vector_store, search

# ============ 1. 重建索引（章节级切分，保证评估环境干净、公平） ============
print("正在重建索引（chunk_size=150，章节级切分）...")
reset_vector_store()
for filename in ["employee_handbook.txt", "it_policy.md", "training_policy.txt"]:
    docs = process_document(DATA_DIR / filename, chunk_size=150, chunk_overlap=20)
    add_documents(docs)
    print(f"  {filename}: {len(docs)} 块")
print()

# ============ 2. 测试集：(问题, [期望关键词...]) ============
TEST_CASES = [
    ("报销超过5000元需要谁审批？", ["财务总监"]),
    ("请假超过3天需要谁审批？", ["部门总监"]),
    ("晋升需要满足什么条件？", ["满1年", "满一年", "B+"]),
    ("员工每年有多少天年假？", ["15天", "15 天", "15"]),
    ("公司密码需要多久修改一次？", ["90天", "90 天", "90"]),
    ("迟到超过30分钟要怎么办？", ["报备"]),
    ("员工每年有几次晋升窗口期？", ["两次", "2次", "3月和9月"]),
    ("外部培训每年报销限额多少？", ["5000"]),
    ("导师每季度津贴是多少？", ["2000"]),
    ("考取职业资格证书奖励多少？", ["1000"]),
    ("每月几号发工资？", ["10号", "10日", "10"]),
    ("核心工作时间是几点到几点？", ["10点", "下午4点"]),
]


# ============ 3. 三种方案（生成部分完全一致，只换检索） ============
def generate(question: str, docs) -> str:
    context = build_context(docs)
    prompt = f"【资料】\n{context}\n\n【用户问题】\n{question}"
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    return get_llm().invoke(messages).content


def answer_vector(question: str) -> str:
    """方案 A：纯向量检索，直接取 top-4"""
    docs = search(question, k=4)
    return generate(question, docs)


def answer_hybrid(question: str) -> str:
    """方案 B：混合检索（向量 + BM25），取 top-4"""
    docs = hybrid_search(question, k=4)
    return generate(question, docs)


def answer_hybrid_rerank(question: str) -> str:
    """方案 C：混合召回 top-20，再用重排模型精排取 top-4"""
    candidates = hybrid_search(question, k=20)
    if len(candidates) > 4:
        texts = [d.page_content for d in candidates]
        indices = rerank(question, texts, top_n=4)
        docs = [candidates[i] for i in indices]
    else:
        docs = candidates
    return generate(question, docs)


def is_correct(answer: str, keywords: list[str]) -> bool:
    return any(kw in answer for kw in keywords)


# ============ 4. 跑评估 ============
print(f"{'问题':<26}{'纯向量':<8}{'混合':<8}{'混合+重排':<8}")
print("-" * 52)

ok_vector = 0
ok_hybrid = 0
ok_hybrid_rerank = 0

for question, keywords in TEST_CASES:
    a = answer_vector(question)
    b = answer_hybrid(question)
    c = answer_hybrid_rerank(question)
    ok_a = is_correct(a, keywords)
    ok_b = is_correct(b, keywords)
    ok_c = is_correct(c, keywords)
    ok_vector += ok_a
    ok_hybrid += ok_b
    ok_hybrid_rerank += ok_c

    mark_a = "✅" if ok_a else "❌"
    mark_b = "✅" if ok_b else "❌"
    mark_c = "✅" if ok_c else "❌"
    print(f"{question:<26}{mark_a:<8}{mark_b:<8}{mark_c:<8}")

print("-" * 52)
total = len(TEST_CASES)
print(f"纯向量检索：    {ok_vector}/{total} 正确（{ok_vector / total:.0%}）")
print(f"混合检索：      {ok_hybrid}/{total} 正确（{ok_hybrid / total:.0%}）")
print(f"混合检索+重排： {ok_hybrid_rerank}/{total} 正确（{ok_hybrid_rerank / total:.0%}）")
