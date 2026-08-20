"""RAG 核心链路：查询路由(Agent) -> 多轮对话改写 -> 检索(召回+重排) -> 生成"""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.services.hybrid import hybrid_search
from app.services.llm import get_llm
from app.services.rerank import rerank

# 检索场景的系统提示词：给大模型定规矩，防止它「胡说八道」
SYSTEM_PROMPT = """你是一个企业知识库助手。请严格遵守以下规则：
1. 只能根据【资料】中的内容回答问题
2. 如果资料中没有相关信息，直接回答"资料中未找到相关信息"，不要编造
3. 回答要简洁、准确
4. 在回答中引用了哪份资料的内容，就在对应句末加上引用标记 [编号]，例如 [1]、[2]（编号对应【资料1】【资料2】）
"""

# 非检索场景的系统提示词：问候、闲聊、感谢等，不需要查资料
SYSTEM_PROMPT_CHAT = """你是一个友好、专业的企业助手。
对于问候、感谢、闲聊等不需要查资料的问题，请友好、简短地回应。
涉及公司制度、政策、流程的问题会走知识库检索（由系统自动判断）。
"""


def build_context(docs: list[Document]) -> str:
    """把检索到的文档块拼成一段上下文"""
    return "\n\n".join(
        f"【资料{i + 1}】(来源：{doc.metadata['source']})\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


def _profile_text(profile: str) -> str:
    """把用户长期画像拼进系统提示词，让 AI 跨会话记得用户是谁"""
    if not profile:
        return ""
    return (
        f"\n\n【当前用户信息】\n{profile}\n"
        "（这是系统长期记住的关于该用户的信息。回答时如果涉及用户个人情况，可以自然参考；"
        "用户问「我叫什么」「我的职位」这类问题时，据此回答。）"
    )


def extract_profile(question: str, answer: str, existing_profile: str) -> str | None:
    """从这轮对话提取值得长期记住的用户信息，返回更新后的画像；无更新返回 None

    这是「长期记忆」：每轮聊完，让大模型判断用户是否透露了新的个人信息
    （姓名、职位、部门、偏好等），有就合并进画像，下次提问时再注入。
    """
    prompt = f"""你是用户画像提取器。根据这轮对话，判断用户是否透露了值得长期记住的个人信息。

规则：
1. 只提取用户明确说出的、相对稳定的个人信息（如姓名、职位、部门、城市、偏好等）
2. 不提取临时的一次性信息（如「我今天想请假」「刚才问了报销」这种）
3. 如果用户没有透露新的值得记住的信息，只输出「无更新」

已有的画像：
{existing_profile or "（暂无）"}

这轮对话：
用户：{question}
助手：{answer}

请输出：
- 如果用户透露了新的值得记住的信息，把新旧信息合并，输出一段简洁的完整画像（自然语言）
- 如果没有新的值得记住的信息，只输出「无更新」"""
    result = get_llm().invoke(prompt)
    content = result.content.strip()
    if not content or content.startswith("无更新"):
        return None
    return content


def rewrite_question(question: str, history: list[dict]) -> str:
    """多轮对话的关键：把依赖上下文的问题改写成独立问题（指代消解）"""
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}"
        for m in history
    )
    prompt = f"""请把用户的最新问题改写成「脱离上下文也能独立理解」的完整问题，用于后续搜索资料。

改写要求：
1. 把指代词（那、它、这个等）换成具体内容
2. 用自然的方式提问，不要照搬上一句话的句式
3. 直接输出改写后的问题，不要解释

示例：
历史在聊「晋升需要什么条件」，用户问「那报销呢」→ 正确改写：「报销制度有哪些规定」

对话历史：
{history_text}

用户最新问题：{question}

改写后的问题："""

    result = get_llm().invoke(prompt)
    return result.content.strip()


def classify_intent(question: str) -> bool:
    """查询路由（Agent 决策）：让 LLM 判断这个问题是否需要检索知识库

    这是「Agent 化」的核心——让大模型自己决定要不要查资料，
    而不是不管三七二十一都去检索。
    """
    prompt = f"""判断下面的用户问题，回答是否需要检索企业知识库。

规则：
- 需要检索：问题在询问公司制度、政策、流程、规范等，必须基于资料回答（例如"报销超过5000谁审批""年假多少天"）
- 不需要检索：简单问候、闲聊、感谢，或与公司制度无关的通用问题（例如"你好""谢谢""你是谁"）

只输出两个词之一：需要检索 或 不需要检索

用户问题：{question}"""
    result = get_llm().invoke(prompt)
    content = result.content.strip()
    # 「不需要检索」里包含「需要」，要先排除，否则会误判
    if "不需要" in content:
        return False
    return True  # 默认需要检索（保守：宁多查、不漏查）


def rerank_documents(query: str, docs: list[Document], top_n: int) -> list[Document]:
    """两阶段检索的第二阶段：用重排模型精排，挑出最相关的 top_n 个"""
    if len(docs) <= top_n:
        return docs
    texts = [doc.page_content for doc in docs]
    indices = rerank(query, texts, top_n)
    return [docs[i] for i in indices]


def retrieve(question: str, k: int, history: list[dict]) -> tuple[str, list[Document]]:
    """改写问题 + 混合检索 + 重排，返回 (改写后的问题, 精排后的文档)"""
    rewritten_query = rewrite_question(question, history) if history else question
    candidates = hybrid_search(rewritten_query, k=max(k * 5, 10))
    docs = rerank_documents(rewritten_query, candidates, top_n=k)
    return rewritten_query, docs


def _append_history(messages: list, history: list[dict]) -> list:
    """把对话历史追加到消息列表"""
    for m in history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    return messages


def build_messages(question: str, docs: list[Document], history: list[dict], profile: str = "") -> list:
    """构建检索场景的消息列表（带资料 + 严格约束提示词 + 用户画像）"""
    context = build_context(docs)
    user_prompt = f"【资料】\n{context}\n\n【用户问题】\n{question}"
    messages = [SystemMessage(content=SYSTEM_PROMPT + _profile_text(profile))]
    _append_history(messages, history)
    messages.append(HumanMessage(content=user_prompt))
    return messages


def build_chat_messages(question: str, history: list[dict], profile: str = "") -> list:
    """构建非检索场景的消息列表（不带资料，用友好提示词 + 用户画像）"""
    messages = [SystemMessage(content=SYSTEM_PROMPT_CHAT + _profile_text(profile))]
    _append_history(messages, history)
    messages.append(HumanMessage(content=question))
    return messages


def build_sources(docs: list[Document]) -> list[dict]:
    """整理来源：给每个文档块编号，附带来源文件和原文片段（用于前端引用高亮）"""
    return [
        {
            "id": i + 1,
            "source": doc.metadata["source"],
            "snippet": doc.page_content,
        }
        for i, doc in enumerate(docs)
    ]


def ask(question: str, k: int = 4, history: list[dict] | None = None, profile: str = "") -> dict:
    """RAG 问答（一次性返回完整回答）

    先做查询路由（Agent 决策）：
      - 需要检索 → 改写 + 混合检索 + 重排 + 生成
      - 不需要检索 → 直接友好回答（跳过检索）
    """
    history = history or []

    if classify_intent(question):
        rewritten_query, docs = retrieve(question, k, history)
        messages = build_messages(question, docs, history, profile)
        answer = get_llm().invoke(messages)
        return {
            "answer": answer.content,
            "sources": build_sources(docs),
            "rewritten_query": rewritten_query if history else None,
            "searched": True,
        }

    messages = build_chat_messages(question, history, profile)
    answer = get_llm().invoke(messages)
    return {
        "answer": answer.content,
        "sources": [],
        "rewritten_query": None,
        "searched": False,
    }


def ask_stream(question: str, k: int = 4, history: list[dict] | None = None, profile: str = ""):
    """RAG 问答（流式）：生成器，逐段 yield 结果

    先做查询路由（Agent 决策），再流式生成。yield 格式（dict）：
      {"type": "meta", "rewritten_query": ..., "sources": [...], "searched": bool}  # 首条
      {"type": "delta", "data": "片段"}                                             # 中间
      {"type": "done"}                                                              # 结束
    """
    history = history or []

    if classify_intent(question):
        rewritten_query, docs = retrieve(question, k, history)
        messages = build_messages(question, docs, history, profile)
        meta = {
            "type": "meta",
            "rewritten_query": rewritten_query if history else None,
            "sources": build_sources(docs),
            "searched": True,
        }
    else:
        messages = build_chat_messages(question, history, profile)
        meta = {
            "type": "meta",
            "rewritten_query": None,
            "sources": [],
            "searched": False,
        }

    yield meta
    for chunk in get_llm().stream(messages):
        text = chunk.content
        if text:
            yield {"type": "delta", "data": text}
    yield {"type": "done"}
