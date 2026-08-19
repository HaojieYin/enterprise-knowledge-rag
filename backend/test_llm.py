"""测试脚本：验证 DeepSeek 大模型能正常调用"""
import sys

from app.services.llm import get_llm

# 解决 Windows 终端中文乱码问题（把输出编码强制设为 UTF-8）
sys.stdout.reconfigure(encoding="utf-8")


def main():
    llm = get_llm()

    question = "请用一句话介绍你自己"
    print("正在调用 DeepSeek 大模型，请稍等...\n")

    # invoke() 表示「调用」大模型，传入一句话，返回一个回答对象
    answer = llm.invoke(question)

    print(f"问题：{question}")
    print(f"回答：{answer.content}")


if __name__ == "__main__":
    main()
