"""项目的配置文件：负责读取 .env 里的密钥和参数"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 找到项目根目录（.env 文件所在的位置）
# __file__ 是当前文件路径：backend/app/config.py
# .parent.parent.parent 依次往上一层：app -> backend -> 项目根目录(CCDEMO)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# 大模型配置：从 .env 里读，如果没填就用后面的默认值
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# Embedding（向量化）配置：硅基流动（因为 DeepSeek 没有 Embedding 接口）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 数据目录：存放文档和向量库
DATA_DIR = PROJECT_ROOT / "backend" / "data"
CHROMA_DIR = DATA_DIR / "chroma"
