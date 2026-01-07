"""配置文件"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # 模型配置
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_model: str = "deepseek-chat"  # DeepSeek API模型
    
    # LLM API配置 (DeepSeek)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "your_deepseek_api_key")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    
    # 本地模型配置 (Ollama)
    use_local_llm: bool = False
    local_llm_model: str = "qwen3:8b"
    local_llm_url: str = "http://localhost:11434"
    
    # 检索配置
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 30  # 初始检索数量（增大候选池）
    top_k_rerank: int = 8      # 最终保留数量
    use_reranker: bool = False  # Reranker在此场景效果不佳
    
    # 混合检索权重 (RRF fusion)
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    
    # 路径配置
    data_dir: str = "./dataset"
    index_dir: str = "./indices"

config = Config()
