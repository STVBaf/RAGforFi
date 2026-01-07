"""重排序模块 - 使用Cross-Encoder精排"""
from typing import List, Tuple
from .chunker import Chunk

class Reranker:
    """Cross-Encoder重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from FlagEmbedding import FlagReranker
        self.reranker = FlagReranker(model_name, use_fp16=True)
    
    def rerank(self, query: str, chunks: List[Tuple[Chunk, float]], 
               top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """重排序候选文档"""
        if not chunks:
            return []
        
        # 构建query-document对
        pairs = [[query, chunk.text] for chunk, _ in chunks]
        
        # 计算相关性分数
        scores = self.reranker.compute_score(pairs)
        
        # 如果只有一个结果，scores是float而不是list
        if isinstance(scores, float):
            scores = [scores]
        
        # 重新排序
        scored_chunks = list(zip([c for c, _ in chunks], scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_k]
