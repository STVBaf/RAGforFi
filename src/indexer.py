"""混合索引模块 - 向量索引 + BM25索引"""
import os
import pickle
from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from .chunker import Chunk

class HybridIndexer:
    """混合索引器：向量检索 + BM25关键词检索"""
    
    def __init__(self, embedding_model: str, index_dir: str = "./indices"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        # Embedding模型
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(embedding_model)
        
        # 向量索引 - 使用numpy简单实现（避免ChromaDB兼容问题）
        self.embeddings = None
        self.documents = []
        
        # BM25索引
        self.bm25 = None
        self.chunks: List[Chunk] = []
    
    def index_chunks(self, chunks: List[Chunk]):
        """索引文档块"""
        self.chunks = chunks
        self.documents = [c.text for c in chunks]
        
        # 1. 向量索引
        print("Encoding documents...")
        self.embeddings = self.embed_model.encode(self.documents, show_progress_bar=True)
        
        # 2. BM25索引
        print("Building BM25 index...")
        tokenized = [self._tokenize(t) for t in self.documents]
        self.bm25 = BM25Okapi(tokenized)
        
        # 保存索引
        self._save_index()
        print(f"✓ Indexed {len(chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        tokens = re.findall(r'\d+\.?\d*%?|\w+', text.lower())
        return tokens
    
    def search(self, query: str, top_k: int = 10, 
               vector_weight: float = 0.6) -> List[Tuple[Chunk, float]]:
        """混合检索"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # 1. 向量检索 - 余弦相似度
        query_emb = self.embed_model.encode([query])[0]
        
        # 归一化
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        doc_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
        
        vector_scores = np.dot(doc_norms, query_norm)
        vector_ranks = np.argsort(-vector_scores)[:top_k]
        
        # 2. BM25检索
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranks = np.argsort(-bm25_scores)[:top_k]
        
        # 3. RRF融合
        scores = {}
        bm25_weight = 1 - vector_weight
        
        for rank, idx in enumerate(vector_ranks):
            scores[idx] = scores.get(idx, 0) + vector_weight / (rank + 1)
        
        for rank, idx in enumerate(bm25_ranks):
            scores[idx] = scores.get(idx, 0) + bm25_weight / (rank + 1)
        
        # 排序返回
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in sorted_results[:top_k]]
    
    def _save_index(self):
        """保存索引"""
        data = {
            "embeddings": self.embeddings,
            "chunks": self.chunks,
            "documents": self.documents,
            "bm25": self.bm25
        }
        with open(f"{self.index_dir}/index.pkl", "wb") as f:
            pickle.dump(data, f)
    
    def load_bm25(self):
        """加载索引"""
        path = f"{self.index_dir}/index.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data["embeddings"]
                self.chunks = data["chunks"]
                self.documents = data["documents"]
                self.bm25 = data["bm25"]
            print(f"✓ Loaded index with {len(self.chunks)} chunks")
