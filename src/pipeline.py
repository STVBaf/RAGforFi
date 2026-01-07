"""RAG Pipeline - 完整流程封装"""
from typing import List, Optional
from dataclasses import dataclass

from .parser import FinanceDocParser, Document
from .chunker import FinanceChunker, Chunk
from .indexer import HybridIndexer

@dataclass
class RAGResult:
    """RAG结果"""
    question: str
    answer: str
    retrieved_chunks: List[Chunk]
    reranked_chunks: List[Chunk]

class FinanceRAGPipeline:
    """金融RAG完整流程"""
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        llm_model: str = "gpt-4",
        index_dir: str = "./indices",
        chunk_size: int = 512,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5,
        use_reranker: bool = True,
        use_local_llm: bool = False,
        local_llm_url: str = "http://localhost:11434/v1",
        api_key: str = None,
        base_url: str = None
    ):
        self.parser = FinanceDocParser()
        self.chunker = FinanceChunker(chunk_size=chunk_size)
        self.indexer = HybridIndexer(embedding_model, index_dir)
        
        self.use_reranker = use_reranker
        self.reranker = None
        
        self.generator = None
        self.use_local_llm = use_local_llm
        self.llm_model = llm_model
        self.local_llm_url = local_llm_url
        self.api_key = api_key
        self.base_url = base_url
        self.reranker_model = reranker_model
        
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
    
    def _get_reranker(self):
        """延迟加载reranker"""
        if self.reranker is None and self.use_reranker:
            from .reranker import Reranker
            self.reranker = Reranker(self.reranker_model)
        return self.reranker
    
    def _get_generator(self):
        """延迟加载生成器"""
        if self.generator is None:
            if self.use_local_llm:
                from .generator import FinanceGeneratorLocal
                self.generator = FinanceGeneratorLocal(
                    model=self.llm_model,
                    base_url=self.local_llm_url
                )
            else:
                from .generator import FinanceGenerator
                self.generator = FinanceGenerator(
                    model=self.llm_model,
                    api_key=self.api_key,
                    base_url=self.base_url
                )
        return self.generator
    
    def index_chunks(self, chunks: List):
        """索引文档块列表"""
        self.indexer.index_chunks(chunks)
        print(f"Indexed {len(chunks)} chunks")
    
    def query(self, question: str) -> RAGResult:
        """完整问答流程"""
        # 1. 检索
        retrieved = self.indexer.search(question, top_k=self.top_k_retrieve)
        
        # 2. 重排序（可选）
        if self.use_reranker:
            reranker = self._get_reranker()
            reranked = reranker.rerank(question, retrieved, top_k=self.top_k_rerank)
        else:
            reranked = retrieved[:self.top_k_rerank]
        
        # 3. 生成
        generator = self._get_generator()
        answer = generator.generate(question, reranked)
        
        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=[c for c, _ in retrieved],
            reranked_chunks=[c for c, _ in reranked]
        )
    
    def load_index(self):
        """加载已有索引"""
        self.indexer.load_bm25()
