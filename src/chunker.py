"""文本分块模块 - 金融文档专用分块策略"""
from dataclasses import dataclass
from typing import List
import re

@dataclass
class Chunk:
    """文本块"""
    text: str
    chunk_type: str  # "text" | "table" | "hybrid"
    metadata: dict

class FinanceChunker:
    """金融文档分块器 - 表格感知"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, content: str, metadata: dict = None) -> List[Chunk]:
        """分块文档，表格单独处理"""
        chunks = []
        metadata = metadata or {}
        
        # 分离表格和文本
        segments = self._split_tables_and_text(content)
        
        for seg_type, seg_content in segments:
            if seg_type == "table":
                # 表格作为独立chunk，保持完整
                chunks.append(Chunk(
                    text=seg_content.strip(),
                    chunk_type="table",
                    metadata={**metadata, "has_table": True}
                ))
            else:
                # 文本按固定大小分块
                text_chunks = self._chunk_text(seg_content)
                for tc in text_chunks:
                    chunks.append(Chunk(
                        text=tc,
                        chunk_type="text",
                        metadata=metadata
                    ))
        
        return chunks
    
    def _split_tables_and_text(self, content: str) -> List[tuple]:
        """分离表格和普通文本"""
        table_pattern = r'(\|.+\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)'
        
        segments = []
        last_end = 0
        
        for match in re.finditer(table_pattern, content):
            # 表格前的文本
            if match.start() > last_end:
                text = content[last_end:match.start()]
                if text.strip():
                    segments.append(("text", text))
            # 表格
            segments.append(("table", match.group()))
            last_end = match.end()
        
        # 剩余文本
        if last_end < len(content):
            text = content[last_end:]
            if text.strip():
                segments.append(("text", text))
        
        return segments
    
    def _chunk_text(self, text: str) -> List[str]:
        """按固定大小分块文本"""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 尝试在句子边界断开
            if end < len(text):
                last_period = max(chunk.rfind("。"), chunk.rfind("."), chunk.rfind("\n"))
                if last_period > self.chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - self.overlap
        
        return [c for c in chunks if c]
