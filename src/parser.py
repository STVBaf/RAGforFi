"""文档解析模块 - 使用Docling处理PDF，保留表格结构"""
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Document:
    """文档结构"""
    content: str
    tables: List[str]  # Markdown格式表格
    metadata: dict

class FinanceDocParser:
    """金融文档解析器"""
    
    def __init__(self):
        self.converter = None  # 按需加载Docling
    
    def _get_converter(self):
        """延迟加载Docling转换器"""
        if self.converter is None:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
        return self.converter
    
    def parse_pdf(self, pdf_path: str) -> Document:
        """解析PDF文档"""
        converter = self._get_converter()
        result = converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        
        # 提取表格
        tables = self._extract_tables(markdown)
        
        return Document(
            content=markdown,
            tables=tables,
            metadata={"source": pdf_path}
        )
    
    def _extract_tables(self, markdown: str) -> List[str]:
        """从Markdown中提取表格"""
        table_pattern = r'(\|.+\|[\r\n]+\|[-:| ]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)'
        return re.findall(table_pattern, markdown)
    
    def table_to_markdown(self, table: List[List[str]]) -> str:
        """将二维列表转为Markdown表格"""
        if not table:
            return ""
        
        lines = []
        # 表头
        lines.append("| " + " | ".join(str(c) for c in table[0]) + " |")
        lines.append("| " + " | ".join(["---"] * len(table[0])) + " |")
        # 数据行
        for row in table[1:]:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        
        return "\n".join(lines)
