"""生成模块 - Chain-of-Thought金融问答"""
from typing import List, Tuple, Optional
import re
from .chunker import Chunk

# 金融CoT提示词模板
FINANCE_COT_PROMPT = """You are a financial analyst. Answer the question based ONLY on the provided context.

## Context
{context}

## Question
{question}

## Instructions
1. Find relevant numbers in the context
2. If calculation is needed:
   - List the values with their meanings
   - Show the calculation step by step
   - Give the final numerical answer
3. If the answer is a percentage, convert to decimal (e.g., 15% = 0.15) unless asked for percentage
4. If information is not in context, say "Cannot determine"

## Answer (provide ONLY the final number or brief answer):
"""

class FinanceGenerator:
    """金融问答生成器 - 支持OpenAI API"""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None, base_url: str = None):
        from openai import OpenAI
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs) if kwargs else OpenAI()
        self.model = model
    
    def generate(self, question: str, chunks: List[Tuple[Chunk, float]], 
                 stream: bool = False) -> str:
        """生成答案"""
        # 构建上下文
        context = self._build_context(chunks)
        
        # 构建提示词
        prompt = FINANCE_COT_PROMPT.format(context=context, question=question)
        
        # 调用LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是专业的金融分析助手，擅长数值计算和财务分析。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # 低温度提高计算准确性
            stream=stream
        )
        
        if stream:
            return response  # 返回流式响应
        
        answer = response.choices[0].message.content
        
        # 验证计算结果T
        verified_answer = self._verify_calculation(answer)
        
        return verified_answer
    
    def _build_context(self, chunks: List[tuple[Chunk, float]]) -> str:
        """构建上下文，优先展示表格"""
        # 分离表格和文本
        tables = []
        texts = []
        
        for chunk, score in chunks:
            if chunk.chunk_type == "table":
                tables.append(f"[相关性: {score:.3f}]\n{chunk.text}")
            else:
                texts.append(f"[相关性: {score:.3f}]\n{chunk.text}")
        
        # 表格优先
        context_parts = []
        if tables:
            context_parts.append("### 相关表格\n" + "\n\n".join(tables))
        if texts:
            context_parts.append("### 相关文本\n" + "\n\n".join(texts))
        
        return "\n\n".join(context_parts)
    
    def _verify_calculation(self, answer: str) -> str:
        """简单验证计算结果（可扩展）"""
        # 提取计算表达式并验证
        calc_pattern = r'(\d+\.?\d*)\s*[\+\-\*\/]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)'
        
        for match in re.finditer(calc_pattern, answer):
            # 这里可以添加更复杂的验证逻辑
            pass
        
        return answer


class FinanceGeneratorLocal:
    """本地模型生成器（使用Ollama原生API）"""
    
    def __init__(self, model: str = "qwen3:8b", base_url: str = "http://localhost:11434"):
        import requests
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False  # 禁用代理
    
    def generate(self, question: str, chunks: List[Tuple[Chunk, float]]) -> str:
        """生成答案"""
        context = "\n\n".join([c.text for c, _ in chunks])
        prompt = FINANCE_COT_PROMPT.format(context=context, question=question)
        
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
