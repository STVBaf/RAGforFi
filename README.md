# RAG for Finance (FinanceBench)

基于FinanceBench数据集的金融垂直领域RAG优化系统。

## 目录

- [系统架构](#系统架构)
- [RAG框架详解](#rag框架详解)
- [核心模块实现](#核心模块实现)
- [实验结果](#实验结果)
- [Oracle上界解释](#oracle上界解释)
- [快速开始](#快速开始)

---

## 系统架构

```
                            ┌─────────────────────────────────────────────────────────┐
                            │                    RAG Pipeline                          │
                            └─────────────────────────────────────────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         │                                            │                                            │
         ▼                                            ▼                                            ▼
┌─────────────────┐                        ┌─────────────────┐                        ┌─────────────────┐
│   INDEXING      │                        │   RETRIEVAL     │                        │   GENERATION    │
│   (离线构建)     │                        │   (在线检索)     │                        │   (答案生成)     │
└────────┬────────┘                        └────────┬────────┘                        └────────┬────────┘
         │                                          │                                          │
         ▼                                          ▼                                          ▼
┌─────────────────┐                        ┌─────────────────┐                        ┌─────────────────┐
│  1. Parser      │                        │  1. Query       │                        │  1. Context     │
│     (文档解析)   │                        │     Enhancement │                        │     Assembly    │
│     ↓           │                        │     (查询增强)   │                        │     (上下文组装) │
│  2. Chunker     │                        │     ↓           │                        │     ↓           │
│     (语义分块)   │                        │  2. Hybrid      │                        │  2. LLM         │
│     ↓           │                        │     Search      │                        │     Prompting   │
│  3. Embedding   │                        │     (混合检索)   │                        │     (提示构造)   │
│     (向量化)     │                        │     ↓           │                        │     ↓           │
│     ↓           │                        │  3. RRF Fusion  │                        │  3. Answer      │
│  4. Index       │                        │     (分数融合)   │                        │     Extraction  │
│     (存储索引)   │                        │                 │                        │     (答案提取)   │
└─────────────────┘                        └─────────────────┘                        └─────────────────┘
```

---

## RAG框架详解

### 什么是 RAG?

**RAG (Retrieval-Augmented Generation)** 是一种结合检索和生成的AI架构：

```
用户问题 ──▶ 检索相关文档 ──▶ 将文档作为上下文 ──▶ LLM生成答案
```

**核心思想**: 不依赖LLM的参数记忆，而是从外部知识库实时检索相关信息，解决：
- ✅ 知识时效性问题（参数知识可能过时）
- ✅ 幻觉问题（基于真实文档生成）
- ✅ 可追溯性（可以引用来源）

### 本项目的 RAG 流程

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              完整 RAG 流程                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 用户问题  │───▶│ Query增强 │───▶│ 混合检索  │───▶│ 上下文组装 │───▶│ LLM生成  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │               │        │
│       ▼               ▼               ▼               ▼               ▼        │
│  "What is      "[2018] 3M:      Vector: 0.85     "Context:        "Answer:    │
│   3M's CAPEX    What is 3M's    BM25: 0.72       CAPEX was        $1,577      │
│   in FY2018?"   CAPEX..."       → RRF融合        $1,577M..."      million"    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块实现

### 1. 文档解析 (Parser)

**文件**: `src/parser.py`

```python
# 使用 Docling 解析 PDF，保留表格结构
from docling.document_converter import DocumentConverter

class FinanceDocParser:
    def parse_pdf(self, pdf_path: str) -> Document:
        result = self.converter.convert(pdf_path)
        markdown = result.document.export_to_markdown()
        return Document(content=markdown, tables=self._extract_tables(markdown))
```

**设计要点**:
- 使用 Docling 保持表格的 Markdown 结构
- 金融报表中表格包含关键数值，必须保持完整性

### 2. 语义分块 (Chunker)

**文件**: `src/chunker.py`

```python
class FinanceChunker:
    def chunk_document(self, content: str, metadata: dict) -> List[Chunk]:
        # 表格感知分块：检测表格边界，避免切割表格
        chunks = []
        for segment in self._split_by_structure(content):
            if self._is_table(segment):
                chunks.append(Chunk(text=segment, chunk_type="table"))
            else:
                # 文本按 chunk_size 分块，保留 overlap
                chunks.extend(self._split_text(segment))
        return chunks
```

**设计要点**:
- **表格感知**: 识别 Markdown 表格边界，保持表格完整
- **重叠分块**: 文本块之间有 overlap，避免边界信息丢失
- **元数据保留**: 每个 chunk 携带来源文档信息

### 3. 混合索引 (Indexer)

**文件**: `src/indexer.py`

```python
class HybridIndexer:
    def __init__(self, embedding_model):
        # 向量索引 (语义检索)
        self.embedder = FlagModel(embedding_model)
        self.vectors = None  # NumPy array
        
        # BM25索引 (关键词检索)
        self.bm25 = None  # BM25Okapi
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        # 1. 向量检索
        vector_results = self._vector_search(query, top_k)
        
        # 2. BM25检索
        bm25_results = self._bm25_search(query, top_k)
        
        # 3. RRF融合
        return self._rrf_fusion(vector_results, bm25_results)
```

**为什么用混合检索?**

| 检索方式 | 优点 | 缺点 |
|----------|------|------|
| 向量检索 | 语义理解强，能匹配同义词 | 精确数字/术语匹配弱 |
| BM25关键词 | 精确匹配强，速度快 | 无法理解语义相似性 |
| **混合检索** | **两者优势结合** | - |

**RRF (Reciprocal Rank Fusion) 算法**:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))

其中:
- k = 60 (常数)
- rank_i(d) = 文档d在第i个检索器中的排名
```

### 4. 查询增强 (Query Enhancement)

**文件**: `main.py`

```python
def enhance_query(question: str, company: str, doc_name: str) -> str:
    # 从文档名提取年份: "3M_2018_10K" -> "2018"
    year = re.search(r'(\d{4})', doc_name).group(1)
    
    # 增强查询
    return f"[{year}] {company}: {question}"
    
# 示例:
# 原始: "What is the capital expenditure?"
# 增强: "[2018] 3M: What is the capital expenditure?"
```

**效果**: +7.5% 准确率提升

### 5. 答案生成 (Generator)

**文件**: `src/generator.py`

```python
class FinanceGenerator:
    def generate(self, question: str, context: List[Tuple[Chunk, float]]) -> str:
        # 组装上下文
        context_text = "\n\n".join([c.text for c, _ in context])
        
        # 金融专用 Prompt
        prompt = f"""You are a financial analyst. Based on the context, answer the question.

Context:
{context_text}

Question: {question}

Instructions:
- If the answer involves calculations, show your work
- If the information is not in the context, say "Cannot determine"
- Be precise with numbers and units

Answer:"""
        
        return self.llm.chat(prompt)
```

**设计要点**:
- **Chain-of-Thought**: 引导显示计算过程
- **精确数值**: 强调保持数字精度
- **诚实回答**: 信息不足时说"无法确定"

---

## 实验结果

### 消融实验 (n=40)

| 排名 | 配置 | 准确率 | vs基线 |
|------|------|--------|--------|
| 1 | Oracle (Upper Bound) | **47.5%** | +17.5% |
| 2 | RAG (k=30→8) | **42.5%** | +12.5% |
| 3 | RAG + QE (k=20→5) | 37.5% | +7.5% |
| 4 | RAG Baseline (k=20→5) | 30.0% | - |
| 5 | RAG (k=10→3) | 30.0% | ±0% |
| 6 | RAG + Reranker | 22.5% | -7.5% |

### 可视化

![消融实验结果](results/ablation_chart.png)

![Top-K 参数影响](results/topk_analysis.png)

![各组件影响](results/component_impact.png)

### 关键发现

1. **Top-K 参数最关键** (+12.5%)
   ```
   k=10→3: 30.0%  ──┐
   k=20→5: 30.0%  ──┼── 增大候选池效果显著
   k=30→8: 42.5%  ──┘
   ```

2. **Query Enhancement 有效** (+7.5%)
   - 添加公司名和年份帮助定位正确文档

3. **Reranker 负面效果** (-7.5%)
   - Cross-Encoder 在此场景不适用
   - 可能将语义相似但非目标的文档排到前面

---

## Oracle上界解释

### 什么是 Oracle?

**Oracle 模式**直接使用数据集提供的 **ground truth evidence**（标准答案对应的原始证据）作为上下文，**绕过检索环节**。

```
┌─────────────────────────────────────────────────────────────────┐
│  正常 RAG:  问题 ──▶ [检索] ──▶ 文档 ──▶ [LLM] ──▶ 答案         │
│                        ↑                                        │
│                   可能检索错误                                   │
├─────────────────────────────────────────────────────────────────┤
│  Oracle:    问题 ──────────▶ GT文档 ──▶ [LLM] ──▶ 答案          │
│                              ↑                                  │
│                         完美检索                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 为什么 Oracle 是上界?

Oracle 代表**完美检索**的理想情况：
- 如果 RAG 能检索到与 Oracle 相同的文档，理论上应达到相同准确率
- RAG 准确率 ≤ Oracle 准确率 (除非检索到更好的文档)

### 为什么 Oracle 不是 100%?

即使给了正确文档，LLM 仍然会出错：

| 原因 | 示例 |
|------|------|
| **复杂推理** | 需要多步金融计算 |
| **格式匹配** | "1,577" vs "$1577.00" |
| **信息不足** | 问题需要额外背景知识 |
| **LLM能力限制** | 数学计算错误 |

### 准确率分解

```
RAG准确率 = 检索质量 × LLM生成能力

Oracle = 100% × 47.5% = 47.5%  (完美检索)
Best RAG = 89% × 47.5% ≈ 42.5%  (检索损失约11%)
```

![RAG vs Oracle](results/rag_vs_oracle.png)

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 DeepSeek API Key (config.py)
openai_api_key: str = "your-api-key"

# 3. 构建索引
python main.py build

# 4. 运行评估
python main.py eval+ 50    # 最佳配置
python main.py oracle 50   # Oracle上界
python main.py ablation 30 # 消融实验

# 5. 生成可视化
python analysis.py

# 6. 交互演示
python main.py demo
```

---

## 项目结构

```
RAGforFi/
├── config.py           # 配置文件 (API Key, 模型参数)
├── main.py             # 主程序入口
├── analysis.py         # 结果分析与可视化
├── src/
│   ├── parser.py       # 文档解析 (Docling PDF)
│   ├── chunker.py      # 表格感知分块
│   ├── indexer.py      # 混合索引 (Vector + BM25 + RRF)
│   ├── reranker.py     # Cross-Encoder 重排序 (可选)
│   ├── generator.py    # LLM 答案生成
│   ├── financebench.py # FinanceBench 数据集加载
│   └── pipeline.py     # RAG Pipeline 封装

```

---

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| Embedding | BAAI/bge-small-en-v1.5 | 轻量级中英文向量模型 |
| Reranker | BAAI/bge-reranker-base | Cross-Encoder (可选) |
| LLM | DeepSeek API | deepseek-chat |
| Vector Index | NumPy | 余弦相似度检索 |
| Keyword Index | BM25Okapi | rank-bm25 库 |
| Fusion | RRF | Reciprocal Rank Fusion |

---

## 参考文献

1. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
2. BGE Embedding: https://huggingface.co/BAAI/bge-small-en-v1.5
3. FinanceBench: https://huggingface.co/datasets/PatronusAI/financebench
