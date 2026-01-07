"""FinanceBench数据集支持 - 更适合RAG评估的金融问答数据集"""
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

@dataclass  
class FinanceBenchItem:
    """FinanceBench数据项"""
    id: str
    company: str
    doc_name: str
    question: str
    answer: str
    evidence: str  # 证据文本
    justification: str
    question_type: str

def load_financebench(cache_dir: str = "./dataset") -> List[FinanceBenchItem]:
    """加载FinanceBench数据集"""
    cache_path = Path(cache_dir) / "financebench.json"
    
    # 尝试从缓存加载
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from cache")
        return [FinanceBenchItem(**item) for item in data]
    
    # 从HuggingFace下载
    print("Downloading FinanceBench from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset('PatronusAI/financebench', split='train')
    
    items = []
    for row in ds:
        # 提取evidence文本
        evidence_text = ""
        if row.get("evidence"):
            evidence_list = row["evidence"]
            if isinstance(evidence_list, list) and len(evidence_list) > 0:
                evidence_text = evidence_list[0].get("evidence_text", "")
        
        items.append(FinanceBenchItem(
            id=row["financebench_id"],
            company=row["company"],
            doc_name=row["doc_name"],
            question=row["question"],
            answer=row["answer"],
            evidence=evidence_text,
            justification=row.get("justification", ""),
            question_type=row.get("question_type", "")
        ))
    
    # 缓存到本地
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([vars(item) for item in items], f, indent=2)
    
    print(f"Downloaded and cached {len(items)} items")
    return items

def build_financebench_corpus(items: List[FinanceBenchItem]) -> List[Dict]:
    """将evidence构建为检索语料库"""
    corpus = []
    seen = set()
    
    for item in items:
        if not item.evidence or item.evidence in seen:
            continue
        seen.add(item.evidence)
        
        corpus.append({
            "text": item.evidence,
            "metadata": {
                "company": item.company,
                "doc_name": item.doc_name,
                "source": "financebench"
            }
        })
    
    print(f"Built corpus with {len(corpus)} unique documents")
    return corpus
