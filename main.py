"""主程序入口 - FinanceBench RAG评估系统"""
import re
from tqdm import tqdm
from config import config

# ==================== 数据加载 ====================

def load_financebench(limit: int = None):
    """加载FinanceBench数据集"""
    from src.financebench import load_financebench as _load
    items = _load(config.data_dir)
    if limit:
        items = items[:limit]
    return items

# ==================== 索引构建 ====================

def build_index():
    """构建FinanceBench索引"""
    print("=" * 50)
    print("Building FinanceBench Index")
    print("=" * 50)
    
    from src.financebench import load_financebench, build_financebench_corpus
    from src.chunker import Chunk
    from src.indexer import HybridIndexer
    
    items = load_financebench(config.data_dir)
    corpus = build_financebench_corpus(items)
    
    # 转换为Chunk
    chunks = []
    for doc in corpus:
        chunks.append(Chunk(
            text=doc["text"],
            chunk_type="text",
            metadata=doc["metadata"]
        ))
    
    indexer = HybridIndexer(config.embedding_model, config.index_dir)
    indexer.index_chunks(chunks)
    print("\n✓ Index built successfully!")

# ==================== 检索优化策略 ====================

def enhance_query(question: str, company: str = "", doc_name: str = "") -> str:
    """Query增强：添加公司名和年份信息"""
    enhanced = question
    
    # 从doc_name提取年份 (e.g., "3M_2018_10K" -> "2018")
    year_match = re.search(r'(\d{4})', doc_name)
    year = year_match.group(1) if year_match else ""
    
    # 添加公司和年份上下文
    if company and company.lower() not in question.lower():
        enhanced = f"{company}: {enhanced}"
    if year and year not in question:
        enhanced = f"[{year}] {enhanced}"
    
    return enhanced

# ==================== 生成器 ====================

def get_generator():
    """获取生成器"""
    if config.use_local_llm:
        from src.generator import FinanceGeneratorLocal
        return FinanceGeneratorLocal(
            model=config.local_llm_model,
            base_url=config.local_llm_url
        )
    else:
        from src.generator import FinanceGenerator
        return FinanceGenerator(
            model=config.llm_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )

# ==================== 评估逻辑 ====================

def evaluate_answer(predicted: str, ground_truth: str) -> bool:
    """评估答案正确性（宽松匹配）"""
    pred_clean = predicted.lower().replace(",", "").replace("$", "").replace("%", "").strip()
    gt_clean = ground_truth.lower().replace(",", "").replace("$", "").replace("%", "").strip()
    
    # 提取数字
    pred_nums = re.findall(r'-?\d+\.?\d*', pred_clean)
    gt_nums = re.findall(r'-?\d+\.?\d*', gt_clean)
    
    # 数值比较 (5%容差)
    if pred_nums and gt_nums:
        try:
            pred_val = float(pred_nums[-1])
            gt_val = float(gt_nums[0])
            if gt_val != 0 and abs(pred_val - gt_val) / abs(gt_val) < 0.05:
                return True
            if gt_val == 0 and abs(pred_val) < 0.01:
                return True
        except:
            pass
    
    # 文本包含匹配
    if gt_clean in pred_clean or pred_clean in gt_clean:
        return True
    
    # 关键词匹配 (对于Yes/No/文本问题)
    gt_keywords = set(re.findall(r'\b\w+\b', gt_clean))
    pred_keywords = set(re.findall(r'\b\w+\b', pred_clean))
    if gt_keywords:
        overlap = len(gt_keywords & pred_keywords) / len(gt_keywords)
        if overlap > 0.6:
            return True
    
    return False

def run_evaluation(limit: int = 50, use_oracle: bool = False, 
                   use_reranker: bool = False, use_query_enhance: bool = False,
                   top_k_retrieve: int = None, top_k_rerank: int = None):
    """评估FinanceBench
    
    Args:
        limit: 评估样本数
        use_oracle: 使用ground truth context
        use_reranker: 使用Cross-Encoder重排序
        use_query_enhance: 使用Query增强
        top_k_retrieve: 初始检索数量(覆盖config)
        top_k_rerank: 重排序保留数量(覆盖config)
    """
    # 使用传入参数或config默认值
    top_k_ret = top_k_retrieve or config.top_k_retrieval
    top_k_rer = top_k_rerank or config.top_k_rerank
    
    # 构建模式描述
    mode_parts = []
    if use_oracle:
        mode_parts.append("Oracle")
    else:
        mode_parts.append(f"RAG(k={top_k_ret}→{top_k_rer})")
        if use_query_enhance:
            mode_parts.append("+QE")
        if use_reranker:
            mode_parts.append("+RR")
    mode = " ".join(mode_parts)
    
    print("=" * 50)
    print(f"FinanceBench Evaluation ({mode})")
    print("=" * 50)
    
    from src.chunker import Chunk
    
    items = load_financebench(limit)
    print(f"Evaluating {len(items)} questions...")
    
    generator = get_generator()
    
    # RAG模式加载索引和重排序器
    indexer = None
    reranker = None
    if not use_oracle:
        from src.indexer import HybridIndexer
        indexer = HybridIndexer(config.embedding_model, config.index_dir)
        indexer.load_bm25()
        
        if use_reranker:
            from src.reranker import Reranker
            reranker = Reranker(config.reranker_model)
    
    results = []
    correct = 0
    
    for item in tqdm(items, desc="Evaluating"):
        try:
            if use_oracle:
                # Oracle: 使用ground truth evidence
                chunk = Chunk(text=item.evidence, chunk_type="text", metadata={})
                context = [(chunk, 1.0)]
            else:
                # Query增强
                query = item.question
                if use_query_enhance:
                    query = enhance_query(query, item.company, item.doc_name)
                
                # 检索 - 使用参数传入的top_k
                context = indexer.search(query, top_k=top_k_ret)
                
                # 重排序
                if use_reranker and context:
                    context = reranker.rerank(query, context, top_k=top_k_rer)
                elif not use_reranker:
                    # 不使用reranker时，也截断到top_k_rerank
                    context = context[:top_k_rer]
            
            # 生成
            answer = generator.generate(item.question, context)
            
            # 评估
            is_correct = evaluate_answer(answer, item.answer)
            if is_correct:
                correct += 1
            
            results.append({
                "id": item.id,
                "company": item.company,
                "question": item.question,
                "predicted": answer[:200],
                "ground_truth": item.answer,
                "correct": is_correct
            })
            
        except Exception as e:
            print(f"\nError: {item.question[:50]}... - {e}")
            results.append({
                "id": item.id,
                "company": item.company,
                "question": item.question,
                "predicted": f"ERROR: {e}",
                "ground_truth": item.answer,
                "correct": False
            })
    
    accuracy = correct / len(items) if items else 0
    
    # 打印结果
    print(f"\n{'=' * 50}")
    print(f"Results ({mode})")
    print("=" * 50)
    print(f"Total: {len(items)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print(f"\nSample results:")
    for r in results[:5]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} [{r['company']}] {r['question'][:45]}...")
        print(f"      Pred: {r['predicted'][:60]}...")
        print(f"      GT: {r['ground_truth'][:60]}...")
    
    return results, accuracy

def run_ablation_study(limit: int = 30):
    """消融实验：对比不同检索策略和Top-K参数"""
    print("=" * 70)
    print("Ablation Study: Comparing RAG Strategies")
    print("=" * 70)
    
    configs = [
        # 基线
        {"name": "Oracle (Upper Bound)", "oracle": True, "rerank": False, "enhance": False, "k_ret": 20, "k_rer": 5},
        {"name": "RAG Baseline (k=20→5)", "oracle": False, "rerank": False, "enhance": False, "k_ret": 20, "k_rer": 5},
        
        # Top-K 对比
        {"name": "RAG (k=10→3)", "oracle": False, "rerank": False, "enhance": False, "k_ret": 10, "k_rer": 3},
        {"name": "RAG (k=30→8)", "oracle": False, "rerank": False, "enhance": False, "k_ret": 30, "k_rer": 8},
        
        # Query增强
        {"name": "RAG + QE (k=20→5)", "oracle": False, "rerank": False, "enhance": True, "k_ret": 20, "k_rer": 5},
        
        # Reranker (需要更大候选池)
        {"name": "RAG + RR (k=30→5)", "oracle": False, "rerank": True, "enhance": False, "k_ret": 30, "k_rer": 5},
        {"name": "RAG + RR (k=50→8)", "oracle": False, "rerank": True, "enhance": False, "k_ret": 50, "k_rer": 8},
        
        # 组合策略
        {"name": "RAG + QE + RR (k=50→8)", "oracle": False, "rerank": True, "enhance": True, "k_ret": 50, "k_rer": 8},
    ]
    
    results_summary = []
    
    for cfg in configs:
        print(f"\n>>> Running: {cfg['name']}")
        _, acc = run_evaluation(
            limit=limit,
            use_oracle=cfg["oracle"],
            use_reranker=cfg["rerank"],
            use_query_enhance=cfg["enhance"],
            top_k_retrieve=cfg["k_ret"],
            top_k_rerank=cfg["k_rer"]
        )
        results_summary.append({"config": cfg["name"], "accuracy": acc})
    
    # 汇总表格
    print("\n" + "=" * 70)
    print("Ablation Study Summary")
    print("=" * 70)
    print(f"{'Configuration':<35} {'Accuracy':>10}")
    print("-" * 47)
    for r in results_summary:
        print(f"{r['config']:<35} {r['accuracy']:>10.2%}")
    
    # 找出最佳配置
    best = max(results_summary, key=lambda x: x['accuracy'])
    print(f"\n✓ Best Config: {best['config']} ({best['accuracy']:.2%})")
    
    return results_summary

def interactive_demo():
    """交互式演示"""
    print("=" * 50)
    print("Interactive Demo")
    print("=" * 50)
    
    from src.indexer import HybridIndexer
    from src.reranker import Reranker
    
    print("\nLoading components...")
    indexer = HybridIndexer(config.embedding_model, config.index_dir)
    indexer.load_bm25()
    
    reranker = Reranker(config.reranker_model)
    generator = get_generator()
    
    print("✓ System ready!")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if not question:
            continue
        
        try:
            # 检索
            retrieved = indexer.search(question, top_k=config.top_k_retrieval)
            print(f"\n[Retrieved {len(retrieved)} chunks]")
            
            # 重排序
            reranked = reranker.rerank(question, retrieved, top_k=config.top_k_rerank)
            print(f"[Reranked to {len(reranked)} chunks]")
            
            # 生成
            answer = generator.generate(question, reranked)
            
            print(f"\n{'─' * 40}")
            print(f"Answer: {answer}")
            print(f"{'─' * 40}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")

def test_retrieval(limit: int = 5):
    """测试检索效果"""
    print("=" * 50)
    print("Testing Retrieval Quality")
    print("=" * 50)
    
    from src.indexer import HybridIndexer
    
    items = load_financebench(limit)
    indexer = HybridIndexer(config.embedding_model, config.index_dir)
    indexer.load_bm25()
    
    for item in items:
        print(f"\n[{item.company}] Q: {item.question[:60]}...")
        print(f"Expected evidence: {item.evidence[:100]}...")
        
        # 普通检索
        query = item.question
        results = indexer.search(query, top_k=3)
        
        print(f"\nRetrieved (basic query):")
        for i, (chunk, score) in enumerate(results):
            match = "✓" if item.evidence[:50] in chunk.text else " "
            print(f"  {match} [{i+1}] (score: {score:.3f}) {chunk.text[:80]}...")
        
        # 增强检索
        enhanced_query = enhance_query(query, item.company, item.doc_name)
        results_enhanced = indexer.search(enhanced_query, top_k=3)
        
        print(f"\nRetrieved (enhanced: {enhanced_query[:50]}...):")
        for i, (chunk, score) in enumerate(results_enhanced):
            match = "✓" if item.evidence[:50] in chunk.text else " "
            print(f"  {match} [{i+1}] (score: {score:.3f}) {chunk.text[:80]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [limit]")
        print("\nCommands:")
        print("  build     - Build FinanceBench index")
        print("  eval      - Evaluate with RAG baseline (k=30→8)")
        print("  eval+     - Evaluate with RAG + QueryEnhance (best config)")
        print("  oracle    - Evaluate with ground truth context (upper bound)")
        print("  ablation  - Run ablation study (compare all strategies)")
        print("  demo      - Interactive demo")
        print("  test      - Test retrieval quality")
        sys.exit(1)
    
    cmd = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if cmd == "build":
        build_index()
    elif cmd == "eval":
        # 使用最佳配置: k=30→8
        run_evaluation(limit or 50, use_oracle=False, 
                       top_k_retrieve=30, top_k_rerank=8)
    elif cmd == "eval+":
        # 最佳配置 + Query Enhancement
        run_evaluation(limit or 50, use_oracle=False, 
                       use_query_enhance=True, use_reranker=False,
                       top_k_retrieve=30, top_k_rerank=8)
    elif cmd == "oracle":
        run_evaluation(limit or 50, use_oracle=True)
    elif cmd == "ablation":
        run_ablation_study(limit or 30)
    elif cmd == "demo":
        interactive_demo()
    elif cmd == "test":
        test_retrieval(limit or 5)
    else:
        print(f"Unknown command: {cmd}")
