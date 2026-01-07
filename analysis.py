"""结果分析与可视化"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 消融实验结果 (n=40)
ABLATION_RESULTS = {
    "Oracle (Upper Bound)": 0.475,
    "RAG (k=30→8)": 0.425,
    "RAG + QE (k=20→5)": 0.375,
    "RAG Baseline (k=20→5)": 0.300,
    "RAG (k=10→3)": 0.300,
    "RAG + RR (k=50→8)": 0.275,
    "RAG + QE + RR (k=50→8)": 0.250,
    "RAG + RR (k=30→5)": 0.225,
}

# 按准确率排序
SORTED_RESULTS = dict(sorted(ABLATION_RESULTS.items(), key=lambda x: x[1], reverse=True))

def plot_ablation_bar_chart():
    """绘制消融实验柱状图"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    configs = list(SORTED_RESULTS.keys())
    accuracies = [v * 100 for v in SORTED_RESULTS.values()]
    
    # 颜色区分
    colors = []
    for cfg in configs:
        if "Oracle" in cfg:
            colors.append('#2ecc71')  # 绿色 - Oracle
        elif "RR" in cfg:
            colors.append('#e74c3c')  # 红色 - Reranker (负面效果)
        elif cfg == "RAG (k=30→8)":
            colors.append('#3498db')  # 蓝色 - 最佳配置
        elif "QE" in cfg:
            colors.append('#9b59b6')  # 紫色 - Query Enhancement
        else:
            colors.append('#95a5a6')  # 灰色 - 基线
    
    bars = ax.barh(configs, accuracies, color=colors, edgecolor='white', height=0.7)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('FinanceBench RAG Ablation Study (n=40)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 55)
    ax.axvline(x=47.5, color='#2ecc71', linestyle='--', alpha=0.7, label='Oracle Upper Bound')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Oracle (Upper Bound)'),
        Patch(facecolor='#3498db', label='Best RAG Config'),
        Patch(facecolor='#9b59b6', label='Query Enhancement'),
        Patch(facecolor='#e74c3c', label='Reranker (Negative)'),
        Patch(facecolor='#95a5a6', label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/ablation_chart.png', dpi=150, bbox_inches='tight')
    plt.savefig('results/ablation_chart.svg', bbox_inches='tight')
    print("✓ Saved: results/ablation_chart.png")
    plt.close()

def plot_topk_analysis():
    """绘制Top-K参数分析图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Top-K 对比数据
    topk_data = {
        'k=10→3': 30.0,
        'k=20→5': 30.0,
        'k=30→8': 42.5,
    }
    
    x = list(topk_data.keys())
    y = list(topk_data.values())
    
    bars = ax.bar(x, y, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='white', width=0.6)
    
    for bar, val in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Top-K Configuration (retrieve → final)', fontsize=12)
    ax.set_title('Impact of Top-K on RAG Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 50)
    
    # 添加趋势线
    ax.plot([0, 1, 2], y, 'ko--', markersize=8, alpha=0.5)
    
    # 标注提升
    ax.annotate('+12.5%', xy=(2, 42.5), xytext=(1.5, 38),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=11, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/topk_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/topk_analysis.png")
    plt.close()

def plot_component_impact():
    """绘制各组件影响分析"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 各组件相对于基线的影响
    baseline = 30.0
    components = {
        'Baseline\n(k=20→5)': 0,
        'Top-K↑\n(k=30→8)': 42.5 - baseline,
        'Query\nEnhancement': 37.5 - baseline,
        'Reranker': 22.5 - baseline,  # 负面影响
    }
    
    x = list(components.keys())
    y = list(components.values())
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#e74c3c']
    
    bars = ax.bar(x, y, color=colors, edgecolor='white', width=0.6)
    
    for bar, val in zip(bars, y):
        label = f'+{val:.1f}%' if val >= 0 else f'{val:.1f}%'
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 1.5
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                label, ha='center', fontsize=12, fontweight='bold',
                color='green' if val > 0 else 'red')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Accuracy Change vs Baseline (%)', fontsize=12)
    ax.set_title('Component Impact Analysis', fontsize=14, fontweight='bold')
    ax.set_ylim(-10, 15)
    
    plt.tight_layout()
    plt.savefig('results/component_impact.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/component_impact.png")
    plt.close()

def plot_rag_vs_oracle():
    """绘制RAG vs Oracle对比"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['RAG Best\n(k=30→8)', 'Oracle\n(Upper Bound)']
    values = [42.5, 47.5]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('RAG Performance vs Oracle Upper Bound', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 55)
    
    # 添加差距标注
    gap = 47.5 - 42.5
    ax.annotate(f'Gap: {gap:.1f}%\n(Retrieval Loss)', 
                xy=(0.5, 45), xytext=(0.5, 50),
                ha='center', fontsize=11, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    # 添加解释
    ax.text(0.5, -8, 
            'Oracle uses ground-truth evidence (perfect retrieval)\n'
            'RAG gap represents retrieval quality loss',
            ha='center', fontsize=10, style='italic', color='gray',
            transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig('results/rag_vs_oracle.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/rag_vs_oracle.png")
    plt.close()

def generate_analysis_report():
    """生成分析报告"""
    report = """
# FinanceBench RAG 实验分析报告

## 1. 实验设置

- **数据集**: FinanceBench (150 questions, 40 samples evaluated)
- **Embedding**: BAAI/bge-small-en-v1.5
- **LLM**: DeepSeek-chat (via API)
- **索引**: NumPy向量 + BM25 混合检索 (RRF融合)

## 2. 消融实验结果

| 排名 | 配置 | 准确率 | vs基线 |
|------|------|--------|--------|
| 1 | Oracle (Upper Bound) | 47.5% | +17.5% |
| 2 | RAG (k=30→8) | 42.5% | +12.5% |
| 3 | RAG + QE (k=20→5) | 37.5% | +7.5% |
| 4 | RAG Baseline (k=20→5) | 30.0% | - |
| 5 | RAG (k=10→3) | 30.0% | ±0% |
| 6 | RAG + RR (k=50→8) | 27.5% | -2.5% |
| 7 | RAG + QE + RR (k=50→8) | 25.0% | -5.0% |
| 8 | RAG + RR (k=30→5) | 22.5% | -7.5% |

## 3. 关键发现

### 3.1 Top-K 参数的影响 (最重要的优化)
- k=10→3: 30.0%
- k=20→5: 30.0%
- k=30→8: **42.5%** (+12.5%)

**结论**: 增大候选池是最有效的优化策略。当检索更多候选文档时，正确文档被包含的概率显著提升。

### 3.2 Query Enhancement 的效果
- 基线: 30.0%
- +QE: 37.5% (+7.5%)

**结论**: 在查询中添加公司名称和年份上下文有助于检索定位到正确的财报文档。

### 3.3 Reranker 的负面效果
- 基线: 30.0%
- +RR: 22.5% (-7.5%)

**结论**: Cross-Encoder Reranker 反而降低了准确率。可能原因：
1. FinanceBench 每个问题只对应一个特定文档
2. Reranker 倾向于选择语义相似但非答案所在的文档
3. 金融术语的语义相似性可能误导重排序

### 3.4 Oracle 上界分析
- Oracle: 47.5% (使用 ground truth evidence)
- Best RAG: 42.5%
- Gap: 5.0%

**结论**: 即使完美检索，LLM 也只能达到 47.5% 准确率，说明：
1. 部分问题需要复杂金融推理
2. 答案格式匹配存在挑战
3. 某些问题可能需要多步计算

## 4. 优化建议

1. **继续增大 Top-K**: 尝试 k=50→10 或更大
2. **改进 Chunking**: 按公司/年份分区索引
3. **优化 LLM Prompt**: 针对金融计算设计更好的 CoT 模板
4. **多文档融合**: 某些问题可能需要跨文档信息

## 5. 可视化

![Ablation Chart](results/ablation_chart.png)
![Top-K Analysis](results/topk_analysis.png)
![Component Impact](results/component_impact.png)
![RAG vs Oracle](results/rag_vs_oracle.png)
"""
    
    Path("results").mkdir(exist_ok=True)
    with open("results/analysis_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("✓ Saved: results/analysis_report.md")

def main():
    """生成所有可视化"""
    Path("results").mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    plot_ablation_bar_chart()
    plot_topk_analysis()
    plot_component_impact()
    plot_rag_vs_oracle()
    generate_analysis_report()
    
    print("\n" + "=" * 50)
    print("All visualizations generated in results/")
    print("=" * 50)

if __name__ == "__main__":
    main()
