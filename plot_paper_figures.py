import numpy as np
import matplotlib.pyplot as plt
import os

def plot_paper_figures():
    # 1. 确认数据路径
    mix_log_dir = "Mix-IFedNCF/log"
    base_log_dir = "IFedNCF/log"

    # 加载数据
    try:
        mix_cold = np.load(os.path.join(mix_log_dir, 'cold_recalls.npy'))
        base_cold = np.load(os.path.join(base_log_dir, 'cold_recalls.npy'))
        mix_warm = np.load(os.path.join(mix_log_dir, 'warm_recalls.npy'))
        base_warm = np.load(os.path.join(base_log_dir, 'warm_recalls.npy'))
    except FileNotFoundError as e:
        print(f"❌ 数据加载失败，请确保两边的训练脚本都已跑完并生成了 npy 文件。\n报错: {e}")
        return

    # 全局字体与样式设置 (学术标准)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.linewidth'] = 1.5
    
    # ---------------------------------------------------------
    # 绘制 图 A：冷启动收敛速度曲线 (交互视频个数 vs Recall)
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    
    # 假设前 20 步交互最能说明问题 (你可以调整截取长度)
    plot_steps = min(20, len(mix_cold))
    x_steps = np.arange(1, plot_steps + 1)
    
    plt.plot(x_steps, mix_cold[:plot_steps], marker='o', linestyle='-', color='#D32F2F', 
             linewidth=2.5, markersize=7, label='Ours (Popularity + Dwell Time)')
    plt.plot(x_steps, base_cold[:plot_steps], marker='s', linestyle='--', color='#1976D2', 
             linewidth=2.5, markersize=7, label='Baseline (Original IFedNCF)')
    
    plt.title('Cold-Start Convergence Efficiency', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Number of Interacted Items', fontsize=13)
    plt.ylabel('Recall@20 (Cold Users)', fontsize=13)
    plt.xticks(np.arange(1, plot_steps + 1, step=2), fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    plt.savefig('Figure_A_ColdStart.pdf', format='pdf', dpi=300)
    print("✅ 成功生成图 A：Figure_A_ColdStart.pdf")

    # ---------------------------------------------------------
    # 绘制 图 B：系统全局通信与收敛稳定性 (通信轮次 vs Recall)
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    
    x_rounds = np.arange(1, len(mix_warm) + 1)
    
    plt.plot(x_rounds, mix_warm, linestyle='-', color='#388E3C', 
             linewidth=2, label='Global System (with Ours)')
    plt.plot(x_rounds, base_warm, linestyle='--', color='#7B1FA2', 
             linewidth=2, alpha=0.8, label='Global System (Baseline)')
    
    # 标出冷启动接入的节点
    COLD_START_ROUND = 50
    plt.axvline(x=COLD_START_ROUND, color='gray', linestyle='-.', linewidth=1.5, alpha=0.7)
    plt.text(COLD_START_ROUND + 1, min(mix_warm) + 0.05, 'Cold Users\nJoined', color='gray', fontsize=10)

    plt.title('Global System Convergence Stability', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Federated Communication Round', fontsize=13)
    plt.ylabel('Recall@20 (Warm Users)', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9, edgecolor='black')
    plt.tight_layout()
    plt.savefig('Figure_B_Stability.pdf', format='pdf', dpi=300)
    print("✅ 成功生成图 B：Figure_B_Stability.pdf")

if __name__ == "__main__":
    plot_paper_figures()