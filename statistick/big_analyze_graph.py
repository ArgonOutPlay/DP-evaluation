import matplotlib.pyplot as plt
import numpy as np


# MAIN DP GRAPHS
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,  
    "xtick.labelsize": 16,
    "ytick.labelsize": 20,
    "legend.fontsize": 20
    })
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})

def plot_thesis_statistics():
    #JUDGE SENSITIVITY
    judges = ['GPT-4o-mini', 'GPT-4.1-mini', 'GPT-5.4-nano', 'GPT-5.4-mini', 'GPT-5.1', 'Lidský průměr']
    deltas = [0.0004, 0.0757, 0.097, 0.1046, 0.1159, 0.207]
    cis = [0.06, 0.05, 0.035, 0.04, 0.045, 0.065, 0.10]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3F3F42", "#445968", "#466b8b", '#1f77b4', "#3d86b9", '#d62728', "#d66d27"]
    for i in range(len(judges)):
        ax.errorbar(judges[i], deltas[i], yerr=cis[i], fmt='o', capsize=8, 
                    markersize=10, color=colors[i], elinewidth=3, markeredgewidth=2)
    
    plt.xticks(rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Nárůst relevance odpovědi (Delta)')
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('stat_1_judge_sensitivity.pdf', bbox_inches='tight')
    plt.show()

    # QUESTION COMPLEXITY COMPARISON
    categories = ['Jednoduché otázky', 'Složité otázky', 'Kombinované otázky']
    rel_deltas = [0.0626, 0.1159, 0.2221]
    rel_cis = [0.05, 0.045, 0.07]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    ax2.bar(x, rel_deltas, yerr=rel_cis, capsize=10, color='#2E8B57', alpha=0.7, label='Zlepšení Relevance')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Průměrné zlepšení (Delta)')
    ax2.set_ylim(0, 0.35)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i, v in enumerate(rel_deltas):
        ax2.text(i, v + 0.08, f"+{v*100:.1f}%", ha='center', fontsize=16)

    plt.tight_layout()
    plt.savefig('stat_2_complexity_utility.pdf', bbox_inches='tight')
    plt.show()

    metrics = ['Věrohodnost','Relevance\nodpovědi', 'Dostatečnost\nkontextu']
    
    # pearson consistency graph
    r_human = [0.607, 0.899, 0.868]
    r_gpt51 = [ 0.106, 0.497, 0.723] 
    r_gpt4o = [ 0.142, 0.267, 0.508] 

    y = np.arange(len(metrics))
    height = 0.25  

    fig, ax = plt.subplots(figsize=(12, 7))
    bar1 = ax.barh(y + height, r_human, height, label='Shoda člověka s člověkem ', color='#d9e7d5', alpha=0.8)
    bar2 = ax.barh(y, r_gpt51, height, label='Shoda člověka s GPT-5.1', color='#3d86b9', alpha=0.8)
    bar3 = ax.barh(y - height, r_gpt4o, height, label='Shoda člověka s GPT-4o-mini', color='#A9A9A9', alpha=0.8)


    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Pearsonův korelační koeficient (r)')
    ax.set_xlim(0, 1.1) 
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=18, frameon=True, shadow=True)

    def add_labels(bars):
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'r={width:.2f}', va='center', fontsize=20, fontweight='bold')

    add_labels(bar1)
    add_labels(bar2)
    add_labels(bar3)

    plt.tight_layout()
    plt.savefig('stat_3_consistency_POSTER.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_thesis_statistics()