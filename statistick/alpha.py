import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ALPHA AND METADATA GRAPH
def plot_full_ablation_study():    
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
    
    #answer relevancy
    rel_no_meta = [0.8629, 0.8074, 0.8589, 0.8881, 0.9017, 0.8935]
    rel_with_meta = [0.7736, 0.7795, 0.8727, 0.8885, 0.8696, 0.8564]
    
    # context suff
    suff_no_meta = [0.607, 0.621, 0.656, 0.669, 0.671, 0.695]
    suff_with_meta = [0.553, 0.571, 0.602, 0.661, 0.668, 0.694]

    plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20
    })

    fig, ax1 = plt.subplots(figsize=(13, 8))

    # Relevancy
    ax1.plot(alphas, rel_no_meta, marker='o', color='#1f77b4', linewidth=3, label='Relevance odpovědi (Bez metadat)')
    ax1.plot(alphas, rel_with_meta, marker='x', linestyle='--', color='#6baed6', linewidth=2, label='Relevance odpovědi (S metadaty)')
    ax1.set_xlabel('Parametr Alfa (0.0 = BM25, 1.0 = Sémantika)', fontsize=20)
    ax1.set_ylabel('Answer Relevancy skóre', color='#1f77b4', fontsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    ax1.set_ylim(0.70, 0.95)

    # Context Suff
    ax2 = ax1.twinx()
    ax2.plot(alphas, suff_no_meta, marker='o', color="#f33434", alpha=0.6, linewidth=2, label='Dostatečnost kontextu (Bez metadat)')
    ax2.plot(alphas, suff_with_meta, marker='x', linestyle='--', color="#A50707", alpha=0.6, linewidth=2, label='Dostatečnost kontextu (S metadaty)')
    ax2.set_ylabel('Context Sufficiency skóre', color='#d62728', fontsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    ax2.set_ylim(0.50, 0.75)

    #title
    plt.title('Vliv parametru Alfa a extrakce metadat na kvalitu systému', fontsize=24, pad=20)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # highlight best config
    ax1.annotate('Vítězná konfigurace\nHybrid α=0.6, Bez metadat', xy=(0.6, 0.9017), xytext=(0.2, 0.92), fontsize=20, arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    plt.tight_layout()
    plt.savefig('ablation_alpha_metadata.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_full_ablation_study()