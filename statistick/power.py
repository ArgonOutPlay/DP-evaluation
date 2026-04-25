import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestPower

#POWER ANALYZE

JUDGE_PAIRS = [
    ("GPT-5.1 (Hist)", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_51_evaluated_gpt-5.1.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_51_evaluated_gpt-5.1.json"),
    ("GPT-5.4-mini (Hist)", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_54mini_evaluated_gpt-5.4-mini.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_54mini_evaluated_gpt-5.4-mini.json"),
    ("GPT-5.4-nano (Hist)", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_54nano_evaluated_gpt-5.4-nano.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_54nano_evaluated_gpt-5.4-nano.json"),
    ("GPT-4.1-mini (Hist)", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_41mini_evaluated_gpt-4.1-mini.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_41mini_evaluated_gpt-4.1-mini.json"),
    ("GPT-4o-mini (Hist)", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_4omini_evaluated_gpt-4o-mini.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_4omini_evaluated_gpt-4o-mini.json")
]

METRICS = ['answer_relevancy', 'faithfulness', 'context_sufficiency']

def calculate_comprehensive_stats():
    all_results = []
    power_analysis = TTestPower()
    n_range = np.arange(10, 501, 10)
    
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 10
    })
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    metric_labels = ['Answer Relevancy', 'Faithfulness', 'Context Sufficiency']

    for m_idx, metric in enumerate(METRICS):
        ax = axes[m_idx]
        
        for name, b_path, a_path in JUDGE_PAIRS:
            if not os.path.exists(b_path) or not os.path.exists(a_path): continue
            
            df_b = pd.read_json(b_path)
            df_a = pd.read_json(a_path)
            merged = pd.merge(df_b, df_a, on="user_input", suffixes=('_b', '_a'))
            diffs = (merged[f'{metric}_a'] - merged[f'{metric}_b']).dropna()
            mu = np.mean(diffs)
                        
            sd = np.std(diffs, ddof=1)
            n = len(diffs)
            eff_size = mu / sd if sd != 0 else 0
            pwr = power_analysis.power(effect_size=eff_size, nobs=n, alpha=0.05, alternative='larger')
            
            try:
                n10 = int(np.ceil(power_analysis.solve_power(effect_size=(0.10/sd), power=0.8, alpha=0.05, alternative='larger')))
                n5 = int(np.ceil(power_analysis.solve_power(effect_size=(0.05/sd), power=0.8, alpha=0.05, alternative='larger')))
            except:
                n10, n5 = ">1000", ">1000"

            all_results.append({
                "Metrika": metric,
                "Soudce": name,
                "Delta": round(mu, 4),
                "SD (Šum)": round(sd, 4),
                "Power": round(pwr, 4),
                "Req. N (10% Delta)": n10,
                "Req. N (5% Delta)": n5
            })

            # graph to predict dataset size based on CI
            me = 1.96 * (sd / np.sqrt(n_range))
            ax.plot(n_range, me, label=f"{name}", linewidth=2)

        ax.set_title(f'Predikce přesnosti: {metric_labels[m_idx]}')
        ax.set_xlabel('Počet otázek (N)')
        ax.set_ylabel('Chyba odhadu (Margin of Error)')
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.4)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 0.3)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ax in axes:
        ax.tick_params(axis='x', labelbottom=True)
    plt.savefig('dataset_prediction.pdf', bbox_inches='tight')
    plt.show()

    return pd.DataFrame(all_results)

df_final = calculate_comprehensive_stats()
df_final = df_final.sort_values(by=['Metrika', 'Soudce'])
print("\n=== KOMPLETNÍ POWER ANALÝZA (Všechny metriky) ===")
print(df_final.to_string(index=False))