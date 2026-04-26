import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from scipy.stats import pearsonr

# SCRIPT TO ANALYZ MOSTOF THE DATA

# DATA PATHS

LA_HUMAN_DATA = [
    {"xls": r"statistick/la_eval_1.xlsx", "key": r"statistick/key_1.json"},
    {"xls": r"statistick/la_eval_2.xlsx", "key": r"statistick/key_2.json"}
]

MY_HUMAN_DATA = [
    {"xls": r"statistick/my_eval_1.xlsx", "key": r"statistick/key_1.json"},
    {"xls": r"statistick/my_eval_2.xlsx", "key": r"statistick/key_2.json"}
]

JUDGE_FILES = {
    "GPT-5.1": r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_51_evaluated_gpt-5.1.json",
    "GPT-5.4-mini": r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_54mini_evaluated_gpt-5.4-mini.json",
    "GPT-5.4-nano": r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_54nano_evaluated_gpt-5.4-nano.json",
    "GPT-4.1-mini": r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_41mini_evaluated_gpt-4.1-mini.json",
    "GPT-4o-mini": r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_4omini_evaluated_gpt-4o-mini.json"
}

PAIRED_HIST = [
    ("GPT-5.1", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_51_evaluated_gpt-5.1.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_51_evaluated_gpt-5.1.json"),
    ("GPT-5.4-mini", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_54mini_evaluated_gpt-5.4-mini.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_54mini_evaluated_gpt-5.4-mini.json"),
    ("GPT-5.4-nano", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_54nano_evaluated_gpt-5.4-nano.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_54nano_evaluated_gpt-5.4-nano.json"),
    ("GPT-4.1-mini", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_41mini_evaluated_gpt-4.1-mini.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_41mini_evaluated_gpt-4.1-mini.json"),
    ("GPT-4o-mini", r"evaluation_results/ragas_nogt/different_models/100_basic_rag_4omini_evaluated_gpt-4o-mini.json", r"evaluation_results/ragas_nogt/different_models/100_v25_4_3_4omini_evaluated_gpt-4o-mini.json")
]

PAIRED_COMBINED = [
    ("GPT-4o-mini", r"evaluation_results/ragas_nogt/gpt4omini/combined_01_ollama_default_config_results_evaluated_gpt-4o-mini.json", r"evaluation_results/ragas_nogt/gpt4omini/combined_incremental_adaptive_v25_4_3_results_evaluated_gpt-4o-mini.json"),
    ("GPT-5.4-mini", r"evaluation_results/ragas_nogt/gpt54mini/combined_01_ollama_default_config_54mini_evaluated_gpt-5.4-mini.json", r"evaluation_results/ragas_nogt/gpt54mini/combined_incremental_adaptive_v25_4_3_54mini_evaluated_gpt-5.4-mini.json")
]

PAIRED_SIMPLE = [
    ("GPT-5.4-mini", r"evaluation_results/ragas_nogt/gpt54mini/100simple_basic_rag_54mini_results_evaluated_gpt-5.4-mini.json", r"evaluation_results/ragas_nogt/gpt54mini/100simple_incremental_adaptive_v25_4_3_54omini_results_evaluated_gpt-5.4-mini.json")
]

# loading functions
#load stuff from excel
def get_col(df, keyword, system):
    for col in df.columns:
        if keyword.lower() in col.lower() and f"_{system}_" in col.upper(): return col
    for col in df.columns:
        if keyword.lower() in col.lower() and (f" {system} " in col or f"_{system}" in col): return col
    return None

def load_human_data(human_data=LA_HUMAN_DATA):
    all_rows = []
    for item in human_data:
        if not os.path.exists(item["xls"]): continue
        df = pd.read_excel(item["xls"])
        with open(item["key"], 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        
        cols = {
            'rel_a': get_col(df, "Rel", "A"), 'rel_b': get_col(df, "Rel", "B"),
            'faith_a': get_col(df, "Faith", "A"), 'faith_b': get_col(df, "Faith", "B"),
            'suff_a': get_col(df, "Suff", "A"), 'suff_b': get_col(df, "Suff", "B")
        }

        for _, row in df.iterrows():
            q = row['Otázka']
            if q not in mapping: continue
            is_a = mapping[q]['System_A'] == 'Adaptive'
            all_rows.append({
                "question": q,
                "h_rel_adapt": row[cols['rel_a']] if is_a else row[cols['rel_b']],
                "h_rel_basic": row[cols['rel_b']] if is_a else row[cols['rel_a']],
                "h_faith_adapt": row[cols['faith_a']] if is_a else row[cols['faith_b']],
                "h_faith_basic": row[cols['faith_b']] if is_a else row[cols['faith_a']],
                "h_suff_adapt": row[cols['suff_a']] if is_a else row[cols['suff_b']],
                "h_suff_basic": row[cols['suff_b']] if is_a else row[cols['suff_a']]
            })
    return pd.DataFrame(all_rows).dropna()

#compute funstions
def calc_stats_safe(a_vals, b_vals):
    mask = ~np.isnan(a_vals) & ~np.isnan(b_vals)
    a, b = a_vals[mask].astype(float), b_vals[mask].astype(float)
    if len(a) < 2: return 0, [0,0], 1, 0, len(a)
    diffs = a - b
    mean_d = np.mean(diffs)
    sem = stats.sem(diffs)
    ci = stats.t.interval(0.95, len(diffs)-1, loc=mean_d, scale=sem)
    _, p = stats.ttest_rel(a, b)
    std_d = np.std(diffs, ddof=1)
    coh = mean_d / std_d if std_d != 0 else 0
    return mean_d, ci, p, coh, len(a)

# pearson - correlation and CI
def calculate_pearson_ci(x, y, confidence=0.95):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4: return 0, (0, 0)
    
    r, _ = pearsonr(x, y)

    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    low_z = z - z_crit * se
    high_z = z + z_crit * se
    low_r = np.tanh(low_z)
    high_r = np.tanh(high_z)
    
    return r, (low_r, high_r)

# START

df_la = load_human_data(LA_HUMAN_DATA)
df_my = load_human_data(MY_HUMAN_DATA)
df_inter = pd.merge(df_la, df_my, on="question", suffixes=('_la', '_my'))

# 000000000000000000000000000000000
print("\n=== 0. MEZIHODNOTITELSKÁ SHODA (n=100 - adaptivní i basic)===")
print(f"{'Metrika':20} | {'Korelace (r)':12} | {'95% CI':18} | {'MAE'}")
print("-" * 70)

metrics_to_stack = [
    ('h_rel_adapt', 'h_rel_basic', 'Answer Relevancy'),
    ('h_faith_adapt', 'h_faith_basic', 'Faithfulness'),
    ('h_suff_adapt', 'h_suff_basic', 'Context Sufficiency')
]

for col_adapt, col_basic, label in metrics_to_stack:
    # Spojíme výsledky pro oba systémy do jednoho vektoru
    la_all = pd.concat([df_inter[f'{col_adapt}_la'], df_inter[f'{col_basic}_la']])
    my_all = pd.concat([df_inter[f'{col_adapt}_my'], df_inter[f'{col_basic}_my']])
    
    r, ci = calculate_pearson_ci(la_all, my_all)
    mae = np.mean(np.abs(la_all - my_all))
    
    print(f"{label:20} | {r:+.3f}      | [{ci[0]:.2f}, {ci[1]:.2f}] | {mae:.3f}")



def print_stat_block(title, paired_list):
    print(f"\n=== {title} ===")
    metrics = ['answer_relevancy', 'faithfulness', 'context_sufficiency']
    for name, pb, pa in paired_list:
        if os.path.exists(pb) and os.path.exists(pa):
            df_a, df_b = pd.read_json(pa), pd.read_json(pb)
            m = pd.merge(df_a, df_b, on="user_input", suffixes=('_a', '_b'))
            print(f"\n--- Soudce: {name} (n={len(m)}) ---")
            print(f"{'Metrika':20} | {'Delta':7} | {'95% CI':16} | {'p-val':8} | {'D'}")
            for metric in metrics:
                d, ci, p, coh, n = calc_stats_safe(m[f'{metric}_a'], m[f'{metric}_b'])
                print(f"{metric:20} | {d:+.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {p:.2e} | {coh:.2f}")

print("\n=== Hodnocení průměrného lidského soudce ===")

# # 111111111111111111111111111111111111111111111111
df_cons = pd.merge(df_la, df_my, on="question", suffixes=('_la', '_my'))
for m in ['rel_adapt', 'rel_basic', 'faith_adapt', 'faith_basic', 'suff_adapt', 'suff_basic']:
    df_cons[f'h_{m}_cons'] = (df_cons[f'h_{m}_la'] + df_cons[f'h_{m}_my']) / 2


print("\n=== 1. KALIBRACE SOUDCŮ: LIDSKÝ KONSENZUS VS AI (n=100 - adaptivní i basic) ===")
print(f"{'Judge':12} | {'Metric':19} | {'Corr':6} | {'Bias':7} | {'SD Err':7} | {'MAE':6}")
print("-" * 85)

metrics_map = {
    'answer_relevancy': ('h_rel_basic_cons', 'h_rel_adapt_cons'),
    'faithfulness': ('h_faith_basic_cons', 'h_faith_adapt_cons'),
    'context_sufficiency': ('h_suff_basic_cons', 'h_suff_adapt_cons')
}

for judge_name, path_basic, path_adapt in PAIRED_HIST:
    if os.path.exists(path_basic) and os.path.exists(path_adapt):
        df_basic = pd.read_json(path_basic)
        df_adapt = pd.read_json(path_adapt)

        merged_basic = pd.merge(df_cons, df_basic, left_on='question', right_on='user_input')
        merged_adapt = pd.merge(df_cons, df_adapt, left_on='question', right_on='user_input')

        if merged_basic.empty or merged_adapt.empty:
            continue

        for ai_m, (hum_basic, hum_adapt) in metrics_map.items():
            if ai_m not in merged_basic.columns or ai_m not in merged_adapt.columns:
                continue

            h_all = pd.concat([
                merged_basic[hum_basic],
                merged_adapt[hum_adapt]
            ], ignore_index=True).astype(float)

            a_all = pd.concat([
                merged_basic[ai_m],
                merged_adapt[ai_m]
            ], ignore_index=True).astype(float)

            corr, _ = pearsonr(h_all, a_all)
            diffs = a_all - h_all

            print(f"{judge_name:12} | {ai_m:19} | {corr:+.3f} | {np.mean(diffs):+.3f} | {np.std(diffs, ddof=1):.3f} | {np.mean(np.abs(diffs)):.3f}")
        print("-" * 85)

# 222222222222 33333333333333333 44444444444444444
print_stat_block("2. STATISTICKÁ VÝZNAMNOST (Historie, n=100)", PAIRED_HIST)
print_stat_block("3. ROBUSTNOST (Combined dataset, n=125)", PAIRED_COMBINED)
print("\n=== 4. LIDSKÉ HODNOCENÍ ===")

human_configs = [
    (df_la, "HODNOTITEL: LA (Nezávislý)"),
    (df_my, "HODNOTITEL: MY (Autor)"),
    (df_cons, "LIDSKÝ KONSENZUS (Průměr)")
]

for df_to_eval, label in human_configs:
    print(f"\n--- {label} (n={len(df_to_eval)}) ---")
    print(f"{'Metrika':20} | {'Delta':7} | {'95% CI':16} | {'p-val':8} | {'Cohen D'}")
    
    # Detekce názvů sloupců (konsenzus má jiné přípony než jednotlivci)
    if "h_rel_adapt_cons" in df_to_eval.columns:
        h_metrics = [
            ('h_rel_adapt_cons', 'h_rel_basic_cons', 'Answer Relevancy'),
            ('h_faith_adapt_cons', 'h_faith_basic_cons', 'Faithfulness'),
            ('h_suff_adapt_cons', 'h_suff_basic_cons', 'Context Sufficiency')
        ]
    else:
        h_metrics = [
            ('h_rel_adapt', 'h_rel_basic', 'Answer Relevancy'),
            ('h_faith_adapt', 'h_faith_basic', 'Faithfulness'),
            ('h_suff_adapt', 'h_suff_basic', 'Context Sufficiency')
        ]
        
    for col_a, col_b, m_name in h_metrics:
        d, ci, p, coh, n = calc_stats_safe(df_to_eval[col_a], df_to_eval[col_b])
        print(f"{m_name:20} | {d:+.3f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {p:.2e} | {coh:.2f}")

df = pd.merge(df_la, df_my, on="question", suffixes=('_la', '_my'))

for m in ['rel_adapt', 'rel_basic', 'faith_adapt', 'faith_basic', 'suff_adapt', 'suff_basic']:
    df[f'h_{m}_cons'] = (df[f'h_{m}_la'].astype(float) + df[f'h_{m}_my'].astype(float)) / 2

print("\n=== PRŮMĚRNÉ LIDSKÉ HODNOCENÍ ===")
metrics = {
    "Answer Relevancy (Adapt)": 'h_rel_adapt_cons',
    "Answer Relevancy (Basic)": 'h_rel_basic_cons',
    "Faithfulness (Adapt)": 'h_faith_adapt_cons',
    "Faithfulness (Basic)": 'h_faith_basic_cons',
    "Context Sufficiency (Adapt)": 'h_suff_adapt_cons',
    "Context Sufficiency (Basic)": 'h_suff_basic_cons'
}

for name, col in metrics.items():
    mean_val = df[col].mean()
    print(f"{name:30}: {mean_val:.3f}")

# 55555555555555555555555555555555555
print_stat_block("5. ANALÝZA JEDNODUCHÝCH OTÁZEK (n=86, Simple Dataset)", PAIRED_SIMPLE)