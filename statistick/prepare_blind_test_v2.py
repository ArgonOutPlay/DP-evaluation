import pandas as pd
import json
import random
import os

def prepare_blind_test_excel(path_basic, path_adaptive, old_key_path, sample_size=25):
    #load datasets
    df_b = pd.read_json(path_basic)
    df_a = pd.read_json(path_adaptive)
    
    #load old file to not have same questions
    if os.path.exists(old_key_path):
        with open(old_key_path, 'r', encoding='utf-8') as f:
            old_key = json.load(f)
        already_evaluated = list(old_key.keys())
        print(f"Nalezeno {len(already_evaluated)} již ohodnocených otázek. Budou vyřazeny z výběru.")
    else:
        already_evaluated = []
        print("Starý klíč nenalezen, vybírám z celého datasetu.")

    # remove dups and timeouts
    df = pd.merge(df_b, df_a, on="user_input", suffixes=('_b', '_a'))
    df = df.drop_duplicates(subset=['user_input'])
    df = df.dropna(subset=["faithfulness_b", "faithfulness_a"])
    
    # remove already evaluated
    df = df[~df['user_input'].isin(already_evaluated)]
    df['is_web'] = df['user_input'].str.contains('hum_', na=False) | df['user_input'].str.contains('web', na=False)
    
    # remove web questions
    hist_candidates = df[~df['is_web']]
    if len(hist_candidates) < sample_size:
        sample_size = len(hist_candidates)
    
    new_selection = hist_candidates.sample(sample_size)

    # randomize blind test
    blind_rows = []
    new_key_data = {}

    for _, row in new_selection.iterrows():
        is_adaptive_a = random.choice([True, False])
        
        if is_adaptive_a:
            new_key_data[row['user_input']] = {"System_A": "Adaptive", "System_B": "Basic"}
            ans_a, ctx_a = row['response_a'], row['retrieved_contexts_a']
            ans_b, ctx_b = row['response_b'], row['retrieved_contexts_b']
        else:
            new_key_data[row['user_input']] = {"System_A": "Basic", "System_B": "Adaptive"}
            ans_a, ctx_a = row['response_b'], row['retrieved_contexts_b']
            ans_b, ctx_b = row['response_a'], row['retrieved_contexts_a']
        
        blind_rows.append({
            "Otázka": row['user_input'],
            "Odpověď_A": ans_a,
            "Odpověď_B": ans_b,
            "Známka_A_Rel(1.0, 0.75, 0.5, 0.25, 0.0)": "",
            "Známka_A_Faith(1.0, 0.75, 0.5, 0.25, 0.0)": "",
            "Známka_A_Suff(1.0, 0.7, 0.3, 0.0)": "",
            "Známka_B_Rel(1.0, 0.75, 0.5, 0.25, 0.0)": "",
            "Známka_B_Faith(1.0, 0.75, 0.5, 0.25, 0.0)": "",
            "Známka_B_Suff(1.0, 0.7, 0.3, 0.0)": "",
            "Kontext_A": "\n---\n".join(ctx_a) if isinstance(ctx_a, list) else str(ctx_a),
            "Kontext_B": "\n---\n".join(ctx_b) if isinstance(ctx_b, list) else str(ctx_b),
        })

    #excell
    df_blind = pd.DataFrame(blind_rows)
    output_excel = "slepý_test_SET2.xlsx"
    output_key = "klíč_SET2.json"
    
    writer = pd.ExcelWriter(output_excel, engine='xlsxwriter')
    df_blind.to_excel(writer, index=False, sheet_name='Slepý Test SET 2')
    workbook  = writer.book
    worksheet = writer.sheets['Slepý Test SET 2']
    wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
    
    worksheet.set_column('A:A', 30, wrap_format)
    worksheet.set_column('B:C', 50, wrap_format)
    worksheet.set_column('D:I', 15, wrap_format)
    worksheet.set_column('J:K', 50, wrap_format)
    worksheet.freeze_panes(1, 0)
    writer.close()

    with open(output_key, "w", encoding='utf-8') as f:
        json.dump(new_key_data, f, indent=4, ensure_ascii=False)
    
    print(f"Soubor '{output_excel}' a '{output_key}' připraveny.")

#MAIN
path_basic = r"E:\RAG-eval\DP-evaluation\evaluation_results\ragas_nogt\gpt51\01_ollama_default_config_results_evaluated_gpt-5.1.json"
path_adaptive = r"E:\RAG-eval\DP-evaluation\evaluation_results\ragas_nogt\gpt51\server_100complex_ollama_incremental_adaptive_v25_4_3_results_evaluated_gpt-5.1.json"
old_key = "klíč_k_hodnocení.json"

prepare_blind_test_excel(path_basic, path_adaptive, old_key)