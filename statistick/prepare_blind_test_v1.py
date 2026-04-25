import pandas as pd
import json
import random

def prepare_blind_test_excel(path_basic, path_adaptive, sample_size=25):
    #load datasets
    df_b = pd.read_json(path_basic)
    df_a = pd.read_json(path_adaptive)
    
    # remove dups and timeouts
    df = pd.merge(df_b, df_a, on="user_input", suffixes=('_b', '_a'))
    df = df.drop_duplicates(subset=['user_input'])
    df = df.dropna(subset=["faithfulness_b", "faithfulness_a"])
    df['delta_rel'] = df['answer_relevancy_a'] - df['answer_relevancy_b']
    df['is_web'] = df['user_input'].str.contains('hum_', na=False) | df['user_input'].str.contains('web', na=False)

    selected_indices = []
    web_df = df[df['is_web']]
    web_samples = web_df.sample(min(8, len(web_df)))
    selected_indices.extend(web_samples.index.tolist())
    
    hist_candidates = df[~df.index.isin(selected_indices) & ~df['is_web']]
    hist_samples_top = hist_candidates.sort_values(by='delta_rel', ascending=False).head(10)
    selected_indices.extend(hist_samples_top.index.tolist())
    
    remaining_pool = df[~df.index.isin(selected_indices)]
    needed = max(0, sample_size - len(selected_indices))
    random_samples = remaining_pool.sample(min(needed, len(remaining_pool)))

    final_selection = pd.concat([web_samples, hist_samples_top, random_samples])

    # randomize blind test
    blind_rows = []
    key_data = {}

    def format_context(ctx_list):
        if not isinstance(ctx_list, list):
            return str(ctx_list)
        formatted = []
        for i, text in enumerate(ctx_list):
            formatted.append(f"--- [ÚRYVEK {i+1}] ---\n{text}")
        return "\n\n".join(formatted)

    for _, row in final_selection.iterrows():
        is_adaptive_a = random.choice([True, False])
        
        if is_adaptive_a:
            key_data[row['user_input']] = {"System_A": "Adaptive", "System_B": "Basic"}
            ans_a, ctx_a = row['response_a'], row['retrieved_contexts_a']
            ans_b, ctx_b = row['response_b'], row['retrieved_contexts_b']
        else:
            key_data[row['user_input']] = {"System_A": "Basic", "System_B": "Adaptive"}
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
            "Kontext_A": format_context(ctx_a),
            "Kontext_B": format_context(ctx_b),
            "POZNÁMKA": ""
        })

    df_blind = pd.DataFrame(blind_rows)
    output_file = "slepý_test_k_hodnocení.xlsx"
    
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df_blind.to_excel(writer, index=False, sheet_name='Slepý Test')

    workbook  = writer.book
    worksheet = writer.sheets['Slepý Test']
    wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
    
    worksheet.set_column('A:A', 30, wrap_format)
    worksheet.set_column('B:C', 50, wrap_format)
    worksheet.set_column('D:I', 12, wrap_format)
    worksheet.set_column('J:K', 60, wrap_format) 
    worksheet.freeze_panes(1, 0)
    writer.close()

    with open("klíč_k_hodnocení.json", "w", encoding='utf-8') as f:
        json.dump(key_data, f, indent=4, ensure_ascii=False)
    
    print(f"DONE.")

#MAIN
path_b = r"E:\RAG-eval\DP-evaluation\evaluation_results\ragas_nogt\gpt4omini\combined_01_ollama_default_config_results_evaluated_gpt-4o-mini.json"
path_a = r"E:\RAG-eval\DP-evaluation\evaluation_results\ragas_nogt\gpt4omini\combined_incremental_adaptive_v25_4_3_results_evaluated_gpt-4o-mini.json"

prepare_blind_test_excel(path_b, path_a)