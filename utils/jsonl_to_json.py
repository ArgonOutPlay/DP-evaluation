import json

input_file = r"E:\RAG-eval\DP-evaluation\datasets\not_mine\lubos_dataset_new_bench_full.jsonl"
output_file = r"E:\RAG-eval\DP-evaluation\datasets\not_mine\lubos_dataset_new_bench_full.json"

data = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)