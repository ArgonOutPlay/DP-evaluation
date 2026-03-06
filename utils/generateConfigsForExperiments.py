import os
import yaml
import copy

model_type = "OPENAI"         #in the template
search_type = "hybrid"        #in the template
# openai_models = ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4.1" , "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]
openai_models = ["gpt-5.1", "gpt-4o-mini", "gpt-4.1-nano"]  # ["gpt-5.1", "gpt-5-mini", "gpt-5-nano"]
temperatures = [0.0]
alphas =  [0.5]                 #[0.3, 0.5, 0.7]
chunk_limits = [3]              #[1, 3, 5, 10]
qt_strategies = ["multi_query"] #["multi_query", "hyde", "nothing"]
max_retries = [3] #[1, 3]
web_search_enabled = [True]
metadata_extraction_allowed = [True]
self_reflection = [True]

template = {
    "class_name": "AdaptiveRagGenerator",
    "params": {
        "model_type" : model_type,
        "search_type" : search_type
    }
}

output_dir = "rag_configurations/experiments/eval_judge"
os.makedirs(output_dir, exist_ok=True)

counter = 0
for strat in qt_strategies:
    for a in alphas:
        for chunk_lim in chunk_limits:
            for model in openai_models:
                temp = 0.0
                mr = 3
                for ws in web_search_enabled:
                    for mea in metadata_extraction_allowed:
                        for sr in self_reflection:
                            config = copy.deepcopy(template)
                            config_id = f"exp_{counter:04d}"
                            config["id"] = f"exp_model_type_{model_type}_search_type_{search_type}_model_{model}_temp_{temp}_alpha_{a}_chunk_limit_{chunk_lim}_qt_strategy_{strat}_max_retries_{mr}_web_search_enabled_{ws}_metadata_extraction_allowed_{mea}_self_reflection_{sr}"
                            config["name"] = f"exp_model_type_{model_type}_search_type_{search_type}_model_{model}_temp_{temp}_alpha_{a}_chunk_limit_{chunk_lim}_qt_strategy_{strat}_max_retries_{mr}_web_search_enabled_{ws}_metadata_extraction_allowed_{mea}_self_reflection_{sr}"
                            config["description"] = f"exp_model_type_{model_type}_search_type_{search_type}_model_{model}_temp_{temp}_alpha_{a}_chunk_limit_{chunk_lim}_qt_strategy_{strat}_max_retries_{mr}_web_search_enabled_{ws}_metadata_extraction_allowed_{mea}_self_reflection_{sr}"
                            config["params"]["model_name"] = model
                            config["params"]["temperature"] = temp
                            config["params"]["alpha"] = a
                            config["params"]["chunk_limit"] = chunk_lim
                            config["params"]["qt_strategy"] = strat
                            config["params"]["max_retries"] = mr
                            config["params"]["web_search_enabled"] = ws
                            config["params"]["metadata_extraction_allowed"] = mea
                            config["params"]["self_reflection"] = sr

                            counter = counter +1

                            file_path = os.path.join(output_dir, f"{config_id}.yaml")
                            with open(file_path, "w", encoding="utf-8") as f:
                                yaml.dump(config, f, sort_keys=False)