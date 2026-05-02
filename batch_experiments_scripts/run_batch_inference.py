import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import json
import asyncio
import tqdm
import logging
from time import time
from dotenv import load_dotenv

#load config (.env)
load_dotenv() #have to be called before config import

#semant app - RAG
from semant_demo.config import config
from semant_demo.weaviate_utils.weaviate_abstraction import WeaviateAbstraction
from semant_demo.rag.rag_factory import rag_load_single_config
from semant_demo.schemas import RagRequest, RagSearch

import warnings
#ignore socket warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BatchInference")

async def run_one_config(rag_config_path, dataset, search, previous_docs = None):
    #get rag
    _, _, rag_generator = rag_load_single_config(config, rag_config_path)
    results = []
    
    #run all questions/entries
    for entry in tqdm.tqdm(dataset, desc=f"Processing: {os.path.basename(rag_config_path)}", leave=False):
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        if (previous_docs == None):
            prev_docs = []
        try:
            start_time = time()
            rag_search = RagSearch(
                search_query=question  #not used in adaptive RAG, only in basic rag
            )
            #create request
            request = RagRequest(
                    question =  question,
                    history = [],
                    rag_search = rag_search,
                    previous_documents=prev_docs      #is used only in case of history would required specific dataset and specific runner
            )
            response = await rag_generator.rag_request(request, search)

            results.append({
                "question": question,
                "rag_answer": response.rag_answer,
                "retrieved_contexts": [ctx.text for ctx in response.sources],
                "ground_truth": ground_truth,
                "time_spent": time() - start_time,
                "id": entry.get("question_id", "unknown")
            })

        except Exception as e:
            logger.error(f"Error for question: {question[:30]} ---> error: {e}")

    return results


async def main():
    #paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #CONFIGS_DIR  = os.path.join(base_dir, "rag_configs")
    CONFIGS_DIR  = os.path.join(base_dir, "test_configs_03")    # test_configs_03
    #DATASET_PATH = os.path.join(base_dir, "datasets/200cze_gpt51_complex_questions_altered.json")
    #DATASET_PATH = os.path.join(base_dir, "datasets/first20_200cze_gpt51_complex_questions_altered.json")
    #DATASET_PATH = os.path.join(base_dir, "datasets/first20_updated_dataset.json")
    #DATASET_PATH = os.path.join(base_dir, "datasets/100_complex_altered.json")
    DATASET_PATH = os.path.join(base_dir, "datasets/200cze_gpt51_syn_human_altered.json")
    #DATASET_PATH = os.path.join(base_dir, "datasets/web_search_questions_new.json")
    #DATASET_PATH = os.path.join(base_dir, "datasets/combined_dataset.json")
    OUTPUT_DIR = os.path.join(base_dir, "..", "experiment_results", "raw_outputs_simple") #_tunning, _web, _simple, _updated, _big, raw_outputs_updated, v25_rerun, nogt_tests, nogt_tests_alpha
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #models gpt-4.1-mini, gpt-5.4-mini, gpt-5.4-nano

    # init searcher
    searcher = await WeaviateAbstraction.create(config=config)

    try:
        # load dataset
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        #get all configs and sort them in order
        yaml_configs = [f for f in os.listdir(CONFIGS_DIR) if f.endswith(".yaml")]
        yaml_configs.sort()

        print(f"Found {len(yaml_configs)} configurations. Starting batch processing...")

        #run all configurations
        for rag_config_name in yaml_configs:
            #get path and name
            config_path = os.path.join(CONFIGS_DIR, rag_config_name)
            config_id = rag_config_name.replace(".yaml", "")
            output = os.path.join(OUTPUT_DIR, f"{config_id}_results.json")

            if os.path.exists(output):
                print(f"Skipping {config_id}, results already exist.")
                continue

            print(f"Running config: {config_id}")

            #run on config
            data = await run_one_config(config_path, dataset, searcher)

            #save results
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            print(f"DONE: {config_id} saved.")

        print("All experiments completed.")
    except Exception as e:
        print(f"An error occurred during batch: {e}")
    finally:
        await searcher.close()

if __name__ == "__main__":
    asyncio.run(main())