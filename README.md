# RAG Evaluation and Synthetic Dataset Generation
- This repository provides tools for automated evaluation and synthetic data generation for RAG (Retrieval-Augmented Generation) systems. It is part of the [semANT](https://github.com/DCGM/semant-demo) project.

## Installation
- Clone the repository
- Install dependencies:
    - ``` pip install -r requirements-evaluate.txt ```
    - ``` pip install -r utils/requirements-generateTests.txt ```
- Install semANT Backend as an editable package:
    - ``` pip install -e path/to/semant_demo_backend ```
- Environment Setup - copy ```env.example ``` to ``` .env ``` and fill in API keys, database and ollama service url.


## Evaluation (``` evaluate.py ```)
- The main program for testing RAG configurations.
- Example: ```python evaluate.py --mode GT --core deepeval --eval_model OPENAI --rag_config_path rag_configurations/01_ollama_default_config.yaml --path_to_dataset datasets/NEW/200_llama_ollama_syn.json > eval_results/new/GT_200_llama_index_ollama_syn_deepeval_eval_ollama_gpt4o_config_01.txt```
- Parameters:
    - ```--mode``` : ```GT``` or ```NOGT```
    - ```--core``` : ```ragas``` or ```deepeval```
    - ```--eval_model``` : ```OPENAI``` or ```OLLAMA```
    - ```--path_to_dataset```
    - ```--rag_config_path```
    - ```--context_precision```: ```ON``` or ```OFF```
    - ```--context_relevancy```: ```ON``` or ```OFF```

## Dataset Generation (``` utils/```)
### Generate synthetic Dataset from Weaviate ( ```generateTests_fromDB.py ```)
- Example: ```python utils/generateTests_fromDB.py --generator llama --model OLLAMA --num_of_generated_tests 1 --output_name datasets/NEW/test2512.json```
### Utils
- Verifies that IDs in a dataset actually exist in the currently connected Weaviate instance - ```check_chunks_in_db.py```
- Converts JSON datasets with Ground Truth to simple TXT question lists for NOGT evaluation - ```generateNOGT.py```



## Project Structure
- ``` datasets/ ``` -  Contains generated and manually revised test sets (JSON/TXT).
- ``` eval_results/ ``` -  Logs and outputs from evaluation runs.
- ``` rag_configurations/ ``` - YAML files defining different RAGs.
- ``` utils/ ``` - Scripts for data preparation and generation.
- ``` deepeval_custom_model.py ``` - A custom wrapper used in ``` evaluate.py ```.

