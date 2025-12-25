# RAG evaluation
- Main repository with RAG implementation is [here](https://github.com/DCGM/semant-demo).

### Evaluation
- evaluate.py
    - set env variables - env.example
    - How to run? Example: ```python evaluate.py --mode GT --core deepeval --eval_model OPENAI --rag_config_path rag_configurations\01_ollama_default_config.yaml --path_to_dataset datasets\NEW\200_llama_ollama_syn.json > eval_results\new\GT_200_llama_index_ollama_syn_deepeval_eval_ollama_gpt4o_config_01.txt```

### Generate synthetick dataset
- utils\generateTests_fromDB.py
    - set env variables - utils\env.example
    - How to run? Example: ```python utils\generateTests_fromDB.py --generator llama --model OLLAMA --num_of_generated_tests 1 --output_name datasets\NEW\test2512.json``` 
- generate dataset for NOGT mode - utils\generateNOGT.py
- check if chunks are in db - utils\check_chunks_in_db.py

### RAG configurations
- configurations for rag in main semant repository
- rag_configurations

### Datasets
- datasets
    - NEW - generated with utils\generateTests_fromDB.py
    - OLD_DB_unknown_ids - generated with utils\generateTests.py (no chunk ids)

### RAG results
- eval_results

