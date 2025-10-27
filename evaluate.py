#env
import os
from dotenv import load_dotenv
#load config (.env)
load_dotenv() #have to be called before config import
#semant app - RAG
from semant_demo.config import config
from semant_demo.weaviate_search import WeaviateSearch
from semant_demo.rag.rag_generator import RagGenerator
from semant_demo.schemas import RagConfig, RagSearch, SearchType
#ragas openai
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai
#ragas ollama
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
#ragas
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms.base import llm_factory

from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
#metrics imported as instances
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
#backend 
import json
import asyncio
#others
from typing import Tuple
import argparse

#colors for better logs in terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

# function that return parameters in required schemes
def createRagSupportParameters(model_name: str, question: str, search_type: str, alpha: float, limit: int, temperature: float = 0.0):
    #configuration of the model - api, model, temperature
        model_config = RagConfig(
            model_name=model_name,
            temperature=temperature
        )
        # create search query, will by used to create weaviate search query
        rag_search = RagSearch(
            search_query = question,
            limit= limit,
            search_type = search_type, #'hybrid', #vector: alpha=1, text: alpha=0
            alpha= alpha,
            min_year= None,
            max_year= None,
            min_date= None,
            max_date=None,
            language= None
        )
        return {
            "ragConfig": model_config,
            "ragSearch": rag_search
        }
        

#load questions/queries and ground truths from json file given path
def loadDataFromJson (path: str) -> Tuple[list[str], list[str]]:
    queries = []
    gts = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                if 'question' and 'ground_truth' in entry:
                    queries.append(entry['question'])
                    gts.append(entry['ground_truth'])

        print(f"Loaded {len(queries)} questions.")
        return queries, gts
    
    except FileNotFoundError:
        raise FileNotFoundError(f"\nRAG EVALUATION ERROR: File not found: {path}")
    except Exception as e:
        raise Exception(f"\nRAG EVALUATION ERROR: while loading file: {path}: {e}")


#load questions/queries from txt file given path
def loadDataFromTXT(path: str) -> list[str]:
    result = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                query = line.strip()
                if query:
                    result.append(query)
                    
        print(f"Loaded {len(result)} questions.")
        return result
    
    except FileNotFoundError:
        raise FileNotFoundError(f"\nRAG EVALUATION ERROR: File not found: {path}")
    except Exception as e:
        raise Exception(f"\nRAG EVALUATION ERROR: while loading file: {path}: {e}")

        

async def main():
    #get required variables
    parser = argparse.ArgumentParser(description="Evaluator for RAG.")
    parser.add_argument("--mode",
                        type=str,
                        default="NOGT",
                        choices=["NOGT", "GT"],
                        help="Evaluation mode: 'NOGT' (.json required) or 'GT' (.txt required) ")
    parser.add_argument("--rag_model",
                        type=str,
                        default="OLLAMA",
                        choices=["OLLAMA","GOOGLE", "OPENAI"],
                        help="Model used by RAG: 'OLLAMA', 'GOOGLE' or 'OPENAI' ")
    parser.add_argument("--temperature",
                    type=float,
                    default=0.0,
                    help="RAG model temperature.")
    parser.add_argument("--search_type",
                    type=str,
                    default="hybrid",
                    choices=["hybrid","vector", "text"],
                    help="Search type used for searching in DB.")
    parser.add_argument("--alpha",
                    type=float,
                    default=0.5,
                    help="Hybrid search alpha parametr.")
    parser.add_argument("--limit",
                    type=int,
                    default=5,
                    help="Limit of chunks that will be retrieved during db search.")
    parser.add_argument("--eval_model",
                    type=str,
                    default="OLLAMA",
                    choices=["OLLAMA", "OPENAI"],
                    help="Model used for evaluation: 'OLLAMA' or 'OPENAI' ")
    parser.add_argument("--precission",
                    type=str,
                    default="OFF",
                    choices=["ON", "OFF"],
                    help="Only relevant in 'NOGT' mode. Precission choices: 'ON' or 'OFF' ")
    parser.add_argument("--path",
                    type=str,
                    default="PATH_MISSING",
                    help="Path to question file (.json with GT mode or .txt with NOGT mode)")
    parser.add_argument("--synthetic_dataset",
                    type=str,
                    default="OFF",
                    choices=["ON", "OFF"],
                    help="Run evaluation with synthetic dataset. Synthetic dataset choices: 'ON' or 'OFF' ")

    args = parser.parse_args()

    print(f"""{Colors.GREEN} 
            Starting evaluation in mode: {args.mode} with RAG model: {args.rag_model}
            and temperature: {args.temperature} and limit: {args.limit}
            and search type: {args.search_type} and alpha parametr: {args.alpha} 
            and evaluation model: {args.eval_model} and precission: {args.precission} 
            and synthetic dataset: {args.synthetic_dataset}. 
            {Colors.RESET} """)

    eval_model = args.eval_model
    rag_model = args.rag_model
    mode = args.mode
    precission_mode = False

    #--- get path ---
    if(args.path == "PATH_MISSING"):
        if(mode == "NOGT"):
            path = os.getenv("PATH_WITHOUT_GT")
        else:
            path = os.getenv("PATH_GT")
    else:
        path = args.path

    if (args.synthetic_dataset == "ON"):
        path = os.getenv("PATH_SYN")

    if (args.precission == "ON"):
        precission_mode = True
 
    #--- load data ---
    try:
        queries = []
        ground_truths = []
        # load from txt
        if (mode == "NOGT" ):
            queries = loadDataFromTXT(path)
        # load from json
        elif (mode == "GT"):
            queries, ground_truths = loadDataFromJson(path)
    except FileNotFoundError as e:
        print("Invalid path, error detail:", e)
        return
    except Exception as e:
        print("Error detail:", e)
        return

    #--- get desired evaluation model ---
    if (eval_model == "OPENAI"):
        eval_model_name = os.getenv("OPENAI_EVAL_MODEL")
        #API key is taken automaticly from env ( os.getenv("OPENAI_API_KEY") )
        llm = llm_factory(eval_model_name)
    elif(eval_model == "OLLAMA"):
        eval_model_name = os.getenv("OLLAMA_EVAL_MODEL")
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        llm = OllamaLLM(model=eval_model_name, base_url = ollama_url)
    else:
        print(f"\n Invalid model: {eval_model}. Possible models: [OPENAI, OLLAMA].")
        return

    print(f"{Colors.YELLOW} --- Model used for evaluation is {eval_model} precisely: {eval_model_name} --- {Colors.RESET}")
    
    #--- evaluation ---
    search = await WeaviateSearch.create(config=config)
    rag_generator = RagGenerator(config=config, search=search)
    try:
        dataset = []
        if (mode == "NOGT"):
            single_context_precision = LLMContextPrecisionWithoutReference(llm=llm)
            precisions = []

            for query in queries:
                ragSP = createRagSupportParameters(
                    model_name=rag_model,
                    question=query,
                    search_type=args.search_type,
                    alpha=args.alpha,
                    limit=args.limit,
                    temperature=args.temperature
                    
                )
                ragResult = await rag_generator.generate_answer(
                    question_string = query,
                    history= [],
                    #rag configuration parameters
                    rag_config = ragSP["ragConfig"],
                    #search parameters
                    rag_search = ragSP["ragSearch"]
                )
                retrieved_contexts_text = [ctx.text for ctx in ragResult["sources"]]
                dataset.append(
                    {
                        "user_input" : query,
                        "retrieved_contexts" : retrieved_contexts_text,
                        "response" : ragResult["answer"]
                    }
                )
                #calculating precission (current eval require GT to be able to calculate precission)
                if(precission_mode == True):
                    sample = SingleTurnSample(
                        user_input=query,
                        response=ragResult["answer"],
                        retrieved_contexts=retrieved_contexts_text
                    )
                    #add precision of sample to list
                    precisions.append(await single_context_precision.single_turn_ascore(sample))

            evaluation_dataset = EvaluationDataset.from_list(dataset)

            print(f"{Colors.YELLOW} ---Starting evaluation --- {Colors.RESET}")

            result = evaluate(dataset=evaluation_dataset,
                              metrics=[faithfulness, answer_relevancy],
                              llm=llm)
            
            print(f"{Colors.GREEN}---Evaluation finished ---{Colors.RESET}")
            print(f"{Colors.GREEN} {result} {Colors.RESET}")
            if(precission_mode == True):
                average_precision = sum(precisions) / len(precisions)
                print(f"{Colors.GREEN} average_precission: {average_precision} {Colors.RESET}")

        elif(mode == "GT"):
            for query, gt in zip(queries, ground_truths):
                ragSP = createRagSupportParameters(
                    model_name=rag_model,
                    question=query,
                    search_type=args.search_type,
                    alpha=args.alpha,
                    limit=args.limit,
                    temperature=args.temperature
                    
                )
                ragResult = await rag_generator.generate_answer(
                    question_string = query,
                    history= [],
                    #rag configuration parameters
                    rag_config = ragSP["ragConfig"],
                    #search parameters
                    rag_search = ragSP["ragSearch"]
                )
                retrieved_contexts_text = [ctx.text for ctx in ragResult["sources"]]
                dataset.append(
                    {
                        "user_input":query,
                        "retrieved_contexts":retrieved_contexts_text,
                        "response":ragResult["answer"],
                        "reference": gt
                    }
                )
            evaluation_dataset = EvaluationDataset.from_list(dataset)

            print(f"{Colors.YELLOW} ---Starting evaluation --- {Colors.RESET}")

            result = evaluate(dataset=evaluation_dataset,
                              metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness],
                              llm=llm)
            
            print(f"{Colors.GREEN}---Evaluation finished ---{Colors.RESET}")
            print(f"{Colors.GREEN} {result} {Colors.RESET}")
        else:
            print(f"{Colors.RED}Invalid mode: {mode}. Possible modes: [GT, NOGT].{Colors.RESET}\n")

    except Exception as e:
        print(f"{Colors.RED} Error detail: {e} {Colors.RESET}\n")
    finally:
        if 'search' in locals():
            await search.close()


if __name__ == "__main__":
    asyncio.run(main())