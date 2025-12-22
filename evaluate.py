#env
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
# wrappers
from deepeval_custom_model import CustomOpenAI
#load config (.env)
load_dotenv() #have to be called before config import
#semant app - RAG
from semant_demo.config import config
from semant_demo.weaviate_search import WeaviateSearch
from semant_demo.rag.rag_factory import rag_load_single_config
from semant_demo.schemas import RagRequest, RagSearch
#ragas openai
from langchain_openai import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
#ragas ollama
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
#ragas
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms.base import llm_factory
#context precission (basicly context relevancy but harsher (very simplified))
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithoutReference
#context relevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import ContextRelevance
#metrics imported as instances
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
#deepeval
from deepeval.test_case import LLMTestCase
from deepeval import evaluate as deepEvaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric
)
from deepeval.evaluate import AsyncConfig
#backend 
import json
import asyncio
#others
from typing import Tuple
import argparse
import warnings
#ignore deepeval warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed <socket*")
#ignore ragas warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed file.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed <ssl.SSLSocket*")
#have to ignote those two warnings, because recommended implementations dont work
warnings.filterwarnings("ignore", category=DeprecationWarning, message="LangchainLLMWrapper*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="LangchainEmbeddingsWrapper*")
import logging
#openai api is returning just one out of three results to ragas, but it should works just fine fith temperature 0 --> ignoring warning 
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)
#colors for better logs in terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

# function that return parameters in required schemes
def createRagSupportParameters(question: str):
        #create search
        rag_search = RagSearch(
            search_query=question   #with history its rephrased
        )
        #create request
        rag_request = RagRequest(
                question =  question,
                history = [],
                rag_search = rag_search
        )
        return rag_request
        

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
    parser.add_argument("--core",
                        type=str,
                        default="ragas",
                        choices=["ragas", "deepeval"],
                        help="Library used for evaluation.")
    parser.add_argument("--eval_model",
                    type=str,
                    default="OLLAMA",
                    choices=["OLLAMA", "OPENAI"],
                    help="Model used for evaluation: 'OLLAMA' or 'OPENAI' ")
    parser.add_argument("--context_precission",
                    type=str,
                    default="OFF",
                    choices=["ON", "OFF"],
                    help="Only relevant in 'NOGT' mode. Precission choices: 'ON' or 'OFF' ")
    parser.add_argument("--context_relevancy",
                    type=str,
                    default="OFF",
                    choices=["ON", "OFF"],
                    help="Does not work with OLLAMA evaluation model. Only relevant in 'NOGT' mode. Relevancy choices: 'ON' or 'OFF' ")
    parser.add_argument("--path_to_dataset",
                    type=str,
                    default="PATH_MISSING",
                    help="Path to question file (.json with GT mode or .txt with NOGT mode)")
    parser.add_argument("--rag_config_path",
                    type=str,
                    default="PATH_MISSING",
                    help="Path to rag configuration file (.yaml)")
    

    args = parser.parse_args()
    print(f"""{Colors.GREEN} 
            Starting evaluation in mode: {args.mode} and core: {args.core}
            and evaluation model: {args.eval_model} and context precission: {args.context_precission} 
            and context relevancy: {args.context_relevancy}.
            {Colors.RESET} """)

    #so deepeval will not timeout
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "6000"
    eval_model = args.eval_model
    rag_config_path = args.rag_config_path
    mode = args.mode
    precission_mode = True if args.context_precission == "ON" else False
    relevancy_mode = True if args.context_relevancy == "ON" else False

    if (relevancy_mode == True and eval_model == "OLLAMA"):
        print(f"{Colors.RED} Context relevancy does not work with OLLAMA evaluator (Is not supported in ragas.). {Colors.RESET}")
        return

    #--- get path ---
    if(args.path_to_dataset == "PATH_MISSING"):
        if(mode == "NOGT"):
            path = os.getenv("PATH_WITHOUT_GT")
        else:
            path = os.getenv("PATH_GT")
    else:
        path = args.path_to_dataset
 
    if (eval_model == "OLLAMA" and (relevancy_mode or precission_mode)):
        print(f"{Colors.YELLOW} In this version of Ragas context relevancy and precission are not supported for Ollama. We recommend you to use context precission instead or use OPENAI for evaluation. {Colors.RESET}")
        relevancy_mode = False

    #--- load data ---
    try:
        queries = []
        ground_truths = []
        # load from txt
        if (mode == "NOGT" ):
            if (path.endswith(".txt") == False):
                print(f"{Colors.RED} Error: In NOGT mode, input file has to be .txt {Colors.RESET}")
                return
            queries = loadDataFromTXT(path)
        # load from json
        elif (mode == "GT"):
            if (path.endswith(".json") == False):
                print(f"{Colors.RED} Error: In GT mode, input file has to be .json {Colors.RESET}")
                return
            queries, ground_truths = loadDataFromJson(path)
    except FileNotFoundError as e:
        print(f"{Colors.RED} Invalid path, error detail: {e} {Colors.RESET}")
        return
    except Exception as e:
        print(f"{Colors.RED} Error detail: {e} {Colors.RESET}")
        return

    #--- get desired evaluation model ---
    #deepeval have it implemented in itself, need to run with this command: deepeval set-ollama model-name
    if (args.core == "ragas"):
        if (eval_model == "OPENAI"):
            eval_model_name = os.getenv("OPENAI_EVAL_MODEL")
            #API key is taken automaticly from env ( os.getenv("OPENAI_API_KEY") )
            #have to use old version, because with ragas recommended version (llm_factory), code will fall because of max token limit
            base_llm = ChatOpenAI(model = eval_model_name, temperature=0, timeout=3000.0)
            llm = LangchainLLMWrapper(base_llm)

            # doesnt work with ragas embeddings, ignore warnings
            openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")
            openai_ragas_emb = LangchainEmbeddingsWrapper(openai_emb)
        elif(eval_model == "OLLAMA"):
            eval_model_name = os.getenv("OLLAMA_EVAL_MODEL")
            ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            llm = OllamaLLM(model=eval_model_name, base_url = ollama_url, request_timeout=3000.0 )  #new version can be used if not using context relevancy with NOGT
            ollama_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
            ollama_ragas_emb = LangchainEmbeddingsWrapper(ollama_emb)
        else:
            print(f"\n Invalid model: {eval_model}. Possible models: [OPENAI, OLLAMA].")
            return
    else:   #deepeval
        if (eval_model == "OPENAI"):
            eval_model_name = os.getenv("OPENAI_EVAL_MODEL")
            llm = CustomOpenAI(model_name=eval_model_name, max_tokens=4000, timeout=3000.0)

        else:
            print(f"{Colors.RED} OLLAMA is not supported as an evaluator with deepeval core. {Colors.RESET}")
            return
    
    #--- evaluation ---
    searcher = await WeaviateSearch.create(config=config)
    #create rag instance
    results = rag_load_single_config(config, rag_config_path)
    _, _, rag_generator = results
    try:
        dataset = []
        precisions = []
        relevancies = []
        if (mode == "NOGT" and args.core == "ragas"):
            single_context_precision_evaluator = LLMContextPrecisionWithoutReference(llm=llm)
            single_context_relevancy_evaluator = ContextRelevance(llm=llm)
        #call rag --> get query and context
        for i, query in enumerate(queries):
            ragSP = createRagSupportParameters(question=query)
            ragResult = await rag_generator.rag_request(ragSP, searcher)
            retrieved_contexts_text = [ctx.text for ctx in ragResult.sources]
            if (mode == "NOGT"):
                #ragas
                if (args.core == "ragas"):
                    dataset.append({"user_input" : query, "retrieved_contexts" : retrieved_contexts_text, "response" : ragResult.rag_answer})
                    #calculating precission (current eval require GT to be able to calculate precission)
                    if(precission_mode == True):
                        sample = SingleTurnSample(
                            user_input=query,
                            response=ragResult.rag_answer,
                            retrieved_contexts=retrieved_contexts_text
                        )
                        #add precision of sample to list
                        precisions.append(await single_context_precision_evaluator.single_turn_ascore(sample))

                    #calculate context relevancy
                    if(relevancy_mode == True):
                        sample = SingleTurnSample(
                            user_input=query,
                            retrieved_contexts=retrieved_contexts_text
                        )
                        #add precision of sample to list
                        relevancies.append(await single_context_relevancy_evaluator.single_turn_ascore(sample))
                else:
                    dataset.append(LLMTestCase(input=query, actual_output=ragResult.rag_answer, retrieval_context=retrieved_contexts_text))
            #GT
            else:
                #ragas
                if (args.core == "ragas"):
                    dataset.append({"user_input":query, "retrieved_contexts":retrieved_contexts_text, "response":ragResult.rag_answer, "reference": ground_truths[i]})
                #deepeval
                else:
                    dataset.append(LLMTestCase(input=query, actual_output=ragResult.rag_answer, retrieval_context=retrieved_contexts_text, expected_output=ground_truths[i]))
        #---eval setup---
        print(f"{Colors.YELLOW} ---Starting evaluation --- {Colors.RESET}")
        result = []
        metrics = []
        if (args.core == "ragas"):
            if (mode == "NOGT"):
                metrics = [faithfulness, answer_relevancy]
            else:
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness]
            evaluation_dataset = EvaluationDataset.from_list(dataset)
            if (eval_model == "OLLAMA"):
                result = evaluate(dataset=evaluation_dataset, metrics=metrics, llm=llm, embeddings=ollama_ragas_emb)
            else: #openai
                result = evaluate(dataset=evaluation_dataset, metrics=metrics, llm=llm, embeddings=openai_ragas_emb)
        else:   #deepeval have to be setuped like this to avoid token overflow - this allow to create own llm instance with token limit
            #NOGT
            metrics = [ContextualRelevancyMetric(model=llm, threshold=0.5), AnswerRelevancyMetric(model=llm, threshold=0.5), FaithfulnessMetric(model=llm, threshold=0.5)]
            if (mode == "GT"):
                metrics.append(ContextualPrecisionMetric(model=llm, threshold=0.5))
                metrics.append(ContextualRecallMetric(model=llm, threshold=0.5))
            async_config_deepeval = AsyncConfig(max_concurrent=5)
            result = deepEvaluate(dataset, metrics=metrics, async_config=async_config_deepeval)


        #--- write results ---
        print(f"{Colors.GREEN}---Evaluation finished ---{Colors.RESET}")
        #ragas
        if (args.core == "ragas"):
            print(f"{Colors.GREEN} {result} {Colors.RESET}")
        #deepeval
        else:
            pass    #deepeval write results in in evaluation function
            
        #only used in ragas with nogt module because newer version of ragas doesnt support it in evaluate without gt -->have to be done like this
        if (args.core == "ragas" and mode == "NOGT"):
            if(precission_mode == True):
                average_precision = sum(precisions) / len(precisions)
                print(f"{Colors.GREEN} average_precission: {average_precision} {Colors.RESET}")
            
            if(relevancy_mode == True):
                average_relevancy = sum(relevancies) / len(relevancies)
                print(f"{Colors.GREEN} average_relevancy: {average_relevancy} {Colors.RESET}")

    except Exception as e:
        print(f"{Colors.RED} Error detail: {e} {Colors.RESET}\n")
    finally:
        if 'search' in locals():
            await searcher.close()

if __name__ == "__main__":
    asyncio.run(main())