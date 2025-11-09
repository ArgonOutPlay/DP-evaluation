import os
from dotenv import load_dotenv
import random
import json
import argparse
import glob
from typing import List

#llama
from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator    # ignore warnings, older version is required
#openai
from llama_index.llms.openai import OpenAI
#ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate

#deepeval
from deepeval.synthesizer import Synthesizer, Evolution
#models
from deepeval.models import GPTModel as DeepEvalOpenAI
from deepeval.models import OllamaModel as DeepEvalOllama

#colors for better logs in terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'

#load .env
load_dotenv()

#function that load data from .jsonl given path
#in this case load all chunks from document
def loadDataFromJsonl (path: str) -> List[str]:
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'text' in data and data['text']:
                    texts.append(data['text'])
            except json.JSONDecodeError:
                continue 
    return texts

ending = "*.jsonl"   #glob
#Used models
GPTmodel = "gpt-4o"
OLLAMAmodel = "gemma3:12b"
#custom prompts used for generating questions and answers with LlamaIndex
generate_question_prompt_template = (
    """
    Here is a piece of text with some information:
    {context_str}
    Your task is to act as user who has not seen this text. \
    Based on the information in the text, formulate ONE natural sounding question about a key fact or detail. 
    Follow these rules:
    1) The question MUST be standalone question that makes sense on its own.
    2) The question MUST be answerable using only the facts present in the context. Do not ask for interpretations, motivations, \
        or feelings that are not explicitly stated. For example, avoid questions like "Why was he..." or "What did she mean by..."
    4) Do NOT mention the text, context, document. For example, do not ask things such as: "Based on the context..." or "Given a text" or "What does text say about..." 
    5) Question should be something a real person would ask to learn specific information.
    6) Respond ONLY with generated question and nothing else.
    """
)

generate_gt_prompt_template = (
    """
    Here is the context:
    {context_str}
    Given the context information and not prior knowledge, answer the following question with whole sentence.
    Question: {query_str}
    Follow these rules:
    1) Answer the question using ONLY the given context. Do not generate any other text. Respond ONLY with the answer.
    2) Do NOT just extract name or a date. Explain the answer based on the context.
    3) Do NOT write things such as "According to text..." or "Based on the context..."
    """
)

def main():
    parser = argparse.ArgumentParser(description="Genereting questions for RAG.")
    parser.add_argument("--generator",
                    type=str,
                    default="llama",
                    choices=["llama", "deepeval"],
                    help="Library used for generating questions.")
    parser.add_argument("--model",
                        type=str,
                        default="OLLAMA",
                        choices=["OLLAMA", "OPENAI"],
                        help="Evaluation models: 'OLLAMA' (server/local) or 'OPENAI' (API) ")
    parser.add_argument("--input",
                    type=str,
                    default="/mnt/ssd2/weaviate_data/all.768/chunks.vec.lang",
                    help="Input path.")
    parser.add_argument("--output",
                    type=str,
                    default="dataset_synthetick_gt.json",
                    help="Output path.")
    parser.add_argument("--num_chunks_to_proc",
                    type=int,
                    default=1)
    parser.add_argument("--num_files_to_proc",
                    type=int,
                    default=1)
    parser.add_argument("--timeout",
                    type=float,
                    default=300.0,
                    help="Timeout for Ollama, default is 300.0s. Only works for llama generator. Its timeout for whole dataset. (one Ollama on sophie ~ 500 questions in 300.0s)")
    parser.add_argument("--show_progress",
                action="store_true",
                help="Show progress while generating data. Information about every sample.")
    
    args = parser.parse_args()

    show_progress = False
    if (args.show_progress):
        show_progress = True

    timeout = args.timeout
    if (timeout < 10.0):
        timeout = 10.0

    num_chunks_to_proc = args.num_chunks_to_proc
    if (num_chunks_to_proc < 0):
        num_chunks_to_proc = 1

    num_files_to_proc = args.num_files_to_proc
    if (num_files_to_proc < 0):
        num_files_to_proc = 1

    print(f"{Colors.GREEN} Generating  with library: {args.generator} and with model: {args.model}, number of tests to generate: {num_chunks_to_proc} from: {num_files_to_proc} documents, using {num_chunks_to_proc} chunks. Reading from: {args.input} and saving to: {args.output}. Ollama timeout is {timeout}. Show progress: {args.show_progress}. {Colors.RESET} ")
    
    #path to data
    in_dir_path = args.input
    out_dir_path = args.output

    #--- load data ---
    try:
        #get all files and select desired amount (base on input arguments)
        files = glob.glob(os.path.join(in_dir_path, ending))
        num_files_final = min(num_files_to_proc, len(files))
        selected_files = random.sample(files, num_files_final)

        #load data from selected files
        data = []
        print(f"Number of files found: {len(files)}, Number of files to process: {len(selected_files)}")
        #load data from files
        for filepath in selected_files:
            data.extend(loadDataFromJsonl(filepath))
        #gen desired amount of chunks to process
        data_reduced = random.sample(data, min(num_chunks_to_proc, len(data)))

        print("----- Data loaded -----")
    except Exception as e:
        print("\nError occured while loading data, error detail:", e)

    #--- generate data ---
    final_data = []
    #LlamaIndex
    if (args.generator == "llama"):
        #convert to desired format
        documents = [Document(text=t) for t in data_reduced]
        # custom prompts
        question_template = PromptTemplate(generate_question_prompt_template)
        answer_template = PromptTemplate(generate_gt_prompt_template)

        try:
            if (args.model == "OPENAI"):    #OPENAI
                generator_llm = OpenAI(model=GPTmodel)

            else:   #OLLAMA
                generator_llm = Ollama(model=OLLAMAmodel, base_url=os.getenv("OLLAMA_URL"), request_timeout = timeout)
        except Exception as e:
            print("\nError occured while creating model instances, error detail:", e)

        try:
            generator = DatasetGenerator.from_documents(
                documents=documents,
                llm=generator_llm,
                num_questions_per_chunk=1,
                show_progress=show_progress,
                # rewrite prompts
                text_question_template=question_template,
                text_qa_template=answer_template
            )

            #generate questions and gt
            gen_out = generator.generate_dataset_from_nodes()

            for i, pair in enumerate(gen_out.qr_pairs):
                final_data.append({
                    "question_id": f"gen_li_{i + 1}",
                    "question": pair[0],
                    "ground_truth": pair[1]
                })

        except Exception as e:
            print("\nError occured while generating data with LlamaIndex, error detail:", e)

    #Deepeval
    else:
        try:
            if (args.model == "OPENAI"):    #OPENAI
                generator_llm = DeepEvalOpenAI(model=GPTmodel)
            else:   #OLLAMA
                generator_llm = DeepEvalOllama(model=OLLAMAmodel, base_url=os.getenv("OLLAMA_URL"))

            generator = Synthesizer(
                model=generator_llm
            )

            final_contexts = [[c] for c in data_reduced]
            generator.generate_goldens_from_contexts(
                    contexts=final_contexts,
                    max_goldens_per_context=1,
                    include_expected_output=True
            )

            for i, golden in enumerate(generator.synthetic_goldens):
                final_data.append({
                    "question_id": f"gen_de_{i + 1}",
                    "question": golden.input,
                    "ground_truth": golden.expected_output
                })
            
        except Exception as e:
            print("\nError occured while generating data with Deepeval, error detail:", e)
        


    #---save data---
    try:
        output_name = f"{str(len(final_data))}_{num_files_final}_{args.generator}_{args.model}_{out_dir_path}"
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        print(f"{Colors.GREEN} ----- Generating completed, saved to file: {output_name} ----- {Colors.RESET}")

    except Exception as e:
        print("\nError occured while saving data, error detail:", e)

if __name__ == "__main__":
    main()