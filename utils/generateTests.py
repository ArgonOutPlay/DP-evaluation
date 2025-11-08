import os
from dotenv import load_dotenv
import random
import json
import pandas
import argparse
import glob
from typing import List

from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator
from tqdm import tqdm
#openai
from llama_index.llms.openai import OpenAI
#ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate

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
#number of generated questions
GPTmodel = "gpt-4o"
OLLAMAmodel = "gemma3:12b"

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
    """
)

def main():
    #parse parameters
    parser = argparse.ArgumentParser(description="Genereting questions for RAG.")
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
    
    args = parser.parse_args()
    print(f"{Colors.GREEN} Generating with model: {args.model}, number of tests to generate: {args.num_chunks_to_proc} from: {args.num_files_to_proc} documents, using {args.num_chunks_to_proc} chunks. Reading from: {args.input} and saving to: {args.output}. {Colors.RESET} ")
    
    num_chunks_to_proc = args.num_chunks_to_proc
    num_files_to_proc = args.num_files_to_proc

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
        #convert to desired format
        documents = [Document(text=t) for t in data_reduced]

        print("----- Data loaded -----")
    except Exception as e:
        print("\nError occured while loading data, error detail:", e)

    # custom prompts
    question_template = PromptTemplate(generate_question_prompt_template)
    answer_template = PromptTemplate(generate_gt_prompt_template)

    #--- generate data ---
    if (args.model == "OPENAI"):    #OPENAI
        generator_llm = OpenAI(model=GPTmodel)

    else:   #OLLAMA
        generator_llm = Ollama(model=OLLAMAmodel, base_url=os.getenv("OLLAMA_URL"), request_timeout = 300.0)

    generator = DatasetGenerator.from_documents(
        documents=documents,
        llm=generator_llm,
        num_questions_per_chunk=1,
        show_progress=True,
        # rewrite prompts
        text_question_template=question_template,
        text_qa_template=answer_template
    )


    #generate questions and gt
    gen_out = generator.generate_dataset_from_nodes()

    #--- save data ---
    final_data = []
    for i, pair in enumerate(gen_out.qr_pairs):
        final_data.append({
            "question_id": f"gen_li_{i + 1}",
            "question": pair[0],
            "ground_truth": pair[1]
        })

    output_name = str(len(final_data)) + "_" +  args.model + "_" + out_dir_path
    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"{Colors.GREEN} ----- Generating completed, saved to file: {output_name} ----- {Colors.RESET}")

if __name__ == "__main__":
    main()