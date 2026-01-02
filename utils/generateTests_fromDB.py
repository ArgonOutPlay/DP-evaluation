import os
from dotenv import load_dotenv
import random

import json
import argparse
import weaviate
import asyncio
import tqdm #progress bar
import nest_asyncio
nest_asyncio.apply()    # to be sure there were problems with llama generative function

#llama
from llama_index.core import Document
from llama_index.core.evaluation import DatasetGenerator    # ignore warnings, older version is required
#openai
from llama_index.llms.openai import OpenAI
#ollama
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate

#deepeval
from deepeval.synthesizer import Synthesizer
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

#load data from weaviate db
async def loadDataFromWeaviate(limit):
    client =  weaviate.use_async_with_custom(
            http_host=os.getenv("WEAVIATE_HOST"), http_port=int(os.getenv("WEAVIATE_REST_PORT")), http_secure=False,
            grpc_host=os.getenv("WEAVIATE_HOST"), grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT")), grpc_secure=False,
        )
    
    await client.connect()
    #load data
    chunks = client.collections.get("Chunks")
    #this is to randomize data
    fetch_limit = limit * 20 if limit < 50000 else limit
    response = await chunks.query.fetch_objects(
        limit=fetch_limit,
        return_properties=["text"]
    )

    data = []
    for item in response.objects:
        if "text" in item.properties:
            data.append({"text" : item.properties["text"], "id" : str(item.uuid)})

    await client.close()

    return data

#Used models
GPTmodel =  os.getenv("OPENAI_MODEL", "gpt-4o") 
OLLAMAmodel = os.getenv("OLLAMA_MODEL", "gemma3:12b")  
#custom prompts used for generating questions and answers with LlamaIndex
generate_question_prompt_template = (
    """
    Here is a piece of text with some information:
    {context_str}
    Your task is to act as user who has not seen this text. Questions will be used to evaluate RAG. \
    Based on the information in the text, formulate ONE natural sounding question about a key fact or detail. 
    Follow these rules:
    1) The question MUST be standalone question that makes sense on its own.
    2) **NO PRONOUNS or VAGUE TERMS:** Do not use words like "he", "she", "it", "the woman", "the man", "the company", "this article". 
       Instead, you MUST replace them with the SPECIFIC NAMES or ENTITIES found in the text (e.g., instead of "Why was she jailed?", ask "Why was Jane Doe jailed?").
    3) The question MUST be answerable using only the facts present in the context. Do not ask for interpretations, motivations, \
        or feelings that are not explicitly stated. For example, avoid questions like "Why was he..." or "What did she mean by..."
    4) If the text does not contain specific names to identify the subject, do not generate a question.
    5) Do NOT mention the text, context, document. For example, do not ask things such as: "Based on the context..." or "Given a text" or "What does text say about..." 
    6) Question should be something a real person would ask to learn specific information.
    7) Respond ONLY with generated question and nothing else.
    """
)

generate_gt_prompt_template = (
    """
    Here is the context:
    {context_str}
    Given the context information and not prior knowledge, answer the following question with whole sentence.
    Question: {query_str}
    Follow these rules:
    1) Answer the question using ONLY the given context. Do not generate any other text.
    2) Do NOT just extract name or a date. Explain the answer based on the context.
    3) Do NOT write things such as "According to text..." or "Based on the context..."
    4) Respond ONLY with the answer.
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
    parser.add_argument("--output_name",
                    type=str,
                    default="dataset_synthetic_gt.json",
                    help="Output path.")
    parser.add_argument("--num_of_generated_tests",
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

    num_of_generated_tests = args.num_of_generated_tests
    if (num_of_generated_tests < 0):
        num_of_generated_tests = 1

    print(f"{Colors.GREEN} Generating  with library: {args.generator} and with model: {args.model}, number of tests to generate: {num_of_generated_tests}, using {num_of_generated_tests} chunks. Reading from: DB and saving to: {args.output_name}. Ollama timeout is {timeout}. Show progress: {args.show_progress}. {Colors.RESET} ")
    
    #path to data output
    out_dir_path = args.output_name

    data_reduced = []
    #--- load data ---
    try:

        data_with_id = asyncio.run(loadDataFromWeaviate(limit=num_of_generated_tests))
        #if db have least data then desired
        data_reduced = random.sample(data_with_id, min(num_of_generated_tests * 2, len(data_with_id)))

        print("----- Data loaded -----")
    except Exception as e:
        print("\nError occured while loading data, error detail:", e)

    #--- generate data ---
    final_data = []
    #LlamaIndex
    if (args.generator == "llama"):
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

        #generate tests one by one because we need to connect it with their chunk id
        i = 0
        for data in tqdm.tqdm(data_reduced, desc="Generating questions"):
            i = i + 1
            #convert to desired format
            single_document = [Document(text=data["text"])]

            try:
                generator = DatasetGenerator.from_documents(
                    documents=single_document,
                    llm=generator_llm,
                    num_questions_per_chunk=1,
                    show_progress=show_progress,
                    # rewrite prompts
                    text_question_template=question_template,
                    text_qa_template=answer_template
                )

                #generate questions and gt
                gen_out = generator.generate_dataset_from_nodes()

                for pair in gen_out.qr_pairs:
                    final_data.append({
                        "question_id": f"gen_li_{i}",
                        "source_chunk_id" : data["id"], 
                        "question": pair[0],
                        "ground_truth": pair[1]
                    })

                #end generation if enaught tests were generated
                if (len(final_data) == num_of_generated_tests):
                    break

            except Exception as e:
                print("\nError occurred while generating data with LlamaIndex, error detail:", e)

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
        except Exception as e:
            print("\nError occurred while generating data with Deepeval, error detail:", e)

        #generate tests one by one because we need to connect it with their chunk id
        i = 0
        for data in tqdm.tqdm(data_reduced, desc="Generating questions"):
            i = i + 1
        
            try:
                final_context = [[data["text"]]]
                generator.generate_goldens_from_contexts(
                        contexts=final_context,
                        max_goldens_per_context=1,
                        include_expected_output=True
                )

                #generate questions and gt
                if (generator.synthetic_goldens):
                    golden = generator.synthetic_goldens[-1]
                    final_data.append({
                        "question_id": f"gen_de_{i}",
                        "source_chunk_id" : data["id"], 
                        "question": golden.input,
                        "ground_truth": golden.expected_output
                    })
                #end generation if enaught tests were generated
                if (len(final_data) == num_of_generated_tests):
                    break
            
            except Exception as e:
                print("\nError occured while generating data with Deepeval, error detail:", e)
    
    #return desired number of data/samples
    if (len(final_data) > num_of_generated_tests):
        final_data = final_data[:num_of_generated_tests]
    #---save data---
    try:
        output_name = out_dir_path
        with open(output_name, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        print(f"{Colors.GREEN} ----- Generating completed, saved to: {output_name} ----- {Colors.RESET}")

    except Exception as e:
        print("\nError occured while saving data, error detail:", e)

if __name__ == "__main__":
    main()