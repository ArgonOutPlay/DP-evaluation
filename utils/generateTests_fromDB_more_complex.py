import os
from dotenv import load_dotenv
import random

import json
import argparse
import weaviate
from weaviate.classes.query import Filter
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
    try:
        chunks_temp = client.collections.get("Chunks")
        lang_filter = Filter.by_property("language").equal("ces")

        # get random chunks -> to obtain documents
        response = await chunks_temp.query.fetch_objects(
            limit=5000,
            filters=lang_filter,
            return_properties=["from_page"],
            return_references=[weaviate.classes.query.QueryReference(link_on="document", return_properties=[] )]
        )

        data = []   #combined chunks - bigger context
        doc_question_count = {}

        #create combined chunks base on seed which are used to obtain documents
        for seed in response.objects:
            if len(data) >= limit: 
                break
            
            # get docs references
            doc_ref = seed.references.get("document")

            # skip invalid docs
            if (not doc_ref or not doc_ref.objects):
                continue
                
            #get doc id and start page
            d_id = str(doc_ref.objects[0].uuid)
            start_page = seed.properties["from_page"]

            # skip already used docs
            cur_doc_count = doc_question_count.get(d_id, 0)
            if cur_doc_count >= 5:
                continue

            # get chunks next to each other
            doc_chunks = await chunks_temp.query.fetch_objects(
                limit=4,
                filters=(
                    Filter.by_ref(link_on="document").by_id().equal(d_id) & 
                    Filter.by_property("from_page").greater_or_equal(start_page) &
                    Filter.by_property("from_page").less_than(start_page + 5) &
                    lang_filter
                ),
                return_properties=["text", "from_page"]
            )

            if len(doc_chunks.objects) < 2: 
                continue

            # sort chunks by ppage number and joint them together to create a big chunk for more complex context
            sorted_chunks = sorted(doc_chunks.objects, key=lambda x: x.properties.get("from_page", 0))
            combined_text = "\n\n[...] \n\n".join([c.properties["text"].strip() for c in sorted_chunks])

            data.append({
                "text": combined_text,
                "id": str(sorted_chunks[0].uuid)
            })
            doc_question_count[d_id] = cur_doc_count + 1
    finally:
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
    
    Your task is to act as a user who wants to understand the deeper context of this document. 
    Questions will be used to evaluate a sophisticated RAG system.
    
    Formulate ONE complex question that requires REASONING or CONNECTING multiple pieces of information from the text.
    
    Follow these rules:
    1) COMPLEXITY: Do not ask simple "When" or "Who" questions if possible. Instead, ask "Why", "How", or "What were the consequences of...".
    2) STANDALONE: The question must make sense on its own without seeing the text. The question should focus on the cause and effect described throughout the whole passage.
    3) NO PRONOUNS: Use specific names and entities (instead of "he", "this event", etc.).
    4) FACTUAL BASIS: The answer must be found in the text, but the question should require the reader to synthesize information.
    5) NO LEAKAGE: Do not include the answer within the question itself.
    6) NO META-TALK: Do not mention "the text", "the document" or "according to...".
    7) **LANGUAGE:** Match the language of the provided text. If the text is in Czech, ask in Czech. If in German, ask in German.
    8) Respond ONLY with the question.
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
    4) **LANGUAGE:** The answer MUST be written in the SAME LANGUAGE as the provided context.
    5) Respond ONLY with the answer.
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
        pbar = tqdm.tqdm(total=num_of_generated_tests, desc="Generating questions")
        for data in data_reduced:
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

                pbar.update(1)

                for pair in gen_out.qr_pairs:
                    question_number = len(final_data) + 1
                    final_data.append({
                        "question_id": f"gen_li_{question_number}",
                        "source_chunk_id" : data["id"], 
                        "question": pair[0],
                        "ground_truth": pair[1],
                        "context": data["text"]
                    })

                #end generation if enaught tests were generated
                if (len(final_data) == num_of_generated_tests):
                    break

            except Exception as e:
                print("\nError occurred while generating data with LlamaIndex, error detail:", e)
        pbar.close()

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
        pbar = tqdm.tqdm(total=num_of_generated_tests, desc="Generating questions")
        for data in data_reduced:
        
            try:
                final_context = [[data["text"]]]
                generator.generate_goldens_from_contexts(
                        contexts=final_context,
                        max_goldens_per_context=1,
                        include_expected_output=True
                )

                pbar.update(1)

                #generate questions and gt
                if (generator.synthetic_goldens):
                    question_number = len(final_data) + 1
                    golden = generator.synthetic_goldens[-1]
                    final_data.append({
                        "question_id": f"gen_de_{question_number}",
                        "source_chunk_id" : data["id"], 
                        "question": golden.input,
                        "ground_truth": golden.expected_output,
                        "context": data["text"]
                    })
                #end generation if enaught tests were generated
                if (len(final_data) == num_of_generated_tests):
                    break
            
            except Exception as e:
                print("\nError occured while generating data with Deepeval, error detail:", e)
        pbar.close()
    
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