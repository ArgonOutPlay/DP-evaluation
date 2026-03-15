import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
import json
import asyncio
import tqdm
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

#load config (.env)
load_dotenv() #have to be called before config import

#semant app - RAG
from semant_demo.config import config
from semant_demo.weaviate_search import WeaviateSearch
from semant_demo.schemas import SearchRequest, SearchResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import warnings
#ignore socket warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

#logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GenerateTestsFromQuestions")

generate_gt_prompt_template = [
    ("system", """
    Jsi odborný historik. Odpovídej plynule v češtině.
    Tvým úkolem je vygenerovat odpovědi na otázky, tyto odpovědi budou využity jako referenční odpovědi při evaluaci rag systému.
    Spoj informace z různých zdrojů tak, aby jednotlivé infromace mohly být citovány.
    
    PRAVIDLA:
    1) Odpověď začni PŘÍMO fakty. Nepoužívej úvody jako "Ohledně...", "Na základě..." nebo "V kontextu...".
    2) Piš v odstavcích, buď věcný, ale ne strohý.
    3) Pokud kontext neobsahuje informaci, v odpovědi ji úplně VYNECHEJ. Nepiš o tom, co v textu není. 
       Jen pokud v kontextu není VŮBEC NIC k tématu, napiš jedinou větu: "K tomuto tématu chybí v dostupných pramenech podklady."
     
    Kontext: \n {context_string} \n
    """),
    ("user", "{question_string}")
]


multiquery_prompt_template = [
    ("system", 
    """
    Jsi expert na vyhledávání v historických archivech. Tvým úkolem je vygenerovat 3 RŮZNÉ varianty vyhledávacího dotazu na základě otázky uživatele.

    Pravidla pro tvorbu dotazů:
    1) ZACHOVEJ ENTITY: Všechna data (např. '10. 12. 1863'), názvy zákonů a vlastní jména ponechej PŘESNĚ tak, jak jsou.
    2) SYNONYMA: Použij synonyma pro hlavní děje a důsledky, abys pokryl různé způsoby, jakými může být událost v archivech popsána.
    3) STEP-BACK DOTAZ: Jeden dotaz zaměř na širší historické souvislosti nebo obecný právní a sociální rámec dané doby.
    
    FORMÁT: Vypiš POUZE samotné dotazy, každý na nový řádek. Žádný úvodní text, žádné číslování."""),
    ("user", "{question_string}")
]

GPTmodel =  "gpt-5.1"
gen_answer = ChatPromptTemplate.from_messages(generate_gt_prompt_template)
gen_multiquery = ChatPromptTemplate.from_messages(multiquery_prompt_template)

#functions to format context for model
def get_clean_doc(doc) -> str:
    doc_info = doc.document_object
    title = getattr(doc_info, "title", "Unknown titel")
    year = getattr(doc_info, "yearIssued", "Unknown year")
    clean_doc = f"DOCUMENT/DOKUMENT: {title} (Year issued/Rok vydání: {year}) "
    clean_doc = clean_doc + f"CONTENT/OBSAH: " + str(doc.text.replace('\\n', ' '))

    return clean_doc

def format_weaviate_context(results: list[SearchResponse]) -> str:
    snippets = []
    for i, res in enumerate(results):
        clean_text = get_clean_doc(res)
        snippets.append(f"\n---SOURCE [doc{i+1}] START ---\n{clean_text}\n--- SOURCE [doc{i+1}] END ---")
    return ("\n".join(snippets))

async def main():
    #paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #DATASET_PATH = os.path.join(base_dir, "..", "datasets/NEW/200cze_gpt51_complex_questions_altered.json")
    DATASET_PATH = os.path.join(base_dir, "..", "datasets/NEW/last80_200cze_gpt51_complex_questions_altered.json")
    OUTPUT_DIR = os.path.join(base_dir, "..", "datasets") 
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # init searcher
    searcher = await WeaviateSearch.create(config=config)

    try:
        # load dataset
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        output = os.path.join(OUTPUT_DIR, f"last80.json")

        api_key = os.getenv("OPENAI_API_KEY")
        
        generator_llm = ChatOpenAI(
                model = GPTmodel,
                api_key = api_key,
                temperature = 0.0
        )
        alpha = 0.5

        final_data = []
        #generate data - answers
        for entry in tqdm.tqdm(dataset, desc=f"Generating questions", leave=False):
            question = entry["question"]
            source_chunk_id = entry["source_chunk_id"]

            try:
                # multiquery to have broader search
                multiquery_chain = gen_multiquery | generator_llm | StrOutputParser()
                result_raw = await multiquery_chain.ainvoke({"question_string" : question})
                queries = [line.strip().lstrip("0123456789.- ") for line in result_raw.split("\n") if len(line.strip()) > 5]
                queries = queries[:3]
                if question not in queries:
                    queries.append(question)
                # query to db in parallel and get contexts
                async def single_search(query):
                    search_request = SearchRequest(
                        query = query,
                        type = "hybrid",
                        hybrid_search_alpha = alpha,
                        limit = 7,
                        min_year = None,
                        max_year = None,
                        min_date = None,
                        max_date = None,
                        language = None,
                        tag_uuids = [],
                        positive = False,
                        automatic = False,
                        is_hyde = False
                    )
                    #call db search
                    return await searcher.search(search_request)
                
                #make answers little bit different - change alpha
                if (len(final_data) % 30 == 0) and (len(final_data) > 0):
                    alpha = min(1.0, alpha + 0.1)

                #call in parallel
                search_tasks = [single_search(query) for query in queries]
                search_responses = await asyncio.gather(*search_tasks)
                
                #remove duplicities and put it together
                unique_chunks = {}
                for response in search_responses:
                    #for chunk in response.results[:3]:
                    for chunk in response.results:
                        chunk_id = getattr(chunk, "id", None)
                        if chunk_id not in unique_chunks:
                            unique_chunks[chunk_id] = chunk

                all_chunks = list(unique_chunks.values())
                all_chunks = all_chunks[:20]
                context =  format_weaviate_context(all_chunks)

                #invoke chain with retrieved context to generate gt / answer
                answer_chain = gen_answer | generator_llm | StrOutputParser()
                ground_truth = await answer_chain.ainvoke({"context_string" : context, "question_string" : question})

                #make item in dataset
                question_number = len(final_data) + 1
                final_data.append({
                    "question_id": f"gen_cust_{question_number}",
                    "source_chunk_id" : source_chunk_id, 
                    "question": question,
                    "ground_truth": ground_truth,
                    "context": context
                })


            except Exception as e:
                print("\nError occurred while generating data, error detail:", e)
  
        #save results
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await searcher.close()

if __name__ == "__main__":
    asyncio.run(main())