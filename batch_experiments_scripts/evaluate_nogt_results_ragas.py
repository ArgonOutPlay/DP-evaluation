import json
import os
import sys
import typing as t
import argparse
from dotenv import load_dotenv
load_dotenv() #have to be called before config import
import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics.base import SingleTurnMetric, MetricWithLLM, MetricType
from ragas.dataset_schema import SingleTurnSample
from ragas.prompt import PydanticPrompt

from dataclasses import dataclass

import warnings
#ignore ragas warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed file.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed <ssl.SSLSocket*")
#have to ignore those two warnings, because recommended implementations dont work
warnings.filterwarnings("ignore", category=DeprecationWarning, message="LangchainLLMWrapper*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="LangchainEmbeddingsWrapper*")
import logging
#openai api is returning just one out of three results to ragas, but it should works just fine with temperature 0 --> ignoring warning 
logging.getLogger("ragas.prompt.pydantic_prompt").setLevel(logging.ERROR)

GLOBAL_METADATA_STORAGE = []

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# custom metric - context sufficiency - does retrieved context provide enaugh information to answer the question?
class SufficiencyInput(BaseModel):
    question: str = Field(description="the user's question to be answered")
    context: str = Field(description="the retrieved context snippets from documents")

class SufficiencyOutput(BaseModel):
    score: float = Field(description="0.0 = Irelevantní, 0.3 = Nedostatečné, 0.7 = Částečné, 1.0 = Úplné")
    classification: str = Field(description="Jedno slovo: [Irrelevant, Insufficient, Partial, Full]")
    missing_info: str = Field(description="Stručný popis v češtině, co v kontextu chybí pro úplnou odpověď. Pokud nechybí nic, napiš 'N/A'.")
    reasoning: str = Field(description="Stručné zdůvodnění v češtině.")

class SufficiencyPrompt(PydanticPrompt[SufficiencyInput, SufficiencyOutput]):
    instruction = """
    Jste expert na analýzu informací. Vaším úkolem je posoudit, zda poskytnutý KONTEXT obsahuje dostatek informací pro zodpovězení OTÁZKY.
    
    Pravidla pro skórování:
    1.0 (Full): Kontext obsahuje všechny klíčové i doplňující informace pro vyčerpávající odpověď.
    0.7 (Partial): Kontext obsahuje hlavní fakt, ale chybí detaily (např. chybí datum, i když událost tam je).
    0.3 (Insufficient): Kontext se tématu dotýká, ale neobsahuje přímou odpověď na jádro otázky.
    0.0 (Irrelevant): Kontext je o něčem úplně jiném.
    
    Výstup generujte v češtině.
    """
    input_model = SufficiencyInput
    output_model = SufficiencyOutput
    
    examples = [
        (
            SufficiencyInput(
                question="Kdy a kde se narodil Karel Čapek?",
                context="Karel Čapek byl český spisovatel, narodil se v roce 1890."
            ),
            SufficiencyOutput(
                score=0.7, 
                classification="Partial", 
                missing_info="Místo narození (obec/město).", 
                reasoning="Rok narození je uveden, ale místo chybí."
            )
        )
    ]

@dataclass
class ContextSufficiency(MetricWithLLM, SingleTurnMetric):
    name: str = "context_sufficiency"

    _required_columns = {
        MetricType.SINGLE_TURN: {"user_input", "retrieved_contexts"}
    }
    sufficiency_prompt: PydanticPrompt = SufficiencyPrompt()

    # can be empty - it is for compatibility with older version
    async def _ascore(self, row: t.Dict, callbacks: t.Any = None) -> float:
        pass

    # this is main ascore
    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: t.Any = None) -> float:
        # create prompt
        prompt_input = SufficiencyInput(
            question=sample.user_input,
            context="\n".join(sample.retrieved_contexts)
        )
        
        # generate results using llm
        prompt_response = await self.sufficiency_prompt.generate(
            data=prompt_input, 
            llm=self.llm
        )

        GLOBAL_METADATA_STORAGE.append({
            "user_input": sample.user_input,
            "sufficiency_classification": prompt_response.classification,
            "sufficiency_missing_info": prompt_response.missing_info,
            "sufficiency_reasoning": prompt_response.reasoning
        })
        
        return float(prompt_response.score)

    def init(self, run_config: t.Any):
        pass

async def main():
    global GLOBAL_METADATA_STORAGE
    GLOBAL_METADATA_STORAGE.clear()
    # get input path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path")
    args = parser.parse_args()

    eval_model_name = os.getenv("OPENAI_EVAL_MODEL")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    #format data into dataset
    formatted_data = []
    for item in data:
        formatted_data.append({
            "user_input": item["question"],
            "response": item["rag_answer"],
            "retrieved_contexts": item["retrieved_contexts"]
        })
    dataset = EvaluationDataset.from_list(formatted_data)
    llm = LangchainLLMWrapper(ChatOpenAI(model=eval_model_name, temperature=0, base_url=base_url, api_key=api_key))
    # config so ragas will not timeout
    run_config = RunConfig(max_workers=16, timeout=1600)

    #evaluation
    context_sufficiency = ContextSufficiency(llm=llm)
    metrics = [faithfulness, answer_relevancy, context_sufficiency]
    
    print(f"Evaluation started (NOGT), input: {args.input}")

    results = evaluate(dataset=dataset, metrics=metrics, llm=llm, run_config=run_config)

    #get path to results
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = "evaluation_results/ragas_nogt"
    os.makedirs(output_dir, exist_ok=True)

    #save detail score  - the most important output 
    results_df = results.to_pandas()

    if GLOBAL_METADATA_STORAGE:
        metadata_df = pd.DataFrame(GLOBAL_METADATA_STORAGE)
        metadata_df = metadata_df.drop_duplicates(subset=['user_input'])
        
        if "user_input" in results_df.columns:
            results_df = pd.merge(results_df, metadata_df, on="user_input", how="left")
            print(f"Successfully merged {len(metadata_df)} metadata entries.")
    else:
        print("Warning: No metadata was collected!")
        
    results_df.to_json(f"{output_dir}/{base_name}_evaluated_{eval_model_name}.json", orient="records", force_ascii=False, indent=4)

    # these outuputs are basicly same and are not that much immportant
    #save averages
    results_df = results_df.round(4)

    summary_dict = results_df.mean(numeric_only=True).to_dict()
    summary_dict["framework"] = "ragas_nogt"
    summary_dict["experiment"] = base_name
    summary_dict["judge"] = eval_model_name

    with open(f"{output_dir}/{base_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False)

    #save markdown
    try:
        summary_df = pd.DataFrame([summary_dict])
        # reorganize columns
        cols = ["experiment", "judge"] + [c for c in summary_df.columns if c not in ["experiment", "judge"]]
        summary_df = summary_df[cols]
        
        with open(f"{output_dir}/{base_name}_report.md", "w", encoding="utf-8") as f:
            f.write(f"# Evaluation Report (Ragas): {base_name}\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n## Detailed Scores per Question\n")
            # save few first lines
            f.write(results_df.head(10).to_markdown(index=False))
    except ImportError:
        pass

    #save csv
    summary_df.to_csv(f"{output_dir}/{base_name}_summary.csv", index=False)

    print(f"Evaluation complete (Ragas), results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())