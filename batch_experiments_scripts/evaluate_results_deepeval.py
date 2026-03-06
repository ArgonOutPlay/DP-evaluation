import os
import json
import warnings
import argparse
import asyncio
import pandas as pd

#deepeval
from deepeval.test_case import LLMTestCase
from deepeval import evaluate as deepEvaluate
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric, 
    FaithfulnessMetric
)
from deepeval.evaluate import AsyncConfig

# wrappers
from deepeval_custom_model import CustomOpenAI

#ignore deepeval warnings
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed transport.*")
warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed <socket*")
#have to ignore those two warnings, because recommended implementations dont work
warnings.filterwarnings("ignore", category=DeprecationWarning, message="LangchainLLMWrapper*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="LangchainEmbeddingsWrapper*")

async def main():
    # get path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path")
    args = parser.parse_args()

    eval_model_name = os.getenv("OPENAI_EVAL_MODEL")
    #timeouts and max tokens - there were problems in evaluate.py with this
    llm = CustomOpenAI(model_name=eval_model_name, max_tokens=4000, timeout=300.0)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # format into deepeval dataset format
    dataset = []
    for item in data:
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["rag_answer"],
            retrieval_context=item["retrieved_contexts"],
            expected_output=item.get("ground_truth", "")
        )
        dataset.append(test_case)

    metrics = [
        AnswerRelevancyMetric(model=llm, threshold=0.5),
        FaithfulnessMetric(model=llm, threshold=0.5),
        ContextualPrecisionMetric(model=llm, threshold=0.5),
        ContextualRecallMetric(model=llm, threshold=0.5)
    ]

    #run eval
    print(f"Evaluation started, input: {args.input}")
    async_config = AsyncConfig(max_concurrent=5)
    result = deepEvaluate(dataset, metrics=metrics, async_config=async_config)

    # get scores
    detailed_results = []
    for test_result in result.test_results:
        metrics_scores = {}
        for m in test_result.metrics_data:
            metrics_scores[m.name.lower().replace(" ", "_")] = round(m.score, 4)
        
        detailed_results.append({
            "question": test_result.input,
            "answer": test_result.actual_output,
            **metrics_scores,
            "success": test_result.success
        })

    #get path to results
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = "evaluation_results/deepeval"
    os.makedirs(output_dir, exist_ok=True)

    #save detail score
    df = pd.DataFrame(detailed_results)
    df.to_json(f"{output_dir}/{base_name}_evaluated_{eval_model_name}.json", orient="records", force_ascii=False, indent=4)

    # save averages
    df_numeric = df.select_dtypes(include=['number'])
    df_numeric = df_numeric.round(4)

    summary_dict = df_numeric.mean().to_dict()
    summary_dict["framework"] = "deepeval"
    summary_dict["experiment"] = base_name
    summary_dict["judge"] = eval_model_name
    summary_dict["pass_rate"] = round((df["success"].sum() / len(df)) * 100, 2)

    with open(f"{output_dir}/{base_name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=4, ensure_ascii=False)
    
    #Marcdown
    try:
        summary_df = pd.DataFrame([summary_dict])
        # reorganize columns
        cols = ["experiment", "judge"] + [c for c in summary_df.columns if c not in ["experiment", "judge"]]
        summary_df = summary_df[cols]

        with open(f"{output_dir}/{base_name}_report.md", "w", encoding="utf-8") as f:
            f.write(f"# Evaluation Report (Deepeval): {base_name}\n")
            f.write(summary_df.to_markdown(index=False))
            f.write("\n## Detailed Scores per Question\n")
            # save few first lines
            f.write(df.head(10).to_markdown(index=False))
    except ImportError:
        pass

    #save csv
    summary_df.to_csv(f"{output_dir}/{base_name}_summary.csv", index=False)

    print(f"Evaluation complete (Deepeval), results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())