#custom prompts used for generating simple questions and answers with LlamaIndex
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
    7) **LANGUAGE:** The question MUST be written in the SAME LANGUAGE as the provided text ({context_str}). If the text is in Czech, ask in Czech. If in German, ask in German.
    8) Respond ONLY with generated question and nothing else.
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
    4) **LANGUAGE:** The answer MUST be written in the SAME LANGUAGE as the provided context ({context_str})
    5) Respond ONLY with the answer.
    """
)