#custom prompts used for generating complex questions and answers with LlamaIndex
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