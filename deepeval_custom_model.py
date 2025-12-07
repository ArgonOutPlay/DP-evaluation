from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self, model_name, max_tokens : int, timeout : float):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.model = ChatOpenAI(
            model = self.model_name,
            max_tokens = self.max_tokens,
            temperature=0,
            timeout= self.timeout
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name
