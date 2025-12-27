from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
import re
import json

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

    #function is necessary because deepeval otherwise just end because of json in input
    def _clean_json(self, text):
        text = text.strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        possible_text = match.group(0) if match else text

        try:
            json.loads(possible_text)
            return possible_text
        except json.JSONDecodeError:
            pass

        #try to fix the json
        try:
            repaired_text = possible_text.strip()
            if repaired_text.count('{') > repaired_text.count('}'):
                repaired_text = repaired_text + '}' * (repaired_text.count('{') - repaired_text.count('}'))
            if repaired_text.count('[') > repaired_text.count(']'):
                repaired_text = repaired_text + ']' * (repaired_text.count('[') - repaired_text.count(']'))

            json.loads(repaired_text)
            return repaired_text
        except json.JSONDecodeError:
            pass
    
        print(f"Invalid JSON was generated. Dummy is returned.")

        return json.dumps(
            {
            "reason" : "JSON parsing failed, invalid output.",
            "score" : 0,
            "verdict": "no",
            "verdicts": [],
            "statements": []
            }
            )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self._clean_json(self.model.invoke(prompt).content)

    async def a_generate(self, prompt: str) -> str:
        res = await self.model.ainvoke(prompt)
        return (self._clean_json(res.content))

    def get_model_name(self):
        return self.model_name
