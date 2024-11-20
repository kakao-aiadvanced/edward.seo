from ai_advanced_edward_seo.utils.file_io import read_text_file

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from loguru import logger

class HallucinationChecker:
    def __init__(self, llm, prompt_file):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["response", "context"],
            template=read_text_file(prompt_file),
        )
        self.chain = (
            self.prompt
            | self.llm
            | JsonOutputParser()
        )

    def check_hallucination(self, response, context):
        combined_context = "\n\n".join(context)
        hallucination_check = self.chain.invoke(
            {
                "response": response,
                "context": combined_context
            }
        )
        return hallucination_check['hallucination'] == 'yes'

