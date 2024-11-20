from ai_advanced_edward_seo.utils.file_io import read_text_file

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from loguru import logger

class QuestionRouter:
    def __init__(self, llm, prompt_file):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", read_text_file(prompt_file)),
                ("human", "question: {question}"),
            ]
        )
        
        self.chain = (
            self.prompt
            | self.llm
            | JsonOutputParser()
        )

    def route(self, question: str):
        return self.chain.invoke({"question": question})