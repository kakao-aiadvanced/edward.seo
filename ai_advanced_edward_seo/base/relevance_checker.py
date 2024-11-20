from ai_advanced_edward_seo.utils.file_io import read_text_file

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from loguru import logger

class RelevanceChecker:
    def __init__(self, llm, prompt_file):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["document_content", "query"],
            template=read_text_file(prompt_file),
        )
        self.chain = (
            self.prompt
            | llm
            | JsonOutputParser()
        )

    def check_relevance(self, docs, queries: list[str]):
        relevant_results = []
        for query in queries:
            for doc in docs:
                try:
                    relevance_check = self.chain.invoke(
                        {
                            "document_content": doc.get_page_content(),
                            "query": query
                        }
                    )
                    relevance_result = relevance_check.get("relevance", "no")
                except ValueError as e:
                    logger.error(f"Error parsing relevance response: {e}")
                    relevance_result = "no"

                relevant_results.append(
                    {
                        "document": doc,
                        "query": query,
                        "relevance": relevance_result
                    }
                )
        return relevant_results

