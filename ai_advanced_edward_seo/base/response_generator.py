from ai_advanced_edward_seo.utils.file_io import read_text_file

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from loguru import logger

class ResponseGenerator:
    def __init__(self, llm, prompt_file):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=read_text_file(prompt_file),
        )

    def generate_response(self, retriever, context, query, stream: bool = False):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retrieved_context = retriever.vectorstore_retriever | format_docs
        response_chain = (
            {
                "context": retrieved_context,
                "query": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        if stream:
            response = []
            for chunk in response_chain.stream(query):
                response.append(chunk)
                print(chunk, end="", flush=True)
        else:
            response = response_chain.invoke(query)
            print(response)
        return response

