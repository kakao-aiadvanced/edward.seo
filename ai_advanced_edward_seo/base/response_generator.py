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

    def generate_from_context(self, context, query, stream: bool = False):
        if isinstance(context, list):
            all_content = ''
            for doc in context:
                all_content += doc.get_page_content() + '\n\n'
        else:
            all_content = context

        response_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        if stream:
            response = []
            for chunk in response_chain.stream({"context": all_content, "query": query}):
                response.append(chunk)
                print(chunk, end="", flush=True)
        else:
            response = response_chain.invoke({"context": all_content, "query": query})
            print(response)
        return response