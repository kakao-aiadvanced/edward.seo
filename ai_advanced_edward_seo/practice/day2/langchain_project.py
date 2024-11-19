import os
import bs4
import argparse

from ai_advanced_edward_seo.utils.api_key import check_api_keys
from ai_advanced_edward_seo.utils.web_driver import check_chrome_driver, reinstall_chrome_driver

from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ['LANGCHAIN_TRACING_V2'] = "true"

llm = ChatOpenAI(model="gpt-4o-mini")

def generate_response(context: str, query: str):
    # Response Generation
    output_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "You are a helpful assistant. Answer the following query based on the context provided.\n\n"
            "Query: {query}\nContext: {context}\n\n"
            "Provide a concise and informative answer."
        ),
    )
   
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
 
    response_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | output_prompt
        | llm
        | StrOutputParser()
    )

    for chunk in response_chain.stream(query):
        print(chunk, end="", flush=True)
    
def langchain_practice(web_urls: list[str], query: str, chunk_size: int = 100, chunk_overlap: int = 20):
    print(f'Web URLs: {web_urls}')

    loader = SeleniumURLLoader(urls=web_urls)

    try:
        docs = loader.load()
    except selenium.common.exceptions.SessionNotCreatedException as e:
        print(f'Reinstalling Chrome Driver due to exception: {e}')
        reinstall_chrome_driver()
        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 6})

    # Chain 1: Retrieval
    relevant_docs = retriever.get_relevant_documents(query)

    # Chain 2: Relevance Check
    relevance_prompt = PromptTemplate(
        input_variables=["document_content", "query"],
        template=(
            "You are a relevance checker. Determine if the following document is relevant to the query.\n\n"
            "Document: {document_content}\nQuery: {query}\n\n"
            "Respond with a JSON object containing:\n"
            '"relevance": "yes" if relevant\n'
            '"relevance": "no" if not relevant.'
        ),
    )
    relevance_chain = LLMChain(
        llm=llm,
        prompt=relevance_prompt,
        output_parser=JsonOutputParser(),
    )

    relevant_results = []
    for doc in relevant_docs:
        try:
            relevance_check = relevance_chain.run({"document_content": doc.page_content, "query": query})
            relevance_result = relevance_check.get("relevance", "no")
        except ValueError as e:
            print(f"Error parsing relevance response: {e}")
            relevance_result = "no"
        relevant_results.append({"document": doc, "relevance": relevance_result})

    # Response Generation
    output_prompt = PromptTemplate(
        input_variables=["query", "context"],
        template=(
            "You are a helpful assistant. Answer the following query based on the context provided.\n\n"
            "Query: {query}\nContext: {context}\n\n"
            "Provide a concise and informative answer."
        ),
    )
   
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
 
    # Generate final response
    relevant_contexts = [res["document"].page_content for res in relevant_results if res["relevance"] == "yes"]
    if not relevant_contexts:
        return False

    response_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | output_prompt
        | llm
        | StrOutputParser()
    )

    response = []
    for chunk in response_chain.stream(query):
        response.append(chunk)
        print(chunk, end="", flush=True)
    
    # Hallucination Check
    hallucination_prompt = PromptTemplate(
        input_variables=["response", "context"],
        template=(
            "You are a hallucination checker. Determine if the response contains information not present in the context.\n\n"
            "Response: {response}\nContext: {context}\n\n"
            "Respond with a JSON object containing:\n"
            '"hallucination": "yes" if the response contains hallucination\n'
            '"hallucination": "no" if the response does not contain hallucination.'
        ),
    )
    hallucination_chain = LLMChain(
        llm=llm,
        prompt=hallucination_prompt,
        output_parser=JsonOutputParser(),
    )

    combined_context = "\n\n".join(relevant_contexts)
    hallucination_check = hallucination_chain.run(
        {
            "response": response,
            "context": combined_context
        }
    )
    
    if hallucination_check['hallucination'] == 'yes':
        print('Oops! The response contains hallucination. Generating a new answer...')
        generate_response(combined_context, query)
    else:
        print(f'The reference source of the response is:')
        print(f'{relevant_contexts}')
        
    return True

def main():
    parser = argparse.ArgumentParser(description="LangChain Basics")
    parser.add_argument("--test-splitter", action='store_true')
    parser.add_argument("--web-urls-file-path", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    
    if not check_api_keys():
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    check_chrome_driver()
  
    all_yes_queries = [
        'agent memory',
        'large language model',
        'prompt engineering',
        'prompt attack'
    ]
    
    all_no_queries = [
        'Elun Musk',
        'I am too tired of doing football',
        'When the Vicent Van Gogh was born?',
        'Why DOGE coin gets popular after Donald Trump has been elected?' 
    ]
    
    with open(args.web_urls_file_path, 'r') as f:
        urls = f.readlines()
        for i, url in enumerate(urls):
            urls[i] = url.strip()

        # test cases to get all yes
        all_relevant = True
        for query in all_yes_queries:
            relevant_doc_exists = langchain_practice(urls, query)
            print(f'Query: {query} (relevant document exists: {relevant_doc_exists})')
            if not relevant_doc_exists:
                all_relevant = False
       
        if all_relevant:
            print("All docs are relevant to the queries") 

        # test cases to get all no
        all_not_relevant = True
        for query in all_no_queries:
            relevant_doc_exists = langchain_practice(urls, query)
            print(f'Query: {query} (relevant document exists: {relevant_doc_exists})')
            if relevant_doc_exists:
                all_not_relevant = False
       
        if all_not_relevant:
            print("All docs are NOT relevant to the queries") 

if __name__ == '__main__':
    main()