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

os.environ['LANGCHAIN_TRACING_V2'] = "true"

llm = ChatOpenAI(model="gpt-4o-mini")
   
def test_splitter(web_url: str, query: str):
    chunk_sizes = [500, 1000, 2000, 3000]
    chunk_overlaps = [100, 200, 300, 400]

    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            langchain_basic(web_url, query, chunk_size, chunk_overlap)

def test_embedding(model, documents: list[str], query: str):
    embeddings_model = model(api_key=os.environ.get("OPENAI_API_KEY"))
    embeddings = embeddings_model.embed_documents(documents)
    embedded_query = embeddings_model.embed_query(query)
    print(f'Test embedding: {model}')
    print(f'- query: {query}')
    for i, emb in enumerate(embedded_query[:len(embeddings)]):
        print(f'Embedding {i}: (similarity: {emb}, doc: {documents[i]})')

def langchain_basic(web_url: str, query: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    print(f'Web URL: {web_url}')
    print(f'Query: {query}')

    loader = SeleniumURLLoader(urls=[web_url])

    try:
        docs = loader.load()
    except selenium.common.exceptions.SessionNotCreatedException as e:
        print(f'Reinstalling Chrome Driver due to exception: {e}')
        reinstall_chrome_driver()
        docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(query)
    print(f'Result from recursive character text splitter: chunk_size({chunk_size}), chunk_overlap({chunk_overlap})')
    print(result)
  
def main():
    parser = argparse.ArgumentParser(description="LangChain Basics")
    parser.add_argument("--test-splitter", action='store_true')
    parser.add_argument("--web-url", type=str, default=None, help="Web URL")
    parser.add_argument("--query", type=str, default=None, help="Query")
    args = parser.parse_args()
    
    if not check_api_keys():
        print("Please set the OPENAI_API_KEY environment variable.")
        return

    check_chrome_driver()
   
    test_embedding(
        OpenAIEmbeddings,
        [
            "Hello, how are you?",
            "RAG is one of the most powerful weapon of LLMs",
            "Elon Musk is the father of DOGE coin"
        ],
        "The most general way for greeting someone."
    )
    
    if args.web_url is not None: 
        if args.test_splitter:
            test_splitter(args.web_url, args.query)
        langchain_basic(args.web_url, args.query)
    
if __name__ == '__main__':
    main()