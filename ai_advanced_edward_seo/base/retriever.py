from ai_advanced_edward_seo.base.doc_container import DocContainer

from langchain_community.document_loaders import SeleniumURLLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from loguru import logger

class Retriever:
    def __init__(self, web_urls, embedding_model='text-embedding-3-small', chunk_size=100, chunk_overlap=20):
        self.web_urls = web_urls
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.loader = SeleniumURLLoader(urls=web_urls)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        
        self.docs = None
        self.splits = None
        self.vectorstore = None
        self.vectorstore_retriever = None

        self.load_documents()

    def load_documents(self):
        try:
            docs = self.loader.load()
        except selenium.common.exceptions.SessionNotCreatedException as e:
            logger.error(f'Reinstalling Chrome Driver due to exception: {e}')
            reinstall_chrome_driver()
            docs = self.loader.load()

        wrapped_docs = []
        for i, doc in enumerate(docs):
            wrapped_docs.append(DocContainer(url=self.web_urls[i], langchain_doc=doc))
        self.docs = wrapped_docs    

    def split_documents(self, docs):
        langchain_docs = [doc.langchain_doc for doc in docs]
        return self.text_splitter.split_documents(langchain_docs)

    def create_vectorstore(self, splits):
        self.vectorstore = Chroma.from_documents(documents=splits, embedding=self.embedding)
        self.vectorstore_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 6})

    def get_relevant_documents(self, query):
        logger.info(f'Finding relevant documents for query: "{query}"')
        
        if self.splits is None:
            self.splits = self.split_documents(self.docs)
        if self.vectorstore is None:
            self.create_vectorstore(self.splits)

        docs = self.vectorstore_retriever.invoke(query)
        wrapped_docs = []

        for doc in docs:
            wrapped_docs.append(DocContainer(url=None, langchain_doc=doc))
        return wrapped_docs

