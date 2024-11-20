import os
import argparse

from ai_advanced_edward_seo.base.doc_container import DocContainer
from ai_advanced_edward_seo.base.question_router import QuestionRouter
from ai_advanced_edward_seo.base.retriever import Retriever
from ai_advanced_edward_seo.base.relevance_checker import RelevanceChecker
from ai_advanced_edward_seo.base.web_search import TavilyWebSearch
from ai_advanced_edward_seo.base.hallucination_checker import HallucinationChecker
from ai_advanced_edward_seo.base.response_generator import ResponseGenerator

from ai_advanced_edward_seo.utils.api_key import check_api_keys
from ai_advanced_edward_seo.utils.web_driver import check_chrome_driver

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from typing import List, TypedDict
from pprint import pprint
from pathlib import Path
from loguru import logger

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[DocContainer]

class Agent:
    def __init__(self, llm, prompt_file_paths: list[str], urls: list[str]):
        self.llm = llm
        self.urls = urls
        
        self.retriever = Retriever(urls)
        self.relevance_checker = RelevanceChecker(llm, prompt_file_paths['relevance_checker'])
        self.hallucination_checker = HallucinationChecker(llm, prompt_file_paths['hallucination_checker'])
        self.response_generator = ResponseGenerator(llm, prompt_file_paths['response_generator'])
        self.question_router = QuestionRouter(llm, prompt_file_paths['question_router'])
        
        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("retrieve", self.retrieve)
        self.workflow.add_node("grade_documents", self.grade_documents)
        self.workflow.add_node("websearch", self.web_search)
        self.workflow.add_node("generate", self.generate)
        self.workflow.add_node("output_answer", self.output_answer)
        
        self.workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )

        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "websearch": "websearch",
                "generate": "generate",
            },
        )
        
        self.workflow.add_edge("websearch", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )

        self.workflow.add_conditional_edges(
            "generate",
            self.check_hallucination,
            {
                "yes": "output_answer",
                "no": "generate",
            },
        ) 

        self.workflow.add_edge("output_answer", END)
        
        self.app = self.workflow.compile()

    def run_query(self, query: str):
        inputs = {"question": query}
        for output in self.app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
                
        if value is not None:
            pprint(value["generation"])

    def web_search(self, state: GraphState) -> GraphState:
        api_key = os.getenv('TAVILY_API_KEY')
        searcher = TavilyWebSearch(api_key)
        
        question = state["question"]
        if "documents" in state:
            documents = state["documents"]
        else:
            documents = []

        search_context = searcher.search(question)
        
        for obj in search_context.results:
            url = obj.url
            content = obj.content
            doc = DocContainer(url=url, langchain_doc=Document(page_content=content))
            documents.append(doc)

        return {"documents": documents, "question": question}
    
    def retrieve(self, state: GraphState) -> GraphState:
        self.retriever.load_documents()
        docs = self.retriever.get_relevant_documents(state["question"])
        return {"documents": docs, "question": state["question"]}
    
    def grade_documents(self, state: GraphState) -> GraphState:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"

        relevance_result = self.relevance_checker.check_relevance(documents, [question])
        for r in relevance_result:
            grade = r["relevance"]

            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(r["document"])
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                # We do not include the document in filtered_docs
                # We set a flag to indicate that we want to run web search
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}
    
    def generate(self, state: GraphState) -> GraphState:
        response = self.response_generator.generate_from_context(state["documents"], state["question"])
        return {"generation": response}
    
    def check_hallucination(self, state: GraphState) -> bool:
        generation = state["generation"]
        documents = state["documents"]
        
        contexts = [doc.get_page_content() for doc in documents]
        hallucination_occured = self.hallucination_checker.check_hallucination(generation, contexts)
        return 'no' if hallucination_occured else 'yes'

    def output_answer(self, state: GraphState) -> GraphState:
        print("---OUTPUT ANSWER---")
        print(state["generation"])
        print("---REFERENCES---")
        for i, doc in enumerate(state["documents"]):
            print(f'[{i}] {doc.url}')
        return state
    
    def route_question(self, state: GraphState) -> str:
        print("---ROUTE QUESTION---")
        question = state["question"]
        print(question)
        source = self.question_router.route(question)
        print(source)
        print(source["datasource"])
        if source["datasource"] == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "websearch"
        elif source["datasource"] == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    
    def decide_to_generate(self, state: GraphState) -> str:
        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
            )
            return "websearch"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"
    
    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        contexts = [doc.get_page_content() for doc in documents]
        hallucination_occured = self.hallucination_checker.check_hallucination(generation, contexts)

        # Prompt
        system = """You are a grader assessing whether an
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "question: {question}\n\n answer: {generation} "),
            ]
        )

        answer_grader = prompt | self.llm | JsonOutputParser()
        answer_grader.invoke({"question": question, "generation": generation})

        # Check hallucination
        if not hallucination_occured:
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

def main():
    cur_dir = Path(__file__).resolve().parent
    package_root_dir = cur_dir.parent.parent
    assets_dir = package_root_dir / 'assets'

    parser = argparse.ArgumentParser(description="LangChain Basics")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--web-urls-file-path", type=str, required=True)
    parser.add_argument("--queries-file-path", type=str, default=None)
    parser.add_argument("--question-router-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_question_router.txt'))
    parser.add_argument("--relevance-checker-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_relevance_checker.txt'))
    parser.add_argument("--hallucination-checker-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_hallucination_checker.txt'))
    parser.add_argument("--response-generator-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_response_generator.txt'))
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    llm = ChatOpenAI(model=args.llm_model)

    prompt_file_paths = {
        'question_router': args.question_router_prompt_file_path,
        'relevance_checker': args.relevance_checker_prompt_file_path,
        'hallucination_checker': args.hallucination_checker_prompt_file_path,
        'response_generator': args.response_generator_prompt_file_path
    }

    mandatory_api_keys = [
        'OPENAI_API_KEY',
        'TAVILY_API_KEY'
    ]
    
    optional_api_keys = [
        'LANGCHAIN_API_KEY'
    ]
    
    if not check_api_keys(mandatory_keys=mandatory_api_keys, optional_keys=optional_api_keys):
        logger.error("Please set the environment variables for mandatory API keys: {mandatory_api_keys}")
        return

    check_chrome_driver()

    urls = []
    queries = []
    
    if args.web_urls_file_path is not None:
        with open(args.web_urls_file_path, 'r') as f:
            urls = [url.strip() for url in f.readlines()]

    if args.queries_file_path is not None:
        with open(args.queries_file_path, 'r') as f:
            queries = [query.strip() for query in f.readlines()]

    if len(queries) == 0 and args.query is not None:
        queries = [args.query]
    else:
        logger.error("Please provide either a query or a queries file path.")
        return

    agent = Agent(llm, prompt_file_paths, urls)
    for q in queries:
        agent.run_query(q)
    
if __name__ == '__main__':
    main()