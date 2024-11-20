import os
import argparse

from ai_advanced_edward_seo.base.retriever import Retriever
from ai_advanced_edward_seo.base.relevance_checker import RelevanceChecker
from ai_advanced_edward_seo.base.hallucination_checker import HallucinationChecker
from ai_advanced_edward_seo.base.response_generator import ResponseGenerator

from ai_advanced_edward_seo.utils.api_key import check_api_keys
from ai_advanced_edward_seo.utils.web_driver import check_chrome_driver, reinstall_chrome_driver

from langchain_openai import ChatOpenAI

from loguru import logger
from pathlib import Path

os.environ['LANGCHAIN_TRACING_V2'] = "true"

def langchain_practice_relevance_check(llm, prompt_file_paths: list[str], web_urls: str, queries: list[str]) -> tuple[bool, bool]:
    retriever = Retriever(web_urls)
    relevance_checker = RelevanceChecker(llm, prompt_file_paths['relevance_checker'])
    relevant_results = relevance_checker.check_relevance(retriever.docs, queries)

    all_relevant = True
    all_not_relevant = True
   
    for result in relevant_results:
        logger.info(f'Query: {result["query"]}')
        logger.info(f'Document: {result["document"].url}')
        logger.info(f'Relevance: {result["relevance"]}')

        if result["relevance"] == "no":
            all_relevant = False
        else:
            all_not_relevant = False
    return all_relevant, all_not_relevant
 
def langchain_practice_all(llm, prompt_file_paths: list[str], web_urls: list[str], query: str) -> bool:
    retriever = Retriever(web_urls)
    relevance_checker = RelevanceChecker(llm, prompt_file_paths['relevance_checker'])

    relevant_docs = retriever.get_relevant_documents(query)
    relevant_results = relevance_checker.check_relevance(relevant_docs, [query])

    relevant_urls = [res["document"].url for res in relevant_results if res["relevance"] == "yes"]
    relevant_contexts = [res["document"].get_page_content() for res in relevant_results if res["relevance"] == "yes"]

    if not relevant_contexts:
        logger.error('No relevant documents found.')
        return False

    hallucination_checker = HallucinationChecker(llm, prompt_file_paths['hallucination_checker'])
    response_generator = ResponseGenerator(llm, prompt_file_paths['response_generator'])
    response = response_generator.generate_response(retriever, relevant_contexts, query, stream=True)

    if hallucination_checker.check_hallucination(response, relevant_contexts):
        logger.warning('Oops! The response contains hallucination. Generating a new answer...')
        response_generator.generate_response(retriever, relevant_contexts, query, stream=True)
    else:
        logger.info(f'The reference source of the response is:')
        logger.info(f'{relevant_urls}')
    return True

def main():
    cur_dir = Path(__file__).resolve().parent
    package_root_dir = cur_dir.parent.parent
    assets_dir = package_root_dir / 'assets'

    parser = argparse.ArgumentParser(description="LangChain Basics")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--web-urls-file-path", type=str, required=True)
    parser.add_argument("--queries-file-path", type=str, default=None)
    parser.add_argument("--relevance-checker-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_relevance_checker.txt'))
    parser.add_argument("--hallucination-checker-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_hallucination_checker.txt'))
    parser.add_argument("--response-generator-prompt-file-path", type=str, default=os.path.join(assets_dir, 'prompt_template_response_generator.txt'))
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    llm = ChatOpenAI(model=args.llm_model)

    prompt_file_paths = {
        'relevance_checker': args.relevance_checker_prompt_file_path,
        'hallucination_checker': args.hallucination_checker_prompt_file_path,
        'response_generator': args.response_generator_prompt_file_path
    }

    mandatory_api_keys = [
        'OPENAI_API_KEY'
    ]
    
    optional_api_keys = [
        'LANGCHAIN_API_KEY'
    ]
    
    if not check_api_keys(mandatory_keys=mandatory_api_keys, optional_keys=optional_api_keys):
        logger.error("Please set the environment variables for mandatory API keys: {mandatory_api_keys}")
        return

    check_chrome_driver()

    with open(args.web_urls_file_path, 'r') as f:
        urls = [url.strip() for url in f.readlines()]

    if args.queries_file_path is not None:
        with open(args.queries_file_path, 'r') as f:
            queries = [query.strip() for query in f.readlines()]

            all_relevant, all_not_relevant = langchain_practice_relevance_check(llm, prompt_file_paths, urls, queries)
            if all_relevant:
                logger.info("All docs are relevant to the queries")
            elif all_not_relevant:
                logger.info("Not all docs are relevant to the queries")
            else:
                logger.info("Some docs are relevant to the queries")
    elif args.query is not None:
        langchain_practice_all(llm, prompt_file_paths, urls, args.query)

if __name__ == '__main__':
    main()
