import os

from abc import ABC, abstractmethod
from tavily import TavilyClient

class SearchResult:
    def __init__(self, url: str, content: str):
        self.url = url
        self.content = content

    def __repr__(self):
        return f"SearchContext(url={self.url}, content={self.content})"

    def __str__(self):
        return f"SearchContext(url={self.url}, content={self.content})"

class SearchContext:
    def __init__(self, query: str):
        self.query = query
        self.results = []
        self.context_response = None
        
    def clear(self):
        self.results = []
        self.context_response = None

    def get_merged_content(self) -> str:
        return '\n'.join([r.content for r in self.results])

    def add_result(self, result: SearchResult):
        self.results.append(result)

    def add_context_response(self, response: str):
        self.context_response = response
        
    def __repr__(self):
        return f"SearchContext(query={self.query}, results={self.results})"
    
    def __str__(self):
        return f"SearchContext(query={self.query}, results={self.results})"
    
class WebSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def search(self, query: str, max_results: int, max_tokens: int):
        pass

class TavilyWebSearch(WebSearch):    
    def __init__(self, api_key):
        super().__init__(api_key)
        self.client = TavilyClient(api_key=self.api_key)
        
    def search(self, query: str, max_results: int = 10, max_tokens: int = 500, search_depth: str = "advanced"):
        context = SearchContext(query=query)
        response = self.client.search(query=query, max_results=max_results)
        
        for obj in response['results']:
            context.add_result(SearchResult(url=obj["url"], content=obj["content"]))    

        response_context = self.client.get_search_context(query=query, search_depth=search_depth, max_tokens=max_tokens)
        context.add_context_response(response_context)
        return context

if __name__ == "__main__":
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Tavily API key is not set.")
    search = TavilyWebSearch(api_key)
    context = search.search("The inventor of python still contributes to the language", max_results=5)
    print(context)
