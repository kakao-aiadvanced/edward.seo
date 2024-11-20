import os
import openai
import tavily

from langchain_openai import ChatOpenAI

def validate_openai_api_key(api_key) -> bool:
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except openai.AuthenticationError as e:
        print("API key validation failed: {e}")
        return False

def validate_langchain_api_key(api_key) -> bool:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        result = llm.invoke("Hello, how are you?")
        return True
    except Exception as e:
        return False
    
def validate_tavily_api_key(api_key) -> bool:
    try:
        client = tavily.TavilyClient(api_key=api_key)
        response = client.search(query="Hello", max_results=1)
        return True
    except tavily.errors.InvalidAPIKeyError as e:
        return False    
    
def check_api_keys(mandatory_keys: list[str], optional_keys: list[str] = []) -> bool:
    for key in mandatory_keys:
        print(f'Checking mandatory key [{key}]: ', end='', flush=True)
        if key not in os.environ:
            print(f'missing environment variable for the key')
            return False
        if key == 'OPENAI_API_KEY':
            if validate_openai_api_key(os.environ[key]):
                print('valid')
            else:
                print('invalid')
                return False
        elif key == 'TAVILY_API_KEY':
            if validate_tavily_api_key(os.environ[key]):
                print('valid')
            else:
                print('invalid')
                return False

    for key in optional_keys:
        print(f'Checking optional key [{key}]: ', end='', flush=True)
        if key not in os.environ:
            print(f'missing environment variable for the key')
            continue
        if key == 'LANGCHAIN_API_KEY':
            if validate_langchain_api_key(os.environ[key]):
                print('valid')
            else:
                print('invalid')
    return True
