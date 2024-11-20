import os
import openai

from langchain_openai import ChatOpenAI

def validate_openai_api_key(api_key) -> bool:
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError as e:
        print("API key validation failed: {e}")
        return False
    else:
        return True

def validate_langchain_api_key(api_key) -> bool:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        result = llm.invoke("Hello, how are you?")
        return True
    except Exception as e:
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
