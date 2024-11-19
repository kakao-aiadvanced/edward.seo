import os
import openai

from langchain_openai import ChatOpenAI

def validate_openai_api_key(api_key) -> bool:
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError as e:
        print("Invalid OpenAI API key or request failed: {e}")
        return False
    else:
        print("Valid OpenAI API key! Request succeeded.")
        return True

def validate_langchain_api_key(api_key) -> bool:
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        result = llm.invoke("Hello, how are you?")
        print("Valid LangChain API key! Request succeeded.")
        return True
    except Exception as e:
        print(f"Invalid API key or request failed: {e}")
        return False
        
def check_api_keys() -> bool:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    langchain_api_key = os.environ.get("LANGCHAIN_API_KEY")

    print(f"OpenAI API Key: {openai_api_key}")
    print(f"Langchain API Key: {langchain_api_key}")

    if openai_api_key is None or len(openai_api_key) == 0:
        return False
    if langchain_api_key is None or len(langchain_api_key) == 0:
        return False

    if not validate_openai_api_key(openai_api_key):
        print("Invalid OpenAI API key.")
        return False
    if not validate_langchain_api_key(langchain_api_key):
        print("Invalid Langchain API key.")
        return False
    return True
 