import langchain
import os

print(f"LangChain path: {langchain.__path__}")
for path in langchain.__path__:
    print(f"Listing {path}:")
    try:
        print(os.listdir(path))
    except Exception as e:
        print(f"Error listing {path}: {e}")
