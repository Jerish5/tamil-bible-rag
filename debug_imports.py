import langchain
print(f"LangChain file: {langchain.__file__}")
print(f"LangChain path: {langchain.__path__}")
try:
    import langchain.chains
    print("Imported langchain.chains")
except ImportError as e:
    print(f"Failed to import langchain.chains: {e}")

try:
    from langchain.chains import RetrievalQA
    print("Imported RetrievalQA")
except ImportError as e:
    print(f"Failed to import RetrievalQA: {e}")
