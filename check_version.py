import langchain
import os

print(f"LangChain Version: {langchain.__version__}")
print(f"Has chains? {'chains' in dir(langchain)}")

try:
    import langchain.chains
    print("Imported langchain.chains")
except ImportError as e:
    print(f"ImportError: {e}")

# Check directory content
path = langchain.__path__[0]
print(f"Directory content of {path}:")
print(os.listdir(path))
