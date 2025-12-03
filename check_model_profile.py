import langchain_core.language_models
print(f"langchain_core file: {langchain_core.language_models.__file__}")
try:
    from langchain_core.language_models import ModelProfile
    print("Successfully imported ModelProfile")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
