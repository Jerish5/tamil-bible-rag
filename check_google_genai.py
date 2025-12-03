try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    print("Successfully imported ChatGoogleGenerativeAI")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
