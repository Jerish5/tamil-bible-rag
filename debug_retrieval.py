import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_PATH = "chroma_db"

def check_retrieval():
    print("\nChecking retrieval...")
    if not os.path.exists(DB_PATH):
        print("Vector DB not found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    query = "தமிழ் விவிலியத்தின் வரலாறு"
    print(f"Query: {query}")
    results = vectorstore.similarity_search(query, k=5)
    
    # Check sources
    print("\nChecking sources in DB...")
    # This is a bit hacky for Chroma, but we can peek
    # Or just search for everything (limit)
    all_docs = vectorstore.get()
    metadatas = all_docs['metadatas']
    sources = set()
    for m in metadatas:
        if m and 'source' in m:
            sources.add(m['source'])
    print(f"Unique sources found: {len(sources)}")
    if "Viviliya Thedal.pdf" in sources:
        print("SUCCESS: 'Viviliya Thedal.pdf' is in the DB.")
    else:
        print("FAILURE: 'Viviliya Thedal.pdf' is NOT in the DB.")
        print(f"Sources: {list(sources)[:10]}...")
    
    if not results:
        print("No results found.")
    else:
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(doc.page_content[:500])
            print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    # Redirect output to file
    import sys
    original_stdout = sys.stdout
    with open("retrieval_debug.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        print("Starting retrieval check...")
        check_retrieval()
        sys.stdout = original_stdout
    print("Debug results written to retrieval_debug.txt")
