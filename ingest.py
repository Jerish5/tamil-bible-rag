import os
import json
import glob
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

# Configuration
DATA_DIR = "data"
PDF_PATH = "Holy Bible - PDF Version.pdf"
DB_PATH = "chroma_db"

def ingest_data():
    documents = []
    
    # 1. Load JSON Bible Data
    print(f"Loading JSON files from {DATA_DIR}...")
    if os.path.exists(DATA_DIR):
        json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
        json_files = [f for f in json_files if "Books.json" not in f]
        print(f"Found {len(json_files)} book files.")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                book_name_english = data.get("book", {}).get("english", "Unknown")
                book_name_tamil = data.get("book", {}).get("tamil", "Unknown")
                
                for chapter in data.get("chapters", []):
                    chapter_num = chapter.get("chapter")
                    for verse in chapter.get("verses", []):
                        verse_num = verse.get("verse")
                        text = verse.get("text")
                        
                        if text:
                            content = f"{book_name_tamil} ({book_name_english}) {chapter_num}:{verse_num} - {text}"
                            metadata = {
                                "source": os.path.basename(json_file),
                                "type": "bible_verse",
                                "book": book_name_english,
                                "chapter": chapter_num,
                                "verse": verse_num
                            }
                            documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        print(f"Loaded {len(documents)} verses from JSON.")
    else:
        print(f"Directory {DATA_DIR} not found.")

    # 2. Load PDF Data
    print(f"Loading PDF from {PDF_PATH}...")
    if os.path.exists(PDF_PATH):
        try:
            loader = PyPDFLoader(PDF_PATH)
            pdf_docs = loader.load()
            print(f"Loaded {len(pdf_docs)} pages from PDF.")
            
            # Split PDF content
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = text_splitter.split_documents(pdf_docs)
            
            # Update metadata for PDF docs
            for doc in split_docs:
                doc.metadata["source"] = PDF_PATH
                doc.metadata["type"] = "pdf_content"
            
            documents.extend(split_docs)
            print(f"Added {len(split_docs)} chunks from PDF.")
            
        except Exception as e:
            print(f"Error loading PDF: {e}")
    else:
        print(f"PDF file {PDF_PATH} not found.")

    print(f"Total documents to ingest: {len(documents)}")

    # Create Embeddings
    print("Creating embeddings (this may take a while)...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': device}
    )

    # Store in Chroma
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)  # Clean up old DB
    
    batch_size = 5000
    total_docs = len(documents)
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        print(f"Ingesting batch {i//batch_size + 1}/{(total_docs//batch_size) + 1}...")
        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_PATH
            )
        else:
            vectorstore.add_documents(batch)
            
    print(f"Vector store created at {DB_PATH}")

if __name__ == "__main__":
    ingest_data()
