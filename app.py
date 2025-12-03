import streamlit as st
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Tamil Bible RAG", page_icon="📖", layout="wide")

# Title and Header
st.title("📖 Tamil Bible RAG System")
st.markdown("Ask questions about the Bible in Tamil or English. If the answer isn't in the Bible, I'll search the web!")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" in os.environ:
        st.success("API Key found in environment.")
    else:
        st.warning("Please enter your Google Gemini API Key to generate answers.")

# Initialize Embeddings (must match ingest.py)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = get_embeddings()

# Load Vector Store
DB_PATH = "chroma_db"

# Check if chroma_db exists, if not, unzip it
if not os.path.exists(DB_PATH):
    zip_path = "chroma_db.zip"
    # Check for split files
    if not os.path.exists(zip_path):
        part1 = "chroma_db.zip.001"
        if os.path.exists(part1):
            with st.spinner("Reassembling database..."):
                with open(zip_path, 'wb') as dest:
                    part_num = 1
                    while True:
                        part_name = f"{zip_path}.{part_num:03d}"
                        if not os.path.exists(part_name):
                            break
                        with open(part_name, 'rb') as source:
                            dest.write(source.read())
                        part_num += 1
    
    if os.path.exists(zip_path):
        import zipfile
        with st.spinner("Extracting database..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
    else:
        st.error("Vector Database not found. Please run `ingest.py` first.")
        st.stop()

if not os.path.exists(DB_PATH):
    st.error("Vector Database not found. Please run `ingest.py` first.")
    st.stop()

vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# QA Chain
if "GOOGLE_API_KEY" in os.environ:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    # Custom Prompt for Bible RAG
    customer_prompt = """
            நீங்கள் திருவிவிலியம் குறித்த கேள்விகளுக்கு பதிலளிக்கும், தமிழில் நிபுணத்துவம் வாய்ந்த உதவியாளர் மற்றும் வல்லுநர்.
            - பயனர் தமிழ் மொழியில் கேள்வி கேட்டால், அவர்களுக்கு தெளிவான, இயல்பான, ஆனால் அருமையாக அமைந்த பதிலை அளியுங்கள்.
            - பதில்கள் தமிழில் மட்டுமே இருக்க வேண்டும், மற்றும் இந்தியக் கத்தோலிக்க திருச்சபையில் பயன்படுத்தப்படும் பதங்களை மட்டும் பயன்படுத்துங்கள்.
            - பதில்கள் Markdown வடிவத்தில் இருக்க வேண்டும்.
            - பதில் திருவிவிலியத்தின் உள்ளடக்கம் மட்டுமே அடிப்படையாக கொள்ள வேண்டும்.
            ### எண்ணுதல் மற்றும் கணக்கிடுதல் (Counting and Calculation):
            - பயனர் 'எத்தனை', 'மொத்தம் எத்தனை' போன்ற எண்ணிக்கை சார்ந்த கேள்விகளைக் கேட்டால், உங்கள் கருவிகள் மூலம் கிடைத்த தகவல்களை முதலில் பகுப்பாய்வு செய்யுங்கள்.
            - அந்த தகவல்களின் அடிப்படையில், மொத்த எண்ணிக்கையைக் கணக்கிட்டு, அந்த எண்ணை உங்கள் பதிலில் தெளிவாகக் குறிப்பிடுங்கள்.


    Context:
    {context}

    Question: {question}

    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(customer_prompt)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # User Input
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Searching Bible..."):
            try:
                result = qa_chain({"query": query})
                answer = result["result"]
                source_docs = result["source_documents"]
                
                # Check for "I don't know" or "Not found" to trigger web search
                lower_answer = answer.lower()
                triggers = [
                    "don't know", "do not know", "not found", "not mentioned",
                    "தெரியவில்லை", "தகவல் இல்லை", "குறிப்பிடப்படவில்லை", "பதில் இல்லை", "இல்லை"
                ]
                
                if any(trigger in lower_answer for trigger in triggers):
                    st.warning("Answer not found in Bible context. Searching the web...")
                    
                    search = DuckDuckGoSearchRun()
                    with st.spinner("Searching the web..."):
                        web_results = search.run(query)
                    
                    # Re-prompt with web results
                    web_template = """You are a helpful assistant. The user asked a question that wasn't found in the Bible database.
                    Here is some information from the web:
                    {web_context}
                    
                    Question: {question}
                    
                    Answer based on the web info (cite source as 'Web Search'). Answer in the SAME language as the question.
                    **CRITICAL**: All Tamil answers MUST be in **Roman Catholic Tamil style** (e.g., use 'Thiruviliyam' for Bible, and standard Catholic terminology)."""
                    
                    prompt_web = PromptTemplate.from_template(web_template)
                    chain_web = LLMChain(llm=llm, prompt=prompt_web)
                    
                    with st.spinner("Generating answer from web..."):
                        web_response = chain_web.run(web_context=web_results, question=query)
                    
                    st.markdown("### Web Answer")
                    st.write(web_response)
                    
                else:
                    st.markdown("### Answer")
                    st.write(answer)
                    
                    st.markdown("---")
                    st.markdown("### Source Verses")
                    for i, doc in enumerate(source_docs):
                        with st.expander(f"Source {i+1} ({doc.metadata.get('book', '?')} {doc.metadata.get('chapter', '?')}:{doc.metadata.get('verse', '?')})"):
                            st.write(doc.page_content)
                            
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please provide an API Key to start asking questions.")
