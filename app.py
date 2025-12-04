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
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import extra_streamlit_components as stx
import time
import datetime

# Load environment variables
load_dotenv()

# Load from Streamlit secrets if not in env (for Cloud deployment)
if "GOOGLE_API_KEY" not in os.environ and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if "DEEPSEEK_API_KEY" not in os.environ and "DEEPSEEK_API_KEY" in st.secrets:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]



# Custom CSS removed for default Streamlit UI

# Login Logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Initialize Cookie Manager
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

# Check for existing cookie
if not st.session_state.logged_in:
    cookie_val = cookie_manager.get(cookie="bible_rag_login")
    if cookie_val == "true":
        st.session_state.logged_in = True

if not st.session_state.logged_in:
    st.title("Bible RAG Login")
    with st.form("login_form"):
        username = st.text_input("Email")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember Me")
        submitted = st.form_submit_button("Sign In")
        
        if submitted:
            if username == "matv001@madhatv.in" and password == "matv@001":
                st.session_state.logged_in = True
                if remember_me:
                    cookie_manager.set("bible_rag_login", "true", expires_at=datetime.datetime.now() + datetime.timedelta(days=30))
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

# Header
st.title("📖 Tamil Bible RAG")

# Logout Button (Top Right)
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("Sign Out", key="logout_btn"):
        st.session_state.logged_in = False
        cookie_manager.delete("bible_rag_login")
        st.rerun()

# Welcome Message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. How can I help you explore the Scriptures today?"}
    ]

# Main Chat Container
# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Initialize Embeddings (must match ingest.py)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

embeddings = get_embeddings()

# Load Vector Store
DB_PATH = "chroma_db"

# Check if chroma_db exists
if not os.path.exists(DB_PATH):
    zip_path = "chroma_db.zip"
    
    # Reassemble zip if it doesn't exist
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
    
    # Extract zip with path sanitization (fix Windows backslashes)
    if os.path.exists(zip_path):
        import zipfile
        with st.spinner("Extracting database..."):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    # Fix Windows path separators for Linux
                    member.filename = member.filename.replace('\\', '/')
                    zip_ref.extract(member, ".")
    else:
        st.error("Vector Database not found. Please run `ingest.py` first.")
        st.stop()

if not os.path.exists(DB_PATH):
    st.error(f"Vector Database still not found at {DB_PATH}.")
    st.write("Current working directory:", os.getcwd())
    st.write("Files in directory:", os.listdir("."))
    st.stop()

vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# QA Chain Setup
# QA Chain Setup
llm = None
if "DEEPSEEK_API_KEY" in os.environ:
    llm = ChatOpenAI(
        model='deepseek-chat', 
        api_key=os.environ.get("DEEPSEEK_API_KEY"), 
        base_url='https://api.deepseek.com',
        temperature=0.3
    )
elif "GOOGLE_API_KEY" in os.environ:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

if llm:
    
    # High-Quality Tamil Roman Catholic Prompt (Theology Teacher Persona)
    customer_prompt = """
    You are a knowledgeable Roman Catholic Theology Teacher (கத்தோலிக்க இறையியல் ஆசிரியர்).
    Your goal is to provide accurate, spiritually enriching answers based on the Tamil Common Bible (திருவிவிலியம்) and Catholic Tradition.

    **Core Instructions:**
    1. **Persona**: Speak with the authority and clarity of a Catechism teacher. Use formal, clear Tamil.
    2. **Priority**: ALWAYS check the provided **Context** first. If the answer is there, use it.
    3. **Fallback**: If the answer is NOT in the context, use your **General Knowledge** as a Catholic scholar.

    **Theological & Historical Context (Apply where relevant):**
    - **Divine Inspiration**: Explain that God is the primary author, but He revealed concepts through human authors who wrote in their own style, language, and social context (சமுதாயச் சூழல், மொழி நடை).
    - **Unity vs. Diversity**: The Bible is ONE book (Unity) but also a "Library/Collection" (நூலகம்/தொகுப்பு) of **73 Books** (46 Old Testament + 27 New Testament).
    - **Timeline**: Mention the span from **10th Century BC to 1st Century AD**.
    - **Formation**: Explain the transition from **Oral Tradition (வாய்மொழி மரபு)** to **Written Text (எழுத்து வடிவம்)**.

    **Response Format (Markdown):**
    
    ### 💡 சுருக்கம் (Summary)
    [Provide a direct, concise 2-3 line answer in Tamil]

    ### 📝 விளக்கம் (Explanation)
    [Provide a detailed, structured explanation. Incorporate the historical/theological context mentioned above if relevant to the question.]

    ### 📖 இறைவார்த்தைகள் (Verses)
    [Quote relevant verses. If from Context, use **Book Chapter:Verse**. If from General Knowledge, cite the reference clearly.]

    Context:
    {context}

    Question: {question}

    Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(customer_prompt)

    # Standard QA Chain with Custom Prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                try:
                    result = qa_chain({"query": prompt})
                    answer = result["result"]
                    source_docs = result["source_documents"]
                    
                    st.write(answer)
                    
                    # Optional: Show sources
                    with st.expander("Source Documents"):
                        for doc in source_docs:
                            st.write(doc.page_content)
                            
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.error("⚠️ No API Key found! Please set DEEPSEEK_API_KEY or GOOGLE_API_KEY.")
    st.markdown("""
    ### How to fix this on Streamlit Cloud:
    1. Click **Manage App** in the bottom right corner.
    2. Click the **three dots** (⋮) next to your app name and select **Settings**.
    3. Go to the **Secrets** section.
    4. Paste your API key in this format:
    ```toml
    DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
    # OR
    GOOGLE_API_KEY = "your_google_api_key_here"
    ```
    5. Click **Save**.
    """)
