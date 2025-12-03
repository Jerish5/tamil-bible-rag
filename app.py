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
import extra_streamlit_components as stx
import time
import datetime

# Load environment variables
load_dotenv()

# Custom CSS for "Bible Gateway" Style Interface
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #fcfcfc;
        font-family: 'Merriweather', 'Georgia', serif;
        color: #333;
    }
    
    /* Hide Default Header/Sidebar */
    header[data-testid="stHeader"] {
        background: white;
        border-bottom: 1px solid #eee;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Navbar Styling */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: white;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: bold;
        color: #8B0000; /* Dark Red like traditional Bibles */
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Chat Container */
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 20px;
    }
    
    /* Message Styling */
    .stChatMessage {
        background-color: transparent;
        border-bottom: 1px solid #f0f0f0;
        padding: 1.5rem 0;
    }
    
    div[data-testid="stChatMessageContent"] {
        background-color: transparent !important;
        color: #333 !important;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Assistant Message Specifics */
    div[data-testid="stChatMessageContent"][aria-label="assistant"] {
        font-family: 'Merriweather', serif;
    }
    
    /* User Message Specifics */
    div[data-testid="stChatMessageContent"][aria-label="user"] {
        background-color: #f5f5f5 !important;
        border-radius: 10px;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif; /* Sans-serif for user input */
    }
    
    /* Fixed Input Area */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 20px;
        background: white;
        z-index: 1000;
        border-top: 1px solid #eee;
    }
    
    div[data-testid="stChatInput"] {
        max-width: 900px;
        margin: 0 auto;
        border-radius: 50px !important;
        border: 1px solid #ddd !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    }
    
    /* Add padding to bottom of main container so content isn't hidden behind input */
    .main-container {
        padding-bottom: 100px;
    }
    
    /* Verse Card */
    .source-card {
        background-color: #fff9f0; /* Warm paper color */
        border-left: 4px solid #8B0000;
        padding: 15px;
        margin-top: 15px;
        font-family: 'Merriweather', serif;
        font-style: italic;
        color: #555;
    }
    
    /* Login Form Clean Style */
    div[data-testid="stForm"] {
        background-color: white;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #eee;
    }
    
    h1, h2, h3 {
        color: #333 !important;
    }
    
    .stButton button {
        background-color: #8B0000 !important;
        color: white !important;
        border-radius: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# Login Logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Initialize Cookie Manager
@st.cache_resource(experimental_allow_widgets=True)
def get_manager():
    return stx.CookieManager()

cookie_manager = get_manager()

# Check for existing cookie
if not st.session_state.logged_in:
    cookie_val = cookie_manager.get(cookie="bible_rag_login")
    if cookie_val == "true":
        st.session_state.logged_in = True

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #8B0000;'>Bible RAG Login</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Email")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember Me")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username == "matv001@madhatv.in" and password == "matv@001":
                    st.session_state.logged_in = True
                    if remember_me:
                        cookie_manager.set("bible_rag_login", "true", expires_at=datetime.datetime.now() + datetime.timedelta(days=30))
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    st.stop()

# Navbar / Header
st.markdown("""
<div class="navbar">
    <div class="logo">
        <span>📖</span> Tamil Bible RAG
    </div>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<div class="main-container">', unsafe_allow_html=True)

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
if "GOOGLE_API_KEY" in os.environ:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    # Custom Prompt for Bible RAG with JSON Output
    customer_prompt = """
    You are a Bible expert assistant specializing in the Tamil Common Bible.
    
    **Role & Tone:**
    - You must act as a knowledgeable Roman Catholic Bible scholar.
    - Use strict "Roman Catholic Tamil" terminology (e.g., use 'Thiruviliyam' for Bible).
    - Your response MUST be in **valid JSON format** strictly adhering to the schema below.
    - Do not include any markdown formatting (like ```json) outside the JSON object. Just return the raw JSON.

    **JSON Schema:**
    {
      "summary": "A 2-3 line concise answer in Tamil.",
      "explanation": "A detailed, multi-paragraph explanation based on the context. Use clear Tamil.",
      "verses": [
         {"reference": "Book Chapter:Verse", "text": "Verse text in Tamil..."}
      ],
      "suggestions": [
         "Next question suggestion 1",
         "Next question suggestion 2",
         "Next question suggestion 3"
      ]
    }

    **Rules:**
    1. **Context Only**: Use ONLY the provided context to answer. If the answer is not in the context, set "summary" to "I don't know" and leave others empty.
    2. **Verses**: Extract verses mentioned in the context. Ensure the reference matches the text.
    3. **Counting**: If asked to count, analyze the context first, then provide the count in the "summary" and details in "explanation".

    Context:
    {context}

    Question: {question}

    Answer (JSON):"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(customer_prompt)
    
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

        padding-bottom: 100px;
    }
    
    /* Verse Card */
    .source-card {
        background-color: #fff9f0; /* Warm paper color */
        border-left: 4px solid #8B0000;
        padding: 15px;
        margin-top: 15px;
        font-family: 'Merriweather', serif;
        font-style: italic;
        color: #555;
    }
    
    /* Login Form Clean Style */
    div[data-testid="stForm"] {
        background-color: white;
        padding: 40px;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #eee;
    }
    
    h1, h2, h3 {
        color: #333 !important;
    }
    
    .stButton button {
        background-color: #8B0000 !important;
        color: white !important;
        border-radius: 5px !important;
    }
</style>
""", unsafe_allow_html=True)

# Login Logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #8B0000;'>Bible RAG Login</h1>", unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username == "matv001@madhatv.in" and password == "matv@001":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    st.stop()

# Navbar / Header
st.markdown("""
<div class="navbar">
    <div class="logo">
        <span>📖</span> Tamil Bible RAG
    </div>
</div>
""", unsafe_allow_html=True)

# Logout Button (Top Right)
col1, col2 = st.columns([8, 1])
with col2:
    if st.button("Sign Out", key="logout_btn"):
        st.session_state.logged_in = False
        st.rerun()

# Welcome Message
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. How can I help you explore the Scriptures today?"}
    ]

# Main Chat Container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

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
if "GOOGLE_API_KEY" in os.environ:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    # Custom Prompt for Bible RAG with JSON Output
    customer_prompt = """
    You are a Bible expert assistant specializing in the Tamil Common Bible.
    
    **Role & Tone:**
    - You must act as a knowledgeable Roman Catholic Bible scholar.
    - Use strict "Roman Catholic Tamil" terminology (e.g., use 'Thiruviliyam' for Bible).
    - Your response MUST be in **valid JSON format** strictly adhering to the schema below.
    - Do not include any markdown formatting (like ```json) outside the JSON object. Just return the raw JSON.

    **JSON Schema:**
    {
      "summary": "A 2-3 line concise answer in Tamil.",
      "explanation": "A detailed, multi-paragraph explanation based on the context. Use clear Tamil.",
      "verses": [
         {"reference": "Book Chapter:Verse", "text": "Verse text in Tamil..."}
      ],
      "suggestions": [
         "Next question suggestion 1",
         "Next question suggestion 2",
         "Next question suggestion 3"
      ]
    }

    **Rules:**
    1. **Context Only**: Use ONLY the provided context to answer. If the answer is not in the context, set "summary" to "I don't know" and leave others empty.
    2. **Verses**: Extract verses mentioned in the context. Ensure the reference matches the text.
    3. **Counting**: If asked to count, analyze the context first, then provide the count in the "summary" and details in "explanation".

    Context:
    {context}

    Question: {question}

    Answer (JSON):"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(customer_prompt)
    
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
            with st.spinner("Searching the Scriptures..."):
                try:
                    result = qa_chain({"query": prompt})
                    raw_answer = result["result"]
                    source_docs = result["source_documents"]
                    
                    # Try to parse JSON
                    import json
                    try:
                        # Clean up potential markdown code blocks if the LLM adds them
                        clean_answer = raw_answer.replace("```json", "").replace("```", "").strip()
                        data = json.loads(clean_answer)
                        
                        # Check for fallback triggers in the summary
                        lower_summary = data.get("summary", "").lower()
                        triggers = [
                            "don't know", "do not know", "not found", "not mentioned",
                            "தெரியவில்லை", "தகவல் இல்லை", "குறிப்பிடப்படவில்லை", "பதில் இல்லை", "இல்லை"
                        ]
                        
                        if any(trigger in lower_summary for trigger in triggers):
                            raise ValueError("Answer not found in context")

                        # 1. Display Summary
                        st.markdown(f"### 💡 சுருக்கம் (Summary)")
                        st.info(data.get("summary", ""))
                        
                        # 2. Display Explanation
                        st.markdown(f"### 📝 விளக்கம் (Explanation)")
                        st.markdown(data.get("explanation", ""))
                        
                        # 3. Display Verses (from JSON or Context)
                        st.markdown(f"### 📖 இறைவார்த்தைகள் (Verses)")
                        
                        # Prefer verses from JSON if available and valid, otherwise fallback to source docs
                        json_verses = data.get("verses", [])
                        if json_verses:
                            for verse in json_verses:
                                ref = verse.get("reference", "Unknown")
                                text = verse.get("text", "")
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>{ref}</strong><br>{text}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # Fallback to source docs if JSON verses are empty
                            for doc in source_docs:
                                content = doc.page_content
                                clean_content = content.split(" - ")[-1] if " - " in content else content
                                st.markdown(f"""
                                <div class="source-card">
                                    {clean_content}
                                </div>
                                """, unsafe_allow_html=True)

                        # 4. Display Suggestions
                        suggestions = data.get("suggestions", [])
                        if suggestions:
                            st.markdown("### 🔍 அடுத்து என்ன கேட்கலாம்? (Suggestions)")
                            cols = st.columns(len(suggestions))
                            for i, suggestion in enumerate(suggestions):
                                # Note: Buttons in chat history might not be clickable in the same way, 
                                # but they serve as visual prompts.
                                st.button(suggestion, key=f"sugg_{i}")

                        # Save structured response to history (as formatted HTML/Markdown)
                        # We reconstruct a nice string for the history
                        history_content = f"**Summary:** {data.get('summary')}\n\n**Explanation:** {data.get('explanation')}"
                        st.session_state.messages.append({"role": "assistant", "content": history_content})

                    except (json.JSONDecodeError, ValueError):
                        # Fallback to Web Search or Raw Text if JSON fails or answer not found
                        st.warning("Answer not found in Bible context or format error. Searching the web...")
                        
                        search = DuckDuckGoSearchRun()
                        web_results = search.run(prompt)
                        
                        web_template = """You are a helpful assistant. The user asked a question that wasn't found in the Bible database.
                        Here is some information from the web:
                        {web_context}
                        
                        Question: {question}
                        
                        Answer based on the web info (cite source as 'Web Search'). Answer in the SAME language as the question.
                        **CRITICAL**: All Tamil answers MUST be in **Roman Catholic Tamil style** (e.g., use 'Thiruviliyam' for Bible, and standard Catholic terminology)."""
                        
                        prompt_web = PromptTemplate.from_template(web_template)
                        chain_web = LLMChain(llm=llm, prompt=prompt_web)
                        
                        web_response = chain_web.run(web_context=web_results, question=prompt)
                        
                        st.markdown("### Web Answer")
                        st.write(web_response)
                        st.markdown("\n\n*Source: Web Search*")
                        
                        st.session_state.messages.append({"role": "assistant", "content": web_response + "\n\n*Source: Web Search*"})
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")
