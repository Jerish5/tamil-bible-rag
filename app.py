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
                    answer = result["result"]
                    source_docs = result["source_documents"]
                    
                    # Check for fallback triggers
                    lower_answer = answer.lower()
                    triggers = [
                        "don't know", "do not know", "not found", "not mentioned",
                        "தெரியவில்லை", "தகவல் இல்லை", "குறிப்பிடப்படவில்லை", "பதில் இல்லை", "இல்லை"
                    ]
                    
                    final_response = answer
                    sources_text = ""

                    if any(trigger in lower_answer for trigger in triggers):
                        st.warning("Answer not found in Bible context. Searching the web...")
                        
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
                        final_response = chain_web.run(web_context=web_results, question=prompt)
                        sources_text = "\n\n*Source: Web Search*"
                    else:
                        # Format sources
                        sources_text = ""
                        for i, doc in enumerate(source_docs):
                            book = doc.metadata.get('book', '?')
                            chapter = doc.metadata.get('chapter', '?')
                            verse = doc.metadata.get('verse', '?')
                            content = doc.page_content
                            # Clean up content for display
                            clean_content = content.split(" - ")[-1] if " - " in content else content
                            sources_text += f"""<div class="source-card">
                            <strong>{book} {chapter}:{verse}</strong><br>
                            {clean_content}
                            </div>"""

                    # Display Answer
                    full_response = final_response + sources_text
                    st.markdown(full_response, unsafe_allow_html=True)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")

