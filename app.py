"""
Main Streamlit application for AI Document Assistant (RAG System) with all enhancements.
Modern, beautiful UI design.
"""
import streamlit as st
import os
from dotenv import load_dotenv
from pdf_utils import extract_text_from_multiple_pdfs, preprocess_text
from rag_pipeline import RAGPipeline
from db import ChatDatabase
from auth import AuthSystem, login_page, check_authentication
from cloud_storage import CloudStorage
from utils import validate_pdf_file, format_timestamp
from typing import List, Tuple

# Load environment variables
load_dotenv()

# Page configuration with modern theme
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, beautiful UI
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .status-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a1a1a;
    }
    
    .status-error {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #1a1a1a;
    }
    
    .status-info {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #1a1a1a;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px dashed #667eea;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "db" not in st.session_state:
    st.session_state.db = ChatDatabase()
if "auth_system" not in st.session_state:
    st.session_state.auth_system = AuthSystem()
if "cloud_storage" not in st.session_state:
    storage_type = os.getenv("STORAGE_TYPE", "local")
    st.session_state.cloud_storage = CloudStorage(storage_type=storage_type)
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_documents" not in st.session_state:
    st.session_state.source_documents = {}
if "model_type" not in st.session_state:
    st.session_state.model_type = os.getenv("MODEL_TYPE", "openai")
if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


def initialize_rag_pipeline(model_type: str = None, model_name: str = None):
    """Initialize RAG pipeline with selected model."""
    model_type = model_type or st.session_state.model_type
    model_name = model_name or st.session_state.model_name
    
    try:
        if model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                return None
            return RAGPipeline(
                openai_api_key=api_key,
                model_type="openai",
                model_name=model_name,
                use_memory=True
            )
        elif model_type == "ollama":
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Check if Ollama is running
            try:
                import requests
                response = requests.get(f"{ollama_url}/api/tags", timeout=2)
                if response.status_code != 200:
                    st.warning(f"⚠️ Ollama server at {ollama_url} is not responding correctly. Make sure Ollama is running.")
            except Exception:
                st.warning(f"""
                ⚠️ **Ollama Not Running**
                
                Cannot connect to Ollama at {ollama_url}. Please:
                1. Open Terminal
                2. Run: `ollama serve`
                3. Wait for "Ollama is running" message
                4. Refresh this page
                """)
            return RAGPipeline(
                model_type="ollama",
                model_name=model_name or "llama2",
                ollama_base_url=ollama_url,
                use_memory=True
            )
        elif model_type == "huggingface":
            return RAGPipeline(
                model_type="huggingface",
                model_name=model_name or "gpt2",
                use_memory=True
            )
        else:
            return None
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None


def process_uploaded_documents(pdf_files, user_id: str = None):
    """Process uploaded PDF files and create vector store."""
    if not pdf_files:
        st.error("⚠️ No files provided. Please upload at least one PDF file.")
        return False
    
    # Validate all files are PDFs
    invalid_files = []
    for file in pdf_files:
        if not validate_pdf_file(file):
            invalid_files.append(file.name)
    
    if invalid_files:
        st.error(f"❌ Invalid files detected: {', '.join(invalid_files)}. Only PDF files are supported.")
        return False
    
    # Create a progress container
    progress_container = st.container()
    
    with progress_container:
        # Step 1: Upload files
        st.info(f"📤 Uploading {len(pdf_files)} file(s)...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Upload files to cloud storage
            uploaded_paths = []
            for idx, file in enumerate(pdf_files):
                status_text.text(f"Uploading {file.name}...")
                progress_bar.progress((idx + 1) / (len(pdf_files) * 3))
                
                try:
                    file_bytes = file.read()
                    file.seek(0)
                    storage_path = st.session_state.cloud_storage.upload_file(
                        file_bytes,
                        file.name,
                        user_id=user_id
                    )
                    uploaded_paths.append(storage_path)
                except Exception as upload_error:
                    st.error(f"❌ Failed to upload {file.name}: {str(upload_error)}")
                    return False
            
            status_text.text("✅ Upload complete!")
            
            # Step 2: Extract text from PDFs
            st.info("📄 Extracting text from PDFs...")
            status_text.text("Extracting text from documents...")
            progress_bar.progress(len(pdf_files) / (len(pdf_files) * 3))
            
            try:
                combined_text = extract_text_from_multiple_pdfs(pdf_files)
                
                if not combined_text or not combined_text.strip():
                    st.error(
                        "❌ **No text could be extracted from the PDFs.**\n\n"
                        "This usually means:\n"
                        "- Your PDFs contain only images/scans (not searchable text)\n"
                        "- The PDFs are corrupted or password-protected\n"
                        "- The PDFs are empty\n\n"
                        "**Solution:** Please upload PDFs with extractable text content."
                    )
                    return False
                
                # Show extraction stats
                word_count = len(combined_text.split())
                char_count = len(combined_text)
                status_text.text(f"✅ Extracted {word_count:,} words ({char_count:,} characters)")
                
                # Preprocess text
                processed_text = preprocess_text(combined_text)
                
            except Exception as extract_error:
                st.error(
                    f"❌ **Text extraction failed:** {str(extract_error)}\n\n"
                    "Please check:\n"
                    "- PDFs are not corrupted\n"
                    "- PDFs contain readable text (not just scanned images)\n"
                    "- PDFs are not password-protected"
                )
                return False
            
            # Step 3: Create embeddings and vector store
            st.info("🔄 Creating embeddings and vector store...")
            status_text.text("Processing documents and creating embeddings (this may take a moment)...")
            progress_bar.progress((len(pdf_files) * 2) / (len(pdf_files) * 3))
            
            try:
                # Initialize RAG pipeline if not already done
                if st.session_state.rag_pipeline is None:
                    status_text.text("Initializing AI model...")
                    st.session_state.rag_pipeline = initialize_rag_pipeline()
                    
                    if st.session_state.rag_pipeline is None:
                        st.error(
                            "❌ **Failed to initialize AI model.**\n\n"
                            "Please check:\n"
                            "- Your API key is set correctly in the .env file\n"
                            "- You have selected a valid model in the sidebar\n"
                            "- For Ollama: ensure the server is running (`ollama serve`)\n"
                            "- For OpenAI: check your API quota and billing"
                        )
                        return False
                
                # Load documents into RAG pipeline
                status_text.text("Creating vector embeddings...")
                st.session_state.rag_pipeline.load_documents(processed_text)
                st.session_state.documents_processed = True
                
                # Complete!
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Show success message with stats
                st.success(
                    f"✅ **Successfully processed {len(pdf_files)} document(s)!**\n\n"
                    f"- **Words extracted:** {word_count:,}\n"
                    f"- **Ready for questions!**"
                )
                
                return True
                
            except Exception as embedding_error:
                error_msg = str(embedding_error)
                
                # Check for specific errors
                if st.session_state.model_type == "openai" and (
                    "quota" in error_msg.lower() or 
                    "429" in error_msg or 
                    "insufficient_quota" in error_msg
                ):
                    st.error(
                        "❌ **OpenAI API Quota Exceeded**\n\n"
                        "Your OpenAI API quota has been exceeded. Options:\n\n"
                        "1. **Check Billing:** Visit https://platform.openai.com/account/billing\n"
                        "2. **Add Payment Method:** Add a payment method to increase quota\n"
                        "3. **Use Ollama (Free!):** Switch to Ollama in Model Settings\n"
                        "4. **Wait:** Free tier quotas reset monthly\n\n"
                        "**Quick Fix:** Switch to Ollama model (free, runs locally) in the sidebar!"
                    )
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    st.error(
                        "❌ **Processing Timeout**\n\n"
                        "The embedding process took too long. This can happen with:\n"
                        "- Very large PDF files\n"
                        "- Slow internet connection (for cloud models)\n"
                        "- Server overload\n\n"
                        "**Try:**\n"
                        "1. Upload smaller PDF files (split large PDFs)\n"
                        "2. Use Ollama for local processing (no internet needed)\n"
                        "3. Try again in a few moments"
                    )
                elif "memory" in error_msg.lower() or "oom" in error_msg.lower():
                    st.error(
                        "❌ **Out of Memory**\n\n"
                        "Your system ran out of memory processing these documents.\n\n"
                        "**Try:**\n"
                        "1. Upload fewer/smaller PDF files\n"
                        "2. Close other applications\n"
                        "3. Use a cloud-based model (OpenAI) instead of local models"
                    )
                else:
                    st.error(
                        f"❌ **Error creating embeddings:** {error_msg}\n\n"
                        "**Troubleshooting:**\n"
                        "1. Try reprocessing the documents\n"
                        "2. Check your model settings in the sidebar\n"
                        "3. Try a different model (OpenAI, Ollama, or HuggingFace)\n"
                        "4. Restart the application"
                    )
                
                return False
                
        except Exception as e:
            st.error(f"❌ **Unexpected error:** {str(e)}\n\nPlease try again or contact support.")
            return False
        finally:
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()



def create_new_session():
    """Create a new chat session."""
    session_id = st.session_state.db.create_session()
    st.session_state.current_session_id = session_id
    st.session_state.messages = []
    st.session_state.source_documents = {}
    return session_id


def load_session(session_id: str):
    """Load messages from a session."""
    messages = st.session_state.db.get_session_messages(session_id)
    st.session_state.messages = messages
    st.session_state.current_session_id = session_id


def get_chat_history() -> List[Tuple[str, str]]:
    """Get chat history as list of (question, answer) tuples."""
    history = []
    i = 0
    while i < len(st.session_state.messages) - 1:
        if (st.session_state.messages[i]["role"] == "user" and 
            st.session_state.messages[i + 1]["role"] == "assistant"):
            history.append((
                st.session_state.messages[i]["content"],
                st.session_state.messages[i + 1]["content"]
            ))
            i += 2
        else:
            i += 1
    return history


def display_chat_messages():
    """Display chat messages in the UI with source highlighting."""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show source documents for assistant messages
            if (message["role"] == "assistant" and 
                st.session_state.show_sources and
                idx in st.session_state.source_documents):
                with st.expander("📄 View Sources", expanded=False):
                    sources = st.session_state.source_documents[idx]
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:500] + "..." if len(source) > 500 else source)
                        st.divider()


def main():
    """Main application function."""
    # Check authentication
    if not check_authentication():
        login_page(st.session_state.auth_system)
        return
    
    # Beautiful header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🤖 AI Document Assistant</div>
        <div class="header-subtitle">Upload PDF documents and ask questions using AI-powered RAG</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for settings and document upload
    with st.sidebar:
        # User profile section
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Welcome, {st.session_state.username}!**")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.rag_pipeline = None
            st.session_state.documents_processed = False
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Model selection with beautiful cards
        st.markdown("### ⚙️ Model Settings")
        
        model_type = st.selectbox(
            "**Model Type**",
            ["openai", "ollama", "huggingface"],
            index=["openai", "ollama", "huggingface"].index(st.session_state.model_type) if st.session_state.model_type in ["openai", "ollama", "huggingface"] else 0,
            help="Choose your AI model provider"
        )
        
        # Ensure a sensible default model name when switching providers
        default_names = {"openai": "gpt-3.5-turbo", "ollama": "llama2", "huggingface": "gpt2"}
        if model_type != st.session_state.model_type:
            st.session_state.model_type = model_type
            st.session_state.model_name = default_names.get(model_type, st.session_state.model_name)
        
        if model_type == "openai":
            model_name = st.selectbox(
                "**OpenAI Model**",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"].index(st.session_state.model_name) if st.session_state.model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"] else 0
            )
        elif model_type == "ollama":
            model_name = st.text_input("**Ollama Model**", value=st.session_state.model_name or "llama2")
        else:  # huggingface
            model_name = st.text_input("**HuggingFace Model**", value=st.session_state.model_name or "gpt2")
        
        if model_type != st.session_state.model_type or model_name != st.session_state.model_name:
            # Guard against incompatible names (e.g., OpenAI names in Ollama)
            if model_type == "ollama" and ("gpt" in (model_name or "").lower()):
                st.warning("The selected model name isn't available in Ollama. Switching to 'llama2'.")
                model_name = "llama2"
            st.session_state.model_type = model_type
            st.session_state.model_name = model_name
            st.session_state.rag_pipeline = initialize_rag_pipeline(model_type, model_name)
            if st.session_state.rag_pipeline:
                st.success("✅ Model updated!")
        
        st.divider()
        
        # Document upload section
        st.markdown("### 📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "**Upload PDF files**",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to analyze"
        )
        
        if uploaded_files:
            st.info(f"📄 {len(uploaded_files)} file(s) selected")
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary", use_container_width=True):
            if uploaded_files:
                success = process_uploaded_documents(uploaded_files, user_id=st.session_state.username)
                if success:
                    # Create new session after processing documents
                    create_new_session()
            else:
                st.warning("⚠️ Please upload at least one PDF file.")
        
        # Show processing status with beautiful badges
        if st.session_state.documents_processed:
            st.markdown('<div class="status-badge status-success">✅ Documents Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge status-info">📤 Upload documents to start</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Session management
        st.markdown("### 💬 Chat Sessions")
        
        # New session button
        if st.button("➕ New Session", use_container_width=True):
            create_new_session()
            st.rerun()
        
        # Load previous sessions
        sessions = st.session_state.db.get_all_sessions()
        if sessions:
            st.markdown("**Previous Sessions:**")
            for session in sessions[:5]:  # Show last 5 sessions
                session_id = session["session_id"]
                updated_at = format_timestamp(session["updated_at"])
                
                # Display session info
                if st.button(
                    f"📝 {updated_at}",
                    key=f"session_{session_id}",
                    use_container_width=True
                ):
                    load_session(session_id)
                    st.rerun()
        
        st.divider()
        
        # Settings
        st.markdown("### ⚙️ Settings")
        st.session_state.show_sources = st.checkbox(
            "**Show Source Documents**",
            value=st.session_state.show_sources,
            help="Display source PDF snippets used in answers"
        )
        
        # API Key status with beautiful styling
        st.markdown("### 🔑 API Status")
        if st.session_state.model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your_openai_api_key_here":
                st.markdown('<div class="status-badge status-success">✅ API Key Configured</div>', unsafe_allow_html=True)
                # Warning about quota
                st.warning("""
                ⚠️ **Quota Notice**
                
                If you see quota errors, consider:
                - Using **Ollama** (free, local) - switch in Model Settings
                - Checking billing: https://platform.openai.com/account/billing
                """)
            else:
                st.markdown('<div class="status-badge status-error">⚠️ API Key Missing</div>', unsafe_allow_html=True)
                with st.expander("📝 How to set API key"):
                    st.markdown("""
                    1. Create a `.env` file in the project root
                    2. Add: `OPENAI_API_KEY=your_actual_api_key_here`
                    3. Get your key from: https://platform.openai.com/api-keys
                    4. Restart the app
                    """)
        elif st.session_state.model_type == "ollama":
            st.markdown('<div class="status-badge status-success">🦙 Using Ollama (Free & Local)</div>', unsafe_allow_html=True)
            st.info("💡 Ollama runs locally - no API costs!")
            # Check if Ollama is actually running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    st.success("✅ Ollama server is running")
                else:
                    st.warning("⚠️ Ollama server may not be responding correctly")
            except:
                st.error("""
                ❌ **Ollama server is not running!**
                
                Please start Ollama:
                1. Open Terminal
                2. Run: `ollama serve`
                3. Wait for "Ollama is running" message
                4. Refresh this page
                """)
        else:
            st.markdown('<div class="status-badge status-info">🤗 Using HuggingFace</div>', unsafe_allow_html=True)
    
    # Main chat interface
    if not st.session_state.documents_processed:
        # Beautiful welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="info-box">
                <h3 style="text-align: center; color: #667eea;">🚀 Get Started</h3>
                <p style="text-align: center;">Upload and process PDF documents in the sidebar to start asking questions!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h2>📄</h2>
                <h3>Multi-Document</h3>
                <p>Upload multiple PDFs at once</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h2>🔍</h2>
                <h3>Semantic Search</h3>
                <p>AI-powered document search</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h2>💾</h2>
                <h3>Chat History</h3>
                <p>All conversations saved</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions
        with st.expander("ℹ️ How to use this app", expanded=True):
            st.markdown("""
            ### 🎯 Quick Start Guide
            
            1. **Select Model**: Choose your AI model in the sidebar (OpenAI, Ollama, or HuggingFace)
            2. **Upload Documents**: Click "Browse files" to upload PDF documents
            3. **Process**: Click "Process Documents" to extract and index the content
            4. **Ask Questions**: Start chatting about your documents!
            
            ### ✨ Features
            
            - 📄 **Multi-Document Support**: Upload and query multiple PDFs simultaneously
            - 🔍 **Semantic Search**: Find relevant information using AI-powered search
            - 💾 **Chat History**: All conversations are automatically saved
            - 🔄 **Session Management**: Create new sessions or continue previous ones
            - 🧠 **Chat Memory**: Conversation context is maintained across turns
            - 📄 **Source Highlighting**: See exact PDF snippets used in answers
            - 👤 **User Authentication**: Secure login system
            - ☁️ **Cloud Storage**: Files stored in cloud (S3) or locally
            - 🤖 **Multiple Models**: Support for OpenAI, Ollama, and HuggingFace
            """)
        
        return
    
    # Create session if none exists
    if st.session_state.current_session_id is None:
        create_new_session()
    
    # Display chat history
    if st.session_state.messages:
        display_chat_messages()
    else:
        # Welcome message
        st.info("👋 Start asking questions about your documents!")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Validate question
        if not prompt.strip():
            st.warning("⚠️ Please enter a valid question.")
            return
        
        # Add user message to chat
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        
        # Save user message to database
        st.session_state.db.save_message(
            st.session_state.current_session_id,
            "user",
            prompt
        )
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            # Create a status container for detailed progress
            status_container = st.empty()
            
            try:
                # Step 1: Retrieve relevant documents
                status_container.info("🔍 Searching documents for relevant information...")
                
                # Get chat history for context
                chat_history = get_chat_history()
                
                # Step 2: Generate answer
                status_container.info("🤔 Generating answer (this may take 10-30 seconds)...")
                
                # Query RAG pipeline with memory
                answer, source_docs = st.session_state.rag_pipeline.query(
                    prompt,
                    k=4,
                    chat_history=chat_history,
                    timeout=120  # 2 minute timeout
                )
                
                # Clear status
                status_container.empty()
                
                # Display response
                st.markdown(answer)
                
                # Store source documents
                message_idx = len(st.session_state.messages)
                st.session_state.source_documents[message_idx] = [
                    doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    for doc in source_docs
                ]
                
                # Display sources if enabled
                if st.session_state.show_sources and source_docs:
                    with st.expander("📄 View Sources", expanded=False):
                        for i, source in enumerate(source_docs, 1):
                            source_text = source.page_content if hasattr(source, 'page_content') else str(source)
                            st.markdown(f"**Source {i}:**")
                            st.text(source_text[:500] + "..." if len(source_text) > 500 else source_text)
                            st.divider()
                
                # Save assistant message
                assistant_message = {"role": "assistant", "content": answer}
                st.session_state.messages.append(assistant_message)
                
                # Save to database
                st.session_state.db.save_message(
                    st.session_state.current_session_id,
                    "assistant",
                    answer
                )
                
            except ValueError as ve:
                # Handle validation errors (empty questions, no relevant docs, etc.)
                status_container.empty()
                error_msg = str(ve)
                
                if "No relevant information" in error_msg:
                    st.warning(
                        f"⚠️ **{error_msg}**\n\n"
                        "**Suggestions:**\n"
                        "- Try rephrasing your question\n"
                        "- Use keywords from your documents\n"
                        "- Ask more specific questions\n"
                        "- Ensure your question relates to the uploaded documents"
                    )
                elif "empty" in error_msg.lower():
                    st.error(
                        f"❌ **{error_msg}**\n\n"
                        "This usually means your PDFs don't contain extractable text. "
                        "Please upload PDFs with readable text content."
                    )
                else:
                    st.warning(f"⚠️ {error_msg}")
                
                # Don't save validation errors to chat history
                
            except Exception as e:
                status_container.empty()
                error_msg = str(e)
                
                # Comprehensive error handling with helpful messages
                if "quota" in error_msg.lower() or "429" in error_msg or "API quota" in error_msg:
                    st.error(
                        "❌ **OpenAI API Quota Exceeded**\n\n"
                        "Your OpenAI API quota has been exceeded.\n\n"
                        "**Options:**\n"
                        "1. **Check Billing:** https://platform.openai.com/account/billing\n"
                        "2. **Add Payment Method:** Increase your quota\n"
                        "3. **Use Ollama (Free!):** Switch to Ollama in Model Settings\n"
                        "4. **Wait:** Free tier quotas reset monthly\n\n"
                        "**Quick Fix:** Switch to Ollama model (free, local) in the sidebar!"
                    )
                    
                elif "Cannot connect to Ollama" in error_msg or "connection" in error_msg.lower():
                    st.error(
                        "❌ **Ollama Connection Error**\n\n"
                        "Cannot connect to Ollama server.\n\n"
                        "**Steps to fix:**\n"
                        "1. Open Terminal\n"
                        "2. Run: `ollama serve`\n"
                        "3. In another terminal, run: `ollama pull llama2`\n"
                        "4. Wait for \"Ollama is running\" message\n"
                        "5. Refresh this page and try again\n\n"
                        "**Check:** Make sure Ollama is installed: `ollama --version`"
                    )
                    
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    st.error(
                        "❌ **Request Timeout**\n\n"
                        "The AI model took too long to respond.\n\n"
                        "**This can happen when:**\n"
                        "- Your question is very complex\n"
                        "- Your documents are very large\n"
                        "- The model server is slow/overloaded\n"
                        "- Internet connection is slow (for cloud models)\n\n"
                        "**Try:**\n"
                        "1. Ask a simpler, more specific question\n"
                        "2. Upload smaller documents\n"
                        "3. Switch to a faster model (e.g., gpt-3.5-turbo)\n"
                        "4. Try again in a moment"
                    )
                    
                elif "Invalid" in error_msg and "API key" in error_msg:
                    st.error(
                        "❌ **Invalid API Key**\n\n"
                        "Your OpenAI API key is invalid or not set correctly.\n\n"
                        "**Steps to fix:**\n"
                        "1. Check your `.env` file in the project root\n"
                        "2. Ensure it contains: `OPENAI_API_KEY=your_actual_key`\n"
                        "3. Get your key from: https://platform.openai.com/api-keys\n"
                        "4. Restart the application\n\n"
                        "**Alternative:** Switch to Ollama (free, no API key needed)"
                    )
                    
                elif "rate limit" in error_msg.lower():
                    st.warning(
                        "⚠️ **Rate Limit Exceeded**\n\n"
                        "You're making requests too quickly.\n\n"
                        "**Please:**\n"
                        "- Wait 10-20 seconds before trying again\n"
                        "- Consider using Ollama (no rate limits)\n"
                        "- Upgrade your OpenAI plan for higher limits"
                    )
                    
                elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                    model_name = st.session_state.model_name
                    if st.session_state.model_type == "ollama":
                        st.error(
                            f"❌ **Ollama Model Not Found**\n\n"
                            f"The model '{model_name}' is not available.\n\n"
                            f"**To install:**\n"
                            f"1. Open Terminal\n"
                            f"2. Run: `ollama pull {model_name}`\n"
                            f"3. Wait for download to complete\n"
                            f"4. Try your question again\n\n"
                            f"**Popular models:** llama2, mistral, codellama"
                        )
                    else:
                        st.error(
                            f"❌ **Model Not Found**\n\n"
                            f"The model '{model_name}' is not available.\n\n"
                            f"**Please:**\n"
                            f"- Check the model name in the sidebar\n"
                            f"- Select a different model\n"
                            f"- Verify the model exists for your provider"
                        )
                        
                elif "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
                    st.error(
                        "❌ **Out of Memory**\n\n"
                        "Your system ran out of memory.\n\n"
                        "**Try:**\n"
                        "1. Close other applications\n"
                        "2. Upload smaller documents\n"
                        "3. Use a cloud model (OpenAI) instead of local\n"
                        "4. Restart the application"
                    )
                    
                elif "No documents loaded" in error_msg:
                    st.error(
                        "❌ **No Documents Loaded**\n\n"
                        "Please upload and process PDF documents first.\n\n"
                        "**Steps:**\n"
                        "1. Click 'Browse files' in the sidebar\n"
                        "2. Select your PDF files\n"
                        "3. Click 'Process Documents'\n"
                        "4. Wait for processing to complete\n"
                        "5. Then ask your question"
                    )
                    
                else:
                    # Generic error with troubleshooting
                    st.error(
                        f"❌ **Error:** {error_msg}\n\n"
                        "**Troubleshooting:**\n"
                        "1. Try asking your question again\n"
                        "2. Rephrase your question\n"
                        "3. Check your model settings in the sidebar\n"
                        "4. Try a different model\n"
                        "5. Restart the application if issues persist\n\n"
                        "**Need help?** Check the model status in the sidebar."
                    )
                
                # Save error to chat history for context
                error_summary = f"Error: {error_msg[:200]}"  # Truncate long errors
                error_message = {"role": "assistant", "content": error_summary}
                st.session_state.messages.append(error_message)
                
                st.session_state.db.save_message(
                    st.session_state.current_session_id,
                    "assistant",
                    error_summary
                )


if __name__ == "__main__":
    main()
