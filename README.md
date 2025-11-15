# AI Document Assistant (RAG System MVP) - Enhanced Edition

A comprehensive Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about them using AI-powered semantic search. This enhanced version includes all advanced features.

## ✨ Features

### Core Features
- 📄 **Multi-Document Support**: Upload and query multiple PDFs simultaneously
- 🔍 **Semantic Search**: AI-powered retrieval finds relevant information from your documents
- 💬 **Chat Interface**: Natural conversation with your documents
- 💾 **Chat History**: All conversations are saved locally in SQLite database
- 🔄 **Session Management**: Create new sessions or continue previous conversations

### Enhanced Features (NEW!)
- 🧠 **Chat Memory**: Conversation context is maintained across turns for better follow-up questions
- 📄 **Source Highlighting**: See exact PDF snippets used in answers with expandable source views
- 👤 **User Authentication**: Secure login system with user registration
- ☁️ **Cloud Storage**: Files stored in cloud (AWS S3) or locally
- 🤖 **Multiple Model Support**: 
  - OpenAI (GPT-3.5-turbo, GPT-4, GPT-4-turbo)
  - Ollama (local models like Llama2)
  - Hugging Face (local transformers models)

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root with your configuration:

```env
# Required for OpenAI models
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model configuration
MODEL_TYPE=openai  # Options: openai, ollama, huggingface
MODEL_NAME=gpt-3.5-turbo  # Model name based on MODEL_TYPE

# Optional: Ollama configuration
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Cloud storage (AWS S3)
STORAGE_TYPE=local  # Options: local, s3
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=your_bucket_name
LOCAL_STORAGE_PATH=uploads  # For local storage
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Login/Register**: Create an account or login with existing credentials
2. **Select Model**: Choose your preferred AI model (OpenAI, Ollama, or HuggingFace) in the sidebar
3. **Upload Documents**: Use the sidebar to upload one or more PDF files
4. **Process Documents**: Click "Process Documents" to extract text and create embeddings
5. **Ask Questions**: Type your questions in the chat interface
6. **View Sources**: Toggle "Show Source Documents" to see which PDF snippets were used
7. **View History**: Access previous chat sessions from the sidebar

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application (with all enhancements)
├── rag_pipeline.py     # RAG pipeline (embeddings, retrieval, LLM with memory)
├── db.py               # SQLite database management (sessions, messages)
├── auth.py             # User authentication system
├── cloud_storage.py    # Cloud storage integration (S3/local)
├── pdf_utils.py        # PDF text extraction utilities
├── utils.py            # Helper functions
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🔧 How It Works

1. **User Authentication**: Users register/login to access the system
2. **Document Upload**: PDFs are uploaded via the web interface and stored in cloud/local storage
3. **Text Extraction**: Text is extracted from all pages using `pdfplumber`
4. **Chunking**: Text is split into smaller, overlapping chunks (1000 chars each)
5. **Embeddings**: Each chunk is converted to a vector using embeddings (OpenAI or HuggingFace)
6. **Vector Database**: Embeddings are stored in FAISS for fast semantic search
7. **Query Processing**: When you ask a question:
   - Your question is converted to an embedding
   - Relevant document chunks are retrieved
   - Conversation history is included for context
   - The LLM generates an answer based on those chunks
   - Source documents are tracked and displayed
8. **Chat History**: All messages are saved to SQLite database with user association

## 🛠️ Technologies Used

### Core Technologies
- **Streamlit**: Web application framework
- **LangChain**: RAG pipeline orchestration with memory support
- **FAISS**: Vector database for semantic search
- **SQLite**: Local database for chat history and user management
- **pdfplumber**: PDF text extraction

### AI Models
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Ollama**: Local LLM support (Llama2, etc.)
- **Hugging Face**: Local transformer models

### Storage
- **Local Storage**: File system storage
- **AWS S3**: Cloud storage integration

### Authentication
- **bcrypt**: Password hashing
- **SQLite**: User database

## 📝 Notes

- The vector database is stored in memory (can be extended to save to disk)
- Chat history is stored in `chat_history.db` SQLite file
- User data is stored in `users.db` SQLite file
- Uploaded files are stored locally in `uploads/` or in AWS S3
- For OpenAI models, requires an active internet connection
- For Ollama models, requires Ollama running locally
- For HuggingFace models, models are downloaded on first use

## 🎯 Model Configuration

### OpenAI Models
- Requires `OPENAI_API_KEY` in `.env`
- Models: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- Fast and accurate, requires API key

### Ollama Models
- Requires Ollama installed and running locally
- Set `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- Models: `llama2`, `mistral`, `codellama`, etc.
- Free and private, runs locally

### Hugging Face Models
- Models downloaded automatically on first use
- Models: `gpt2`, `bert-base-uncased`, etc.
- Free and private, runs locally
- May require significant disk space and memory

## 🔐 Security Features

- Password hashing using SHA-256
- User authentication required for access
- User-specific file storage
- Session management

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Cloud Deployment Options

1. **Hugging Face Spaces** (Free)
   - Push code to GitHub
   - Create new Space on Hugging Face
   - Connect GitHub repository
   - Set environment variables in Space settings

2. **Streamlit Cloud** (Free)
   - Push code to GitHub
   - Connect to Streamlit Cloud
   - Deploy from GitHub repository

3. **AWS EC2 / VPS**
   - Install dependencies
   - Set up environment variables
   - Run with `streamlit run app.py --server.port 8501`

4. **Docker** (Recommended for production)
   - Create Dockerfile
   - Build and run container
   - Set environment variables

## 📊 Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `password_hash`: Hashed password
- `email`: Optional email
- `created_at`: Account creation timestamp
- `last_login`: Last login timestamp

### Sessions Table
- `session_id`: Primary key (UUID)
- `created_at`: Session creation timestamp
- `updated_at`: Last update timestamp

### Messages Table
- `id`: Primary key
- `session_id`: Foreign key to sessions
- `role`: 'user' or 'assistant'
- `content`: Message content
- `timestamp`: Message timestamp

## 🎓 What You'll Learn

By completing this enhanced MVP, you demonstrate:

- Understanding of modern AI architectures (RAG, embeddings, LLMs)
- Full-stack ML engineering — from data ingestion to deployment
- Practical coding skills in Python, LangChain, and Streamlit
- Database design for persistence (SQLite)
- User authentication and security
- Cloud storage integration
- Local model integration
- Real-world project thinking (how to handle multi-document, scalable workflows)

## 🔮 Future Enhancements

- [ ] Multi-user chat rooms
- [ ] Document versioning
- [ ] Advanced search filters
- [ ] Export chat history
- [ ] Document annotations
- [ ] Real-time collaboration
- [ ] API endpoints for programmatic access
- [ ] Webhook integrations
- [ ] Advanced analytics dashboard


