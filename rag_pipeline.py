"""
RAG pipeline for document embedding, retrieval, and question answering.
Updated for LangChain 1.0+ API.
"""
import os
from typing import List, Optional, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline


class RAGPipeline:
    """Manages RAG pipeline: embeddings, vector store, and retrieval."""
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model_type: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        ollama_base_url: Optional[str] = None,
        use_memory: bool = True
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            openai_api_key: OpenAI API key (if None, reads from environment)
            model_type: "openai", "ollama", or "huggingface"
            model_name: Model name to use
            ollama_base_url: Base URL for Ollama (default: http://localhost:11434)
            use_memory: Whether to use conversation memory
        """
        self.model_type = model_type
        self.model_name = model_name
        self.use_memory = use_memory
        
        # Initialize embeddings based on model type
        if model_type == "openai":
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )
        elif model_type == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:  # ollama
            # For Ollama, use HuggingFace embeddings (free, local, no API needed)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        
        # Initialize LLM based on model type
        if model_type == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
            )
        elif model_type == "ollama":
            ollama_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm = OllamaLLM(
                model=model_name,
                base_url=ollama_url,
                timeout=300.0  # 5 minute timeout
            )
        else:  # huggingface
            # For HuggingFace, we'll use a pipeline
            try:
                from transformers import pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    max_length=512,
                    temperature=0.7
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            except ImportError:
                # Fallback to OpenAI if HuggingFace fails
                try:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.7,
                        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
                    )
                except:
                    raise ImportError("Neither HuggingFace transformers nor OpenAI is available. Please install required packages.")
            except Exception as e:
                # Fallback to OpenAI if HuggingFace fails
                try:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.7,
                        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
                    )
                except:
                    raise Exception(f"Failed to initialize HuggingFace model: {str(e)}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Vector store (will be initialized when documents are loaded)
        self.vector_store: Optional[FAISS] = None
        self.documents_loaded = False
        
        # Conversation history (simple list-based memory)
        self.conversation_history: List[Tuple[str, str]] = []
    
    def process_documents(self, text: str) -> List[Document]:
        """
        Process and chunk documents.
        
        Args:
            text: Combined text from PDFs
            
        Returns:
            List of Document objects
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Convert to Document objects
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        return documents
    
    def create_vector_store(self, documents: List[Document]):
        """
        Create vector store from documents.
        
        Args:
            documents: List of Document objects
        """
        if not documents:
            raise ValueError("No documents provided")
        
        # Create FAISS vector store from documents
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        self.documents_loaded = True
    
    def load_documents(self, text: str):
        """
        Load and process documents, then create vector store.
        
        Args:
            text: Combined text from PDFs
        """
        documents = self.process_documents(text)
        self.create_vector_store(documents)
    
    def query(
        self, 
        question: str, 
        k: int = 4,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        timeout: int = 120
    ) -> Tuple[str, List[Document]]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            k: Number of document chunks to retrieve
            chat_history: List of (question, answer) tuples for conversation context
            timeout: Maximum time in seconds to wait for response (default: 120)
            
        Returns:
            Tuple of (answer, source_documents)
        """
        # Validate inputs
        if not question or not question.strip():
            raise ValueError("Question cannot be empty. Please ask a valid question.")
        
        if not self.documents_loaded or self.vector_store is None:
            raise ValueError("No documents loaded. Please upload and process documents first.")
        
        # Create retrieval chain with error handling
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        except Exception as e:
            raise Exception(f"Failed to create retriever: {str(e)}. Please try reprocessing your documents.")
        
        # Get relevant documents with timeout and validation
        relevant_docs = []
        try:
            # Try new API first
            relevant_docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older LangChain versions
            try:
                relevant_docs = retriever.get_relevant_documents(question)
            except Exception as e:
                raise Exception(f"Failed to retrieve documents: {str(e)}")
        except Exception as e:
            raise Exception(f"Error during document retrieval: {str(e)}")
        
        # Validate retrieved documents
        if not relevant_docs or len(relevant_docs) == 0:
            raise ValueError(
                "No relevant information found in the documents for your question. "
                "Try rephrasing your question or ensure your documents contain relevant content."
            )
        
        # Extract context from documents
        context_parts = []
        for doc in relevant_docs:
            if hasattr(doc, 'page_content') and doc.page_content.strip():
                context_parts.append(doc.page_content)
        
        if not context_parts:
            raise ValueError(
                "Retrieved documents are empty. Please ensure your PDFs contain extractable text."
            )
        
        context = "\n\n".join(context_parts)
        
        # Validate context length
        if len(context.strip()) < 10:
            raise ValueError(
                "Retrieved context is too short. Your documents may not contain enough relevant information."
            )
        
        # Build conversation history
        history = chat_history or self.conversation_history
        if history and len(history) > 0:
            context_history = "\n".join([
                f"Q: {q}\nA: {a}" for q, a in history[-3:]  # Last 3 exchanges
            ])
            prompt_template = """Use the following pieces of context and conversation history to answer the question at the end.
If you don't know the answer based on the provided context, just say that you don't know.
Don't try to make up an answer. Use only the information from the context.
Provide a clear, concise, and helpful answer.

Previous conversation:
{chat_history}

Context:
{context}

Question: {question}

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "chat_history"]
            )
            formatted_prompt = prompt.format(
                context=context,
                question=question,
                chat_history=context_history
            )
        else:
            prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the provided context, just say that you don't know.
Don't try to make up an answer. Use only the information from the context.
Provide a clear, concise, and helpful answer.

Context:
{context}

Question: {question}

Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            formatted_prompt = prompt.format(
                context=context,
                question=question
            )
        
        # Generate answer using LLM with comprehensive error handling
        answer = None
        error_details = None
        
        try:
            if self.model_type == "openai":
                # For ChatOpenAI, use invoke with messages
                from langchain_core.messages import HumanMessage
                try:
                    response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
                    answer = response.content if hasattr(response, 'content') else str(response)
                except Exception as openai_error:
                    error_msg = str(openai_error).lower()
                    if "quota" in error_msg or "429" in error_msg or "insufficient_quota" in error_msg:
                        raise Exception("OpenAI API quota exceeded. Please check your billing or switch to Ollama (free, local) in Model Settings.")
                    elif "401" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
                        raise Exception("Invalid OpenAI API key. Please check your .env file and ensure OPENAI_API_KEY is set correctly.")
                    elif "timeout" in error_msg or "timed out" in error_msg:
                        raise Exception(f"Request timed out. The model is taking too long to respond. Try: 1) Using a simpler question, 2) Uploading smaller documents, or 3) Switching to a faster model.")
                    elif "rate limit" in error_msg:
                        raise Exception("Rate limit exceeded. Please wait a moment and try again, or switch to Ollama (free, local).")
                    else:
                        raise Exception(f"OpenAI API error: {str(openai_error)}")
                        
            elif self.model_type == "ollama":
                # For Ollama, use invoke with string input
                try:
                    response = self.llm.invoke(formatted_prompt)
                    answer = str(response) if response else ""
                except Exception as ollama_error:
                    error_msg = str(ollama_error).lower()
                    if "connection" in error_msg or "refused" in error_msg or "errno 61" in error_msg:
                        raise Exception(
                            "Cannot connect to Ollama server. Please ensure: "
                            "1) Ollama is installed, "
                            "2) Run 'ollama serve' in Terminal, "
                            "3) The model is downloaded (e.g., 'ollama pull llama2')"
                        )
                    elif "timeout" in error_msg or "timed out" in error_msg:
                        raise Exception(
                            f"Ollama request timed out after {timeout} seconds. "
                            "This can happen with large documents or complex questions. "
                            "Try: 1) Asking a simpler question, 2) Using smaller documents, or 3) Increasing timeout."
                        )
                    elif "model" in error_msg and "not found" in error_msg:
                        raise Exception(
                            f"Ollama model '{self.model_name}' not found. "
                            f"Please run: ollama pull {self.model_name}"
                        )
                    else:
                        raise Exception(f"Ollama error: {str(ollama_error)}")
                        
            else:
                # For other LLMs (HuggingFace), use invoke directly
                try:
                    response = self.llm.invoke(formatted_prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                except Exception as hf_error:
                    error_msg = str(hf_error).lower()
                    if "out of memory" in error_msg or "oom" in error_msg:
                        raise Exception(
                            "Out of memory error. HuggingFace models require significant RAM. "
                            "Try: 1) Using a smaller model, 2) Switching to OpenAI or Ollama, or 3) Closing other applications."
                        )
                    elif "model" in error_msg and "not found" in error_msg:
                        raise Exception(f"HuggingFace model '{self.model_name}' not found. Please check the model name.")
                    else:
                        raise Exception(f"HuggingFace error: {str(hf_error)}")
                        
        except Exception as e:
            # If we already formatted the error above, re-raise it
            if "API quota" in str(e) or "Cannot connect" in str(e) or "not found" in str(e):
                raise
            
            # Otherwise, try fallback invocation
            try:
                response = self.llm(formatted_prompt)
                answer = response if isinstance(response, str) else str(response)
            except Exception as fallback_error:
                raise Exception(
                    f"Failed to generate response. Primary error: {str(e)}. "
                    f"Fallback also failed: {str(fallback_error)}. "
                    "Please try: 1) Reprocessing your documents, 2) Restarting the app, or 3) Switching models."
                )
        
        # Validate answer
        if not answer or not answer.strip():
            raise ValueError(
                "The model returned an empty response. This may happen if: "
                "1) The question is unclear, "
                "2) The documents don't contain relevant information, or "
                "3) There's a model configuration issue. "
                "Please try rephrasing your question or check your model settings."
            )
        
        # Clean up answer
        answer = answer.strip()
        
        # Update conversation history
        if self.use_memory:
            self.conversation_history.append((question, answer))
            # Keep only last 10 exchanges
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
        
        return answer, relevant_docs
    
    def get_relevant_chunks(self, question: str, k: int = 4) -> List[str]:
        """
        Get relevant document chunks for a question (for source citation).
        
        Args:
            question: User's question
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant text chunks
        """
        if not self.documents_loaded or self.vector_store is None:
            return []
        
        # Get relevant documents (using invoke for LangChain 1.0+)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older LangChain versions
            docs = retriever.get_relevant_documents(question)
        
        return [doc.page_content for doc in docs]
    
    def save_vector_store(self, file_path: str):
        """
        Save vector store to disk.
        
        Args:
            file_path: Path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        self.vector_store.save_local(file_path)
    
    def load_vector_store(self, file_path: str):
        """
        Load vector store from disk.
        
        Args:
            file_path: Path to load the vector store from
        """
        self.vector_store = FAISS.load_local(
            file_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.documents_loaded = True
    
    def clear_documents(self):
        """Clear loaded documents and reset vector store."""
        self.vector_store = None
        self.documents_loaded = False
        self.conversation_history = []
