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
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[str, List[Document]]:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            k: Number of document chunks to retrieve
            chat_history: List of (question, answer) tuples for conversation context
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if not self.documents_loaded or self.vector_store is None:
            raise ValueError("No documents loaded. Please upload and process documents first.")
        
        # Create retrieval chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        
        # Get relevant documents (using invoke for LangChain 1.0+)
        try:
            relevant_docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older LangChain versions
            relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Build conversation history
        history = chat_history or self.conversation_history
        if history and len(history) > 0:
            context_history = "\n".join([
                f"Q: {q}\nA: {a}" for q, a in history[-3:]  # Last 3 exchanges
            ])
            prompt_template = """Use the following pieces of context and conversation history to answer the question at the end.
If you don't know the answer based on the provided context, just say that you don't know.
Don't try to make up an answer. Use only the information from the context.

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
        
        # Generate answer using LLM
        try:
            if self.model_type == "openai":
                # For ChatOpenAI, use invoke with messages
                from langchain_core.messages import HumanMessage
                response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
                answer = response.content if hasattr(response, 'content') else str(response)
            elif self.model_type == "ollama":
                # For Ollama, use invoke with string input
                # OllamaLLM expects a string, not a list
                response = self.llm.invoke(formatted_prompt)
                # OllamaLLM returns a string directly
                answer = str(response) if response else ""
            else:
                # For other LLMs (HuggingFace), use invoke directly
                response = self.llm.invoke(formatted_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            # Fallback: try direct invocation for compatibility
            try:
                response = self.llm(formatted_prompt)
                answer = response if isinstance(response, str) else str(response)
            except Exception as fallback_error:
                raise Exception(f"Error generating response: {str(e)}. Fallback also failed: {str(fallback_error)}")
        
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
