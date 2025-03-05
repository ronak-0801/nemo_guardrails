import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging

# Configure logging for console output only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to handle async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()



# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class RAGChatbot:
    def __init__(self, pdf_directory="docs"):
        logger.info("Initializing RAGChatbot")
        # Initialize ChromaDB
        try:
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(
                name="RAG_guardrails",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
        
        self.pdf_directory = pdf_directory
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.config = RailsConfig.from_path("config")
        self.app = LLMRails(config=self.config)
        
        # Share the collection with actions.py
        global collection, embedder
        collection = self.collection
        embedder = self.embedder
        
        self.app.register_action(
            "retrieve_context",
            lambda query: self.retrieve_context(query)
        )
        self.openai_client = client

    def process_pdf_to_documents(self):
        """Process PDFs into document chunks"""
        if not os.path.exists(self.pdf_directory):
            os.makedirs(self.pdf_directory)
            logger.info(f"Created directory: {self.pdf_directory}")
            return []

        documents = []
        doc_id = 0
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
        logger.info(f"Processing {len(pdf_files)} PDF files")

        for file in pdf_files:
            try:
                pdf_path = os.path.join(self.pdf_directory, file)
                loader = PyPDFLoader(pdf_path)
                pdf_documents = loader.load()
                
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=20,
                    separator="\n"
                )
                chunks = text_splitter.split_documents(pdf_documents)
                logger.info(f"Split {file} into {len(chunks)} chunks")
                
                for chunk in chunks:
                    documents.append({
                        "id": f"doc_{doc_id}",
                        "text": chunk.page_content,
                        "metadata": {"source": file}
                    })
                    doc_id += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file}: {str(e)}")

        return documents

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to ChromaDB"""
        if not documents:
            logger.warning("No documents provided to add_documents")
            return
            
        try:
            embeddings = self.embedder.encode([doc['text'] for doc in documents]).tolist()
            self.collection.add(
                ids=[doc['id'] for doc in documents],
                embeddings=embeddings,
                documents=[doc['text'] for doc in documents],
                metadatas=[doc.get('metadata', {}) for doc in documents]
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise

    # def retrieve_context(self, query: str, k: int = 3) -> str:
    #     """Retrieve relevant context from ChromaDB"""
    #     try:
    #         query_embedding = self.embedder.encode(query).tolist()
    #         collection_size = self.collection.count()
            
    #         if collection_size == 0:
    #             return ""
            
    #         k = min(k, collection_size)
    #         results = self.collection.query(
    #             query_embeddings=[query_embedding],
    #             n_results=k,
    #             include=["documents", "metadatas"]
    #         )
            
    #         if not results['documents'] or not results['documents'][0]:
    #             return ""
            
    #         context_parts = []
    #         for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
    #             source = metadata.get('source', 'Unknown source')
    #             context_parts.append(f"From {source}:\n{doc}")
            
    #         return "\n\n---\n\n".join(context_parts)
            
    #     except Exception as e:
    #         print(f"Error in retrieve_context: {str(e)}")
    #         return ""

    # def chat(self, message: str) -> str:
    #     """Chat with the bot"""

    #     try:
    #         # context = self.retrieve_context(message)

    #         # prompt = f"User query: {message}\n\nContext:\n{context}\n\nAnswer based only on the context above."

    #         response = self.app.generate(messages=[{
    #             "role": "user",
    #             "content": message
    #         }])

    #         content = response.get('content', '')
    #         if not content:
    #             return "I apologize, but I encountered an error."

    #         return content
    #     except Exception as e:
    #         print(f"Error in chat: {str(e)}")
    #         return "I apologize, but I encountered an error. Please try again."

if __name__ == "__main__":
    try:
        chatbot = RAGChatbot()
        
        # Process and add PDF documents
        documents = chatbot.process_pdf_to_documents()
        if documents:
            chatbot.add_documents(documents)
        else:
            logger.warning("No documents to process. Please add PDFs to the 'docs' directory.")
        
        # Chat loop
        print("\nBot: Hello! I'm ready to help you with questions. (type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
                
            try:
                response = chatbot.app.generate(messages=[{
                    "role": "user",
                    "content": user_input
                }])
                print(f"Bot: {response.get('content', '')}")
                
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                print("Bot: I apologize, but I encountered an error. Please try again.")
                
    except Exception as e:
        logger.critical(f"Critical error in main application: {str(e)}")


        



