import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio
# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Apply nest_asyncio to handle async operations
nest_asyncio.apply()

class RAGChatbot:
    def __init__(self):
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="simple_rag_demo",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize the rails configuration
        self.config = RailsConfig.from_path("config")
        
        # Create the rails app with context
        self.app = LLMRails(
            config=self.config,
            verbose=True  # Add verbose for debugging
        )
        
        # Register the context
        self.app.register_action(
            "retrieve_context",
            lambda query: self.retrieve_context(query)
        )

        # Initialize OpenAI client
        self.openai_client = client

    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the vector store
        documents: List of dicts with 'id', 'text', and optional 'metadata'
        """
        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedder.encode(texts).tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[doc.get('metadata', {}) for doc in documents]
        )

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from the vector store"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            # Get collection size
            collection_size = self.collection.count()
            print(f"Debug - Collection size: {collection_size}")
            
            # Adjust k to not exceed collection size
            k = min(k, collection_size)
            
            if collection_size == 0:
                return "No context available in the database."
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            print(f"Debug - Query results: {results}")
            
            # Check if we got any results
            if not results['documents'] or not results['documents'][0]:
                return "No relevant context found."
                
            # Combine all retrieved documents into a single context
            context = "\n".join(results['documents'][0])
            return context
            
        except Exception as e:
            print(f"Error in retrieve_context: {str(e)}")
            return "Error retrieving context."

    def chat(self, message: str) -> str:
        """
        Chat with the bot using RAG
        """
        try:
            response = self.app.generate(messages=[{
                "role": "user",
                "content": message
            }])
            
            return response['content']
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = RAGChatbot()
    
    # Add sample documents
    sample_docs = [
        {
            "id": "1",
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "metadata": {"source": "docs"}
        },
        {
            "id": "2",
            "text": "Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming.",
            "metadata": {"source": "docs"}
        },
        {
            "id": "3",
            "text": "Technova is founded in 2023",
            "metadata": {"source": "docs"}
        },
        {
            "id": "4",
            "text": "Python is widely used in data science, machine learning, web development, and automation tasks.",
            "metadata": {"source": "docs"}
        }
    ]
    chatbot.add_documents(sample_docs)
    
    # Simple chat loop
    print("Bot: Hello! Ask me anything about Python. (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
