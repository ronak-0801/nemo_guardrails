from typing import Optional
from nemoguardrails.actions import action
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize global variables for vector store access
chroma_client = None
collection = None
embedder = None

def init_vector_store():
    """Initialize vector store components"""
    global chroma_client, collection, embedder
    if chroma_client is None:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name="simple_rag_demo",
            metadata={"hnsw:space": "cosine"}
        )
        embedder = SentenceTransformer('all-MiniLM-L6-v2')

@action(is_system_action=True)
async def retrieve_context(query: str = None) -> str:
    """Retrieve relevant context from the vector store."""
    try:
        # Initialize vector store if not already initialized
        init_vector_store()
        
        # Get query embedding
        query_embedding = embedder.encode(query).tolist()
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,  # Get only the most relevant result
            where={"source": "docs"}  # Filter by source
        )
        
        # Debug print
        print(f"Debug - Query: {query}")
        print(f"Debug - Results: {results}")
        
        # Check if we got any results and they are relevant
        if (results['documents'] and 
            results['documents'][0] and 
            results['distances'] and 
            results['distances'][0] and 
            results['distances'][0][0] < 0.5):  # Stricter threshold
            
            # Get the most relevant document
            context = results['documents'][0][0]
            return context
        
        return "I don't have enough information to answer that question."
        
    except Exception as e:
        print(f"Error in retrieve_context: {str(e)}")
        return "I don't have enough information to answer that question."

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    bot_response = context.get("bot_message")

    # A quick hard-coded list of proprietary terms. You can also read this from a file.
    proprietary_terms = ["proprietary", "proprietary1", "proprietary2"]

    for term in proprietary_terms:
        if term in bot_response.lower():
            return True

    return False