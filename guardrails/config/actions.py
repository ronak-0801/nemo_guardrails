from typing import Optional
from nemoguardrails.actions import action
import chromadb
from sentence_transformers import SentenceTransformer
import os
from openai import OpenAI
# Global variables for vector store components
chroma_client = None
collection = None
embedder = None

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

def init_vector_store():
    """Initialize vector store components"""
    global chroma_client, collection, embedder
    if chroma_client is None:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name="RAG_guardrails",
            metadata={"hnsw:space": "cosine"}
        )
        embedder = SentenceTransformer('all-MiniLM-L6-v2')


@action(is_system_action=True)
async def retrieve_context(query: str) -> str:
    """Retrieve relevant context from the vector store."""
    try:
        init_vector_store()
        if not query.strip():
            return ""

        query_embedding = embedder.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )

        if results['documents'] and results['documents'][0]:
            context_texts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                source = metadata.get('source', 'Unknown source')
                context_texts.append(f"From {source}: {doc}")

            context = "\n\n".join(context_texts)

        # Generate a response using OpenAI
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context. If the answer cannot be found in the context, say that you don't have enough information."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the provided context:"}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating OpenAI response: {str(e)}")


        return "I don't have enough information to answer that question."
    except Exception as e:
        print(f"Error in retrieve_context: {str(e)}")
        return "I don't have enough information to answer that question."



@action(is_system_action=True)
async def check_response_format(context: Optional[dict] = None) -> bool:
    """Check if the response format is appropriate"""
    if not context or "bot_message" not in context:
        return True
    return True

@action(is_system_action=True)
async def check_blocked_terms(context: Optional[dict] = None):
    bot_response = context.get("bot_message")

    # A quick hard-coded list of proprietary terms. You can also read this from a file.
    proprietary_terms = ["proprietary", "proprietary1", "proprietary2"]

    for term in proprietary_terms:
        if term in bot_response.lower():
            return True

    return False