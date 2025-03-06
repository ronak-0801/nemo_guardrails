import os
from typing import List, Dict
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import asyncio
import torch

torch.classes.__path__ = []
# Apply nest_asyncio to handle async operations
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

class RAGChatbot:
    def __init__(self, pdf_directory="docs", persist_directory="chroma_db"):
        # Initialize ChromaDB with persistence
        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.chroma_client.get_or_create_collection(
                name="RAG_guardrails",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise
        
        self.pdf_directory = pdf_directory
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.config = RailsConfig.from_path("config")
        self.app = LLMRails(config=self.config, verbose=True)
        
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
            return []

        documents = []
        doc_id = 0
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]

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
                
                for chunk in chunks:
                    documents.append({
                        "id": f"doc_{doc_id}",
                        "text": chunk.page_content,
                        "metadata": {"source": file}
                    })
                    doc_id += 1
                    
            except Exception as e:
                continue

        return documents

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to ChromaDB"""
        if not documents:
            return
            
        try:
            embeddings = self.embedder.encode([doc['text'] for doc in documents]).tolist()
            self.collection.add(
                ids=[doc['id'] for doc in documents],
                embeddings=embeddings,
                documents=[doc['text'] for doc in documents],
                metadatas=[doc.get('metadata', {}) for doc in documents]
            )
            
        except Exception as e:
            raise

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context from ChromaDB"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            collection_size = self.collection.count()
            
            if collection_size == 0:
                return ""
            
            k = min(k, collection_size)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return ""
            
            context_parts = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                source = metadata.get('source', 'Unknown source')
                context_parts.append(f"From {source}:\n{doc}")
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            return ""

    def chat(self, query: str, context: str) -> str:
        """Chat with the bot"""
        try:
            prompt = f"User query: {query}\n\nContext:\n{context}\n\nAnswer based only on the context above."

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context. If the answer cannot be found in the context, say that you don't have enough information."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.choices[0].message.content
            if not content:
                return "I apologize, but I encountered an error."

            return content
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try again. chat error: {str(e)}"


async def main():
    try:
        chatbot = RAGChatbot()
        
        # Process and add PDF documents
        documents = chatbot.process_pdf_to_documents()
        if documents:
            chatbot.add_documents(documents)
        else:
            print("No documents to process. Please add PDFs to the 'docs' directory.")
        
        # Chat loop
        print("\nBot: Hello! I'm ready to help you with questions. (type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
                
            try:
                chatbot.app.register_action(
                    action=chatbot.retrieve_context,
                    name="retrieve_context",
                )
                               
                chatbot.app.register_action(
                    action=chatbot.chat,
                    name="chat",
                )

                
                response = await chatbot.app.generate_async(prompt=user_input)
                info = chatbot.app.explain()
                info.print_llm_calls_summary()
                
                print(f"Bot: {response}")
                # print(info)

            except Exception as e:
                print(f"Error generating response: {str(e)}")
                print(f"Bot: I apologize, but I encountered an error. Please try again. main error: {str(e)}")
                
    except Exception as e:
        print(f"Critical error in main application: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
