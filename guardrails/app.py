import streamlit as st
from chatbot import RAGChatbot
import os
import asyncio
from functools import wraps

# Add debug prints at the start
print("Starting application...")
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add initial debug message to Streamlit
st.write("Debug: Application starting...")

# Helper function to run async functions in Streamlit
def async_to_sync(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper

CHROMA_DB_PATH = "chroma_db"  # Directory to store the ChromaDB files

def initialize_session_state():
    """Initialize session state variables"""
    try:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chatbot" not in st.session_state:
            with st.spinner("Initializing chatbot..."):
                # Create the ChromaDB directory if it doesn't exist
                if not os.path.exists(CHROMA_DB_PATH):
                    os.makedirs(CHROMA_DB_PATH)
                # Initialize chatbot with persistent storage path
                st.session_state.chatbot = RAGChatbot(persist_directory=CHROMA_DB_PATH)
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "documents_processed" not in st.session_state:
            st.session_state.documents_processed = False
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        raise

def display_chat_history():
    """Display chat history with improved styling"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

def handle_file_upload():
    """Handle PDF file upload with improved UI"""
    with st.container():
        uploaded_files = st.file_uploader(
            "📄 Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            key="file_uploader",
            help="Upload one or more PDF documents to chat about"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📎 {len(uploaded_files)} file(s) selected")
            with col2:
                process_button = st.button("Process Files", type="primary", use_container_width=True)
                
            if process_button and not st.session_state.processing:
                st.session_state.processing = True
                
                with st.spinner("📚 Processing documents..."):
                    try:
                        # Create docs directory if it doesn't exist
                        if not os.path.exists("docs"):
                            os.makedirs("docs")
                        
                        # Save uploaded files
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join("docs", uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # Process and add documents
                        documents = st.session_state.chatbot.process_pdf_to_documents()
                        if documents:
                            st.session_state.chatbot.add_documents(documents)
                            st.session_state.documents_processed = True
                            st.success(f"✅ Successfully processed {len(documents)} document chunks!")
                        else:
                            st.warning("⚠️ No documents were processed. Please check the uploaded files.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
                    
                    finally:
                        st.session_state.processing = False

@async_to_sync
async def process_user_input(user_input: str):
    """Process user input and generate response"""
    if user_input:
        try:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Register actions for each conversation
                        st.session_state.chatbot.app.register_action(
                            action=st.session_state.chatbot.retrieve_context,
                            name="retrieve_context",
                        )
                        
                        st.session_state.chatbot.app.register_action(
                            action=st.session_state.chatbot.chat,
                            name="chat",
                        )
                        
                        # Generate response using the chatbot
                        response = await st.session_state.chatbot.app.generate_async(prompt=user_input)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.markdown(response)
                        
                    except Exception as e:
                        error_message = f"Error generating response: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")

def main():
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        .stSpinner > div > div {
            border-top-color: #4CAF50 !important;
        }
        .stAlert {
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title with emoji and styling
    st.markdown("# 🤖 RAG Chatbot Assistant")
    st.markdown("---")
    
    try:
        # Initialize session state
        initialize_session_state()
        
        # Sidebar with improved layout
        with st.sidebar:
            st.markdown("### 📚 Document Management")
            handle_file_upload()
            
            st.markdown("---")
            
            # Control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("New Session", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            ### 📖 How to use:
            1. 📤 Upload PDF documents
            2. 🔄 Click 'Process Files'
            3. 💬 Start chatting in the main area
            4. 🔍 Get AI-powered responses
            
            ### 🛠️ Controls:
            - **Clear Chat**: Removes chat history
            - **New Session**: Starts fresh session
            """)
        
        # Main chat interface
        if not st.session_state.documents_processed:
            st.info("👋 Welcome! Please upload and process some documents to start chatting.")
        else:
            # Chat interface
            display_chat_history()
            
            # Chat input with placeholder
            if user_input := st.chat_input("Ask me anything about the documents..."):
                process_user_input(user_input)
    
    except Exception as e:
        st.error(f"❌ Critical error in main application: {str(e)}")

if __name__ == "__main__":
    main() 