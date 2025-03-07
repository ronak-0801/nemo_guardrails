import streamlit as st
from chatbot import RAGChatbot
import os
import asyncio
from functools import wraps
import time
import re

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                if not os.path.exists(CHROMA_DB_PATH):
                    os.makedirs(CHROMA_DB_PATH)
                st.session_state.chatbot = RAGChatbot(persist_directory=CHROMA_DB_PATH)
        if "processing" not in st.session_state:
            st.session_state.processing = False
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        raise

def display_chat_history():
    """Display chat history with improved styling"""
    for idx, message in enumerate(st.session_state.messages):
        # Add message container with custom styling
        with st.chat_message(
            message["role"],
            avatar="👤" if message["role"] == "user" else "🤖"
        ):
            # Add typing effect only for the latest assistant message
            if message["role"] == "assistant" and idx == len(st.session_state.messages) - 1:
                with st.empty():
                    displayed_message = ""
                    full_message = message["content"]
                    words = full_message.split()
                    
                    # Display word by word for smoother animation
                    for i, word in enumerate(words):
                        displayed_message += word + " "
                        st.markdown(
                            f"""
                            <div class='chat-content {message["role"]}-content'>
                                {displayed_message}{'▌' if i < len(words)-1 else ''}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        time.sleep(0.05)  # Slightly faster typing
            else:
                # Regular messages with custom styling
                st.markdown(
                    f"""
                    <div class='chat-content {message["role"]}-content'>
                        {message["content"]}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

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
                            st.success(f"✅ Successfully processed {len(documents)} document chunks!")
                        else:
                            st.warning("⚠️ No documents were processed. Please check the uploaded files.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
                    
                    finally:
                        st.session_state.processing = False

def format_response(response: str) -> str:
    """Format the assistant's response for better readability"""
    # Add markdown formatting for better structure
    response = response.replace("•", "\n•")  # Better bullet points
    response = response.replace(":", ":\n")  # Line break after colons
    
    # Highlight important information
    response = re.sub(r'\*\*(.*?)\*\*', r'**\1**', response)  # Bold
    response = re.sub(r'`(.*?)`', r'`\1`', response)  # Code
    
    return response

@async_to_sync
async def process_user_input(user_input: str):
    """Process user input and generate response"""
    if user_input:
        try:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
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
                        
                        # Generate and format response
                        response = await st.session_state.chatbot.app.generate_async(prompt=user_input)
                        formatted_response = format_response(response)
                        

                        info = st.session_state.chatbot.app.explain()
                        info.print_llm_calls_summary()
                        # Add assistant response to chat history
                        st.session_state.messages.append(
                            {"role": "assistant", "content": formatted_response}
                        )
                        st.markdown(formatted_response)
                        
                    except Exception as e:
                        error_message = f"❌ Error: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_message}
                        )
        except Exception as e:
            st.error(f"❌ Error processing input: {str(e)}")

def main():
    # Update custom CSS for better chat styling
    CUSTOM_CSS = """
    <style>
        /* Main container styling */
        .main-container {
            padding: 2rem;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            animation: fadeIn 0.5s;
        }
        
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        
        .assistant-message {
            background-color: #f3f3f3;
            border-left: 5px solid #4CAF50;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: 500;
            border-radius: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        
        /* Spinner styling */
        .stSpinner > div > div {
            border-top-color: #4CAF50 !important;
        }
        
        /* Alert styling */
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Enhanced chat styling */
        .chat-content {
            padding: 1rem;
            border-radius: 12px;
            margin: 0.5rem 0;
            font-size: 1rem;
            line-height: 1.5;
            animation: slideIn 0.3s ease-out;
        }
        
        .user-content {
            background: linear-gradient(135deg, #6B9FFF 0%, #4C8DFF 100%);
            color: white;
            box-shadow: 0 2px 5px rgba(107, 159, 255, 0.2);
        }
        
        .assistant-content {
            background: linear-gradient(135deg, #F5F7FA 0%, #E4E7EB 100%);
            color: #2C3E50;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
        
        /* Chat container improvements */
        .stChatMessage {
            padding: 0.5rem 1rem;
            transition: transform 0.2s ease;
        }
        
        .stChatMessage:hover {
            transform: translateX(2px);
        }
        
        /* Chat input styling */
        .stChatInputContainer {
            padding: 1rem;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Animations */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Typing indicator animation */
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        
        /* Message separator */
        .stChatMessage:not(:last-child) {
            margin-bottom: 1rem;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        /* Timestamp styling */
        .message-timestamp {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
            text-align: right;
        }
    </style>
    """

    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Main title with emoji and styling
    st.markdown(
        """
        <div style='text-align: center; padding: 2rem;'>
            <h1>🤖 RAG Chatbot Assistant</h1>
            <p style='font-size: 1.2rem; color: #666;'>
                Your intelligent document analysis companion
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    try:
        # Initialize session state
        initialize_session_state()
        
        # Sidebar with improved layout
        with st.sidebar:
            st.markdown("### 📚 Document Management")
            handle_file_upload()
            
            st.markdown("---")
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("Clear DB", use_container_width=True):
                    try:
                        # Get all document IDs
                        doc_count = st.session_state.chatbot.collection.count()
                        if doc_count > 0:
                            results = st.session_state.chatbot.collection.get()
                            if results and 'ids' in results:
                                st.session_state.chatbot.collection.delete(
                                    ids=results['ids']
                                )
                            st.success("✅ Document database cleared!")
                            st.rerun()
                        else:
                            st.info("Database is already empty")
                    except Exception as e:
                        st.error(f"❌ Error clearing database: {str(e)}")
            with col3:
                if st.button("New Session", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
            
            # Display document count
            doc_count = st.session_state.chatbot.collection.count()
            if doc_count > 0:
                st.success(f"📚 {doc_count} document chunks in database")
            
            st.markdown("---")
            st.markdown("""
            ### 📖 How to use:
            1. 📤 Upload PDF documents
            2. 🔄 Click 'Process Files'
            3. 💬 Start chatting in the main area
            4. 🔍 Get AI-powered responses
            
            ### 🛠️ Controls:
            - **Clear Chat**: Removes chat history
            - **Clear DB**: Removes all documents
            - **New Session**: Restarts the application
            """)
        
        # Main chat interface
        doc_count = st.session_state.chatbot.collection.count()
        if doc_count == 0:
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