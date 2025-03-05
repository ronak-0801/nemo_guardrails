import streamlit as st
import os
from chatbot import RAGChatbot

def initialize_chatbot():
    """Initialize the chatbot and store it in session state"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = RAGChatbot()

def initialize_chat_history():
    """Initialize chat history in session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Display all messages in the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input():
    """Handle user input and generate chatbot response"""
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(
        page_title="TechCorp AI Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session states
    initialize_chatbot()
    initialize_chat_history()

    # Sidebar
    with st.sidebar:
        st.title("📚 Document Status")
        
        # Show PDF directory status
        pdf_dir = st.session_state.chatbot.pdf_directory
        if os.path.exists(pdf_dir):
            pdfs = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
            st.success(f"📁 PDF Directory: {pdf_dir}")
            if pdfs:
                st.write("📑 Loaded Documents:")
                for pdf in pdfs:
                    st.write(f"- {pdf}")
            else:
                st.warning("No PDF documents found")
        else:
            st.error(f"PDF directory not found: {pdf_dir}")

        # Add file uploader
        uploaded_file = st.file_uploader("Upload new PDF document", type="pdf")
        if uploaded_file is not None:
            # Create docs directory if it doesn't exist
            os.makedirs(pdf_dir, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(pdf_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Reload PDFs
            st.session_state.chatbot.load_pdfs()
            st.rerun()

        # Add refresh button
        if st.button("🔄 Refresh Documents"):
            st.session_state.chatbot.load_pdfs()
            st.rerun()

    # Main chat interface
    st.title("🤖 TechCorp AI Assistant")
    st.write("Ask me anything about TechCorp!")

    # Display chat history
    display_chat_history()

    # Handle user input
    handle_user_input()

if __name__ == "__main__":
    main() 