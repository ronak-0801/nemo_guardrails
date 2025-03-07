# AI Chatbot with PDF Support

A versatile chatbot implementation with two variants: a basic chatbot and an enhanced version with PDF document interaction capabilities.

## Overview

This project offers two implementations of an AI chatbot:

1. **Basic Chatbot (NeMo Demo)**
   - Simple conversational AI implementation
   - Direct question-answering capabilities

2. **Enhanced Chatbot (Guardrails)**
   - PDF document processing and interaction
   - Streamlit web interface
   - Advanced conversation controls
   - Document-based question answering

## Project Structure

```
.
├── nemo_demo/
│   └── chatbot.py         # Basic chatbot implementation
├── guardrails/
│   ├── app.py            # Streamlit web application
│   ├── chatbot.py        # PDF-enabled chatbot implementation
│   └── config/
│       ├── actions.py    # Custom actions for PDF processing
│       ├── config.yml    # Configuration settings
│       ├── prompts.yml   # System prompts and templates
│       └── rails/
│           └── rails.co  # Conversation control specifications
└── requirements.txt      # Project dependencies
```

## Features

### Basic Chatbot (NeMo Demo)
- Simple text-based conversations
- Quick setup and minimal dependencies
- Straightforward question-answering capabilities

### Enhanced Chatbot (Guardrails)
- PDF document upload and processing
- Interactive Streamlit web interface
- Document-based question answering
- Conversation history tracking
- Configurable system behaviors

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ronak-0801/nemo_guardrails.git 
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Basic Chatbot
```bash
cd nemo_demo
python chatbot.py
```


### Running the PDF-Enabled Chatbot

1. Start the Streamlit application:
```bash
cd guardrails
streamlit run app.py
```

2. Access the web interface in your browser (typically http://localhost:8501)
3. Upload your PDF document
4. Start asking questions about the document



## Configuration

The PDF-enabled chatbot can be customized through configuration files:

- `guardrails/config/config.yml`: General settings and parameters
- `guardrails/config/prompts.yml`: System prompts and response templates
- `guardrails/config/rails/rails.co`: Conversation control rules
- `guardrails/config/actions.py`: PDF processing and custom actions

## Acknowledgments

- [NeMo Framework](https://github.com/NVIDIA/NeMo)
- [Guardrails Framework](https://github.com/ShreyaR/guardrails)
