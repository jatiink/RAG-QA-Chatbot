# RAG QA Chatbot
 
This is a Python application that serves as an assistant for question-answering tasks based on text extracted from PDF files. It utilizes various libraries and tools for text processing, retrieval, and interaction.

## Overview

The application performs the following main functions:
- Extracts text from uploaded PDF files.
- Splits the text into smaller chunks for efficient processing.
- Converts text into embeddings using language models.
- Provides a conversational interface for users to ask questions.
- Retrieves relevant context from the processed text to answer questions concisely.

## Features

- **PDF Text Extraction**: Utilizes PyPDF2 library to extract text content from PDF files.
- **Text Chunking**: Divides extracted text into smaller chunks for better handling and processing.
- **Language Models**: Employs the Llama language model for question answering and embedding generation.
- **Question-Answering Interface**: Provides users with a simple interface to ask questions and receive answers.
- **Context Retrieval**: Retrieves relevant context from the processed text to generate concise answers.
- **Source Document References**: Optionally includes references to the source documents used for answering questions.

## Setup

1. **Install Dependencies**: Ensure all required Python libraries and packages are installed. You can use `pip` to install dependencies listed in `requirements.txt`.
   
'''
pip install -r requirements.txt
'''

2. **Run the Application**: Execute the main Python script to start the application.

'''
chainlit run main.py
'''

3. **Upload PDF Files**: Users can upload one or more PDF files containing text content to begin processing.

4. **Ask Questions**: Once the processing is complete, users can ask questions, and the assistant will provide answers based on the processed text.

## Usage

- **Uploading Files**: Users can upload PDF files via the provided interface.
- **Asking Questions**: Users can type their questions in the chat interface and receive answers in response.
- **Viewing Answers**: Answers are displayed in the chat interface, along with references to the source documents if available.