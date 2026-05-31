# PDF Analysis Assistant

A professional Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, store their contents in a vector database, and ask questions about the uploaded documents through a conversational interface.

## Features

* Upload and process multiple PDF documents
* Automatic text chunking and vector embedding generation
* Semantic search using Chroma vector database
* Question answering based on document content
* Source citation with PDF filename and page number
* Interactive Streamlit web interface
* Local embeddings using Sentence Transformers
* Persistent vector database storage

## Technologies Used

* Python
* Streamlit
* LangChain
* Chroma Vector Database
* Sentence Transformers
* Hugging Face Embeddings
* PyPDF
* Ollama / Hugging Face LLMs

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the application:

```bash
streamlit run rag_web.py
```

Open the provided local URL in your browser.

### Steps

1. Upload one or more PDF documents.
2. Click **Update Memory** to process and index the documents.
3. Ask questions in the chat interface.
4. View generated answers along with document sources and page references.

## Project Structure

```text
.
├── rag_web.py
├── chroma_db/
├── pdfs/
├── requirements.txt
└── README.md
```

## Future Improvements

* Hybrid search (semantic + keyword search)
* Conversation memory
* PDF highlighting and source preview
* Multi-language support
* Advanced reranking models
* User authentication

## License

This project is provided for educational and research purposes.

```
```
