# Simple RAG System

A simple Retrieval-Augmented Generation (RAG) system built with FastAPI that allows you to upload PDF documents and query them using natural language. The system uses vector embeddings for semantic search and provides contextual answers based on your documents.

## Features

- **PDF Document Upload**: Upload PDF files and automatically extract text content
- **Vector Embeddings**: Generate embeddings using Ollama's `nomic-embed-text` model
- **Vector Database**: Store and query embeddings using ChromaDB
- **Semantic Search**: Find relevant document sections based on your queries
- **AI-Powered Responses**: Get contextual answers using OpenRouter's GPT models
- **RESTful API**: Clean FastAPI endpoints for easy integration

## Architecture

```
PDF Upload → Text Extraction → Embedding Generation → Vector Storage (ChromaDB)
                                                           ↓
User Query → Embedding Generation → Vector Search → Context Retrieval → AI Response
```

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- ChromaDB account and credentials
- OpenRouter API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Simple-RAG-System
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn langchain-community openai python-dotenv chromadb ollama
   ```

3. **Install and setup Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for installation instructions)
   ollama pull nomic-embed-text
   ```

4. **Environment Setup**
   Create a `.env` file in the project root with the following variables:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   CHROMA_API_KEY=your_chromadb_api_key
   CHROMA_TENANT_ID=your_chromadb_tenant_id
   CHROMA_DATABASE=your_chromadb_database_name
   ```

## Usage

1. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```

2. **Upload a PDF document**
   ```bash
   curl -X POST "http://localhost:8000/uploadfile/" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_document.pdf"
   ```

3. **Query your documents**
   ```bash
   curl -X POST "http://localhost:8000/chat/" \
        -H "Content-Type: application/json" \
        -d '{"text": "What is this document about?"}'
   ```

## API Endpoints

### `GET /`
- **Description**: Health check endpoint
- **Response**: Simple status message

### `POST /uploadfile/`
- **Description**: Upload and process a PDF file
- **Parameters**: 
  - `file`: PDF file (multipart/form-data)
- **Response**: 
  ```json
  {
    "filename": "document.pdf",
    "page_content": "First page content...",
    "embedding": [0.1, 0.2, ...]
  }
  ```

### `POST /chat/`
- **Description**: Query your uploaded documents
- **Parameters**:
  - `text`: Your question (string)
- **Response**:
  ```json
  {
    "query": "Your question",
    "answer": "AI-generated answer based on your documents",
    "similar_documents": [
      {
        "page_content": "Relevant content...",
        "page_number": 1,
        "filename": "document.pdf"
      }
    ]
  }
  ```

## Project Structure

```
Simple-RAG-System/
├── main.py                    # FastAPI application and endpoints
├── utils/
│   ├── __init__.py
│   ├── chromadb_operations.py # ChromaDB vector operations
│   └── generate_embeddings.py # Embedding generation using Ollama
├── README.md
└── .env                       # Environment variables (create this)
```

## Configuration

### ChromaDB Setup
1. Create a ChromaDB Cloud account
2. Create a database in ChromaDB Cloud
3. The collection `my_collection` will be automatically created when you first run the application
4. The collection uses 768-dimensional embeddings (for nomic-embed-text)

### Model Configuration
- **Embedding Model**: `nomic-embed-text` (via Ollama)
- **LLM Model**: `openai/gpt-oss-20b:free` (via OpenRouter)

## Dependencies

- **FastAPI**: Web framework
- **LangChain**: Document processing
- **Ollama**: Local embedding generation
- **ChromaDB**: Vector database
- **OpenRouter**: LLM API access
- **OpenAI**: API client

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Troubleshooting

### Common Issues

1. **Ollama not running**: Make sure Ollama is installed and the `nomic-embed-text` model is pulled
2. **ChromaDB connection failed**: Verify your API key, tenant ID, and database name in the `.env` file
3. **OpenRouter API errors**: Check your API key and account status
