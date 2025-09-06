# Simple RAG System

A simple Retrieval-Augmented Generation (RAG) system built with FastAPI that allows you to upload PDF documents and query them using natural language. The system uses vector embeddings for semantic search and provides contextual answers based on your documents.

## Features

- **PDF Document Upload**: Upload PDF files and automatically extract text content
- **Vector Embeddings**: Generate embeddings using Ollama's `nomic-embed-text` model
- **Vector Database**: Store and query embeddings using AstraDB (DataStax)
- **Semantic Search**: Find relevant document sections based on your queries
- **AI-Powered Responses**: Get contextual answers using OpenRouter's GPT models
- **RESTful API**: Clean FastAPI endpoints for easy integration

## Architecture

```
PDF Upload → Text Extraction → Embedding Generation → Vector Storage (AstraDB)
                                                           ↓
User Query → Embedding Generation → Vector Search → Context Retrieval → AI Response
```

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- AstraDB account and credentials
- OpenRouter API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Simple-RAG-System
   ```

2. **Install dependencies**
   ```bash
   pip install fastapi uvicorn langchain-community openai python-dotenv astrapy ollama
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
   APPLICATION_TOKEN=your_astradb_application_token
   API_ENDPOINT=your_astradb_api_endpoint
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
│   ├── astradb_operations.py  # AstraDB vector operations
│   └── generate_embeddings.py # Embedding generation using Ollama
├── README.md
└── .env                       # Environment variables (create this)
```

## Configuration

### AstraDB Setup
1. Create a database in AstraDB
2. Create a collection named `pdf_vectors` with vector configuration:
   - Dimension: 768 (for nomic-embed-text)
   - Metric: COSINE

### Model Configuration
- **Embedding Model**: `nomic-embed-text` (via Ollama)
- **LLM Model**: `openai/gpt-oss-20b:free` (via OpenRouter)

## Dependencies

- **FastAPI**: Web framework
- **LangChain**: Document processing
- **Ollama**: Local embedding generation
- **AstraDB**: Vector database
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
2. **AstraDB connection failed**: Verify your credentials and endpoint in the `.env` file
3. **OpenRouter API errors**: Check your API key and account status

### Getting Help

If you encounter issues:
1. Check the logs for error messages
2. Verify all environment variables are set correctly
3. Ensure all services (Ollama, AstraDB) are accessible
4. Open an issue on GitHub with detailed error information