from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.generate_embeddings import generate_embeddings, prompt_embedding
from utils.chromadb_operations import vector_upload, vector_query  # CHROMA import
import logging

app = FastAPI()
load_dotenv()

logger = logging.getLogger("uvicorn")

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

def clean_text(text: str) -> str:
    return ' '.join(text.replace('\n', ' ').split())

@app.get("/")
def root():
    return "Simple RAG application"

@app.post("/uploadfile/")
async def upload_file(file: UploadFile):
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        loader = PyPDFLoader(tmp.name)
        cleaned_pages = [clean_text(page.page_content) for page in loader.load_and_split()]
        embeddings = generate_embeddings(cleaned_pages, file.filename)
        vector_upload(embeddings, cleaned_pages)

        return {
            "filename": file.filename,
            "page_content": cleaned_pages[0],
            "embedding": embeddings[0]["embedding"]
        }
    finally:
        os.unlink(tmp.name)

@app.post("/chat/")
async def chat(text: str):
    embedding = prompt_embedding(text)
    logger.info(f"Generated embedding for query: {embedding}")
    logger.info(f"Length of embedding: {len(embedding)}")
    logger.info("Querying vector database for similar documents...")

    matched_contents = vector_query(embedding, top_k=3)  # Now returns a list of strings

    combined_context = "\n\n---\n\n".join(matched_contents)

    system_content = (
        "You are a helpful assistant that helps users find information "
        "about the documents provided. Use the context to answer the question. "
        "If you don't know, say so without making up an answer.\n\n"
        f"Context:\n{combined_context}"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": text}
    ]

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=messages,
    )

    answer = completion.choices[0].message.content

    return {
        "query": text,
        "answer": answer,
        # The content-only version for Chroma:
        "similar_documents":len(matched_contents)
        
    }
