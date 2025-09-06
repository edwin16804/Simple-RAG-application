from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.generate_embeddings import generate_embeddings, prompt_embedding
from utils.astradb_operations import vector_upload, vector_query

app = FastAPI()
load_dotenv()

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
    matched_docs = vector_query(embedding, top_k=3)

    combined_context = "\n\n---\n\n".join(doc.get("page_content") for doc in matched_docs if doc.get("page_content"))

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
        "similar_documents": [
            {
                "page_content": doc.get("page_content"),
                "page_number": doc.get("page_number"),
                "filename": doc.get("filename")
            } for doc in matched_docs
        ]
    }
