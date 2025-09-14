from fastapi import FastAPI, UploadFile
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.generate_embeddings import generate_embeddings, prompt_embedding
from utils.chromadb_operations import vector_upload, vector_query  # CHROMA import
import logging
import requests
import base64
import json
from markdown import markdown
from bs4 import BeautifulSoup

app = FastAPI()
load_dotenv()

logger = logging.getLogger("uvicorn")

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {github_token}",
    "X-GitHub-Api-Version": "2022-11-28",
}


def fetch_repos_with_readme(username: str):
    """Fetch repos for a user and their README contents."""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        logger.error(f"Failed to fetch repos: {response.status_code} - {response.text}")
        return []

    repos = response.json()
    repo_list = []

    for repo in repos:
        repo_name = repo["name"]
        readme_url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
        readme_text = None

        readme_response = requests.get(readme_url, headers=headers)
        if readme_response.status_code == 200:
            data = readme_response.json()
            try:
                content = data.get("content", "")
                readme_text = base64.b64decode(content).decode("utf-8")
                html = markdown(readme_text)
                soup = BeautifulSoup(html, "html.parser")
                readme_content = soup.get_text(separator="\n").strip()
            except Exception as e:
                logger.error(f"Failed to decode README for {repo_name}: {e}")
        else:
            logger.warning(f"No README for {repo_name}")

        repo_list.append({
            "repo_name": repo_name,
            "repo_link": repo["html_url"],
            "readme": readme_content or ""
        })

    # Save JSON file
    with open("repos.json", "w", encoding="utf-8") as f:
        json.dump(repo_list, f, indent=4)

    logger.info(f"Saved repos.json with {len(repo_list)} repositories")
    return repo_list


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

if client:
    logger.info("OpenAI client initialized successfully.")
else:
    logger.error("Failed to initialize OpenAI client.")


def clean_text(text: str) -> str:
    return ' '.join(text.replace('\n', ' ').split())


@app.on_event("startup")
def startup_event():
    """Fetch repos at startup and save JSON."""
    fetch_repos_with_readme("edwin16804")


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

    matched_contents = vector_query(embedding, top_k=3)

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
        "similar_documents": len(matched_contents)
    }


# @app.get("/file/{filename}")
# async def read_file_name(filename: str):
#     return {"filename": filename}