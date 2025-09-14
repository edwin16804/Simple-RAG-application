from fastapi import FastAPI, UploadFile, File, Form
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from openai import OpenAI
from dotenv import load_dotenv
from utils.generate_embeddings import generate_embeddings_pdf, prompt_embedding, generate_repo_embeddings
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

def clean_text(text: str) -> str:
    return ' '.join(text.replace('\n', ' ').split())

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
        readme_content = ""

        readme_response = requests.get(readme_url, headers=headers)
        if readme_response.status_code == 200:
            data = readme_response.json()
            try:
                content = data.get("content", "")
                readme_text = base64.b64decode(content).decode("utf-8")
                html = markdown(readme_text)
                soup = BeautifulSoup(html, "html.parser")
                readme_content = soup.get_text(separator="\n").strip()
                readme_content = clean_text(readme_content)
            except Exception as e:
                logger.error(f"Failed to decode README for {repo_name}: {e}")
        else:
            logger.warning(f"No README for {repo_name}")

        repo_list.append({
            "repo_name": repo_name,
            "repo_link": repo["html_url"],
            "readme": readme_content
        })

    # Save JSON file
    with open("repos.json", "w", encoding="utf-8") as f:
        json.dump(repo_list, f, indent=4)

    logger.info(f"Saved repos.json with {len(repo_list)} repositories")
    return repo_list

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts and cleans text from PDF file path."""
    loader = PyPDFLoader(file_path)
    cleaned_pages = [clean_text(page.page_content) for page in loader.load_and_split()]
    return "\n".join(cleaned_pages)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

if client:
    logger.info("OpenAI client initialized successfully.")
else:
    logger.error("Failed to initialize OpenAI client.")


@app.get("/")
def root():
    return "RAG Project"


@app.get("/fetch-repos")
def fetch_repos():
    """Fetch repos, generate README embeddings, and upload them to Chroma."""
    repo_json = fetch_repos_with_readme("edwin16804")

    embeddings = generate_repo_embeddings("repos.json")

    # Wrap into vector_upload-compatible format
    formatted_embeddings = []
    for idx, repo in enumerate(embeddings):
        formatted_embeddings.append({
            "embedding": repo["embedding"],
            "content": repo["readme"],  # store README text
            "page_number": idx + 1,     # dummy number
            "filename": repo["repo_name"]
        })

    vector_upload(formatted_embeddings, [e["content"] for e in formatted_embeddings])

    return {"status": "success", "message": f"Fetched {len(repo_json)} repos and uploaded embeddings."}


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """ PDF upload + embedding logic """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        loader = PyPDFLoader(tmp.name)
        cleaned_pages = [clean_text(page.page_content) for page in loader.load_and_split()]
        embeddings = generate_embeddings_pdf(cleaned_pages, file.filename)
        vector_upload(embeddings, cleaned_pages)

        return {
            "filename": file.filename,
            "page_content": cleaned_pages[0] if cleaned_pages else "",
            "embedding": embeddings[0]["embedding"] if embeddings else []
        }
    finally:
        os.unlink(tmp.name)


@app.post("/chat/")
async def tailored_resume(job_pdf: UploadFile = File(...), text: str = Form(...)):
    """Takes job description PDF + user query, fetches relevant projects, tailors resume"""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        contents = await job_pdf.read()
        tmp.write(contents)
        tmp.close()

        job_description = extract_text_from_pdf(tmp.name)

        jd_embedding = prompt_embedding(job_description)
        matched_contents = vector_query(jd_embedding, top_k=3)
        combined_context = "\n\n---\n\n".join(matched_contents)

        # Step 3: Build system prompt for tailoring resume
        system_content = (
            "You are an assistant that tailors resumes based on job descriptions and GitHub projects. "
            "You will be given the job description and README contents of GitHub projects. "
            "Select the most relevant projects, rewrite the professional summary to align with the job, "
            "and generate a LaTeX resume code using the provided template. "
            "Keep formatting consistent and ensure the tailored resume highlights the most suitable projects.\n\n"
            f"Job Description:\n{job_description}\n\n"
            f"Context from GitHub Projects:\n{combined_context}"
            f"\n\nLaTeX Template:\n{LATEX_TEMPLATE}"
            "Respond only with the complete LaTeX code for the tailored resume."
            "Also answer questions related to the resume or about the job description"
            "Also after generating the tailored response , give reasons for selecting the projects you selected"
            "Also highlight the appropriate skills I have worked on that are relevant to the job description"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text}
        ]

        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=messages,
        )

        llm_response = completion.choices[0].message.content

        return {
            "query": text,
            "job_description_excerpt": job_description[:500],  # preview first 500 chars
            "llm_answer": llm_response
        }

    finally:
        os.unlink(tmp.name)



# ---------------- Template ----------------
LATEX_TEMPLATE = r"""
\documentclass[10pt]{article}
\usepackage[top=0.4in, bottom=0.4in, left=0.7in, right=0.7in]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{parskip}

% Formatting tweaks
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.2em}
\titleformat{\section}{\large\bfseries}{}{0em}{}[\titlerule]
\setlist[itemize]{noitemsep, topsep=0pt, leftmargin=*}
\pagenumbering{gobble}

\begin{document}

% ----- HEADER -----
\begin{center}
    {\LARGE \textbf{Edwin Chazhoor}} \\
    \vspace{0.2em}
    +91-7069940011 \quad \textbar \quad
    \href{mailto:edwinchazhoor0408@gmail.com}{edwinchazhoor0408@gmail.com} \quad \textbar \quad
    \href{https://github.com/edwin16804}{github.com/edwin16804} \quad \textbar \quad
    \href{https://www.linkedin.com/in/edwin-chazhoor/}{linkedin.com/in/edwin-chazhoor}
\end{center}

% ----- PROFESSIONAL SUMMARY -----
\section*{Professional Summary}
[TO_BE_FILLED]

% ----- EDUCATION -----
\section*{Education}
\textbf{Vellore Institute of Technology, Chennai} \hfill Jul 2022 -- Present \\
B.Tech in Computer Science Engineering (AI \& Machine Learning) \hfill CGPA: 8.94/10

% ----- TECHNICAL SKILLS -----
\section*{Technical Skills}
\textbf{Languages:} Python, SQL, Java , C/C++ \\
\textbf{Frameworks:} React.js, FastAPI, Flask, Hugging Face Transformers, Scikit-learn \\
\textbf{Machine Learning:} Deep Learning, Speech-to-Text, TTS, Transformers, NLP \\
\textbf{DevOps:} Docker, GitHub Actions, CI/CD \\
\textbf{Cloud:} AWS Lambda, API Gateway, DynamoDB , AstraDB , ChromaDB\\
\textbf{Additional Tools:} OpenTelemetry, SigNoz, LangChain , OpenRouter , Ollama

% ----- WORK EXPERIENCE -----
\section*{Work Experience}
\textbf{AI Backend Engineer Intern, Premji Invest} \hfill May 2025 -- Jul 2025
\begin{itemize}
    \item Developed multilingual voice assistant web app using large language models with FastAPI and Next.js.
    \item Deployed AI agent orchestration with Model Context Protocol (MCP).
    \item Integrated OpenTelemetry + SigNoz for monitoring and debugging.
    \item Automated CI/CD with Docker + GitHub Actions.
\end{itemize}

% ----- PROJECTS -----
\section*{Projects}
[TO_BE_FILLED]

\end{document}
"""