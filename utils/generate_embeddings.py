import ollama

def generate_embeddings_pdf(pages, filename):
    results = []
    for idx, page in enumerate(pages):
        embedding = ollama.embeddings(model='nomic-embed-text', prompt=page)["embedding"]
        results.append({
            "embedding": embedding,
            "content": page,
            "page_number": idx + 1,
            "filename": filename
        })
    return results

import ollama
import json
import os

def generate_repo_embeddings(json_file="../repos.json"):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found. Please generate it first.")

    with open(json_file, "r", encoding="utf-8") as f:
        repos = json.load(f)

    results = []
    for repo in repos:
        readme = repo.get("readme", "")
        if not readme.strip():
            continue  # skip repos without README

        embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=readme
        )["embedding"]

        results.append({
            "repo_name": repo["repo_name"],
            "repo_link": repo["repo_link"],
            "readme": readme,
            "embedding": embedding
        })

    return results

    # # Save embeddings to a new JSON file
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4)

    # print(f"Saved {len(results)} repo embeddings to {output_file}")

    
def prompt_embedding(text):
    return ollama.embeddings(model='nomic-embed-text', prompt=text)["embedding"]
