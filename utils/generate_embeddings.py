import ollama

def generate_embeddings(pages, filename):
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

def prompt_embedding(text):
    return ollama.embeddings(model='nomic-embed-text', prompt=text)["embedding"]
