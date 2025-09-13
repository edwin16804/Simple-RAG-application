# import chromadb



# client = chromadb.CloudClient(
#   api_key='ck-tieDHeNUuhYG4tEoapHyNYdZ3h5RT1yGUqtya2eBLuh',
#   tenant='87e1dc83-c9b7-4de7-9246-7f4291fe8b60',
#   database='Simple-RAG-app'
# )


# collection = client.get_or_create_collection(
#     name="my_collection",
#     embedding_function=None
# )

# if not collection:
#     raise ValueError("Collection 'my_collection' does not exist.")
# else:
#     print("Collection 'my_collection' found or created successfully.")

import chromadb
import os
import uuid
from dotenv import load_dotenv

load_dotenv()


# Setup Chroma Cloud client (fill with your actual credentials)
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT_ID"),
    database=os.getenv("CHROMA_DATABASE")
)

# Ensure the collection exists
collection = client.get_or_create_collection(
    name="my_collection", # analogous to your AstraDB collection name
    embedding_function=None  # Use direct embedding input, not built-in function
)
if not collection:
    raise ValueError("Collection 'pdf_vectors' could not be found or created.")

def vector_upload(embeddings, contents):
    """
    Uploads a batch of embeddings and their metadata/content to Chroma Cloud.
    embeddings: list of dicts, each with keys: embedding, content, page_number, filename
    contents: not used here (for API consistency)
    """
    ids = [str(uuid.uuid4()) for _ in embeddings]
    collection.add(
        embeddings=[item["embedding"] for item in embeddings],
        documents=[item["content"] for item in embeddings],
        metadatas=[
            {"page_number": item["page_number"], "filename": item["filename"]}
            for item in embeddings
        ],
        ids=ids
    )
    return True

def vector_query(embedding, top_k=3):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    # Return a list of strings (page contents)
    return results["documents"][0]
