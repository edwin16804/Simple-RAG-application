from astrapy import DataAPIClient
from dotenv import load_dotenv
from astrapy.constants import VectorMetric
from astrapy.info import CollectionDefinition, CollectionVectorOptions
import os

load_dotenv()
token = os.getenv("APPLICATION_TOKEN")
endpoint = os.getenv("API_ENDPOINT")

client = DataAPIClient()
db = client.get_database(endpoint, token=token)
if not db:
    raise ValueError("Database connection failed.")

def vector_upload(embeddings , contents):
    # definition = CollectionDefinition(
    #         vector=CollectionVectorOptions(
    #             dimension=768,  
    #             metric=VectorMetric.COSINE
    #         )
    #     )
    # collection = db.create_collection("pdf_vectors", definition=definition)

    collection = db.get_collection("pdf_vectors")      
    for item in embeddings:
        doc = {
            "embedding_vector": item["embedding"],
            "page_content": item["content"],
            "page_number": item["page_number"],
            "filename": item["filename"]
        }
        collection.insert_one(doc)
    return True


def vector_query(embedding, top_k=3):
    collection = db.get_collection("pdf_vectors")
    if not collection:
        raise ValueError("Collection 'pdf_vectors' does not exist.")

    cursor = collection.find(
        {},
        sort={"$vector": embedding},
        projection={"page_content": True, "page_number": True, "filename": True},
        limit=top_k,
        include_similarity=True
    )

    documents = []
    for doc in cursor:
        print(doc)
        documents.append(doc)
    return documents


