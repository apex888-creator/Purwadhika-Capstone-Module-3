from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from qdrant_client.models import (
    Distance,
    VectorParams,
)
from qdrant_client import QdrantClient
import pandas as pd
from dotenv import load_dotenv
import os
from uuid import uuid5, NAMESPACE_DNS
import logging
from langchain_qdrant import QdrantVectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO)

logging.info("Initializing Qdrant client...")
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def load_data(csv_path: str, limit=1) -> list[Document]:
    logging.info(f"Loading data from {csv_path} with limit {limit}...")
    df = pd.read_csv(csv_path)
    df = df.head(limit)

    documents = []

    for idx, row in df.iterrows():
        content = " ".join([
            str(row.get('Series_Title', '')),
            str(row.get('Genre', '')),
            str(row.get('Overview', ''))
        ])
        if not content.strip():
            continue

        doc = Document(
            page_content=content,
            metadata={
                'title': str(row.get('Series_Title', 'Unknown')),
                'year': str(row.get('Released_Year', 'Unknown')),
                'certificate': str(row.get('Certificate', 'Unknown')),
                'runtime': str(row.get('Runtime', 'Unknown')),
                'genre': str(row.get('Genre', 'Unknown')),
                'imdb_rating': float(row.get('IMDB_Rating', 0.0)),
                'director': str(row.get('Director', 'Unknown')),
                'row_index': idx,
                'source': 'imdb_top_1000.csv'
            }
        )
        documents.append(doc)
    logging.info(f"Loaded {len(documents)} documents.")
    return documents


def generate_deterministic_id(title: str, index: int) -> str:
    """Membuat UUID yang unik tapi konsisten berdasarkan judul film."""
    unique_string = f"{title}_{index}"
    return str(uuid5(NAMESPACE_DNS, unique_string))


def main():
    documents = load_data('dataset/imdb_top_1000.csv', limit=100)

    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]

    if not os.getenv("QDRANT_COLLECTION_NAME") in collection_names:
        logging.info(
            f"Creating collection {os.getenv('QDRANT_COLLECTION_NAME')}...")
        qdrant_client.create_collection(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    uuids = [
        generate_deterministic_id(doc.metadata.get(
            'title', 'unknown'), doc.metadata.get('row_index', i))
        for i, doc in enumerate(documents)
    ]

    logging.info(
        f"Adding {len(documents)} documents to the vector store with batching...")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = uuids[i:i + batch_size]

        logging.info(
            f"Uploading batch {i // batch_size + 1} (items {i} to {i + len(batch_docs)})...")
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)

    logging.info("All documents added successfully.")


if __name__ == "__main__":
    main()
