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
from uuid import uuid4
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

    uuids = [str(uuid4()) for _ in range(len(documents))]
    logging.info(f"Adding {len(documents)} documents to the vector store...")
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embedding=embeddings,
    )
    vector_store.add_documents(documents=documents, ids=uuids)
    logging.info("Documents added successfully.")


if __name__ == "__main__":
    main()
