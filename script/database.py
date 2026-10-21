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
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

def load_data(excel_path: str, limit=1) -> list[Document]:
    logging.info(f"Loading data from {excel_path} with limit {limit}...")
    df = pd.read_excel(excel_path)
    df = df.head(limit)

    documents = []
    
    for idx, row in df.iterrows():
        content = str(row.get('Resume_str', ''))
        if not content.strip():
            continue

        doc = Document(
            page_content=content,
            metadata={
                'category': str(row.get('Category', 'Unknown')),
                'row_index': idx,
                'source': 'dataset.xlsx'
            }
        )
        documents.append(doc)
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def main():
    documents = load_data('dataset/dataset.xlsx', limit=100)

    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]

    if not os.getenv("QDRANT_COLLECTION_NAME") in collection_names:
        logging.info(f"Creating collection {os.getenv('QDRANT_COLLECTION_NAME')}...")
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