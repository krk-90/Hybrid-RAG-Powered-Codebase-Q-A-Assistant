import os
from dotenv import load_dotenv
import chromadb
import hashlib
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "document_embed")
REQUIRED_ENV_VARS = ["CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE", "GOOGLE_API_KEY"]

def _check_env():
    missing = [k for k in REQUIRED_ENV_VARS if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")
                                   
def get_chroma_client():
    _check_env()
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )

def store_chunks(chunks, client=None):
    if not chunks:
        print("  [WARN] no chunks to store, skipping embed/store step")
        return None
    
    _check_env()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        api_key=os.environ["GOOGLE_API_KEY"],
    )

    client = client or get_chroma_client()

    ids = [
        hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
        for chunk in chunks
    ]

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name=COLLECTION_NAME,
            ids=ids,
        )
    except Exception as e:
        print("store_chunks", COLLECTION_NAME, e)
        raise

    print(f"Stored {len(chunks)} chunks in collection '{COLLECTION_NAME}'")
    return vector_store   