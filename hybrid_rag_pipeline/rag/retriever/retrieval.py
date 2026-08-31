import os
import json
import argparse
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langsmith import traceable
from langchain_core.documents import Document

from ingest.processing import load_docs,chunk_docs

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "rag-tracing")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "document_embed")

if not os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGSMITH_API_KEY"):
    print("[WARN] LANGCHAIN_API_KEY / LANGSMITH_API_KEY not set — @traceable calls will not report to LangSmith.")

_bm25_docs_cache = None

@traceable(name= "get vector")
def get_vector():
    embeddings =GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                            api_key=os.environ["GOOGLE_API_KEY"])
    client = chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

def get_embedded_text(limit:int = None,include_embeddings:bool = False, batch_size: int = 100, verbose: bool = True):
    client = chromadb.CloudClient(
            api_key=os.environ["CHROMA_API_KEY"],
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"],
        )
    collection = client.get_collection(COLLECTION_NAME)

    include = ["documents", "metadatas"]
    if include_embeddings:
        include.append("embeddings")

    all_ids, all_docs, all_metas, all_embeds = [], [], [], []
    offset = 0
    while True:
        batch = collection.get(limit=batch_size, offset=offset, include=include)
        if not batch["ids"]:
            break

        all_ids.extend(batch["ids"])
        all_docs.extend(batch["documents"])
        all_metas.extend(batch["metadatas"])
        if include_embeddings:
            all_embeds.extend(batch["embeddings"])

        offset += batch_size
        if limit and offset >= limit:
            break

    if verbose:
        for i, doc_id in enumerate(all_ids):
            print(f"ID: {doc_id}")
            print(f"Text: {all_docs[i][:200]}...")
            print(f"Metadata: {all_metas[i]}")
            if include_embeddings:
                print(f"Embedding (dim={len(all_embeds[i])}): {all_embeds[i][:5]}...")
            print("-" * 40)

    return {
        "ids": all_ids,
        "documents": all_docs,
        "metadatas": all_metas,
        **({"embeddings": all_embeds} if include_embeddings else {}),
    }

def _get_bm25_docs(force_refresh: bool = False):
    global _bm25_docs_cache
    if _bm25_docs_cache is None or force_refresh:
        stored = get_embedded_text(verbose=False)
        _bm25_docs_cache = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(stored["documents"], stored["metadatas"])
        ]
    return _bm25_docs_cache

def retrieve(k: int = 4, refresh_bm25: bool = False):
    vector_store = get_vector()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = _get_bm25_docs(force_refresh=refresh_bm25)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k

    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever, bm25],
        weights=[0.4, 0.6],
    )
    return ensemble_retriever

def main():
    parser = argparse.ArgumentParser(description="Query or inspect the RAG vector store")
    parser.add_argument("--query", type=str, help="Query text to retrieve relevant chunks for")
    parser.add_argument("--k", type=int, default=4, help="Number of results to retrieve (default: 4)")
    parser.add_argument("--refresh-bm25", action="store_true",
                         help="Force-refresh the BM25 index from Chroma before querying")
    parser.add_argument("--inspect", action="store_true",
                         help="Print stored documents/metadata instead of running a query")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limit number of documents fetched when using --inspect")
    parser.add_argument("--json", action="store_true",
                         help="Output retrieved results as JSON instead of pretty-printed text")

    args = parser.parse_args()
    if args.inspect:
        get_embedded_text(limit=args.limit, verbose=True)
        return
    if not args.query:
        parser.error("--query is required unless --inspect is set")
    retriever = retrieve(k=args.k, refresh_bm25=args.refresh_bm25)
    results = retriever.invoke(args.query)
    if args.json:
        payload = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        for i, doc in enumerate(results, start=1):
            print(f"[{i}] {doc.metadata}")
            print(doc.page_content[:300].strip())
            print("-" * 40)

if __name__ == "__main__":
    main()