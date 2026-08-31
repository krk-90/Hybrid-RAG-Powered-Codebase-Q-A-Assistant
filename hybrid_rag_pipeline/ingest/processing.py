import os
from langchain_community.document_loaders import (
    UnstructuredImageLoader,
    UnstructuredFileLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langsmith import traceable
import argparse

from hybrid_rag_pipeline.Database.chroma_db import store_chunks, get_chroma_client
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "rag-tracing")

if not os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGSMITH_API_KEY"):
    print("[WARN] LANGCHAIN_API_KEY / LANGSMITH_API_KEY not set — @traceable calls will not report to LangSmith.")

@traceable(name="load documents")
def load_docs(path:str):
    try:
        if os.path.isdir(path):
            loader =  [
                ("pdf","**/*.pdf",UnstructuredPDFLoader),
                ("word","**/*.docx",UnstructuredWordDocumentLoader),
                ("image","**/*.png",UnstructuredImageLoader),
                ("image","**/*.jpg",UnstructuredImageLoader),
                ("generic","**/*.txt",UnstructuredFileLoader),
            ]

            documents = []
            for label,glob,loader_cls in loader:
                loader = DirectoryLoader(path,glob=glob,loader_cls=loader_cls)
                docs = loader.load()
                documents.extend(docs)
                print(f"loaded {len(docs)} {label} documents.")

        elif path.endswith(".pdf"):
            documents = UnstructuredPDFLoader(path).load() 
        elif path.endswith(".docx"):
            documents = UnstructuredWordDocumentLoader(path).load()
        elif path.endswith(".png") or path.endswith(".jpg"):
            documents = UnstructuredImageLoader(path).load()
        elif path.endswith(".txt"):
            documents = UnstructuredFileLoader(path).load()
        else:
            raise ValueError(f"unsupported file type : {path}")
        print(f"total loaded {len(documents)} documents from {path}.")
        return documents
    except Exception as e:
        print(f"loading failed :{e}")
        return []

@traceable(name="document chunking")
def chunk_docs(documents,chunk_size = 800,over_lap = 150):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"],
        chunk_size=chunk_size,
        chunk_overlap=over_lap,
    )
    chunks = splitter.split_documents(documents)
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",required=True,help="file or folder to ingest")
    parser.add_argument("--chunk_size",type=int,default=800)
    parser.add_argument("--over_lap", type=int, default=150)
    args = parser.parse_args()
    documents = load_docs(args.path)
    if not documents:
        print("[WARN] no documents loaded, aborting")
        return
    chunks = chunk_docs(documents,args.chunk_size,args.over_lap)
    client = get_chroma_client()
    store_chunks(chunks=chunks,client=client)

if __name__ == "__main__":
    main()    