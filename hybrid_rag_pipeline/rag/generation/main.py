import os
from langchain_groq import ChatGroq
import argparse
from langsmith import traceable
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from functools import lru_cache

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__),".env"))
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "rag-tracing")
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "document_embed")

if not os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGSMITH_API_KEY"):
    print("[WARN] LANGCHAIN_API_KEY / LANGSMITH_API_KEY not set — @traceable calls will not report to LangSmith.")

from hybrid_rag_pipeline.rag.retriever.retrieval import retrieve

PROMPT_TEMPLATE = """You are a helpful coding assistant reviewing and answering code questions based on the
provided context. If the answer isn't in the context, say you don't know —
do not make up information.
 
Context:
{context}
 
Question: {input}
 
Answer:
"""

@lru_cache(maxsize=1)
def llm_setup():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    llm_model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature= 0.3,
        api_key= api_key,
        max_tokens=3000,
        timeout=15,
        max_retries=2,
        streaming=True
        )
    return llm_model

def get_rag_chain(llm_model,retriever):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    combine_docs = create_stuff_documents_chain(llm_model,prompt=prompt)
    rag_chain = create_retrieval_chain(retriever,combine_docs)
    return rag_chain

@traceable(name="rag query")
def ask_streaming(rag_chain, query: str):
    """Streams the answer; collects sources from the first chunk that has them."""
    full_answer = ""
    sources = []
    for chunk in rag_chain.stream({"input": query}):
        if "context" in chunk and not sources:
            sources = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page"),
                    "snippet": doc.page_content[:200],
                }
                for doc in chunk["context"]
            ]
        if "answer" in chunk:
            print(chunk["answer"], end="", flush=True)
            full_answer += chunk["answer"]
    print()
    return {"answer": full_answer, "retrieved": sources}

def main():
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("query", type=str, help="Your question")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved chunks")
    args = parser.parse_args()

    llm_model = llm_setup()
    retriever = retrieve()
    rag_chain = get_rag_chain(llm_model, retriever)

    print("Answer:")
    result = ask_streaming(rag_chain, args.query)   
    if args.show_sources:
        print("\nSources:")
        for i, doc in enumerate(result["retrieved"], 1):
            print(f"  [{i}] {doc['source']} (page {doc['page']})")
            print(f"      {doc['snippet']}...")

if __name__ == "__main__":
    main()