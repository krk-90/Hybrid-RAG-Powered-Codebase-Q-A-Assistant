import os
from functools import lru_cache
 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

@lru_cache(maxsize=1)
def get_cross_encoder(model_name:str = RERANKER_MODEL):
    return HuggingFaceCrossEncoder(model_name = model_name)

def get_reranker(top_n: int = 4, model_name: str = RERANKER_MODEL):
    return CrossEncoderReranker(model=get_cross_encoder(model_name),top_n=top_n)

def wrap_with_reranking(base_retriever, top_n: int = 4, model_name: str = RERANKER_MODEL):
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(top_n=top_n, model_name=model_name),
        base_retriever=base_retriever,
    )

def rerank_documents(query:str,documents:list,top_n:int = 4,model_name : str = RERANKER_MODEL):
    compressor = get_reranker(top_n=top_n,model_name=model_name)
    return compressor.compress_documents(documents,query)