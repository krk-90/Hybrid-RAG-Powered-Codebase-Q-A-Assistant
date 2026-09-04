import os
import shutil
import tempfile
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional,List
from fastapi import FastAPI,HTTPException,Request,UploadFile,File, logger, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from slowapi import Limiter ,_rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from contextlib import asynccontextmanager
from langsmith import traceable
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "rag-tracing")

if not os.environ.get("LANGCHAIN_API_KEY") and not os.environ.get("LANGSMITH_API_KEY"):
    print("[WARN] LANGCHAIN_API_KEY / LANGSMITH_API_KEY not set — @traceable calls will not report to LangSmith.")


from hybrid_rag_pipeline.rag.generation.main import llm_setup,get_rag_chain,ask_streaming
from hybrid_rag_pipeline.rag.retriever.retrieval import retrieve
from hybrid_rag_pipeline.ingest.processing import load_docs,chunk_docs
from hybrid_rag_pipeline.Database.chroma_db import store_chunks,get_chroma_client
from hybrid_rag_pipeline.Database.relational_db import init_db, get_db
from hybrid_rag_pipeline.Database.models import QueryLog
from app.backend.auth.oauth import router as auth_router
from app.backend.auth.security import get_current_user, SupabaseUser
from sqlalchemy.ext.asyncio import AsyncSession

limiter = Limiter(key_func=get_remote_address)

state:dict = {
    "llm_model":None,
    "retreiver":None,
    "rag_chain":None,
    "chroma_client":None,
}

@asynccontextmanager
@traceable(name="lifespan")
async def lifespan(app:FastAPI):
    print("startup loading LLM,retriever,and rag chain....")
    await init_db()
    state["llm_model"] = llm_setup()
    state["chroma_client"] = get_chroma_client()
    state["retreiver"] = retrieve()
    state["rag_chain"] = get_rag_chain(state["llm_model"],state["retreiver"])
    print("startup ready.")

    yield
    print("shutdown [cleaning up]...")
    state.clear()

app = FastAPI(title="Hybrid Rag pipeline API.",escription="CODINEG CHATBOT POWERED BY HYBRID RAG PIPELINE API",version="1.0.0",lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded,_rate_limit_exceeded_handler)

_cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS", "")
CORS_ALLOWED_ORIGINS = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    "null", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".txt"}

class Queryrequest(BaseModel):
    query:str = Field(...,min_length=1,description="The question to ask")
    k:int = Field(4,ge=1,le=20,description="number of final result of reranking")
    show_sources:bool = Field(False,description="include retrieved source of chunk in response")

class sourcechunk(BaseModel):
    source:str
    page:Optional[int] = None
    snippet:str

class Queryresponse(BaseModel):
    answer: str
    sources: List[sourcechunk] = []

class IngestResponse(BaseModel):
    filename: str
    chunks_stored: int
    status: str    

def _refresh_rag_chain():
    state["retreiver"] = retrieve(refresh_bm25 =True)
    state["rag_chain"] = get_rag_chain(state["llm_model"],state["retreiver"])

@app.get("/") 
def endpoint():
    return {"message":"your chatbot is ready for chat."}

@app.get("/health")
@traceable(name="health")
def health():
    return {
        "llm_ready": state.get("llm_model") is not None,
        "retriever_ready": state.get("retriever") is not None,
        "rag_chain_ready": state.get("rag_chain") is not None,
    }

@app.post("/ingest",response_model=IngestResponse)
@limiter.limit("5/minute")
@traceable(name="ingest")
async def ingest(request:Request,file:UploadFile = File(...)):
    ext =Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                )
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_path =os.path.join(temp_dir,file.filename)
        with open(tmp_path,"wb") as f:
            shutil.copyfileobj(file.file,f)

        documents = load_docs(tmp_path)    
        if not documents:
            raise HTTPException(status_code=422, detail="No content could be extracted from the file.")
        chunks = chunk_docs(documents)
        vector_store = store_chunks(chunks=chunks,client=state["chroma_client"])
        if vector_store is None:
            raise HTTPException(status_code=500, detail="Failed to store chunks in the vector store.")

    await asyncio.to_thread(_refresh_rag_chain)
    
    return IngestResponse(filename=file.filename, chunks_stored=len(chunks), status="stored")

@app.post("/query", response_model=Queryresponse)
@limiter.limit("10/minute")
@traceable(name="query")
async def query(
    request: Request,
    body: Queryrequest,
    user: SupabaseUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if state.get("rag_chain") is None:
        raise HTTPException(status_code=503, detail="RAG chain is not ready yet.")
    try:
        result = await asyncio.to_thread(
            ask_streaming,
            state["rag_chain"],
            body.query
        )
        sources = [sourcechunk(**s) for s in result.get("retrieved", [])] if body.show_sources else []

        db.add(QueryLog(user_id=user.id, query=body.query, kind="query"))
        await db.commit()

        return Queryresponse(answer=result["answer"], sources=sources)
    except Exception as e:
        logger.exception("Query failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
@limiter.limit("10/minute")
@traceable(name="streaming_query")
async def query_stream(request: Request, body: Queryrequest, user: SupabaseUser = Depends(get_current_user)):
    if state.get("rag_chain") is None:
        raise HTTPException(status_code=503, detail="RAG chain is not ready yet.")

    try:
        def event_generator():
            try:
                for chunk in state["rag_chain"].stream({"input": body.query}):
                    if "answer" in chunk:
                        yield f"data: {chunk['answer']}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            except Exception as e:
                yield f"event: error\ndata: {str(e)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.exception("Streaming query failed.")
        raise HTTPException(status_code=500, detail=str(e))