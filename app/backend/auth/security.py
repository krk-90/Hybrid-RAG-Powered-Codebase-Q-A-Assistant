import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from supabase import create_client

load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=True)
bearer_scheme = HTTPBearer()
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])

class SupabaseUser(BaseModel):
    id: str
    email: Optional[str] = None
    role: Optional[str] = None

def decode_supabase_token(token: str) -> dict:
    try:
        response = supabase.auth.get_user(token)
        if not response.user:
            raise ValueError("Supabase returned no user")
        return {
            "sub": str(response.user.id),
            "email": response.user.email,
            "role": (response.user.user_metadata or {}).get("role"),
        }
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> SupabaseUser:
    token = credentials.credentials
    payload = decode_supabase_token(token)

    return SupabaseUser(
        id=payload["sub"],
        email=payload.get("email"),
        role=payload.get("role"),
    )