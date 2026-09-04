import os
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from supabase import create_client
from app.backend.auth.security import SupabaseUser, get_current_user
from hybrid_rag_pipeline.Database.models import UserAccount
from hybrid_rag_pipeline.Database.relational_db import get_db

router =APIRouter(prefix="/auth",tags=["auth"])
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: SupabaseUser

class SignupResponse(BaseModel):
    message: str
    user: SupabaseUser
    access_token: str | None = None

async def save_login(user: SupabaseUser, db: AsyncSession) -> None:
    account = await db.get(UserAccount, user.id)
    if account is None:
        account = UserAccount(id=user.id, email=user.email, role=user.role)
        db.add(account)
    else:
        account.email = user.email
        account.role = user.role
        account.last_login_at = datetime.now(timezone.utc)
    await db.commit()

@router.post("/signup", response_model=SignupResponse, status_code=201)
async def signup(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = supabase.auth.sign_up({
            "email": str(body.email),
            "password": body.password,
        })
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to create account")

    if not result.user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to create account")

    user = SupabaseUser(
        id=str(result.user.id),
        email=result.user.email,
        role=(result.user.user_metadata or {}).get("role"),
    )
    await save_login(user, db)

    if result.session:
        return SignupResponse(
            message="Account created and logged in",
            user=user,
            access_token=result.session.access_token,
        )

    return SignupResponse(
        message="Account created. Check your email to confirm your account.",
        user=user,
    )

@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        result = supabase.auth.sign_in_with_password({
            "email": str(body.email),
            "password": body.password,
        })
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not result.session or not result.user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login failed")

    user = SupabaseUser(
        id=str(result.user.id),
        email=result.user.email,
        role=(result.user.user_metadata or {}).get("role"),
    )
    await save_login(user, db)
    return LoginResponse(access_token=result.session.access_token, user=user)

@router.get("/me", response_model=SupabaseUser)
async def me(user: SupabaseUser = Depends(get_current_user)):
    return user