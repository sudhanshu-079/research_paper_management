from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pymongo.errors import DuplicateKeyError

from app.core import security
from app.database.connection import get_db
from app.deps import current_user
from app.schemas.auth import LoginIn, RefreshIn, RegisterIn, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_out(u: dict) -> UserOut:
    return UserOut(id=str(u["_id"]), email=u["email"], full_name=u["full_name"])


def _tokens(user_id: str) -> TokenOut:
    return TokenOut(access_token=security.create_access_token(user_id),
                    refresh_token=security.create_refresh_token(user_id))


@router.post("/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn):
    if err := security.validate_password_strength(body.password):
        raise HTTPException(422, err)
    doc = {
        "email": body.email.lower(),
        "password_hash": security.hash_password(body.password),
        "full_name": body.full_name.strip(),
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = get_db().users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "Email already registered")
    return _tokens(str(result.inserted_id))


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn):
    user = get_db().users.find_one({"email": body.email.lower()})
    # Same error for "no such user" and "wrong password" so emails can't be probed
    if not user or not security.verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect email or password")
    return _tokens(str(user["_id"]))


@router.post("/refresh", response_model=TokenOut)
def refresh(body: RefreshIn):
    user_id = security.decode_token(body.refresh_token, "refresh")
    if not user_id:
        raise HTTPException(401, "Invalid or expired refresh token")
    return _tokens(user_id)


@router.get("/me", response_model=UserOut)
def me(user: dict = Depends(current_user)):
    return _user_out(user)
