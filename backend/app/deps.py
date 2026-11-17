from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_token
from app.database.connection import get_db

bearer = HTTPBearer(auto_error=False)


def to_object_id(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(404, "Not found")


def current_user(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(401, "Not authenticated")
    user_id = decode_token(creds.credentials, "access")
    if not user_id:
        raise HTTPException(401, "Invalid or expired token")
    user = get_db().users.find_one({"_id": to_object_id(user_id)})
    if not user:
        raise HTTPException(401, "User no longer exists")
    return user
