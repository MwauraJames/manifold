# app/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client
from configs.config import Settings

settings = Settings()

supabase: Client = create_client(settings.supabase_url, settings.supabase_key)

# Adds the "Authorize" Bearer token prompt in Swagger UI (/docs)
security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Validates the Supabase JWT Bearer token from the Authorization header
    and returns the authenticated user's UUID (`user_id`).
    """
    token = credentials.credentials
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token.",
            )
        return user_response.user.id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
        )