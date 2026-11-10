# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException

from account.auth import get_current_user, supabase
from dependencies import logger
from schemas import (
    AuthCredentials,
    ForgotPasswordRequest,
    RefreshRequest,
    ResetPasswordVerifyRequest,
    UpdatePasswordAuthenticatedRequest,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", status_code=201)
def sign_up(credentials: AuthCredentials):
    logger.info("[Auth:Signup] Attempting signup for email=%s", credentials.email)
    try:
        res = supabase.auth.sign_up(
            {"email": credentials.email, "password": credentials.password}
        )
        if not res.user:
            raise HTTPException(400, "Could not register user.")

        logger.info("[Auth:Signup] Registered user_id=%s | email=%s", res.user.id, res.user.email)
        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "access_token": res.session.access_token if res.session else None,
            "refresh_token": res.session.refresh_token if res.session else None,
            "confirmation_required": res.session is None,
        }
    except Exception as e:
        logger.warning("[Auth:Signup] Failed for email=%s: %s", credentials.email, e)
        raise HTTPException(400, f"Sign-up failed: {str(e)}")


@router.post("/login")
def login(credentials: AuthCredentials):
    logger.info("[Auth:Login] Attempting login for email=%s", credentials.email)
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": credentials.email, "password": credentials.password}
        )
        logger.info("[Auth:Login] Success for user_id=%s", res.user.id)
        return {
            "user_id": res.user.id,
            "email": res.user.email,
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "expires_in": res.session.expires_in,
        }
    except Exception as e:
        logger.warning("[Auth:Login] Failed for email=%s: %s", credentials.email, e)
        raise HTTPException(401, f"Invalid email or password: {str(e)}")


@router.post("/refresh")
def refresh_session(body: RefreshRequest):
    try:
        res = supabase.auth.refresh_session(body.refresh_token)
        logger.info("[Auth:Refresh] Token refreshed for user_id=%s", res.user.id)
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user_id": res.user.id,
        }
    except Exception as e:
        logger.warning("[Auth:Refresh] Failed: %s", e)
        raise HTTPException(401, f"Could not refresh session: {str(e)}")


@router.get("/me")
def get_me(user_id: str = Depends(get_current_user)):
    return {"user_id": user_id}


@router.post("/forgot-password")
def forgot_password(body: ForgotPasswordRequest):
    logger.info("[Auth:ForgotPassword] Requested for email=%s", body.email)
    try:
        options = {"redirect_to": body.redirect_to} if body.redirect_to else {}
        supabase.auth.reset_password_for_email(body.email.strip(), options=options)
        return {
            "status": "sent",
            "message": "If an account with that email exists, a password recovery code/link has been sent.",
        }
    except Exception as e:
        logger.warning("[Auth:ForgotPassword] Error for email=%s: %s", body.email, e)
        raise HTTPException(400, f"Could not initiate password reset: {str(e)}")


@router.post("/reset-password")
def reset_password_with_otp(body: ResetPasswordVerifyRequest):
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters.")

    try:
        verify_res = supabase.auth.verify_otp(
            {
                "email": body.email.strip(),
                "token": body.token.strip(),
                "type": "recovery",
            }
        )
        if not verify_res.session:
            raise HTTPException(400, "Invalid or expired recovery code.")

        supabase.auth.set_session(
            verify_res.session.access_token,
            verify_res.session.refresh_token,
        )
        supabase.auth.update_user({"password": body.new_password})
        logger.info("[Auth:ResetPassword] Password updated via OTP for user_id=%s", verify_res.user.id)

        return {
            "status": "password_updated",
            "user_id": verify_res.user.id,
            "email": verify_res.user.email,
            "access_token": verify_res.session.access_token,
            "refresh_token": verify_res.session.refresh_token,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[Auth:ResetPassword] Failed for email=%s: %s", body.email, e)
        raise HTTPException(400, f"Password reset failed: {str(e)}")


@router.post("/update-password")
def update_password_with_tokens(body: UpdatePasswordAuthenticatedRequest):
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters.")

    try:
        supabase.auth.set_session(body.access_token, body.refresh_token)
        res = supabase.auth.update_user({"password": body.new_password})
        if not res.user:
            raise HTTPException(400, "Could not update password.")

        logger.info("[Auth:UpdatePassword] Password updated via token for user_id=%s", res.user.id)
        return {
            "status": "password_updated",
            "user_id": res.user.id,
            "email": res.user.email,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("[Auth:UpdatePassword] Failed: %s", e)
        raise HTTPException(401, f"Invalid or expired recovery session: {str(e)}")