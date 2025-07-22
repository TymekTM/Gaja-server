#!/usr/bin/env python3
"""
GAJA Assistant - Production Authentication System
Implementuje bezpieczną autentyfikację zgodną ze standardami produkcyjnymi.
"""

import json
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import HTTPException, status
from loguru import logger
from passlib.context import CryptContext

# Konfiguracja bezpieczeństwa
PWD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = secrets.token_urlsafe(32)  # Dynamiczny klucz dla JWT
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30


class SecurityManager:
    """Zarządza bezpieczeństwem autentyfikacji."""

    def __init__(self):
        self.failed_attempts: dict[str, dict[str, Any]] = {}

    def hash_password(self, password: str) -> str:
        """Hashuje hasło używając bcrypt z solą."""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")

        return PWD_CONTEXT.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Weryfikuje hasło w sposób odporny na timing attacks."""
        try:
            return PWD_CONTEXT.verify(plain_password, hashed_password)
        except Exception as e:
            logger.warning(f"Password verification failed: {e}")
            return False

    def is_account_locked(self, email: str) -> bool:
        """Sprawdza czy konto jest zablokowane."""
        if email not in self.failed_attempts:
            return False

        attempts = self.failed_attempts[email]
        if attempts["count"] >= MAX_LOGIN_ATTEMPTS:
            locked_until = attempts["locked_until"]
            if datetime.now(UTC) < locked_until:
                return True
            else:
                # Odblokuj konto
                del self.failed_attempts[email]
                return False
        return False

    def record_failed_attempt(self, email: str) -> None:
        """Rejestruje nieudaną próbę logowania."""
        now = datetime.now(UTC)

        if email not in self.failed_attempts:
            self.failed_attempts[email] = {"count": 0, "locked_until": None}

        self.failed_attempts[email]["count"] += 1

        if self.failed_attempts[email]["count"] >= MAX_LOGIN_ATTEMPTS:
            self.failed_attempts[email]["locked_until"] = now + timedelta(
                minutes=LOCKOUT_DURATION_MINUTES
            )
            logger.warning(
                f"Account {email} locked due to {MAX_LOGIN_ATTEMPTS} failed attempts"
            )

    def clear_failed_attempts(self, email: str) -> None:
        """Czyści nieudane próby po udanym logowaniu."""
        if email in self.failed_attempts:
            del self.failed_attempts[email]

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Tworzy bezpieczny JWT token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire, "iat": datetime.now(UTC), "type": "access"})

        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, data: dict[str, Any]) -> str:
        """Tworzy refresh token."""
        to_encode = data.copy()
        expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({"exp": expire, "iat": datetime.now(UTC), "type": "refresh"})

        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def verify_token(self, token: str, token_type: str = "access") -> dict[str, Any]:
        """Weryfikuje JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Sprawdź typ tokenu
            if payload.get("type") != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                )

            # Sprawdź ważność
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, UTC) < datetime.now(UTC):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
                )

            return payload

        except jwt.ExpiredSignatureError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
            ) from e
        except jwt.JWTError as e:
            logger.warning(f"JWT validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            ) from e

    def generate_secure_password(self, length: int = 16) -> str:
        """Generuje bezpieczne hasło."""
        import string

        if length < 12:
            raise ValueError("Password length must be at least 12 characters")

        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        password = "".join(secrets.choice(characters) for _ in range(length))

        # Zapewnij przynajmniej jeden znak z każdej kategorii
        password_list = list(password)
        password_list[0] = secrets.choice(string.ascii_lowercase)
        password_list[1] = secrets.choice(string.ascii_uppercase)
        password_list[2] = secrets.choice(string.digits)
        password_list[3] = secrets.choice("!@#$%^&*")

        secrets.SystemRandom().shuffle(password_list)
        return "".join(password_list)

    def sanitize_log_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Usuwa wrażliwe dane przed logowaniem."""
        sensitive_keys = {
            "password",
            "token",
            "secret",
            "key",
            "auth",
            "credential",
            "hash",
            "salt",
            "api_key",
            "private_key",
            "access_token",
            "refresh_token",
        }

        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value

        return sanitized

    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a new user with hashed password."""
        try:
            # Validate password strength
            if len(password) < 8:
                logger.error("Password must be at least 8 characters long")
                return False

            # Hash password
            hashed_password = self.hash_password(password)

            # Store user (simplified file-based storage for testing)
            users_file = "databases/users.json"
            os.makedirs(os.path.dirname(users_file), exist_ok=True)

            # Load existing users
            users = {}
            if os.path.exists(users_file):
                try:
                    with open(users_file, encoding="utf-8") as f:
                        users = json.load(f)
                except Exception:
                    users = {}

            # Check if user already exists
            if username in users:
                logger.error(f"User {username} already exists")
                return False

            # Add new user
            users[username] = {
                "username": username,
                "password_hash": hashed_password,
                "role": role,
                "created_at": datetime.now().isoformat(),
                "is_active": True,
            }

            # Save users
            with open(users_file, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=2)

            logger.info(f"Created user: {username}")
            return True

        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            return False

    def authenticate_user(self, username: str, password: str) -> dict[str, Any]:
        """Authenticate user and return result with JWT token."""
        try:
            # Check if account is locked
            if self.is_account_locked(username):
                return {
                    "success": False,
                    "error": "Account is temporarily locked",
                    "locked_until": self.failed_attempts[username].get("locked_until"),
                }

            # Load users
            users_file = "databases/users.json"
            if not os.path.exists(users_file):
                self.record_failed_attempt(username)
                return {"success": False, "error": "Invalid credentials"}

            with open(users_file, encoding="utf-8") as f:
                users = json.load(f)

            # Check if user exists
            user_data = users.get(username)
            if not user_data:
                self.record_failed_attempt(username)
                return {"success": False, "error": "Invalid credentials"}

            # Check if user is active
            if not user_data.get("is_active", True):
                return {"success": False, "error": "Account is disabled"}

            # Verify password
            if not self.verify_password(password, user_data["password_hash"]):
                self.record_failed_attempt(username)
                return {"success": False, "error": "Invalid credentials"}

            # Clear failed attempts on successful login
            if username in self.failed_attempts:
                del self.failed_attempts[username]

            # Generate JWT token
            token_data = {
                "sub": username,
                "role": user_data.get("role", "user"),
                "exp": datetime.now(UTC)
                + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            }

            token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

            return {
                "success": True,
                "token": token,
                "user": {"username": username, "role": user_data.get("role", "user")},
            }

        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
            return {"success": False, "error": "Authentication failed"}

    def validate_token(self, token: str) -> dict[str, Any]:
        """Validate JWT token."""
        try:
            # Decode token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            # Extract user info
            username = payload.get("sub")
            role = payload.get("role", "user")

            if not username:
                return {"valid": False, "error": "Invalid token payload"}

            return {"valid": True, "username": username, "role": role}

        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token has expired"}
        except jwt.JWTError as e:
            return {"valid": False, "error": f"Invalid token: {e}"}
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return {"valid": False, "error": "Token validation failed"}


# Globalna instancja
security_manager = SecurityManager()
