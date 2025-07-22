#!/usr/bin/env python3
"""
GAJA Assistant - Input Validation and Sanitization System
System walidacji i sanityzacji danych wejściowych zapobiegający atakom.
"""

import html
import json
import re
from typing import Any
from urllib.parse import urlparse

from loguru import logger
from pydantic import BaseModel, EmailStr, Field, validator


class SecurityValidator:
    """Główny system walidacji bezpieczeństwa."""

    # Wzorce ataków do wykrycia
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>",
        r"<link[^>]*>",
        r"<meta[^>]*>",
    ]

    SQL_INJECTION_PATTERNS = [
        r"(\bunion\b|\bselect\b|\binsert\b|\bdelete\b|\bupdate\b|\bdrop\b|\bcreate\b|\balter\b)",
        r"(--|#|/\*|\*/)",
        r"(\bor\b|\band\b)\s+(\d+\s*=\s*\d+|\w+\s*=\s*\w+)",
        r"'\s*(or|and)\s*'",
        r'"\s*(or|and)\s*"',
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\.\./|\.\.",
        r"/etc/passwd",
        r"/bin/\w+",
        r"cmd\.exe",
        r"powershell",
        r"eval\s*\(",
        r"exec\s*\(",
    ]

    LDAP_INJECTION_PATTERNS = [
        r"[*()\\]",
        r"\x00",
        r"[<>{}]",
    ]

    def __init__(self):
        """Initialize validator."""
        pass

    def contains_xss(self, text: str) -> bool:
        """Check if text contains XSS patterns."""
        if not isinstance(text, str):
            return False

        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def contains_sql_injection(self, text: str) -> bool:
        """Check if text contains SQL injection patterns."""
        if not isinstance(text, str):
            return False

        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def contains_command_injection(self, text: str) -> bool:
        """Check if text contains command injection patterns."""
        if not isinstance(text, str):
            return False

        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def is_safe_field_name(self, field_name: str) -> bool:
        """Check if field name is safe."""
        if not isinstance(field_name, str):
            return False

        # Allow only alphanumeric characters, underscores, and hyphens
        pattern = r"^[a-zA-Z0-9_-]+$"
        return bool(re.match(pattern, field_name)) and len(field_name) <= 100

    def sanitize_input(self, text: str) -> str:
        """Sanitize input text."""
        if not isinstance(text, str):
            text = str(text)

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

        # HTML escape
        text = html.escape(text, quote=True)

        # Remove dangerous patterns
        for pattern in self.XSS_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text.strip()

    @classmethod
    def sanitize_string(cls, input_str: str, max_length: int = 1000) -> str:
        """Sanityzuje string usuwając potencjalnie niebezpieczne znaki."""
        if not isinstance(input_str, str):
            input_str = str(input_str)

        # Ogranicz długość
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
            logger.warning(f"Input truncated to {max_length} characters")

        # Usuń znaki kontrolne
        input_str = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", input_str)

        # Escape HTML
        input_str = html.escape(input_str, quote=True)

        # Usuń potencjalnie niebezpieczne wzorce
        for pattern in cls.XSS_PATTERNS:
            input_str = re.sub(pattern, "", input_str, flags=re.IGNORECASE)

        return input_str.strip()

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Waliduje format email."""
        if not email or len(email) > 254:
            return False

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if not re.match(email_pattern, email):
            return False

        # Sprawdź czy nie zawiera niebezpiecznych znaków
        dangerous_chars = ["<", ">", '"', "'", "&", "\\", "/", "`", ";"]
        if any(char in email for char in dangerous_chars):
            return False

        return True

    @classmethod
    def validate_password(cls, password: str) -> dict[str, Any]:
        """Waliduje siłę hasła."""
        result: dict[str, Any] = {
            "valid": False,
            "score": 0,
            "requirements": {
                "min_length": False,
                "uppercase": False,
                "lowercase": False,
                "digit": False,
                "special_char": False,
                "no_common_words": False,
            },
            "suggestions": [],
        }

        if not password:
            result["suggestions"].append("Password is required")
            return result

        # Minimalna długość
        if len(password) >= 12:
            result["requirements"]["min_length"] = True
            result["score"] += 2
        elif len(password) >= 8:
            result["score"] += 1
            result["suggestions"].append(
                "Password should be at least 12 characters long"
            )
        else:
            result["suggestions"].append("Password must be at least 8 characters long")

        # Wielkie litery
        if re.search(r"[A-Z]", password):
            result["requirements"]["uppercase"] = True
            result["score"] += 1
        else:
            result["suggestions"].append("Add uppercase letters")

        # Małe litery
        if re.search(r"[a-z]", password):
            result["requirements"]["lowercase"] = True
            result["score"] += 1
        else:
            result["suggestions"].append("Add lowercase letters")

        # Cyfry
        if re.search(r"\d", password):
            result["requirements"]["digit"] = True
            result["score"] += 1
        else:
            result["suggestions"].append("Add numbers")

        # Znaki specjalne
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            result["requirements"]["special_char"] = True
            result["score"] += 1
        else:
            result["suggestions"].append("Add special characters")

        # Sprawdź popularne hasła
        common_passwords = {
            "password",
            "123456",
            "123456789",
            "qwerty",
            "abc123",
            "password123",
            "admin",
            "letmein",
            "welcome",
            "monkey",
        }

        if password.lower() not in common_passwords:
            result["requirements"]["no_common_words"] = True
            result["score"] += 1
        else:
            result["suggestions"].append("Avoid common passwords")

        # Hasło jest bezpieczne jeśli spełnia podstawowe wymagania
        result["valid"] = (
            result["requirements"]["min_length"]
            and result["requirements"]["uppercase"]
            and result["requirements"]["lowercase"]
            and result["requirements"]["digit"]
            and result["requirements"]["no_common_words"]
        )

        return result

    @classmethod
    def detect_injection_attack(cls, input_str: str) -> dict[str, Any]:
        """Wykrywa potencjalne ataki injection."""
        if not isinstance(input_str, str):
            input_str = str(input_str)

        input_lower = input_str.lower()
        detected_attacks = []

        # SQL Injection
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, input_lower, re.IGNORECASE):
                detected_attacks.append("sql_injection")
                break

        # XSS
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, input_str, re.IGNORECASE):
                detected_attacks.append("xss")
                break

        # Command Injection
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, input_str):
                detected_attacks.append("command_injection")
                break

        # LDAP Injection
        for pattern in cls.LDAP_INJECTION_PATTERNS:
            if re.search(pattern, input_str):
                detected_attacks.append("ldap_injection")
                break

        return {
            "is_suspicious": len(detected_attacks) > 0,
            "detected_attacks": detected_attacks,
            "risk_level": "high" if detected_attacks else "low",
        }

    @classmethod
    def validate_url(cls, url: str) -> dict[str, Any]:
        """Waliduje URL pod kątem bezpieczeństwa."""
        result = {"valid": False, "secure": False, "issues": []}

        try:
            parsed = urlparse(url)

            # Sprawdź podstawowy format
            if not parsed.scheme or not parsed.netloc:
                result["issues"].append("Invalid URL format")
                return result

            result["valid"] = True

            # Sprawdź protokół
            if parsed.scheme.lower() not in ["http", "https", "ftp", "ftps"]:
                result["issues"].append("Unsupported protocol")
                return result

            # Sprawdź czy używa HTTPS
            if parsed.scheme.lower() == "https":
                result["secure"] = True
            else:
                result["issues"].append("Non-HTTPS URL")

            # Sprawdź podejrzane domeny
            suspicious_domains = [
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
                "10.",
                "192.168.",
                "172.",
            ]

            netloc_lower = parsed.netloc.lower()
            for suspicious in suspicious_domains:
                if suspicious in netloc_lower:
                    result["issues"].append("Internal/local domain detected")
                    break

            # Sprawdź długość
            if len(url) > 2048:
                result["issues"].append("URL too long")

        except Exception as e:
            result["issues"].append(f"URL parsing error: {str(e)}")

        return result

    @classmethod
    def validate_json_input(
        cls, json_str: str, max_size: int = 1024 * 1024
    ) -> dict[str, Any]:
        """Waliduje JSON input."""
        result: dict[str, Any] = {"valid": False, "parsed_data": None, "issues": []}

        # Sprawdź rozmiar
        if len(json_str) > max_size:
            result["issues"].append(f"JSON too large (max {max_size} bytes)")
            return result

        try:
            # Parsuj JSON
            data = json.loads(json_str)
            result["parsed_data"] = data
            result["valid"] = True

            # Sprawdź głębokość zagnieżdżenia
            max_depth = cls._check_json_depth(data)
            if max_depth > 10:
                result["issues"].append("JSON too deeply nested")

            # Sprawdź liczbę kluczy
            key_count = cls._count_json_keys(data)
            if key_count > 1000:
                result["issues"].append("Too many JSON keys")

        except json.JSONDecodeError as e:
            result["issues"].append(f"Invalid JSON: {str(e)}")
        except Exception as e:
            result["issues"].append(f"JSON validation error: {str(e)}")

        return result

    @classmethod
    def _check_json_depth(cls, obj: Any, current_depth: int = 0) -> int:
        """Sprawdza głębokość zagnieżdżenia JSON."""
        if current_depth > 20:  # Zapobiegaj stack overflow
            return current_depth

        if isinstance(obj, dict):
            return max(
                [cls._check_json_depth(v, current_depth + 1) for v in obj.values()],
                default=current_depth,
            )
        elif isinstance(obj, list):
            return max(
                [cls._check_json_depth(item, current_depth + 1) for item in obj],
                default=current_depth,
            )
        else:
            return current_depth

    @classmethod
    def _count_json_keys(cls, obj: Any) -> int:
        """Liczy wszystkie klucze w JSON."""
        count = 0

        if isinstance(obj, dict):
            count += len(obj)
            for value in obj.values():
                count += cls._count_json_keys(value)
        elif isinstance(obj, list):
            for item in obj:
                count += cls._count_json_keys(item)

        return count

    def validate_request_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize request data."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")

            validated_data = {}

            for key, value in data.items():
                # Validate key
                if not self.is_safe_field_name(key):
                    raise ValueError(f"Invalid field name: {key}")

                # Validate and sanitize value
                if isinstance(value, str):
                    # Check for malicious patterns
                    if self.contains_xss(value):
                        raise ValueError(f"XSS detected in field: {key}")

                    if self.contains_sql_injection(value):
                        raise ValueError(f"SQL injection detected in field: {key}")

                    if self.contains_command_injection(value):
                        raise ValueError(f"Command injection detected in field: {key}")

                    # Sanitize the value
                    validated_data[key] = self.sanitize_input(value)

                elif isinstance(value, int | float | bool):
                    validated_data[key] = value

                elif isinstance(value, dict):
                    # Recursively validate nested dictionaries
                    validated_data[key] = self.validate_request_data(value)

                elif isinstance(value, list):
                    # Validate list items
                    validated_list = []
                    for item in value:
                        if isinstance(item, str):
                            if self.contains_xss(item) or self.contains_sql_injection(
                                item
                            ):
                                raise ValueError(
                                    f"Malicious content in list item: {item}"
                                )
                            validated_list.append(self.sanitize_input(item))
                        elif isinstance(item, dict):
                            validated_list.append(self.validate_request_data(item))
                        else:
                            validated_list.append(item)
                    validated_data[key] = validated_list

                else:
                    # For other types, convert to string and validate
                    str_value = str(value)
                    if self.contains_xss(str_value) or self.contains_sql_injection(
                        str_value
                    ):
                        raise ValueError(f"Malicious content detected: {str_value}")
                    validated_data[key] = self.sanitize_input(str_value)

            return validated_data

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise ValueError(f"Input validation failed: {e}")


# Pydantic modele z walidacją
class SecureLoginRequest(BaseModel):
    """Bezpieczny model logowania."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(
        ..., min_length=8, max_length=128, description="User password"
    )
    remember_me: bool = Field(False, description="Remember login")

    @validator("email")
    def validate_email_security(cls, v):
        if not SecurityValidator.validate_email(v):
            raise ValueError("Invalid email format or contains dangerous characters")
        return v.lower().strip()

    @validator("password")
    def validate_password_security(cls, v):
        # Sprawdź ataki injection
        attack_check = SecurityValidator.detect_injection_attack(v)
        if attack_check["is_suspicious"]:
            raise ValueError("Password contains suspicious patterns")

        return v


class SecureUserInput(BaseModel):
    """Bezpieczny model dla danych wejściowych użytkownika."""

    query: str = Field(..., min_length=1, max_length=10000, description="User query")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context"
    )
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")

    @validator("query")
    def validate_query_security(cls, v):
        # Sanityzuj input
        sanitized = SecurityValidator.sanitize_string(v, max_length=10000)

        # Sprawdź ataki
        attack_check = SecurityValidator.detect_injection_attack(sanitized)
        if attack_check["is_suspicious"]:
            logger.warning(
                f"Suspicious query detected: {attack_check['detected_attacks']}"
            )
            raise ValueError("Query contains potentially dangerous content")

        return sanitized

    @validator("context")
    def validate_context_security(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Context must be a dictionary")

        # Sprawdź rozmiar
        context_str = json.dumps(v)
        if len(context_str) > 50000:  # 50KB limit
            raise ValueError("Context too large")

        # Waliduj każdą wartość w kontekście
        for key, value in v.items():
            if isinstance(value, str):
                attack_check = SecurityValidator.detect_injection_attack(value)
                if attack_check["is_suspicious"]:
                    raise ValueError(f"Suspicious content in context key: {key}")

        return v

    def validate_request_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize request data."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")

            validated_data = {}

            for key, value in data.items():
                # Validate key
                if not self.is_safe_field_name(key):
                    raise ValueError(f"Invalid field name: {key}")

                # Validate and sanitize value
                if isinstance(value, str):
                    # Check for malicious patterns
                    if self.contains_xss(value):
                        raise ValueError(f"XSS detected in field: {key}")

                    if self.contains_sql_injection(value):
                        raise ValueError(f"SQL injection detected in field: {key}")

                    if self.contains_command_injection(value):
                        raise ValueError(f"Command injection detected in field: {key}")

                    # Sanitize the value
                    validated_data[key] = self.sanitize_input(value)

                elif isinstance(value, int | float | bool):
                    validated_data[key] = value

                elif isinstance(value, dict):
                    # Recursively validate nested dictionaries
                    validated_data[key] = self.validate_request_data(value)

                elif isinstance(value, list):
                    # Validate list items
                    validated_list = []
                    for item in value:
                        if isinstance(item, str):
                            if self.contains_xss(item) or self.contains_sql_injection(
                                item
                            ):
                                raise ValueError(
                                    f"Malicious content in list item: {item}"
                                )
                            validated_list.append(self.sanitize_input(item))
                        elif isinstance(item, dict):
                            validated_list.append(self.validate_request_data(item))
                        else:
                            validated_list.append(item)
                    validated_data[key] = validated_list

                else:
                    # For other types, convert to string and validate
                    str_value = str(value)
                    if self.contains_xss(str_value) or self.contains_sql_injection(
                        str_value
                    ):
                        raise ValueError(f"Malicious content detected: {str_value}")
                    validated_data[key] = self.sanitize_input(str_value)

            return validated_data

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise ValueError(f"Input validation failed: {e}")


class SecureAPIKeyInput(BaseModel):
    """Bezpieczny model dla kluczy API."""

    provider: str = Field(
        ..., min_length=1, max_length=50, description="API provider name"
    )
    api_key: str = Field(..., min_length=10, max_length=200, description="API key")

    @validator("provider")
    def validate_provider(cls, v):
        # Tylko dozwolone znaki
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Provider name contains invalid characters")

        # Lista dozwolonych providerów
        allowed_providers = {
            "openai",
            "anthropic",
            "deepseek",
            "azure",
            "google",
            "weather",
            "news",
            "spotify",
            "github",
            "slack",
        }

        if v.lower() not in allowed_providers:
            raise ValueError(f"Unknown provider: {v}")

        return v.lower()

    @validator("api_key")
    def validate_api_key(cls, v):
        # Sprawdź czy nie jest to placeholder
        if v.startswith("YOUR_") or v == "your_key_here":
            raise ValueError("Invalid API key placeholder")

        # Sprawdź podstawowy format
        if len(v) < 10 or len(v) > 200:
            raise ValueError("API key length is invalid")

        # Sprawdź czy zawiera tylko dozwolone znaki
        if not re.match(r"^[a-zA-Z0-9_.-]+$", v):
            raise ValueError("API key contains invalid characters")

        return v


def create_input_validator() -> SecurityValidator:
    """Tworzy instancję walidatora."""
    return SecurityValidator()


if __name__ == "__main__":
    """Test validation system."""
    validator = SecurityValidator()

    # Test sanityzacji
    test_inputs = [
        "Normal text input",
        "<script>alert('xss')</script>",
        "SELECT * FROM users WHERE id = 1; DROP TABLE users;",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "Normal text with 'quotes' and \"double quotes\"",
    ]

    print("=== Input Sanitization Test ===")
    for test_input in test_inputs:
        sanitized = validator.sanitize_string(test_input)
        attack_check = validator.detect_injection_attack(test_input)

        print(f"\nOriginal: {test_input}")
        print(f"Sanitized: {sanitized}")
        print(f"Attacks detected: {attack_check['detected_attacks']}")

    # Test walidacji hasła
    print("\n=== Password Validation Test ===")
    test_passwords = [
        "weak",
        "StrongP@ssw0rd123",
        "password123",
        "VerySecurePassword!2024",
    ]

    for password in test_passwords:
        result = validator.validate_password(password)
        print(f"\nPassword: {password}")
        print(f"Valid: {result['valid']}")
        print(f"Score: {result['score']}")
        print(f"Suggestions: {result['suggestions']}")

    print("\n✅ Validation system test completed!")
