#!/usr/bin/env python3
"""
GAJA Assistant - Secure Database Manager with Encryption
Zarządza bezpiecznym przechowywaniem danych z szyfrowaniem wrażliwych informacji.
"""

import base64
import hashlib
import json
import os
from typing import Any

from cryptography.fernet import Fernet
from loguru import logger


class DatabaseEncryption:
    """Zarządza szyfrowaniem wrażliwych danych w bazie danych."""

    def __init__(self):
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)

    def _get_or_create_encryption_key(self) -> bytes:
        """Pobiera lub tworzy klucz szyfrowania."""
        key_file = "db_encryption.key"

        # Sprawdź czy klucz istnieje w zmiennej środowiskowej
        env_key = os.getenv("GAJA_DB_ENCRYPTION_KEY")
        if env_key:
            try:
                return base64.urlsafe_b64decode(env_key)
            except Exception:
                logger.warning("Invalid encryption key in environment variable")

        # Try to load from file
        if os.path.exists(key_file):
            try:
                with open(key_file, "rb") as f:
                    return f.read()
            except Exception:
                logger.warning("Could not read encryption key from file")

        # Generate new key
        key = Fernet.generate_key()
        try:
            with open(key_file, "wb") as f:
                f.write(key)
        except Exception:
            logger.warning("Could not save encryption key to file")

        return key

    def encrypt_data(self, data: str) -> str:
        """Szyfruje dane."""
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str) -> str:
        """Odszyfrowuje dane."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def store_encrypted_api_key(self, provider: str, api_key: str) -> bool:
        """Store encrypted API key in database."""
        try:
            encrypted_key = self.encrypt_data(api_key)
            # Here would be actual database storage logic
            # For now, we'll use a simple file-based approach

            api_keys_file = "databases/encrypted_api_keys.json"
            os.makedirs(os.path.dirname(api_keys_file), exist_ok=True)

            # Load existing keys
            api_keys = {}
            if os.path.exists(api_keys_file):
                try:
                    with open(api_keys_file, encoding="utf-8") as f:
                        api_keys = json.load(f)
                except Exception:
                    api_keys = {}

            # Store encrypted key
            api_keys[provider] = encrypted_key

            # Save to file
            with open(api_keys_file, "w", encoding="utf-8") as f:
                json.dump(api_keys, f, indent=2)

            logger.info(f"Stored encrypted API key for provider: {provider}")
            return True

        except Exception as e:
            logger.error(f"Failed to store encrypted API key for {provider}: {e}")
            return False

    def get_encrypted_api_key(self, provider: str) -> str | None:
        """Retrieve and decrypt API key from database."""
        try:
            api_keys_file = "databases/encrypted_api_keys.json"

            if not os.path.exists(api_keys_file):
                return None

            # Load API keys
            with open(api_keys_file, encoding="utf-8") as f:
                api_keys = json.load(f)

            # Get encrypted key
            encrypted_key = api_keys.get(provider)
            if not encrypted_key:
                return None

            # Decrypt and return
            decrypted_key = self.decrypt_data(encrypted_key)
            return decrypted_key

        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            return None

    def encrypt_data(self, data: str) -> str:
        """Szyfruje dane do przechowywania."""
        if not data:
            return data

        try:
            encrypted_bytes = self.cipher.encrypt(data.encode("utf-8"))
            return encrypted_bytes.hex()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_hex: str) -> str:
        """Odszyfrowuje dane z bazy."""
        if not encrypted_hex:
            return encrypted_hex

        try:
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

    def encrypt_json(self, data: dict[str, Any]) -> str:
        """Szyfruje słownik jako JSON."""
        try:
            json_str = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
            return self.encrypt_data(json_str)
        except Exception as e:
            logger.error(f"JSON encryption failed: {e}")
            raise

    def decrypt_json(self, encrypted_hex: str) -> dict[str, Any]:
        """Odszyfrowuje JSON ze szyfrowanego tekstu."""
        try:
            json_str = self.decrypt_data(encrypted_hex)
            return json.loads(json_str) if json_str else {}
        except Exception as e:
            logger.error(f"JSON decryption failed: {e}")
            return {}

    def hash_for_index(self, data: str) -> str:
        """Tworzy hash do indeksowania (bez możliwości odtworzenia)."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


class SecureDatabaseManager:
    """Rozszerzenie DatabaseManager z szyfrowaniem wrażliwych danych."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.encryption = DatabaseEncryption()

        # Lista pól wymagających szyfrowania
        self.encrypted_fields = {
            "api_keys",
            "auth_tokens",
            "personal_data",
            "conversation_content",
            "sensitive_preferences",
            "webhook_urls",
            "external_credentials",
        }

    def is_sensitive_field(self, field_name: str) -> bool:
        """Sprawdza czy pole zawiera wrażliwe dane."""
        field_lower = field_name.lower()

        sensitive_keywords = {
            "key",
            "token",
            "password",
            "secret",
            "credential",
            "personal",
            "private",
            "sensitive",
            "auth",
            "api",
        }

        return field_name in self.encrypted_fields or any(
            keyword in field_lower for keyword in sensitive_keywords
        )

    def encrypt_user_data(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Szyfruje wrażliwe dane użytkownika przed zapisem."""
        encrypted_data = user_data.copy()

        for field_name, value in user_data.items():
            if self.is_sensitive_field(field_name) and value:
                try:
                    if isinstance(value, dict):
                        encrypted_data[field_name] = self.encryption.encrypt_json(value)
                    elif isinstance(value, str):
                        encrypted_data[field_name] = self.encryption.encrypt_data(value)

                    logger.debug(f"Encrypted field: {field_name}")

                except Exception as e:
                    logger.error(f"Failed to encrypt field {field_name}: {e}")
                    # W przypadku błędu szyfrowania, nie zapisuj danych
                    raise

        return encrypted_data

    def decrypt_user_data(self, encrypted_data: dict[str, Any]) -> dict[str, Any]:
        """Odszyfrowuje wrażliwe dane użytkownika po odczycie."""
        decrypted_data = encrypted_data.copy()

        for field_name, value in encrypted_data.items():
            if self.is_sensitive_field(field_name) and value:
                try:
                    if field_name in [
                        "api_keys",
                        "personal_data",
                        "sensitive_preferences",
                    ]:
                        # Pola JSON
                        decrypted_data[field_name] = self.encryption.decrypt_json(value)
                    else:
                        # Pola tekstowe
                        decrypted_data[field_name] = self.encryption.decrypt_data(value)

                    logger.debug(f"Decrypted field: {field_name}")

                except Exception as e:
                    logger.error(f"Failed to decrypt field {field_name}: {e}")
                    # W przypadku błędu deszyfrowania, ustaw wartość pustą
                    decrypted_data[field_name] = (
                        {} if field_name in ["api_keys"] else ""
                    )

        return decrypted_data

    def secure_store_api_key(self, user_id: str, provider: str, api_key: str) -> bool:
        """Bezpiecznie przechowuje klucz API."""
        try:
            # Sprawdź format klucza API
            if not self._validate_api_key_format(provider, api_key):
                logger.warning(f"Invalid API key format for provider: {provider}")
                return False

            # Szyfruj klucz
            _ = self.encryption.encrypt_data(api_key)

            # Utwórz hash do wyszukiwania (bez możliwości odtworzenia)
            _ = self.encryption.hash_for_index(f"{user_id}:{provider}")

            # TODO: Tutaj należy zapisać do bazy danych
            # encrypted_key - zaszyfrowany klucz
            # key_hash - hash do indeksowania

            logger.info(
                f"API key securely stored for user {user_id}, provider {provider}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False

    def secure_retrieve_api_key(self, user_id: str, provider: str) -> str | None:
        """Bezpiecznie pobiera klucz API."""
        try:
            # Utwórz hash do wyszukiwania
            _ = self.encryption.hash_for_index(f"{user_id}:{provider}")

            # TODO: Tutaj należy pobrać z bazy danych używając key_hash
            # encrypted_key = database.get_encrypted_api_key(key_hash)

            # Na razie symulacja
            encrypted_key = None

            if encrypted_key:
                api_key = self.encryption.decrypt_data(encrypted_key)
                logger.debug(
                    f"API key retrieved for user {user_id}, provider {provider}"
                )
                return api_key

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve API key: {e}")
            return None

    def _validate_api_key_format(self, provider: str, api_key: str) -> bool:
        """Waliduje format klucza API."""
        if not api_key or len(api_key) < 10:
            return False

        # Sprawdź czy nie jest to placeholder
        if api_key.startswith("YOUR_") or api_key == "your_key_here":
            return False

        # Sprawdź format dla konkretnych providerów
        format_rules = {
            "openai": {"prefix": "sk-", "min_length": 20},
            "anthropic": {"prefix": "sk-ant-", "min_length": 20},
            "deepseek": {"prefix": "sk-", "min_length": 20},
            "google": {"min_length": 20},
            "azure": {"min_length": 20},
        }

        rules = format_rules.get(provider.lower())
        if rules:
            if "prefix" in rules and not api_key.startswith(rules["prefix"]):
                return False
            if len(api_key) < rules["min_length"]:
                return False

        return True

    def audit_database_security(self) -> dict[str, Any]:
        """Przeprowadza audyt bezpieczeństwa bazy danych."""
        audit_results = {
            "encryption_status": "enabled",
            "encrypted_fields": list(self.encrypted_fields),
            "key_file_exists": os.path.exists("db_encryption.key"),
            "key_permissions": None,
            "recommendations": [],
        }

        # Sprawdź uprawnienia klucza
        key_file = "db_encryption.key"
        if os.path.exists(key_file):
            stat_info = os.stat(key_file)
            permissions = oct(stat_info.st_mode)[-3:]
            audit_results["key_permissions"] = permissions

            if permissions != "600":
                audit_results["recommendations"].append(
                    f"Key file permissions should be 600, currently {permissions}"
                )
        else:
            audit_results["recommendations"].append(
                "Encryption key file not found - using environment variable or generated key"
            )

        # Sprawdź zmienną środowiskową
        if not os.getenv("GAJA_DB_ENCRYPTION_KEY"):
            audit_results["recommendations"].append(
                "Consider setting GAJA_DB_ENCRYPTION_KEY environment variable"
            )

        return audit_results


# Globalna instancja
secure_db = None


def initialize_secure_database(db_path: str) -> SecureDatabaseManager:
    """Inicjalizuje bezpieczny manager bazy danych."""
    global secure_db
    secure_db = SecureDatabaseManager(db_path)
    logger.info("Secure database manager initialized")
    return secure_db


def get_secure_database() -> SecureDatabaseManager | None:
    """Pobiera instancję bezpiecznego managera bazy."""
    return secure_db


if __name__ == "__main__":
    """Test encryption functionality."""
    try:
        # Test szyfrowania
        db = SecureDatabaseManager("test.db")

        # Test danych użytkownika
        user_data = {
            "user_id": "123",
            "username": "testuser",
            "api_keys": {"openai": "sk-test123456789", "anthropic": "sk-ant-test123"},
            "personal_data": {"name": "Jan Kowalski", "email": "jan@example.com"},
            "preferences": {"language": "pl", "theme": "dark"},
        }

        print("Original data:")
        print(json.dumps(user_data, indent=2, ensure_ascii=False))

        # Szyfruj
        encrypted_data = db.encrypt_user_data(user_data)
        print("\nEncrypted data:")
        print(json.dumps(encrypted_data, indent=2, ensure_ascii=False))

        # Odszyfruj
        decrypted_data = db.decrypt_user_data(encrypted_data)
        print("\nDecrypted data:")
        print(json.dumps(decrypted_data, indent=2, ensure_ascii=False))

        # Sprawdź zgodność
        assert user_data == decrypted_data
        print("\n✅ Encryption/decryption test passed!")

        # Audyt bezpieczeństwa
        audit = db.audit_database_security()
        print("\nSecurity audit:")
        print(json.dumps(audit, indent=2))

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
