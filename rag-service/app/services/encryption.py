"""
Encryption service for secure Q&A logging.
Provides field-level encryption for sensitive data before storage.
"""

import os
import base64
import json
import hashlib
from typing import Optional, Dict, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """Handles encryption and decryption of Q&A data for secure logging."""
    
    def __init__(self, key_path: Optional[str] = None, use_env_key: Optional[bool] = None):
        """
        Initialize the encryption service.
        
        Args:
            key_path: Path to file containing encryption key
            use_env_key: Whether to check environment variable for key
        """
        configured_path = key_path or getattr(settings, 'encryption_key_path', None) or os.environ.get('RAG_ENCRYPTION_KEY_PATH')
        # Default to secure filesystem location rather than the repository tree
        self.key_path = configured_path or '/etc/cbthis/rag-encryption.key'
        # Fallback to settings flag when use_env_key is not explicitly provided
        if use_env_key is None:
            self.use_env_key = getattr(settings, 'use_env_encryption_key', True)
        else:
            self.use_env_key = use_env_key
        self._cipher = None
        self._key_version = "v1"
        self._initialize_cipher()
    
    def _initialize_cipher(self):
        """Initialize the Fernet cipher with the encryption key."""
        key = self._load_or_generate_key()
        self._cipher = Fernet(key)
    
    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate a new one."""
        # First, check environment variable if enabled
        if self.use_env_key:
            env_key = os.environ.get('RAG_ENCRYPTION_KEY')
            if env_key:
                try:
                    # Validate the key format
                    key_bytes = base64.urlsafe_b64decode(env_key.encode())
                    if len(key_bytes) == 32:  # Fernet key is 32 bytes
                        return env_key.encode()
                    logger.error('RAG_ENCRYPTION_KEY must decode to 32 bytes')
                except Exception as e:
                    logger.warning(f"Invalid encryption key in environment: {e}")
            else:
                logger.debug('RAG_ENCRYPTION_KEY environment variable not set')

        # Next, check configured key path on disk
        if self.key_path and os.path.exists(self.key_path):
            try:
                with open(self.key_path, 'rb') as f:
                    key = f.read().strip()
                    Fernet(key)  # Validate key
                    return key
            except Exception as e:
                logger.error(f"Failed to load encryption key from file '{self.key_path}': {e}")

        raise RuntimeError(
            "Encryption key is not configured. Set RAG_ENCRYPTION_KEY or provide a secure file via "
            "settings.encryption_key_path / RAG_ENCRYPTION_KEY_PATH before starting the service."
        )
    
    def encrypt_text(self, text: str) -> Tuple[str, str]:
        """
        Encrypt a text string.
        
        Args:
            text: Plain text to encrypt
            
        Returns:
            Tuple of (encrypted_text, key_version)
        """
        if not text:
            return "", self._key_version
        
        try:
            encrypted = self._cipher.encrypt(text.encode('utf-8'))
            # Convert to base64 for storage
            encrypted_b64 = base64.urlsafe_b64encode(encrypted).decode('utf-8')
            return encrypted_b64, self._key_version
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_text(self, encrypted_text: str, key_version: str = "v1") -> str:
        """
        Decrypt an encrypted text string.
        
        Args:
            encrypted_text: Base64 encoded encrypted text
            key_version: Version of key used for encryption
            
        Returns:
            Decrypted plain text
        """
        if not encrypted_text:
            return ""
        
        try:
            # Decode from base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode('utf-8'))
            # Decrypt
            decrypted = self._cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def encrypt_dict(self, data: Dict[str, Any], fields_to_encrypt: list) -> Dict[str, Any]:
        """
        Encrypt specific fields in a dictionary.
        
        Args:
            data: Dictionary containing data
            fields_to_encrypt: List of field names to encrypt
            
        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()
        
        for field in fields_to_encrypt:
            if field in data and data[field]:
                encrypted_value, version = self.encrypt_text(str(data[field]))
                encrypted_data[f"{field}_encrypted"] = encrypted_value
                encrypted_data[f"{field}_encryption_version"] = version
                # Replace original with placeholder
                encrypted_data[field] = f"[ENCRYPTED-{version}]"
        
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], fields_to_decrypt: list) -> Dict[str, Any]:
        """
        Decrypt specific fields in a dictionary.
        
        Args:
            data: Dictionary containing encrypted data
            fields_to_decrypt: List of field names to decrypt
            
        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()
        
        for field in fields_to_decrypt:
            encrypted_field = f"{field}_encrypted"
            version_field = f"{field}_encryption_version"
            
            if encrypted_field in data and data[encrypted_field]:
                try:
                    version = data.get(version_field, "v1")
                    decrypted_value = self.decrypt_text(data[encrypted_field], version)
                    decrypted_data[field] = decrypted_value
                    # Remove encrypted fields from result
                    decrypted_data.pop(encrypted_field, None)
                    decrypted_data.pop(version_field, None)
                except Exception as e:
                    logger.error(f"Failed to decrypt field {field}: {e}")
                    decrypted_data[field] = f"[DECRYPTION_FAILED]"
        
        return decrypted_data
    
    def generate_hash(self, text: str) -> str:
        """Generate a SHA256 hash of the text for indexing."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """
        Rotate the encryption key.
        
        Args:
            new_key: New key to use, or None to generate
            
        Returns:
            The new encryption key
        """
        old_cipher = self._cipher
        old_version = self._key_version
        
        # Generate or use provided key
        if new_key is None:
            new_key = Fernet.generate_key()
        
        # Update cipher
        self._cipher = Fernet(new_key)
        self._key_version = f"v{int(old_version[1:]) + 1}"
        
        # Save new key
        try:
            with open(self.key_path, 'wb') as f:
                f.write(new_key)
            os.chmod(self.key_path, 0o600)
            logger.info(f"Rotated encryption key to version {self._key_version}")
        except Exception as e:
            # Rollback on failure
            self._cipher = old_cipher
            self._key_version = old_version
            logger.error(f"Failed to rotate key: {e}")
            raise
        
        return new_key
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about the current encryption key."""
        return {
            "version": self._key_version,
            "key_path": self.key_path,
            "using_env_key": self.use_env_key and bool(os.environ.get('RAG_ENCRYPTION_KEY')),
            "key_exists": os.path.exists(self.key_path) or bool(os.environ.get('RAG_ENCRYPTION_KEY'))
        }


# Singleton instance
_encryption_service = None


def get_encryption_service() -> EncryptionService:
    """Get the singleton encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service
