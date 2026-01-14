#!/usr/bin/env python3
"""
Encryption key management utility for RAG query logging.
Provides commands for key generation, rotation, and verification.
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime
import json
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.encryption import EncryptionService, get_encryption_service
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EncryptionManager:
    """Manages encryption keys and operations."""
    
    def __init__(self):
        self.encryption_service = get_encryption_service()
    
    def generate_key(self, output_path: Optional[str] = None) -> str:
        """Generate a new encryption key."""
        from cryptography.fernet import Fernet
        
        key = Fernet.generate_key()
        key_str = key.decode('utf-8')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(key_str)
            os.chmod(output_path, 0o600)
            logger.info(f"Key saved to: {output_path}")
        
        return key_str
    
    def show_key_info(self):
        """Display current key information."""
        info = self.encryption_service.get_key_info()
        
        print("\n=== Encryption Key Information ===")
        print(f"Version: {info['version']}")
        print(f"Key Path: {info['key_path']}")
        print(f"Using Environment Key: {info['using_env_key']}")
        print(f"Key Exists: {info['key_exists']}")
        
        if info['key_exists']:
            print("\n✓ Encryption is properly configured")
        else:
            print("\n⚠️  No encryption key found!")
            print("Generate a key using: python manage_encryption.py generate")
    
    async def rotate_key(self, backup_old: bool = True):
        """Rotate the encryption key."""
        old_info = self.encryption_service.get_key_info()
        
        if backup_old and os.path.exists(old_info['key_path']):
            # Backup old key
            backup_path = f"{old_info['key_path']}.{old_info['version']}.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with open(old_info['key_path'], 'rb') as old_f:
                with open(backup_path, 'wb') as backup_f:
                    backup_f.write(old_f.read())
            os.chmod(backup_path, 0o600)
            logger.info(f"Old key backed up to: {backup_path}")
        
        # Generate and set new key
        new_key = self.encryption_service.rotate_key()
        new_info = self.encryption_service.get_key_info()
        
        logger.info(f"Key rotated from {old_info['version']} to {new_info['version']}")
        print(f"\n✓ Key rotation complete: {old_info['version']} → {new_info['version']}")
        
        # Check if we need to re-encrypt existing data
        print("\n⚠️  Note: Existing encrypted data will still use the old key.")
        print("The system supports multiple key versions, so old data remains accessible.")
        print("To re-encrypt existing data with the new key, run:")
        print("  python manage_encryption.py reencrypt")
    
    def test_encryption(self):
        """Test encryption and decryption."""
        test_query = "What is the travel rate for Halifax?"
        test_response = "The travel rate for Halifax is $150 per day according to section 4.2.1"
        
        print("\n=== Testing Encryption ===")
        print(f"Test Query: {test_query}")
        print(f"Test Response: {test_response[:50]}...")
        
        try:
            # Test encryption
            encrypted_query, query_version = self.encryption_service.encrypt_text(test_query)
            encrypted_response, response_version = self.encryption_service.encrypt_text(test_response)
            
            print(f"\n✓ Encryption successful")
            print(f"  Query encrypted length: {len(encrypted_query)} chars")
            print(f"  Response encrypted length: {len(encrypted_response)} chars")
            print(f"  Encryption version: {query_version}")
            
            # Test decryption
            decrypted_query = self.encryption_service.decrypt_text(encrypted_query, query_version)
            decrypted_response = self.encryption_service.decrypt_text(encrypted_response, response_version)
            
            if decrypted_query == test_query and decrypted_response == test_response:
                print("\n✓ Decryption successful - data matches!")
            else:
                print("\n✗ Decryption failed - data mismatch!")
                
        except Exception as e:
            print(f"\n✗ Encryption test failed: {e}")
            return False
        
        return True
    
    def verify_setup(self):
        """Verify the encryption setup is working correctly."""
        print("\n=== Verifying Encryption Setup ===")
        
        # Check configuration
        print("\nConfiguration:")
        print(f"  Query logging enabled: {settings.enable_query_logging}")
        print(f"  Encryption enabled: {settings.encrypt_query_logs}")
        print(f"  Anonymization enabled: {settings.anonymize_query_logs}")
        print(f"  Retention days: {settings.query_retention_days}")
        
        # Check key
        self.show_key_info()
        
        # Test encryption
        if self.test_encryption():
            print("\n✓ All encryption tests passed!")
        else:
            print("\n✗ Encryption tests failed!")
            print("\nTroubleshooting:")
            print("1. Ensure encryption key exists")
            print("2. Check file permissions on key file")
            print("3. Verify RAG_ENCRYPTION_KEY environment variable if using env key")


async def main():
    parser = argparse.ArgumentParser(description="Manage encryption for RAG query logging")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate a new encryption key')
    generate_parser.add_argument('--output', '-o', help='Output path for key file')
    generate_parser.add_argument('--show', action='store_true', help='Display the generated key')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show current key information')
    
    # Rotate command
    rotate_parser = subparsers.add_parser('rotate', help='Rotate the encryption key')
    rotate_parser.add_argument('--no-backup', action='store_true', help='Do not backup old key')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test encryption/decryption')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify complete setup')
    
    # Export key command
    export_parser = subparsers.add_parser('export', help='Export key as environment variable')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = EncryptionManager()
    
    if args.command == 'generate':
        key = manager.generate_key(args.output)
        if args.show:
            print(f"\nGenerated key: {key}")
        else:
            print("\n✓ Key generated successfully")
            print("\nTo use this key, either:")
            print(f"1. Set environment variable: export RAG_ENCRYPTION_KEY='{key}'")
            print(f"2. Key is saved at: {args.output or manager.encryption_service.key_path}")
    
    elif args.command == 'info':
        manager.show_key_info()
    
    elif args.command == 'rotate':
        await manager.rotate_key(backup_old=not args.no_backup)
    
    elif args.command == 'test':
        manager.test_encryption()
    
    elif args.command == 'verify':
        manager.verify_setup()
    
    elif args.command == 'export':
        info = manager.encryption_service.get_key_info()
        if os.path.exists(info['key_path']):
            with open(info['key_path'], 'r') as f:
                key = f.read().strip()
            print(f"export RAG_ENCRYPTION_KEY='{key}'")
        else:
            print("✗ No key file found. Generate one first with: python manage_encryption.py generate")


if __name__ == "__main__":
    asyncio.run(main())