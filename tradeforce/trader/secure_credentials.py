from __future__ import annotations
from typing import TYPE_CHECKING

import os
import stat
import gc
import base64
from getpass import getpass
from contextlib import AbstractContextManager
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
from cryptography.fernet import InvalidToken
from cryptography.exceptions import InvalidSignature

# Prevent circular import for type checking
if TYPE_CHECKING:
    from tradeforce import Tradeforce


class SecureCredentials:
    def __init__(self, root: Tradeforce):
        self.root = root
        self.log = root.logging.get_logger(__name__)
        self.api_key: str | None = None
        self.api_secret: str | None = None
        self.credentials_file = os.path.join(root.config.creds_path, "credentials")
        self.encrypted_credentials_data: bytes | None = None
        self.credentials_salt: bytes | None = None
        self.credentials_exist_on_file = False

    def load_encrypted_credentials(self) -> None:
        encrypted_credentials_and_salt = self.read_credentials_file()
        if encrypted_credentials_and_salt:
            self.extract_salt_and_credentials(encrypted_credentials_and_salt)
            self.credentials_exist_on_file = True
            self.log.info("Encrypted credentials loaded.")
        else:
            self.get_raw_credentials()

    def read_credentials_file(self):
        try:
            with open(self.credentials_file, "rb") as file:
                return file.read()
        except FileNotFoundError:
            self.log.info(
                "Credentials file not found. Provide your API key and secret. It will be encrypted and stored."
            )
        except IOError as error:
            self.log.error(f"Error loading credentials file: {error}")

    def extract_salt_and_credentials(self, encrypted_credentials_and_salt):
        self.credentials_salt = encrypted_credentials_and_salt[:16]
        self.encrypted_credentials_data = encrypted_credentials_and_salt[16:]

    def get_raw_credentials(self):
        if not self.encrypted_credentials_data:
            self.api_key = os.environ.get("BFX_API_KEY") or getpass("Bitfinex API key:")
            self.api_secret = os.environ.get("BFX_API_SECRET") or getpass("Bitfinex API secret:")

    def get_password(self):
        password = getpass("Password:")
        self.encryption_key = self.generate_encryption_key(password)
        self.securely_clear_sensitive_data(password=password)

    def generate_encryption_key(self, password: str):
        if not self.credentials_salt:
            self.credentials_salt = os.urandom(16)
        try:
            key_derivation_function = PBKDF2HMAC(
                algorithm=hashes.SHA3_256(),
                length=32,
                salt=self.credentials_salt,
                iterations=500000,
                backend=default_backend(),
            )
            return base64.urlsafe_b64encode(key_derivation_function.derive(password.encode()))
        except Exception:
            self.log.error("Error generating encryption key.")
            return None

    def save_encrypted_credentials(self):
        if self.are_credentials_and_key_available():
            encrypted_api_key, encrypted_api_secret = self.encrypt_credentials()
            self.securely_clear_sensitive_data(api_key=self.api_key, api_secret=self.api_secret)
            self.write_encrypted_credentials_to_file(encrypted_api_key, encrypted_api_secret)

    def are_credentials_and_key_available(self):
        if not self.api_key or not self.api_secret:
            self.log.error("No credentials (api_key, api_secret) found. Cannot save credentials.")
            return False
        if not self.encryption_key:
            self.log.error("No encryption key found. Cannot save credentials.")
            return False
        return True

    def encrypt_credentials(self):
        encryption_tool = Fernet(self.encryption_key)
        encrypted_api_key = encryption_tool.encrypt(self.api_key.encode())
        encrypted_api_secret = encryption_tool.encrypt(self.api_secret.encode())
        return encrypted_api_key, encrypted_api_secret

    def write_encrypted_credentials_to_file(self, encrypted_api_key, encrypted_api_secret):
        try:
            with open(self.credentials_file, "wb") as file:
                file.write(self.credentials_salt + encrypted_api_key + encrypted_api_secret)
            os.chmod(self.credentials_file, stat.S_IRUSR | stat.S_IWUSR)
        except IOError as error:
            self.log.error(error)
            SystemExit("Error saving encrypted credentials")

    def decrypt_credentials(self):
        if not self.encrypted_credentials_data:
            return None

        encrypted_api_key = self.encrypted_credentials_data[: len(self.encrypted_credentials_data) // 2]
        encrypted_api_secret = self.encrypted_credentials_data[len(self.encrypted_credentials_data) // 2 :]

        try:
            decrypted_api_key = Fernet(self.encryption_key).decrypt(encrypted_api_key)
            decrypted_api_secret = Fernet(self.encryption_key).decrypt(encrypted_api_secret)
        except (InvalidSignature, InvalidToken):
            self.log.error("Invalid password")
            return None
        except Exception:
            self.log.error("Error decrypting credentials.")
            return None
        else:
            self.log.info("Credentials loaded.")
            return {"auth_key": decrypted_api_key.decode(), "auth_sec": decrypted_api_secret.decode()}

    def decrypted_credentials(self):
        return DecryptedCredentials(self)

    def securely_clear_sensitive_data(self, **data_dict):
        for data_name, data_value in data_dict.items():
            if data_value and isinstance(data_value, str):
                data_dict[data_name] = self.securely_overwrite(data_value)
        gc.collect()

    @staticmethod
    def securely_overwrite(data: str) -> None:
        data_len = len(data)
        data_bytes = bytearray(data, "utf-8")
        for i in range(data_len):
            data_bytes[i] = 0
        return None


def load_credentials(root: Tradeforce) -> SecureCredentials:
    """Load API credentials from credential config file.
    Returns None if file is not found or credentials are not valid.
    """
    secure_credentials = SecureCredentials(root)
    secure_credentials.load_encrypted_credentials()
    secure_credentials.get_password()
    if not secure_credentials.credentials_exist_on_file:
        secure_credentials.save_encrypted_credentials()
    return secure_credentials


class DecryptedCredentials(AbstractContextManager):
    def __init__(self, secure_credentials: SecureCredentials):
        self.secure_credentials = secure_credentials
        self.decrypted_data = None

    def __enter__(self):
        self.decrypted_data = self.secure_credentials.decrypt_credentials()
        return self.decrypted_data

    def __exit__(self, exc_type, exc_value, traceback):
        if self.decrypted_data:
            self.secure_credentials.securely_clear_sensitive_data(**self.decrypted_data)
        self.decrypted_data = None
        return False
