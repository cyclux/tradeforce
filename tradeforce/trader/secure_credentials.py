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
    """
    Initialize the SecureCredentials instance.

    Manages the loading, encryption, and decryption of API credentials.

    Args:
        root: The main Tradeforce instance. Provides access to other modules.
    """

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
        """
        Load encrypted credentials from the credentials file.

        If the credentials file is not found or the credentials are not valid,
        prompt the user for API key and secret.
        """
        encrypted_credentials_and_salt = self.read_credentials_file()
        if encrypted_credentials_and_salt:
            self.extract_salt_and_credentials(encrypted_credentials_and_salt)
            self.credentials_exist_on_file = True
            self.log.info("Encrypted credentials loaded.")
        else:
            self.get_raw_credentials()

    def read_credentials_file(self):
        """
        Read the credentials file and return its content.

        Returns:
            The content of the credentials file or None if not found or an error occurred.
        """
        try:
            with open(self.credentials_file, "rb") as file:
                return file.read()
        except FileNotFoundError:
            self.log.info(
                "Credentials file not found. Provide your API key and secret. It will be encrypted and stored."
            )
        except IOError as error:
            self.log.error(f"Error loading credentials file: {error}")

    def extract_salt_and_credentials(self, encrypted_credentials_and_salt: bytes):
        """
        Extract salt and encrypted credentials from the provided data.

        Args:
            encrypted_credentials_and_salt: Data containing salt and encrypted credentials.
        """

        self.credentials_salt = encrypted_credentials_and_salt[:16]
        self.encrypted_credentials_data = encrypted_credentials_and_salt[16:]

    def get_raw_credentials(self):
        """
        Get raw credentials (API key and secret) from environment variables or user input.
        """
        if not self.encrypted_credentials_data:
            self.api_key = os.environ.get("BFX_API_KEY") or getpass("Bitfinex API key:")
            self.api_secret = os.environ.get("BFX_API_SECRET") or getpass("Bitfinex API secret:")

    def get_password(self):
        """
        Get the user's password and generate the encryption key based on it.
        Securely clears the password from memory after use.
        """
        password = getpass("Password:")
        self.encryption_key = self.generate_encryption_key(password)
        self.securely_clear_sensitive_data(password=password)

    def generate_encryption_key(self, password: str):
        """
        Generate an encryption key from the given password and the salt.

        Args:
            password: The user's password.

        Returns:
            The encryption key derived from the password and salt.
        """
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
        """
        Save encrypted API key and secret to the credentials file.
        Securely clears raw API key and secret from memory after use.
        """
        if self.are_credentials_and_key_available():
            encrypted_api_key, encrypted_api_secret = self.encrypt_credentials()
            self.securely_clear_sensitive_data(api_key=self.api_key, api_secret=self.api_secret)
            self.write_encrypted_credentials_to_file(encrypted_api_key, encrypted_api_secret)

    def are_credentials_and_key_available(self):
        """
        Check if credentials (API key, API secret) and encryption key are available.

        Returns:
            True if credentials and key are available, otherwise False.
        """
        if not self.api_key or not self.api_secret:
            self.log.error("No credentials (api_key, api_secret) found. Cannot save credentials.")
            return False
        if not self.encryption_key:
            self.log.error("No encryption key found. Cannot save credentials.")
            return False
        return True

    def encrypt_credentials(self):
        """
        Encrypt API key and secret using the encryption key.

        Returns:
            Tuple containing encrypted API key and encrypted API secret.
        """
        encryption_tool = Fernet(self.encryption_key)
        encrypted_api_key = encryption_tool.encrypt(self.api_key.encode())
        encrypted_api_secret = encryption_tool.encrypt(self.api_secret.encode())
        return encrypted_api_key, encrypted_api_secret

    def write_encrypted_credentials_to_file(self, encrypted_api_key, encrypted_api_secret):
        """
        Write encrypted API key, API secret, and salt to the credentials file.
        Set file permissions to read/write for the user only.

        Args:
            encrypted_api_key: Encrypted API key.
            encrypted_api_secret: Encrypted API secret.
        """
        try:
            with open(self.credentials_file, "wb") as file:
                file.write(self.credentials_salt + encrypted_api_key + encrypted_api_secret)
            os.chmod(self.credentials_file, stat.S_IRUSR | stat.S_IWUSR)
        except IOError as error:
            self.log.error(error)
            SystemExit("Error saving encrypted credentials")

    def decrypt_credentials(self):
        """
        Decrypt the API key and secret using the encryption key.

        Returns:
            A dictionary containing the decrypted API key and secret, or None if an error occurred.

        """
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
        """
        Get an instance of the DecryptedCredentials context manager for securely handling decrypted data.

        Returns:
            An instance of the DecryptedCredentials context manager.
        """
        return DecryptedCredentials(self)

    def securely_clear_sensitive_data(self, **data_dict):
        """
        Securely clear sensitive data from memory.

        Args:
            **data_dict: A dictionary containing the names and values of sensitive data.
        """
        for data_name, data_value in data_dict.items():
            if data_value and isinstance(data_value, str):
                data_dict[data_name] = self.securely_overwrite(data_value)
        gc.collect()

    @staticmethod
    def securely_overwrite(data: str) -> None:
        """
        Overwrite the given string data with zeros to securely clear it from memory.

        Args:
            data: The string data to securely overwrite.
        """
        data_len = len(data)
        data_bytes = bytearray(data, "utf-8")
        for i in range(data_len):
            data_bytes[i] = 0


def load_credentials(root: Tradeforce) -> SecureCredentials:
    """
    Load API credentials from the credentials config file.

    Args:
        root: The main Tradeforce instance.

    Returns:
        An instance of the SecureCredentials class with loaded encrypted credentials.
    """
    secure_credentials = SecureCredentials(root)
    secure_credentials.load_encrypted_credentials()
    secure_credentials.get_password()
    if not secure_credentials.credentials_exist_on_file:
        secure_credentials.save_encrypted_credentials()
    return secure_credentials


class DecryptedCredentials(AbstractContextManager):
    """
    Context manager for handling decrypted credentials.

    Args:
        secure_credentials: An instance of the SecureCredentials class.
    """

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
