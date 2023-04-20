""" trader/secure_credentials.py

Module: tradeforce.trader.secure_credentials
--------------------------------------------

Manages secure storage and handling of API credentials.

Provides the SecureCredentials class for securely loading, encrypting,
decrypting, and storing API credentials. Credentials are encrypted
using a user-provided password and stored in an encrypted credentials file.

The DecryptedCredentials context manager is provided to securely handle
decrypted credentials in a limited scope. Sensitive data is securely
cleared from memory after use.

Classes:
    SecureCredentials: Manages the loading, encryption, and decryption of API credentials.
    DecryptedCredentials: Context manager for securely handling decrypted credentials.

Main function:
    load_credentials: Load API credentials from the credentials config file.

Example:
    import tradeforce.trader.secure_credentials as secure_credentials

    (...)

    secure_credentials = load_credentials(self.root)
    with secure_credentials.decrypted_credentials() as credentials:
        bfx_api = Client(
            credentials["auth_key"],
            credentials["auth_sec"],
            ws_host=WS_HOST,
            rest_host=REST_HOST,
            logLevel=self.config.log_level_ws_live,
        )
    return bfx_api

    (...)

"""

from __future__ import annotations
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Type

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


def load_credentials(root: Tradeforce) -> SecureCredentials:
    """Load API credentials from the credentials config file
        utilizing the SecureCredentials class.

    Main function to be used for loading API credentials
    from the credentials config file.

    Params:
        root: The main Tradeforce instance providing access
        to the config and logging or any other module.

    Returns:
        An instance of the SecureCredentials class with loaded encrypted credentials.
    """
    secure_credentials = SecureCredentials(root)
    secure_credentials._load_encrypted_credentials()
    secure_credentials._get_password()
    if not secure_credentials.credentials_exist_on_file:
        secure_credentials._save_encrypted_credentials()
    return secure_credentials


class SecureCredentials:
    """Initialize the SecureCredentials instance.

    Manages the loading, encryption, and decryption of API credentials.

    Params:
        root: The main Tradeforce instance providing access
        to the config and logging or any other module.
    """

    def __init__(self, root: Tradeforce):
        self.root = root
        self.log = root.logging.get_logger(__name__)
        self.api_key: str | None = None
        self.api_secret: str | None = None
        self.credentials_file = os.path.join(root.config.credentials_path, "credentials")
        self.encrypted_credentials_data: bytes | None = None
        self.credentials_salt: bytes | None = None
        self.credentials_exist_on_file = False

    def _load_encrypted_credentials(self) -> None:
        """Load encrypted credentials from the credentials file.

        If the credentials file is not found or the credentials are not valid,
        prompt the user for API key and secret.
        """
        encrypted_credentials_and_salt = self._read_credentials_file()
        if encrypted_credentials_and_salt:
            self._extract_salt_and_credentials(encrypted_credentials_and_salt)
            self.credentials_exist_on_file = True
            self.log.info("Encrypted credentials loaded.")
        else:
            self._get_raw_credentials()

    def _read_credentials_file(self) -> bytes | None:
        """Read the credentials file and return its content.

        Returns:
            The content of the credentials file or None if not found or an error occurred.
        """
        try:
            with open(self.credentials_file, "rb") as file:
                return file.read()
        except FileNotFoundError:
            SystemExit("Credentials file not found. Provide your API key and secret. It will be encrypted and stored.")
        except IOError as error:
            SystemExit(f"Error loading credentials file: {error}")
        return None

    def _extract_salt_and_credentials(self, encrypted_credentials_and_salt: bytes) -> None:
        """Extract salt and encrypted credentials from the provided data.

        Params:
            encrypted_credentials_and_salt: Data containing salt and encrypted credentials.
        """

        self.credentials_salt = encrypted_credentials_and_salt[:16]
        self.encrypted_credentials_data = encrypted_credentials_and_salt[16:]

    def _get_raw_credentials(self) -> None:
        """Get raw credentials (API key and secret)
        from environment variables or user input.
        """
        if not self.encrypted_credentials_data:
            self.api_key = os.environ.get("BFX_API_KEY") or getpass("Bitfinex API key:")
            self.api_secret = os.environ.get("BFX_API_SECRET") or getpass("Bitfinex API secret:")

    def _get_password(self) -> None:
        """Get the user's password and generate the encryption key based on it.

        Securely clears the password from memory after use.
        """
        password = getpass("Password:")
        self.encryption_key = self._generate_encryption_key(password)
        self._securely_clear_sensitive_data(password=password)

    def _generate_encryption_key(self, password: str) -> bytes | None:
        """Generate an encryption key from the given password and the salt.

        Params:
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

    def _save_encrypted_credentials(self) -> None:
        """Save encrypted API key and secret to the credentials file.

        Securely clears raw API key and secret from memory after use.
        """
        if self._are_credentials_and_key_available():
            encrypted_api_key, encrypted_api_secret = self._encrypt_credentials()
            self._securely_clear_sensitive_data(api_key=self.api_key, api_secret=self.api_secret)
            self._write_encrypted_credentials_to_file(encrypted_api_key, encrypted_api_secret)

    def _are_credentials_and_key_available(self) -> bool:
        """Check if credentials (API key, API secret) and encryption key are available.

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

    def _encrypt_credentials(self) -> tuple[bytes, bytes]:
        """Encrypt API key and secret using the encryption key.

        Returns:
            Tuple containing encrypted API key and encrypted API secret.
        """
        if not self.encryption_key:
            self.log.error("No encryption key found. Cannot encrypt credentials.")
            return b"", b""

        if not self.api_key or not self.api_secret:
            self.log.error("No credentials (api_key, api_secret) found. Cannot encrypt credentials.")
            return b"", b""

        encryption_tool = Fernet(self.encryption_key)
        encrypted_api_key = encryption_tool.encrypt(self.api_key.encode())
        encrypted_api_secret = encryption_tool.encrypt(self.api_secret.encode())
        return encrypted_api_key, encrypted_api_secret

    def _write_encrypted_credentials_to_file(self, encrypted_api_key: bytes, encrypted_api_secret: bytes) -> None:
        """Write encrypted API key, API secret, and salt to the credentials file.

        Set file permissions to read/write for the user only.

        Params:
            encrypted_api_key: Encrypted API key.
            encrypted_api_secret: Encrypted API secret.
        """
        if not self.credentials_salt:
            self.log.error("No salt found. Cannot save encrypted credentials.")
            return

        try:
            with open(self.credentials_file, "wb") as file:
                file.write(self.credentials_salt + encrypted_api_key + encrypted_api_secret)
            os.chmod(self.credentials_file, stat.S_IRUSR | stat.S_IWUSR)
        except IOError as error:
            self.log.error(error)
            SystemExit("Error saving encrypted credentials")

    def _decrypt_credentials(self) -> dict[str, str] | None:
        """Decrypt the API key and secret using the encryption key.

        Returns:
            Dict containing the decrypted API key and secret, or None if an error occurred.

        """
        if not self.encrypted_credentials_data:
            return None
        if not self.encryption_key:
            return None

        encrypted_api_key = self.encrypted_credentials_data[: len(self.encrypted_credentials_data) // 2]
        encrypted_api_secret = self.encrypted_credentials_data[len(self.encrypted_credentials_data) // 2 :]

        try:
            decrypted_api_key = Fernet(self.encryption_key).decrypt(encrypted_api_key)
            decrypted_api_secret = Fernet(self.encryption_key).decrypt(encrypted_api_secret)
        except (InvalidSignature, InvalidToken):
            self.log.error("Invalid password")
            return {}
        except Exception:
            self.log.error("Error decrypting credentials.")
            return {}
        else:
            self.log.info("Credentials loaded.")
            return {"auth_key": decrypted_api_key.decode(), "auth_sec": decrypted_api_secret.decode()}

    def decrypted_credentials(self) -> DecryptedCredentials:
        """Get an instance of the DecryptedCredentials context manager
            for securely handling decrypted data.

        Returns:
            An instance of the DecryptedCredentials context manager.
        """
        return DecryptedCredentials(self)

    def _securely_clear_sensitive_data(self, **data_dict: str | None) -> None:
        """Securely clear sensitive data from memory.

        Params:
            **data_dict: Dict containing the names and values of sensitive data.
        """
        for data_name, data_value in data_dict.items():
            if data_value and isinstance(data_value, str):
                data_dict[data_name] = self._securely_overwrite(data_value)
        gc.collect()

    @staticmethod
    def _securely_overwrite(data: str) -> None:
        """Overwrite the given string data with zeros to securely clear it from memory.

        Params:
            data: The string data to securely overwrite.
        """
        data_len = len(data)
        data_bytes = bytearray(data, "utf-8")
        for i in range(data_len):
            data_bytes[i] = 0


class DecryptedCredentials(AbstractContextManager):
    """Context manager for handling decrypted credentials.

    Params:
        secure_credentials: An instance of the SecureCredentials class.
    """

    def __init__(self, secure_credentials: SecureCredentials) -> None:
        self.secure_credentials = secure_credentials
        # self.decrypted_data = dict[str, str] | None

    def __enter__(self) -> dict[str, str]:
        self.decrypted_data = self.secure_credentials._decrypt_credentials()
        if not self.decrypted_data:
            raise Exception("Error decrypting credentials.")
        return self.decrypted_data

    def __exit__(
        self, exc_type: Type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> Literal[False]:
        if self.decrypted_data:
            self.secure_credentials._securely_clear_sensitive_data(**self.decrypted_data)
        self.decrypted_data = None
        return False
