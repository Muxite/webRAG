import os
import secrets
from typing import Optional


class ApiKeyProvider:
    """
    ApiKeyProvider manages retrieval or generation of the gateway API key.
    :return key: Current API key to be used by the API
    """

    def __init__(self, env_var: str = "GATEWAY_API_KEY", test_flag: str = "TEST_MODE") -> None:
        """
        Initialize provider with environment variable names.
        :param env_var: Environment variable holding the API key
        :param test_flag: Environment variable that enables test behavior
        :return None: Nothing is returned
        """
        self.env_var = env_var
        self.test_flag = test_flag
        self._key: Optional[str] = None

    def _generate_key(self) -> str:
        """
        Generate a cryptographically strong random API key for testing.
        :return key: The generated API key string
        """
        return secrets.token_urlsafe(32)

    def get_expected_key(self) -> Optional[str]:
        """
        Obtain the expected API key from the environment or generate one in tests.
        :return key: The API key string or None when not available
        """
        if self._key:
            return self._key

        existing = os.environ.get(self.env_var)
        if existing:
            self._key = existing
            return self._key

        if os.environ.get(self.test_flag) == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
            key = self._generate_key()
            os.environ[self.env_var] = key
            self._key = key
            return self._key

        return None
