import os
import secrets
from typing import Optional


class ApiKeyProvider:
    """
    ApiKeyProvider manages retrieval or generation of the gateway API key.
    :return key: Current API key to be used by the API
    """

    def __init__(
        self,
        env_var: str = "GATEWAY_API_KEY",
        test_flag: str = "TEST_MODE",
        special_keys_file_env: str = "GATEWAY_SPECIAL_KEYS_FILE",
        default_special_keys_file: str = "special_api_keys.txt",
    ) -> None:
        """
        Initialize provider with environment variable names.
        :param env_var: Environment variable holding the API key
        :param test_flag: Environment variable that enables test behavior
        :return None: Nothing is returned
        """
        self.env_var = env_var
        self.test_flag = test_flag
        self.special_keys_file_env = special_keys_file_env
        self.default_special_keys_file = default_special_keys_file
        self._key: Optional[str] = None
        self._allowed_keys_cache: Optional[set[str]] = None

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

    def _load_special_keys(self) -> set[str]:
        """
        Load special API keys from a file. Lines starting with '#' or blank lines are ignored.
        The file path is taken from env var defined by `special_keys_file_env` or defaults to
        `default_special_keys_file` relative to the current working directory.
        :return keys: A set of keys loaded from the file (may be empty)
        """
        keys: set[str] = set()
        path = os.environ.get(self.special_keys_file_env, self.default_special_keys_file)
        try:
            if not path:
                return keys
            if not os.path.isfile(path):
                return keys
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    keys.add(s)
        except Exception:
            return set()
        return keys

    def get_allowed_keys(self) -> set[str]:
        """
        Returns a set of all allowed API keys:
        - The main expected key (env var or generated during tests)
        - Any keys listed in the special keys file
        """
        if self._allowed_keys_cache is not None:
            return self._allowed_keys_cache

        keys: set[str] = set()
        expected = self.get_expected_key()
        if expected:
            keys.add(expected)
        special = self._load_special_keys()
        keys.update(special)
        self._allowed_keys_cache = keys
        return keys

    def is_valid(self, provided_key: Optional[str]) -> bool:
        """Check whether provided key is among allowed keys."""
        if not provided_key:
            return False
        return provided_key in self.get_allowed_keys()
