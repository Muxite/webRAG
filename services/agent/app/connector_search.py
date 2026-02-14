from agent.app.connector_http import ConnectorHttp
from shared.connector_config import ConnectorConfig
from typing import Optional, Dict, List


class ConnectorSearch(ConnectorHttp):
    """Manage an searching api session for a connector."""
    def __init__(self, connector_config: ConnectorConfig):
        super().__init__(connector_config)
        self.config = connector_config
        self.search_api_key = self.config.search_api_key
        self.search_api_ready = False
        self.url = "https://api.search.brave.com/res/v1/web/search"

    async def init_search_api(self) -> bool:
        """
        Verifies the connection to the Search API by making a test query.
        Sets the readiness flag `self.search_api_ready`.
        """
        if self.search_api_ready:
            return True

        if not self.search_api_key:
            self.logger.error("Cannot initialize Search API without an API key.")
            return False

        self.logger.info("Probing Search API...")
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.search_api_key
        }
        params = {"q": "health check", "count": 1}

        result = await self.request(
            "GET", self.url, retries=2, headers=headers, params=params
        )

        if result.error or result.status != 200:
            self.logger.warning(f"Search API health probe failed with status {result.status}: {result.data}")
            self.search_api_ready = False
            return False

        self.logger.info("Search API OPERATIONAL")
        self.search_api_ready = True
        return True

    async def query_search(self, query: str, count: int = 10) -> Optional[List[Dict[str, str]]]:
        """
        Send a search request to the configured Search API endpoint.
        :param query: Search query string
        :param count: Number of results to return (default 10)
        :return: List of search results or None if request failed or bad response
        """
        if not await self.init_search_api():
            self.logger.warning("Setup failed.")
            return None

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.search_api_key
        }
        params = {
            "q": query,
            "count": count
        }

        result = await self.request("GET", url, retries=3, headers=headers, params=params)

        if result.error:
            raise RuntimeError(f"Search API query failed: status={result.status} data={result.data}")

        data = result.data
        if hasattr(data, "data"):
            data = getattr(data, "data")

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected search response type: {type(data).__name__}")

        def _collect(items: list) -> List[Dict[str, str]]:
            collected: List[Dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                url_value = item.get("url") or item.get("link") or item.get("href")
                if not url_value:
                    nested = item.get("results")
                    if isinstance(nested, list):
                        collected.extend(_collect(nested))
                    continue
                collected.append(
                    {
                        "title": item.get("title") or item.get("name") or "",
                        "url": url_value,
                        "description": item.get("description") or item.get("snippet") or "",
                    }
                )
            return collected

        try:
            web_results = []
            if isinstance(data.get("web"), dict):
                web_results = data.get("web", {}).get("results") or []
            if isinstance(web_results, list) and web_results:
                return _collect(web_results)

            mixed = data.get("mixed", {})
            if isinstance(mixed, dict):
                mixed_items: List[Dict[str, str]] = []
                for key, value in mixed.items():
                    if isinstance(value, list):
                        mixed_items.extend(_collect(value))
                if mixed_items:
                    return mixed_items

            return []
        except Exception as exc:
            raise RuntimeError(f"Search parse failed: {data} ({exc})")