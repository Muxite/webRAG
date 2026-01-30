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

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.search_api_key
        }
        params = {"q": "health check", "count": 1}

        result = await self.request(
            "GET", self.url, retries=2, headers=headers, params=params
        )

        if result.error or result.status != 200:
            self.logger.warning(f"Search API health check failed: {result.status}")
            self.search_api_ready = False
            return False

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
            self.logger.error(f"Search API query failed with {result.status}, {result.data}")
            return None

        try:
            web_results = result.data["web"]["results"]
            return [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", "")
                }
                for item in web_results
            ]
        except (KeyError, TypeError, IndexError):
            self.logger.warning(f"Unexpected Search API response structure: {result}")
            return None