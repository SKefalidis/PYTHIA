import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin
from src.config import GRAPH_SEARCH_SETTINGS
from src.logging import log, LogComponent, LogLevel, LogType


class GraphSearchClient:
    """HTTP client responsible for all interactions with the Graph Search service."""

    def __init__(self, endpoint: Optional[str] = None, timeout: Optional[int] = None):
        base_endpoint = endpoint or GRAPH_SEARCH_SETTINGS.endpoint
        # Ensure trailing slash for clean urljoin
        self.base_url = base_endpoint if base_endpoint.endswith("/") else base_endpoint + "/"
        self.timeout = timeout or GRAPH_SEARCH_SETTINGS.timeout
        self.headers = {"Content-Type": "application/json"}

    def _post(self, path: str, payload: Optional[Dict[str, Any]] = None, return_response: bool = False) -> Optional[Dict[str, Any]]:
        url = urljoin(self.base_url, path.lstrip("/"))
        try:
            response = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
            if response.status_code == 200:
                if return_response:
                    return response
                return response.json()
            log(f"Graph search request to {url} failed with status {response.status_code}: {response.text}",
                LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
        except requests.exceptions.ConnectionError as e:
            log(f"Connection error to graph search server at {url}: {e}",
                LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            raise e
        except Exception as e:
            log(f"Error querying graph search server at {url}: {e} ({type(e).__name__})",
                LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
            raise e

    def find_all_paths(self, uri_start: str, uri_end: str, filter_predicates: bool = False, additional_hops: int = 1) -> requests.Response:
        """POST /pythia-all-paths"""
        payload = {
            "uri_start": uri_start,
            "uri_end": uri_end,
            "fp": "1" if filter_predicates else "0",
            "additional_hops": additional_hops
        }
        return self._post("pythia-all-paths", payload, return_response=True)

    def cancel_tasks(self) -> bool:
        """POST /cancel"""
        res = self._post("cancel")
        if res is not None:
            log("Sent cancellation request for graph search tasks.", LogComponent.PATH_EXTRACTOR, LogLevel.APPLICATION, LogType.NORMAL)
            return True
        log("Failed to send cancellation request for graph search tasks.", LogComponent.PATH_EXTRACTOR, LogLevel.WARNING)
        return False

    def load_graph(self, source_dir: str, filter_classes: Optional[bool] = None) -> bool:
        """POST /load-graph (centralizes setup_graph_tool from utils.py)"""
        if filter_classes is None:
            filter_classes = GRAPH_SEARCH_SETTINGS.filter_classes
        payload = {
            "source_dir": source_dir,
            "filter_classes": "1" if filter_classes else "0",
        }
        return self._post("load-graph", payload) is not None

    def get_active_graph(self) -> Optional[str]:
        """POST /status: returns the currently loaded graph directory name, or None if no graph is loaded."""
        data = self._post("status")
        if data and "graph" in data:
            return data["graph"]
        return None