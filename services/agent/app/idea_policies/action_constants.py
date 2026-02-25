from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorType(str, Enum):
    """
    Standard error type classifications for action failures.
    
    Used to categorize different types of errors that can occur during action execution.
    """
    BLOCKED_SITE = "BlockedSite"
    INVALID_URL = "InvalidURL"
    TIMEOUT = "Timeout"
    NETWORK_ERROR = "NetworkError"
    PARSE_ERROR = "ParseError"
    VALIDATION_ERROR = "ValidationError"
    UNKNOWN = "Unknown"


class ResultStatus(str, Enum):
    """
    Standard status values for action results and execution states.
    
    Used to track the state of actions and nodes throughout execution.
    """
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"


class ActionResultKey(str, Enum):
    """
    Standard keys for action result dictionaries.
    
    All action results (search, visit, think, save, merge) use these keys
    to ensure consistency across the codebase.
    """
    ACTION = "action"
    SUCCESS = "success"
    ERROR = "error"
    ERROR_TYPE = "error_type"
    RETRYABLE = "retryable"
    CONTEXT = "context"
    NODE_ID = "node_id"
    TIMESTAMP = "timestamp"
    
    QUERY = "query"
    COUNT = "count"
    INTENT = "intent"
    RESULTS = "results"
    
    URL = "url"
    LINK = "link"
    CONTENT = "content"
    CONTENT_TOTAL_CHARS = "content_total_chars"
    CONTENT_IS_TRUNCATED = "content_is_truncated"
    CONTENT_FULL = "content_full"
    CONTENT_WITH_LINKS = "content_with_links"
    LINK_CONTEXTS = "link_contexts"
    
    THINKING_CONTENT = "thinking_content"
    TITLE = "title"
    DETAILS = "details"
    
    DOCUMENT_COUNT = "count"
    
    SYNTHESIZED = "synthesized"
    CHILD_COUNT = "child_count"
    RAW_RESPONSE = "raw_response"
    
    VECTOR_CONTEXT = "vector_context"
    HTTP_STATUS = "http_status"
    TRACEBACK_SUMMARY = "traceback_summary"
    ROOT_CAUSE = "root_cause"


class PromptKey(str, Enum):
    """
    Standard keys for LLM prompt message structures.
    
    Used when building prompts for expansion, evaluation, merge, etc.
    """
    ROLE = "role"
    CONTENT = "content"
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContextKey(str, Enum):
    """
    Standard keys for context dictionaries passed to actions and prompts.
    
    These fields are extracted from node details and passed as context
    to help actions understand what they're working with.
    """
    QUERY = "query"
    URL = "url"
    LINK = "link"
    PROMPT = "prompt"
    TEXT = "text"
    INTENT = "intent"
    COUNT = "count"
    TITLE = "title"
    PARENT_GOAL = "parent_goal"
    PARENT_JUSTIFICATION = "parent_justification"
    JUSTIFICATION = "justification"
    RATIONALE = "rationale"
    WHY_THIS_NODE = "why_this_node"


class ActionResultBuilder:
    """
    Builder for creating consistent action result dictionaries.
    
    Provides clear methods for building success and failure results,
    making it easy to see what fields are included in each type of result.
    """
    
    @staticmethod
    def success(
        action: str,
        node_id: Optional[str] = None,
        **additional_fields: Any
    ) -> Dict[str, Any]:
        """
        Build a successful action result.
        
        :param action: Action type (e.g., "search", "visit")
        :param node_id: Optional node identifier
        :param additional_fields: Additional fields specific to the action type
        :returns: Standardized success result dict
        """
        result: Dict[str, Any] = {
            ActionResultKey.ACTION.value: action,
            ActionResultKey.SUCCESS.value: True,
        }
        if node_id:
            result[ActionResultKey.NODE_ID.value] = node_id
        result.update(additional_fields)
        return result
    
    @staticmethod
    def failure(
        action: str,
        error: str,
        error_type: Optional[str] = None,
        retryable: bool = False,
        node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **additional_fields: Any
    ) -> Dict[str, Any]:
        """
        Build a failed action result.
        
        :param action: Action type (e.g., "search", "visit")
        :param error: Error message string
        :param error_type: Optional error type classification
        :param retryable: Whether the action can be retried
        :param node_id: Optional node identifier
        :param context: Optional context dict with action parameters
        :param additional_fields: Additional fields specific to the failure
        :returns: Standardized failure result dict
        """
        result: Dict[str, Any] = {
            ActionResultKey.ACTION.value: action,
            ActionResultKey.SUCCESS.value: False,
            ActionResultKey.ERROR.value: error,
            ActionResultKey.RETRYABLE.value: retryable,
        }
        if error_type:
            result[ActionResultKey.ERROR_TYPE.value] = error_type
        if node_id:
            result[ActionResultKey.NODE_ID.value] = node_id
        if context:
            result[ActionResultKey.CONTEXT.value] = context
        result.update(additional_fields)
        return result


class PromptBuilder:
    """
    Builder for creating LLM prompt message structures.
    
    Provides clear methods for building system/user message pairs,
    making it obvious how prompts are assembled.
    """
    
    @staticmethod
    def build_messages(
        system_content: str,
        user_content: str
    ) -> List[Dict[str, str]]:
        """
        Build a standard two-message prompt (system + user).
        
        :param system_content: System instruction content
        :param user_content: User message content
        :returns: List of message dicts with role and content
        """
        return [
            {
                PromptKey.ROLE.value: PromptKey.SYSTEM.value,
                PromptKey.CONTENT.value: system_content,
            },
            {
                PromptKey.ROLE.value: PromptKey.USER.value,
                PromptKey.CONTENT.value: user_content,
            },
        ]
    
    @staticmethod
    def system_message(content: str) -> Dict[str, str]:
        """
        Create a system message.
        
        :param content: System instruction content
        :returns: Message dict with role="system"
        """
        return {
            PromptKey.ROLE.value: PromptKey.SYSTEM.value,
            PromptKey.CONTENT.value: content,
        }
    
    @staticmethod
    def user_message(content: str) -> Dict[str, str]:
        """
        Create a user message.
        
        :param content: User message content
        :returns: Message dict with role="user"
        """
        return {
            PromptKey.ROLE.value: PromptKey.USER.value,
            PromptKey.CONTENT.value: content,
        }


class ContextBuilder:
    """
    Builder for creating context dictionaries from node details.
    
    Extracts relevant fields from node details to pass as context
    to actions or prompts, making it clear what information is available.
    """
    
    @staticmethod
    def from_node_details(
        details: Dict[str, Any],
        keys: List[str]
    ) -> Dict[str, Any]:
        """
        Extract specified keys from node details as context.
        
        :param details: Node details dictionary
        :param keys: List of key names to extract
        :returns: Context dict with only the specified keys
        """
        context: Dict[str, Any] = {}
        for key in keys:
            if key in details:
                context[key] = details[key]
        return context
    
    @staticmethod
    def for_search(details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context dict for search actions.
        
        Extracts: query, prompt, intent, count
        
        :param details: Node details dictionary
        :returns: Context dict with search-related fields
        """
        from agent.app.idea_policies.base import DetailKey
        return {
            ContextKey.QUERY.value: details.get(DetailKey.QUERY.value) or details.get(DetailKey.PROMPT.value),
            ContextKey.INTENT.value: details.get(DetailKey.INTENT.value),
            ContextKey.COUNT.value: details.get(DetailKey.COUNT.value),
        }
    
    @staticmethod
    def for_visit(details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context dict for visit actions.
        
        Extracts: url, link, intent
        
        :param details: Node details dictionary
        :returns: Context dict with visit-related fields
        """
        from agent.app.idea_policies.base import DetailKey
        return {
            ContextKey.URL.value: details.get(DetailKey.URL.value) or details.get(DetailKey.LINK.value),
            ContextKey.INTENT.value: details.get(DetailKey.INTENT.value),
        }
    
    @staticmethod
    def for_think(details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build context dict for think actions.
        
        Extracts: text, prompt, title
        
        :param details: Node details dictionary
        :returns: Context dict with think-related fields
        """
        from agent.app.idea_policies.base import DetailKey
        return {
            ContextKey.TEXT.value: details.get(DetailKey.TEXT.value),
            ContextKey.PROMPT.value: details.get(DetailKey.PROMPT.value),
            ContextKey.TITLE.value: details.get("title"),
        }


class MergedResultsCompactor:
    """
    Helper to compact merged results for LLM consumption.
    Removes large fields and keeps only essential information.
    """
    
    @staticmethod
    def compact_for_llm(merged_results: List[Dict[str, Any]], max_items: int = 20) -> List[Dict[str, Any]]:
        """
        Compact merged results by removing large fields and keeping only essential info.
        Removes content_full, content_with_links, links_full, and other large fields.
        
        :param merged_results: Full merged results list.
        :param max_items: Maximum number of items to include (truncate if more).
        :returns: Compacted merged results.
        """
        compacted = []
        for item in merged_results[:max_items]:
            if not isinstance(item, dict):
                continue
            
            compact_item = {
                "title": item.get("title", ""),
                "status": item.get("status", ""),
                "score": item.get("score"),
            }
            
            # Compact the result field - remove large content fields
            result = item.get("result")
            if isinstance(result, dict):
                compact_result = {
                    "action": result.get("action", ""),
                    "success": result.get("success", False),
                }
                
                # Keep essential identifiers
                if result.get("url"):
                    compact_result["url"] = result.get("url")
                elif result.get("link"):
                    compact_result["url"] = result.get("link")
                
                if result.get("query"):
                    compact_result["query"] = result.get("query")
                elif result.get("prompt"):
                    compact_result["query"] = result.get("prompt")
                
                # Keep only truncated content (not content_full or content_with_links)
                # Try to get meaningful content snippet, not just truncated start
                content = result.get("content", "")
                if content:
                    max_content_chars = 200  # Keep content very short
                    if len(content) > max_content_chars:
                        # Try to get a more meaningful snippet if content looks like URLs/links
                        if content.startswith("http") and "\n" not in content[:100]:
                            # Looks like just URLs, skip content field
                            pass
                        else:
                            # Get first meaningful chunk
                            compact_result["content"] = content[:max_content_chars].strip() + "...[truncated]"
                    else:
                        compact_result["content"] = content
                
                # Keep only summary of results, not full list
                results_list = result.get("results", [])
                if isinstance(results_list, list) and len(results_list) > 0:
                    compact_result["results_count"] = len(results_list)
                    # Only include first 2 results with minimal data
                    compact_result["results_sample"] = [
                        {
                            "title": r.get("title", "")[:100] if isinstance(r, dict) else str(r)[:100],
                            "url": r.get("url", "")[:100] if isinstance(r, dict) else None,
                        }
                        for r in results_list[:2]
                    ]
                
                # Keep only essential links info (count and sample, not full list)
                links = result.get("links", [])
                if isinstance(links, list) and len(links) > 0:
                    compact_result["links_count"] = len(links)
                    compact_result["links_sample"] = links[:3]  # Only first 3
                
                # Keep error info if present (truncated)
                if result.get("error"):
                    error = str(result.get("error", ""))
                    compact_result["error"] = error[:150] + ("...[truncated]" if len(error) > 150 else "")
                
                # Keep intent if present
                if result.get("intent"):
                    compact_result["intent"] = result.get("intent")
                
                compact_item["result"] = compact_result
            else:
                compact_item["result"] = result
            
            # Keep evaluation if present (but compact it)
            evaluation = item.get("evaluation")
            if evaluation and isinstance(evaluation, dict):
                compact_eval = {
                    "score": evaluation.get("score"),
                }
                if evaluation.get("rationale"):
                    rationale = str(evaluation.get("rationale", ""))
                    compact_eval["rationale"] = rationale[:200] + ("...[truncated]" if len(rationale) > 200 else "")
                compact_item["evaluation"] = compact_eval
            elif evaluation:
                compact_item["evaluation"] = evaluation
            
            compacted.append(compact_item)
        
        if len(merged_results) > max_items:
            compacted.append({
                "title": f"... and {len(merged_results) - max_items} more items (truncated)",
                "status": "truncated",
                "result": {"action": "info", "note": f"Total items: {len(merged_results)}, showing first {max_items}"}
            })
        
        return compacted


class NodeDetailsExtractor:
    """
    Helper functions for extracting common fields from node details.
    
    Centralizes repeated patterns of extracting fields with fallbacks.
    """
    
    @staticmethod
    def get_query(details: Dict[str, Any], fallback_title: Optional[str] = None) -> Optional[str]:
        """
        Extract query from node details with fallbacks.
        
        Tries: query -> prompt -> title (if provided)
        
        :param details: Node details dictionary
        :param fallback_title: Optional title to use as final fallback
        :returns: Query string or None
        """
        from agent.app.idea_policies.base import DetailKey
        return (
            details.get(DetailKey.QUERY.value)
            or details.get(DetailKey.PROMPT.value)
            or fallback_title
        )
    
    @staticmethod
    def get_url(details: Dict[str, Any]) -> Optional[str]:
        """
        Extract URL from node details with fallback to link.
        
        Tries: url -> link
        
        :param details: Node details dictionary
        :returns: URL string or None
        """
        from agent.app.idea_policies.base import DetailKey
        return details.get(DetailKey.URL.value) or details.get(DetailKey.LINK.value)
    
    @staticmethod
    def get_justification(details: Dict[str, Any]) -> str:
        """
        Extract justification from node details with multiple fallbacks.
        
        Tries: justification -> rationale -> why_this_node -> ""
        
        :param details: Node details dictionary
        :returns: Justification string (empty if none found)
        """
        from agent.app.idea_policies.base import DetailKey
        return (
            details.get(DetailKey.JUSTIFICATION.value)
            or details.get(DetailKey.RATIONALE.value)
            or details.get(DetailKey.WHY_THIS_NODE.value)
            or ""
        )
    
    @staticmethod
    def get_action(details: Dict[str, Any]) -> Optional[str]:
        """
        Extract action type from node details.
        
        :param details: Node details dictionary
        :returns: Action type string or None
        """
        from agent.app.idea_policies.base import DetailKey
        return details.get(DetailKey.ACTION.value)
    
    @staticmethod
    def is_merge_action(details: Dict[str, Any]) -> bool:
        """
        Check if node details indicate a merge action.
        
        :param details: Node details dictionary
        :returns: True if action is merge
        """
        from agent.app.idea_policies.base import DetailKey, IdeaActionType
        return details.get(DetailKey.ACTION.value) == IdeaActionType.MERGE.value


class ActionResultExtractor:
    """
    Helper functions for extracting fields from action result dictionaries.
    
    Centralizes repeated patterns of extracting result fields.
    """
    
    @staticmethod
    def is_success(result: Dict[str, Any]) -> bool:
        """
        Check if action result indicates success.
        
        :param result: Action result dictionary
        :returns: True if successful
        """
        return bool(result.get(ActionResultKey.SUCCESS.value))
    
    @staticmethod
    def get_error(result: Dict[str, Any], default: str = "") -> str:
        """
        Extract error message from action result.
        
        :param result: Action result dictionary
        :param default: Default error message if not found
        :returns: Error message string
        """
        return result.get(ActionResultKey.ERROR.value, default)
    
    @staticmethod
    def get_url(result: Dict[str, Any]) -> Optional[str]:
        """
        Extract URL from action result.
        
        :param result: Action result dictionary
        :returns: URL string or None
        """
        return result.get(ActionResultKey.URL.value)
    
    @staticmethod
    def get_query(result: Dict[str, Any]) -> Optional[str]:
        """
        Extract query from action result.
        
        :param result: Action result dictionary
        :returns: Query string or None
        """
        return result.get(ActionResultKey.QUERY.value)
    
    @staticmethod
    def get_results(result: Dict[str, Any]) -> List[Any]:
        """
        Extract results list from action result.
        
        :param result: Action result dictionary
        :returns: Results list (empty if not found)
        """
        return result.get(ActionResultKey.RESULTS.value, [])
    
    @staticmethod
    def is_retryable(result: Dict[str, Any]) -> bool:
        """
        Check if action result indicates it's retryable.
        
        :param result: Action result dictionary
        :returns: True if retryable
        """
        return bool(result.get(ActionResultKey.RETRYABLE.value))
