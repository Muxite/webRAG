"""Memory management for IdeaDAG nodes."""
import hashlib
import json
import logging
import uuid
from typing import List, Dict, Any, Optional
from agent.app.connector_chroma import ConnectorChroma


class MemoryManager:
    """Manages memory operations for IdeaDAG nodes."""
    
    def __init__(self, connector_chroma: ConnectorChroma, namespace: str, chunk_size: int = 800, chunk_overlap: int = 100):
        self.connector_chroma = connector_chroma
        self.namespace = namespace
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        namespace_hash = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]
        self.collection_name = f"mem_{namespace_hash}"
        self._logger = logging.getLogger(__name__)
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        node_context: Optional[Dict[str, Any]] = None,
        n_results: int = 5,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories from ChromaDB."""
        if not self.connector_chroma:
            return []
        
        try:
            if node_context:
                context_parts = []
                if node_context.get("title"):
                    context_parts.append(node_context["title"])
                if node_context.get("action"):
                    context_parts.append(f"action: {node_context['action']}")
                from agent.app.idea_policies.action_constants import ActionResultKey
                error = node_context.get(ActionResultKey.ERROR.value) or node_context.get("error")
                if error:
                    context_parts.append(f"error: {error}")
                if context_parts:
                    query = f"{query} {' '.join(context_parts)}"
            
            where = None
            if memory_type:
                where = {"memory_type": memory_type}
            
            results = await self.connector_chroma.query_chroma(
                collection=self.collection_name,
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
            
            if not results:
                return []
            
            memories = []
            documents = results.get("documents", [[]])[0] or []
            metadatas = results.get("metadatas", [[]])[0] or []
            distances = results.get("distances", [[]])[0] or []
            ids = results.get("ids", [[]])[0] or []
            
            for i, doc in enumerate(documents):
                memory = {
                    "content": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else 1.0,
                    "id": ids[i] if i < len(ids) else None,
                }
                memories.append(memory)
            
            self._logger.debug(f"Retrieved {len(memories)} memories for query: {query[:100]}")
            return memories
        
        except Exception as e:
            self._logger.warning(f"Failed to retrieve memories: {e}")
            return []
    
    async def retrieve_memories_split(
        self,
        query: str,
        node_context: Optional[Dict[str, Any]] = None,
        n_internal: int = 3,
        n_observations: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve memories split by type."""
        if not self.connector_chroma:
            return {"internal_thoughts": [], "observations": []}
        
        if node_context:
            context_parts = []
            if node_context.get("title"):
                context_parts.append(node_context["title"])
            if node_context.get("action"):
                context_parts.append(f"action: {node_context['action']}")
            if node_context.get("error"):
                context_parts.append(f"error: {node_context['error']}")
            if context_parts:
                query = f"{query} {' '.join(context_parts)}"
        
        internal_thoughts = await self.retrieve_relevant_memories(
            query=query,
            node_context=node_context,
            n_results=n_internal,
            memory_type="internal_thought",
        )
        observations = await self.retrieve_relevant_memories(
            query=query,
            node_context=node_context,
            n_results=n_observations,
            memory_type="observation",
        )
        
        return {"internal_thoughts": internal_thoughts, "observations": observations}
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                search_start = max(start, end - int(self.chunk_size * 0.2))
                sentence_end = max(
                    text.rfind(". ", search_start, end),
                    text.rfind(".\n", search_start, end),
                    text.rfind("! ", search_start, end),
                    text.rfind("? ", search_start, end),
                    text.rfind("\n\n", search_start, end),
                )
                if sentence_end > search_start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - self.chunk_overlap)
            if start >= len(text):
                break
        
        return chunks
    
    async def write_memory(
        self,
        content: str,
        node_id: str,
        node_title: str,
        action_type: Optional[str] = None,
        success: Optional[bool] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = None,
        links: Optional[List[str]] = None,
        link_contexts: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Write memory to ChromaDB with chunking."""
        if not self.connector_chroma:
            return False
        
        if not content or not content.strip():
            return False
        
        try:
            chunks = self._chunk_text(content)
            if not chunks:
                return False
            
            if not memory_type:
                from agent.app.idea_policies.base import IdeaActionType
                if action_type in (IdeaActionType.VISIT.value, IdeaActionType.SEARCH.value):
                    memory_type = "observation"
                else:
                    memory_type = "internal_thought"
            
            base_metadata = {
                "node_id": node_id,
                "node_title": node_title[:100],
                "namespace": self.namespace[:50],
                "memory_type": memory_type,
            }
            
            if action_type:
                base_metadata["action_type"] = action_type
            if success is not None:
                base_metadata["success"] = str(success)
            if error:
                base_metadata["error"] = error[:300]
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, str) and len(v) > 200:
                        base_metadata[k] = v[:200]
                    else:
                        base_metadata[k] = v
            
            ids = []
            metadatas = []
            documents = []
            
            links_list = links or []
            link_contexts_dict = link_contexts or {}
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{node_id}_{i:02d}"
                chunk_metadata = dict(base_metadata)
                chunk_metadata["chunk_index"] = str(i)
                chunk_metadata["total_chunks"] = str(len(chunks))
                
                chunk_with_links = chunk
                from agent.app.idea_policies.base import IdeaActionType
                if links_list and action_type == IdeaActionType.VISIT.value:
                    chunk_with_links = chunk + "\n\n--- Links from this page ---\n"
                    for link in links_list[:10]:
                        context = link_contexts_dict.get(link, "")
                        link_line = f"- {link}"
                        if context:
                            link_line += f" ({context})"
                        chunk_with_links += link_line + "\n"
                
                ids.append(chunk_id)
                metadatas.append(chunk_metadata)
                documents.append(chunk_with_links)
            
            success_flag = await self.connector_chroma.add_to_chroma(
                collection=self.collection_name,
                ids=ids,
                metadatas=metadatas,
                documents=documents,
            )
            
            if success_flag:
                self._logger.debug(f"Wrote {len(chunks)} memory chunk(s) for node {node_id}: {node_title[:50]}")
            
            return success_flag
        
        except Exception as e:
            self._logger.warning(f"Failed to write memory: {e}")
            return False
    
    async def write_node_result(
        self,
        node_id: str,
        node_title: str,
        action_type: Optional[str],
        result: Dict[str, Any],
    ) -> bool:
        """Write node execution result as memory with chunking and link association."""
        content_parts = []
        links_for_chunks = []
        
        from agent.app.idea_policies.action_constants import ActionResultKey, ActionResultExtractor
        from agent.app.idea_policies.base import IdeaActionType
        if ActionResultExtractor.is_success(result):
            if action_type == IdeaActionType.VISIT.value:
                url = ActionResultExtractor.get_url(result) or ""
                content_full = result.get("content_full", "")
                content_limited = result.get(ActionResultKey.CONTENT.value, "")
                links_full = result.get("links_full", [])
                links_limited = result.get("links", [])
                link_contexts = result.get(ActionResultKey.LINK_CONTEXTS.value, {})
                
                content_parts.append(f"Visited: {url}")
                
                if content_full:
                    content_parts.append(content_full)
                elif content_limited:
                    content_parts.append(content_limited)
                
                links_for_chunks = links_full or links_limited or []
                
                if links_full:
                    content_parts.append(f"\nLinks found on page ({len(links_full)} total):")
                    for link in links_full:
                        context = link_contexts.get(link, "")
                        link_line = f"- {link}"
                        if context:
                            link_line += f" ({context})"
                        content_parts.append(link_line)
                elif links_limited:
                    content_parts.append(f"\nLinks found on page ({len(links_limited)} shown):")
                    for link in links_limited:
                        context = link_contexts.get(link, "")
                        link_line = f"- {link}"
                        if context:
                            link_line += f" ({context})"
                        content_parts.append(link_line)
            elif action_type == IdeaActionType.SEARCH.value:
                query = ActionResultExtractor.get_query(result) or ""
                results = ActionResultExtractor.get_results(result)
                content_parts.append(f"Searched: {query}")
                if results:
                    content_parts.append(f"Found {len(results)} results:")
                    for r in results[:5]:  # Include more results
                        title = r.get("title", "")
                        url = r.get("url", "")
                        desc = r.get("description", "")[:200] if r.get("description") else ""
                        if title:
                            result_line = f"- {title}"
                            if url:
                                result_line += f" ({url})"
                            if desc:
                                result_line += f": {desc}"
                            content_parts.append(result_line)
        else:
            error = ActionResultExtractor.get_error(result, default="Unknown error")
            content_parts.append(f"Action failed: {error}")
            if action_type == IdeaActionType.VISIT.value:
                url = ActionResultExtractor.get_url(result) or ""
                if url:
                    content_parts.append(f"Failed URL: {url}")
            elif action_type == IdeaActionType.SEARCH.value:
                query = ActionResultExtractor.get_query(result) or ""
                if query:
                    content_parts.append(f"Failed query: {query}")
        
        content = "\n".join(content_parts)
        if not content.strip():
            return False
        
        compact_metadata = {"step_type": "node_result"}
        if action_type == IdeaActionType.VISIT.value and result.get(ActionResultKey.URL.value):
            compact_metadata["url"] = result.get(ActionResultKey.URL.value)[:200]
            links_full = result.get("links_full", [])
            links_count = len(links_full) if links_full else len(result.get("links", []))
            if links_count > 0:
                compact_metadata["links_count"] = str(links_count)
            content_full = result.get("content_full", "")
            if content_full:
                compact_metadata["content_length"] = str(len(content_full))
        elif action_type == IdeaActionType.SEARCH.value and result.get(ActionResultKey.QUERY.value):
            compact_metadata["query"] = result.get(ActionResultKey.QUERY.value)[:200]
        
        return await self.write_memory(
            content=content,
            node_id=node_id,
            node_title=node_title,
            action_type=action_type,
            success=ActionResultExtractor.is_success(result),
            error=ActionResultExtractor.get_error(result),
            metadata=compact_metadata,
            links=links_for_chunks,
            link_contexts=result.get(ActionResultKey.LINK_CONTEXTS.value, {}) if action_type == IdeaActionType.VISIT.value else {},
        )
    
    def format_memories_for_llm(self, memories: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        """Format memories for LLM prompts."""
        if not memories:
            return "No relevant memories found."
        
        formatted = []
        total_chars = 0
        
        for mem in memories:
            content = mem.get("content", "")
            metadata = mem.get("metadata", {})
            node_title = metadata.get("node_title", "Unknown")
            action_type = metadata.get("action_type", "")
            success = metadata.get("success", "")
            error = metadata.get("error", "")
            
            mem_text = f"[Memory: {node_title}"
            if action_type:
                mem_text += f", action={action_type}"
            if success:
                mem_text += f", success={success}"
            if error:
                mem_text += f", error={error[:100]}"
            mem_text += f"]\n{content}"
            
            if total_chars + len(mem_text) > max_chars:
                break
            
            formatted.append(mem_text)
            total_chars += len(mem_text)
        
        return "\n\n".join(formatted) if formatted else "No relevant memories found."
