"""Radix cache for prefix caching.

Simplified from sglang.srt.mem_cache.radix_cache.RadixCache
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SimTreeNode:
    """Node in radix tree for prefix caching.

    Attributes:
        key: Token sequence for this node
        value: KV cache indices for these tokens
        children: Child nodes indexed by first token
        parent: Parent node
        lock_ref: Reference count for locking
        last_access_time: Last time this node was accessed
    """

    key: List[int] = field(default_factory=list)
    value: List[int] = field(default_factory=list)
    children: Dict[int, SimTreeNode] = field(default_factory=dict)
    parent: Optional[SimTreeNode] = None
    lock_ref: int = 0
    last_access_time: float = 0.0

    def __post_init__(self):
        """Initialize computed fields."""
        if not hasattr(self, 'key'):
            self.key = []
        if not hasattr(self, 'value'):
            self.value = []


class SimRadixCache:
    """Radix cache for prefix matching.

    Simplified version of SGLang's RadixCache that supports prefix
    matching but uses simplified eviction logic.
    """

    def __init__(
        self,
        disable: bool = False,
        eviction_strategy: str = "lru"
    ):
        """Initialize radix cache.

        Args:
            disable: Whether to disable caching
            eviction_strategy: Eviction strategy ("lru", "lfu", "mru")
        """
        self.disable = disable
        self.eviction_strategy = eviction_strategy
        self.root = SimTreeNode()
        self.evictable_size = 0

    def match_prefix(
        self,
        token_ids: List[int]
    ) -> Tuple[List[int], int, Optional[SimTreeNode]]:
        """Find longest matching prefix in cache.

        Args:
            token_ids: Token sequence to match

        Returns:
            Tuple of:
            - prefix_indices: KV cache indices for matched prefix
            - matched_length: Number of tokens matched
            - last_node: Last matching node in tree
        """
        if self.disable or not token_ids:
            return [], 0, None

        node = self.root
        matched_indices = []
        matched_length = 0
        remaining_tokens = token_ids[:]

        while remaining_tokens:
            first_token = remaining_tokens[0]

            if first_token not in node.children:
                break

            child = node.children[first_token]

            # Check how much of child.key matches remaining_tokens
            match_len = 0
            for i, token in enumerate(child.key):
                if i >= len(remaining_tokens):
                    break
                if token != remaining_tokens[i]:
                    break
                match_len += 1

            if match_len == 0:
                break

            # Matched this child
            matched_indices.extend(child.value[:match_len])
            matched_length += match_len
            remaining_tokens = remaining_tokens[match_len:]

            if match_len < len(child.key):
                # Partial match of child
                break

            node = child

        return matched_indices, matched_length, node if matched_length > 0 else None

    def insert(
        self,
        token_ids: List[int],
        kv_indices: List[int],
        last_node: Optional[SimTreeNode] = None
    ) -> SimTreeNode:
        """Insert token sequence into radix cache.

        Args:
            token_ids: Token sequence to insert
            kv_indices: KV cache indices for these tokens
            last_node: Last matched node (for incremental insertion)

        Returns:
            Leaf node after insertion
        """
        if self.disable or not token_ids:
            return self.root

        node = last_node if last_node is not None else self.root
        remaining_tokens = token_ids[:]
        remaining_indices = kv_indices[:]

        while remaining_tokens:
            first_token = remaining_tokens[0]

            if first_token in node.children:
                # Child exists, merge or split
                child = node.children[first_token]

                # For simplicity, just continue to child
                # (Full implementation would handle splits)
                node = child
                overlap = min(len(child.key), len(remaining_tokens))
                remaining_tokens = remaining_tokens[overlap:]
                remaining_indices = remaining_indices[overlap:]
            else:
                # Create new child
                new_node = SimTreeNode(
                    key=remaining_tokens[:],
                    value=remaining_indices[:],
                    parent=node
                )
                node.children[first_token] = new_node
                self.evictable_size += len(remaining_tokens)
                return new_node

        return node

    def cache_unfinished_req(
        self,
        token_ids: List[int],
        kv_indices: List[int],
        num_tokens_done: int,
    ) -> None:
        """Cache partially-prefilled tokens for a chunked request.

        Mirrors RadixCache.cache_unfinished_req: inserts the already-computed
        portion of a chunked request into the tree so it can be reused as a
        prefix in the next scheduling round.

        Args:
            token_ids: Full token sequence
            kv_indices: KV cache indices for computed portion
            num_tokens_done: Number of tokens actually computed
        """
        if self.disable or num_tokens_done <= 0:
            return

        # Insert only the computed portion into the cache
        self.insert(
            token_ids=token_ids[:num_tokens_done],
            kv_indices=kv_indices[:num_tokens_done],
        )

    def evict(self, num_tokens: int, evict_callback=None) -> int:
        """Evict tokens from cache using eviction strategy.

        Args:
            num_tokens: Number of tokens to evict
            evict_callback: Optional callback for eviction notification

        Returns:
            Number of tokens actually evicted
        """
        if self.disable or num_tokens <= 0:
            return 0

        evicted = 0

        # Simple LRU eviction: find evictable leaves
        leaves = self._find_evictable_leaves()

        # Sort by last access time (LRU)
        if self.eviction_strategy == "lru":
            leaves.sort(key=lambda n: n.last_access_time)
        elif self.eviction_strategy == "mru":
            leaves.sort(key=lambda n: n.last_access_time, reverse=True)

        # Evict leaves until we have enough space
        for leaf in leaves:
            if evicted >= num_tokens:
                break

            # Remove leaf
            if leaf.parent and len(leaf.key) > 0:
                first_token = leaf.key[0]
                if first_token in leaf.parent.children:
                    del leaf.parent.children[first_token]
                    tokens_freed = len(leaf.value)
                    evicted += tokens_freed
                    self.evictable_size -= tokens_freed

                    if evict_callback:
                        evict_callback(leaf.value)

        return evicted

    def _find_evictable_leaves(self) -> List[SimTreeNode]:
        """Find evictable leaf nodes.

        Returns:
            List of leaf nodes that can be evicted
        """
        leaves = []

        def dfs(node: SimTreeNode):
            if node.lock_ref > 0:
                return

            if not node.children:
                # Leaf node
                if node != self.root:
                    leaves.append(node)
            else:
                for child in node.children.values():
                    dfs(child)

        dfs(self.root)
        return leaves

    def total_size(self) -> int:
        """Get total number of cached tokens.

        Returns:
            Total cached tokens
        """
        return self.evictable_size

    def reset(self):
        """Reset cache to initial state."""
        self.root = SimTreeNode()
        self.evictable_size = 0
