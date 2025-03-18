"""Service for managing memory graph operations and interactions."""

from typing import Any, Dict, List

from app.models.character import Character
from app.models.game_session import GameSession
from app.services.memory_graph import MemoryGraph


class MemoryService:
    """Service for managing memory graphs and memory-related operations."""

    def __init__(self, openai_client=None, memory_graph_config=None):
        """Initialize the memory service.

        Args:
            openai_client: OpenAI client instance for embeddings
            memory_graph_config: Configuration for memory graphs
        """
        self.openai_client = openai_client
        self.memory_graph_config = memory_graph_config or {
            "openai_client": self.openai_client,
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-4o-mini",
        }
        self.memory_graphs = {}

    def get_session_memory_graph(self, session_id: str) -> MemoryGraph:
        """Get or create a memory graph for the specified session.

        Args:
            session_id: The game session ID

        Returns:
            The memory graph for this session
        """
        if session_id not in self.memory_graphs:
            # Create a new memory graph for this session with
            # a session-specific storage directory
            storage_dir = f"data/memory_graph/{session_id}"
            memory_graph = MemoryGraph(
                openai_client=self.memory_graph_config["openai_client"],
                embedding_model=self.memory_graph_config["embedding_model"],
                llm_model=self.memory_graph_config["llm_model"],
                storage_dir=storage_dir,
            )

            # Attempt to load existing memories for this session if available
            try:
                memory_graph.load()
                print(
                    f"Loaded memory graph for session {session_id} with {len(memory_graph.nodes)} nodes"  # noqa: E501
                )
            except Exception as e:
                print(
                    f"No existing memory graph found for session {session_id} or error loading: {e}"  # noqa: E501
                )

            self.memory_graphs[session_id] = memory_graph

        return self.memory_graphs[session_id]

    def delete_memory_node(self, session_id: str, node_id: str) -> bool:
        """Delete a specific memory node from a session's memory graph.

        Args:
            session_id: The game session ID
            node_id: The ID of the node to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        memory_graph = self.get_session_memory_graph(session_id)
        return memory_graph.delete_node(node_id)

    def get_memory_nodes(
        self, session_id: str, sort_by: str = "timestamp", reverse: bool = True
    ) -> List[Dict]:
        """Get all memory nodes for a session.

        Args:
            session_id: The game session ID
            sort_by: Field to sort by ('timestamp', 'importance', 'type')
            reverse: If True, sort in descending order

        Returns:
            List of memory nodes
        """
        memory_graph = self.get_session_memory_graph(session_id)
        return memory_graph.get_all_nodes(sort_by, reverse)

    def get_node_relations(
        self, session_id: str, node_id: str
    ) -> List[Dict]:
        """Get relations for a specific node.

        Args:
            session_id: The game session ID
            node_id: The node ID to get relations for

        Returns:
            List of related nodes with relationship info
        """
        memory_graph = self.get_session_memory_graph(session_id)
        return memory_graph.get_node_relations(node_id)

    def add_memory_context(self, messages: List[Dict[str, Any]],
                           memory_graph: MemoryGraph, character: Character,
                           query: str = None):
        """Add relevant memory context to the messages list.

        Args:
            messages: List of messages to augment with memory
            memory_graph: Memory graph to query
            character: Player character
            query: Optional specific query to use (defaults to latest user message)

        Returns:
            None (modifies messages in place)
        """
        if not memory_graph or not character:
            return

        # Get the current user query if not provided
        if not query:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

        if not query:
            return

        # Query memory graph for relevant context
        relevant_context = memory_graph.get_relevant_context(
            query,
            node_limit=5,
            max_tokens=10000
        )

        if relevant_context:
            # Create a memory context message
            memory_context = "\n\n**Relevant Memory Context:**\n" + relevant_context

            # Find the system message to append memory context to
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i]["content"] += memory_context
                    break

    def add_interaction_to_memory(self, session: GameSession, character: Character, function_args: Dict):  # noqa: E501
        """Add an interaction record to the memory graph.

        Args:
            session: Current game session
            character: Player character
            function_args: Function arguments containing interaction details

        Returns:
            None
        """
        if not session or not character:
            return

        memory_graph = self.get_session_memory_graph(session.id)

        # Add general interaction record
        description = function_args.get("description", "").strip()
        if description:
            memory_graph.add_node(
                content=description,
                node_type="interaction",
                importance=0.7,
            )
