import uuid
import random
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

class MemoryNode:
    """A node in the memory graph representing a game event, character, location, or item."""
    
    def __init__(self, content: str, node_type: str, importance: float = 0.5):
        """
        Initialize a new memory node.
        
        Args:
            content: The full content of the memory
            node_type: Type of memory (event, character, location, item)
            importance: Initial importance score (0-1)
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = None  # Will be populated when get_embedding is called
        self.summary = None    # Will be populated when generate_summary is called
        self.metadata = {
            'type': node_type,
            'timestamp': datetime.utcnow().isoformat(),
            'importance': importance,
            'references': set(),  # IDs of nodes referenced by this node
            'access_count': 0
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        node_dict = {
            'id': self.id,
            'content': self.content,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'summary': self.summary,
            'metadata': self.metadata.copy()
        }
        # Convert set to list for JSON serialization
        node_dict['metadata']['references'] = list(self.metadata['references'])
        return node_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Create a node from a dictionary."""
        node = cls(data['content'], data['metadata']['type'], data['metadata']['importance'])
        node.id = data['id']
        if data['embedding'] is not None:
            node.embedding = np.array(data['embedding'])
        node.summary = data['summary']
        node.metadata = data['metadata']
        # Convert list back to set
        node.metadata['references'] = set(node.metadata['references'])
        return node


class MemoryGraph:
    """Graph-based memory system for maintaining narrative coherence in RPG adventures."""
    
    def __init__(self, openai_client=None, embedding_model="text-embedding-3-large", 
                 llm_model="gpt-4o-mini", storage_dir="data/memory_graph"):
        """
        Initialize the memory graph.
        
        Args:
            openai_client: OpenAI client instance
            embedding_model: Model to use for embeddings
            llm_model: Model to use for LLM operations
            storage_dir: Directory to persist memory graph
        """
        self.nodes: Dict[str, MemoryNode] = {}
        self.relations: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.openai_client = openai_client
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.storage_dir = storage_dir
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI API."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return np.zeros(1536)  # Default embedding size for ada-002
    
    def generate_summary(self, content: str, max_tokens: int = 200) -> str:
        """Generate a concise summary of the content."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Summarize the following game event in a single concise sentence:"},
                    {"role": "user", "content": content}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            # Return truncated content as fallback
            return content[:200] + "..." if len(content) > 200 else content

    def add_node(self, content: str, node_type: str, importance: float = 0.5) -> str:
        """
        Add a new node to the memory graph.
        
        Args:
            content: The content of the memory
            node_type: Type of node (event, character, location, item)
            importance: Initial importance score (0-1)
            
        Returns:
            The ID of the newly created node
        """
        node = MemoryNode(content, node_type, importance)
        node.embedding = self.get_embedding(content)
        node.summary = self.generate_summary(content)
        self.nodes[node.id] = node
        
        # Discover relations to existing nodes
        if len(self.nodes) > 1:
            self._discover_relations(node)
        
        # Persist changes
        self._save_node(node)
        
        return node.id
    
    def _discover_relations(self, new_node: MemoryNode, max_sample_size: int = 10) -> None:
        """
        Use LLM to establish contextual connections between new node and existing ones.
        
        Args:
            new_node: The newly added node
            max_sample_size: Maximum number of existing nodes to sample for relation discovery
        """
        # Sample existing nodes (limited to avoid overwhelming the LLM)
        existing_nodes = random.sample(
            list(self.nodes.values()), 
            min(max_sample_size, len(self.nodes) - 1)
        )
        
        # Skip if no existing nodes
        if not existing_nodes:
            return
        
        # Construct prompt for relation discovery
        node_descriptions = "\n".join([
            f"[{i}] ID: {node.id}, Type: {node.metadata['type']}, Summary: {node.summary}"
            for i, node in enumerate(existing_nodes)
        ])
        
        prompt = f"""Analyze the following new game event and identify any relationships with existing memories:

New Event: {new_node.summary}
Event Type: {new_node.metadata['type']}
Event ID: {new_node.id}

Existing Memories:
{node_descriptions}

For each relationship you find, output a SINGLE line in this EXACT format:
RELATION_TYPE|{new_node.id}|[existing_memory_id]

Valid relation types: REFERENCES, CONTINUES, CONTRADICTS, CAUSES, INVOLVES_SAME_CHARACTER, INVOLVES_SAME_LOCATION

Output ONLY the relationship lines, nothing else. If no relationships exist, output "NO_RELATIONS".
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            
            relation_text = response.choices[0].message.content.strip()
            if relation_text == "NO_RELATIONS":
                return
                
            # Parse relations
            for line in relation_text.split('\n'):
                if not line.strip():
                    continue
                    
                try:
                    relation_type, src_id, dst_id = line.strip().split('|')
                    # Validate IDs
                    if src_id not in self.nodes or dst_id not in self.nodes:
                        for node in existing_nodes:
                            if node.id in dst_id:  # Handle if the model included brackets
                                dst_id = node.id
                                break
                    
                    if src_id in self.nodes and dst_id in self.nodes:
                        self.relations[relation_type].append((src_id, dst_id))
                        # Update references
                        self.nodes[src_id].metadata['references'].add(dst_id)
                except ValueError:
                    print(f"Invalid relation format: {line}")
                    continue
                    
        except Exception as e:
            print(f"Error discovering relations: {str(e)}")
    
    def get_relevant_context(self, query: str, node_limit: int = 5, 
                             max_tokens: int = 10000) -> str:
        """
        Retrieve relevant memories based on the current situation.
        
        Uses a hybrid approach:
        1. Semantic similarity search
        2. Graph-based importance
        3. Recency and access frequency
        
        Args:
            query: The current game situation or query
            node_limit: Maximum number of nodes to return
            max_tokens: Approximate token budget for context
            
        Returns:
            Formatted context string with relevant memories
        """
        query_embedding = self.get_embedding(query)
        
        # Find semantically similar nodes
        similarity_scores = {}
        for node_id, node in self.nodes.items():
            if node.embedding is not None:
                similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
                similarity_scores[node_id] = similarity
        
        # Calculate final relevance scores (combining multiple factors)
        relevance_scores = {}
        current_time = datetime.utcnow()
        
        for node_id, node in self.nodes.items():
            # Skip nodes without embeddings
            if node_id not in similarity_scores:
                continue
                
            # Base score is semantic similarity
            score = similarity_scores[node_id] * 0.5
            
            # Add importance factor
            score += node.metadata['importance'] * 0.3
            
            # Add recency factor (inverse time difference, normalized)
            node_time = datetime.fromisoformat(node.metadata['timestamp'])
            time_diff = (current_time - node_time).total_seconds()
            recency = 1.0 / (1.0 + 0.0001 * time_diff)  # Decay function
            score += recency * 0.1
            
            # Add access count factor (normalized)
            access_factor = min(node.metadata['access_count'] / 10.0, 1.0) * 0.1
            score += access_factor
            
            relevance_scores[node_id] = score
        
        # Get top-ranked nodes
        ranked_nodes = sorted(
            relevance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:node_limit]
        
        # Format selected nodes as context string
        context_parts = []
        total_tokens = 0
        token_estimate = 0
        
        for node_id, score in ranked_nodes:
            node = self.nodes[node_id]
            
            # Increment access count
            node.metadata['access_count'] += 1
            self._save_node(node)
            
            # Estimate tokens (rough approximation)
            node_tokens = len(node.content.split()) * 1.3
            
            if total_tokens + node_tokens > max_tokens:
                # Use summary instead of full content if we're approaching the limit
                summary_tokens = len(node.summary.split()) * 1.3
                if total_tokens + summary_tokens <= max_tokens:
                    memory_text = f"Memory [{node.metadata['type']}]: {node.summary}"
                    token_estimate = summary_tokens
                else:
                    # Skip if even summary would exceed the limit
                    continue
            else:
                memory_text = f"Memory [{node.metadata['type']}]: {node.content}"
                token_estimate = node_tokens
            
            context_parts.append(memory_text)
            total_tokens += token_estimate
        
        if not context_parts:
            return "No relevant memories found."
            
        return "\n\n".join(context_parts)
    
    def _save_node(self, node: MemoryNode) -> None:
        """Save a node to disk."""
        node_path = os.path.join(self.storage_dir, f"{node.id}.json")
        with open(node_path, 'w') as f:
            json.dump(node.to_dict(), f)
    
    def _save_relations(self) -> None:
        """Save all relations to disk."""
        relations_path = os.path.join(self.storage_dir, "relations.json")
        # Convert to serializable format
        serializable_relations = {
            rel_type: [(src, dst) for src, dst in rel_list]
            for rel_type, rel_list in self.relations.items()
        }
        with open(relations_path, 'w') as f:
            json.dump(serializable_relations, f)
    
    def save(self) -> None:
        """Save the entire memory graph to disk."""
        for node in self.nodes.values():
            self._save_node(node)
        self._save_relations()
    
    def load(self) -> None:
        """Load the memory graph from disk."""
        # Clear existing data
        self.nodes.clear()
        self.relations.clear()
        
        # Load nodes
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json') and filename != "relations.json":
                node_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(node_path, 'r') as f:
                        node_data = json.load(f)
                        node = MemoryNode.from_dict(node_data)
                        self.nodes[node.id] = node
                except Exception as e:
                    print(f"Error loading node {filename}: {str(e)}")
        
        # Load relations
        relations_path = os.path.join(self.storage_dir, "relations.json")
        if os.path.exists(relations_path):
            try:
                with open(relations_path, 'r') as f:
                    relations_data = json.load(f)
                    for rel_type, rel_list in relations_data.items():
                        self.relations[rel_type] = rel_list
            except Exception as e:
                print(f"Error loading relations: {str(e)}")
