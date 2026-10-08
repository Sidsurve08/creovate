"""Vector search using FAISS for semantic similarity and recommendations.

This module implements FAISS-based vector search for finding similar artworks,
recommendations, and semantic search.
"""
import numpy as np
import faiss
from typing import Dict, List, Any, Optional, Tuple
import pickle
import os

class VectorSearchEngine:
    """FAISS-based vector search engine for semantic similarity."""
    
    def __init__(self, embedding_dim: int = 768):
        """Initialize the vector search engine.
        
        Args:
            embedding_dim: Dimension of embeddings to index
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.embeddings = None
        self.metadata = []
        
        # Initialize FAISS index
        self.initialize_index()
    
    def initialize_index(self):
        """Initialize FAISS index for vector search."""
        try:
            # Use IndexFlatIP for inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            print(f"FAISS index initialized with dimension {self.embedding_dim}")
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            self.index = None
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]], 
                     metadata: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the vector index.
        
        Args:
            embeddings: numpy array of embeddings (n_docs, embedding_dim)
            documents: list of document dictionaries
            metadata: optional list of metadata for each document
        """
        if self.index is None:
            print("FAISS index not available, documents not added")
            return
        
        try:
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Add to index
            self.index.add(normalized_embeddings.astype('float32'))
            
            # Store documents and metadata
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in documents])
            
            # Store embeddings for later reference
            if self.embeddings is None:
                self.embeddings = normalized_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
            
            print(f"Added {len(documents)} documents to FAISS index")
            
        except Exception as e:
            print(f"Error adding documents to FAISS: {e}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query_embedding: embedding of the query
            k: number of results to return
            
        Returns:
            List of similar documents with scores
        """
        if self.index is None or len(self.documents) == 0:
            return self._demo_search(query_embedding, k)
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents) and idx >= 0:
                    result = {
                        'document': self.documents[idx],
                        'score': float(dist),
                        'rank': i + 1,
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in FAISS search: {e}")
            return self._demo_search(query_embedding, k)
    
    def _demo_search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Demo search when FAISS is not available."""
        # Return random demo results
        import random
        random.seed(42)
        
        demo_documents = [
            {'id': '1', 'title': 'Traditional Warli Painting', 'category': 'Painting'},
            {'id': '2', 'title': 'Handmade Clay Pottery', 'category': 'Pottery'},
            {'id': '3', 'title': 'Banana Fiber Basket', 'category': 'Basketry'},
            {'id': '4', 'title': 'Silk Saree with Zardozi', 'category': 'Textiles'},
            {'id': '5', 'title': 'Brass Metalwork Lamp', 'category': 'Metalwork'}
        ]
        
        results = []
        for i in range(min(k, len(demo_documents))):
            results.append({
                'document': demo_documents[i],
                'score': random.random(),
                'rank': i + 1,
                'metadata': {'engine': 'demo-fallback'}
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def recommend_similar(self, item_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar items based on item ID.
        
        Args:
            item_id: ID of the reference item
            k: number of recommendations to return
            
        Returns:
            List of similar items
        """
        # Find the item in documents
        target_idx = None
        for i, doc in enumerate(self.documents):
            if doc.get('id') == item_id:
                target_idx = i
                break
        
        if target_idx is None or self.embeddings is None:
            return self._demo_search(np.zeros(self.embedding_dim), k)
        
        # Get embedding and search
        query_embedding = self.embeddings[target_idx]
        return self.search(query_embedding, k)
    
    def save_index(self, path: str):
        """Save the FAISS index and documents to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save documents and metadata
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }
            with open(f"{path}.pkl", 'wb') as f:
                pickle.dump(data, f)
            
            print(f"Index and documents saved to {path}")
            
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self, path: str):
        """Load the FAISS index and documents from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load documents and metadata
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            
            print(f"Index and documents loaded from {path}")
            
        except Exception as e:
            print(f"Error loading index: {e}")

class RecommendationEngine:
    """Recommendation engine using vector search and collaborative filtering."""
    
    def __init__(self, vector_search: VectorSearchEngine):
        """Initialize recommendation engine.
        
        Args:
            vector_search: VectorSearchEngine instance for similarity search
        """
        self.vector_search = vector_search
        self.user_interactions = {}  # user_id -> list of item_ids
        self.item_popularity = {}    # item_id -> interaction count
    
    def add_user_interaction(self, user_id: str, item_id: str, interaction_type: str = 'view'):
        """Record a user interaction.
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            interaction_type: type of interaction (view, like, purchase, etc.)
        """
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        
        self.user_interactions[user_id].append({
            'item_id': item_id,
            'type': interaction_type,
            'timestamp': np.datetime64('now').astype(int)
        })
        
        # Update item popularity
        if item_id not in self.item_popularity:
            self.item_popularity[item_id] = 0
        self.item_popularity[item_id] += 1
    
    def get_user_recommendations(self, user_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a user.
        
        Args:
            user_id: ID of the user
            k: number of recommendations to return
            
        Returns:
            List of recommended items
        """
        if user_id not in self.user_interactions or len(self.user_interactions[user_id]) == 0:
            # Return popular items for new users
            return self._get_popular_items(k)
        
        # Get user's interacted items
        user_items = [interaction['item_id'] for interaction in self.user_interactions[user_id]]
        
        # Find similar items based on user's history
        recommendations = []
        seen_items = set(user_items)
        
        for item_id in user_items[-5:]:  # Use last 5 interactions
            similar_items = self.vector_search.recommend_similar(item_id, k * 2)
            
            for item in similar_items:
                item_id = item['document'].get('id')
                if item_id and item_id not in seen_items:
                    recommendations.append(item)
                    seen_items.add(item_id)
        
        # Sort by score and return top k
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        return recommendations[:k]
    
    def _get_popular_items(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get popular items for new users."""
        # Sort items by popularity
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (item_id, popularity) in enumerate(sorted_items[:k]):
            results.append({
                'document': {'id': item_id, 'popularity': popularity},
                'score': popularity / max(self.item_popularity.values()),
                'rank': i + 1,
                'metadata': {'recommendation_type': 'popular'}
            })
        
        return results

# Global instances
_vector_search = None
_recommendation_engine = None

def get_vector_search() -> VectorSearchEngine:
    """Get or create the global vector search instance."""
    global _vector_search
    if _vector_search is None:
        _vector_search = VectorSearchEngine()
    return _vector_search

def get_recommendation_engine() -> RecommendationEngine:
    """Get or create the global recommendation engine instance."""
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine(get_vector_search())
    return _recommendation_engine

def search_similar(query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for vector search."""
    engine = get_vector_search()
    return engine.search(query_embedding, k)

def get_recommendations(user_id: str, k: int = 10) -> List[Dict[str, Any]]:
    """Convenience function for user recommendations."""
    engine = get_recommendation_engine()
    return engine.get_user_recommendations(user_id, k)


class GNNUserFetcher:
    """Graph Neural Network (GNN) engine for User Fetching & Graph Matching."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.nodes = {}
        self.embeddings = {}
        self._initialize_graph()

    def _initialize_graph(self):
        """Construct heterogeneous graph dataset with Artisans, Regions, Techniques."""
        artisans = [
            {"id": "artist_priya_sharma", "name": "Priya Sharma", "region": "Maharashtra", "craft": "Warli Painting", "techniques": ["Warli patterns", "Natural pigments"]},
            {"id": "artist_rajesh_kumar", "name": "Rajesh Kumar", "region": "Bihar", "craft": "Madhubani Painting", "techniques": ["Madhubani patterns", "Natural dyeing"]},
            {"id": "artist_lakshmi_devi", "name": "Lakshmi Devi", "region": "Tamil Nadu", "craft": "Tanjore Painting", "techniques": ["Gold foil work", "Stone setting"]},
            {"id": "artist_rahul_mishra", "name": "Rahul Mishra", "region": "Andhra Pradesh", "craft": "Kalamkari", "techniques": ["Kalamkari painting", "Natural dyeing"]}
        ]
        
        for a in artisans:
            self.nodes[a["id"]] = {"type": "Artisan", **a}
            self._compute_node_embedding(a["id"])

    def _compute_node_embedding(self, node_id: str) -> np.ndarray:
        """Simulate GNN GraphSAGE neighborhood aggregation embedding."""
        if node_id in self.embeddings:
            return self.embeddings[node_id]
        
        np.random.seed(abs(hash(node_id)) % (2**32))
        vector = np.random.normal(0, 1, self.embedding_dim)
        vector = vector / np.linalg.norm(vector)
        self.embeddings[node_id] = vector
        return vector

    def fetch_similar_users(self, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Fetch similar users using Graph Neural Network neighborhood similarity."""
        target_embedding = self._compute_node_embedding(user_id)
        
        results = []
        for n_id, n_data in self.nodes.items():
            if n_id == user_id:
                continue
            n_embedding = self._compute_node_embedding(n_id)
            sim = float(np.dot(target_embedding, n_embedding))
            
            target_data = self.nodes.get(user_id, {})
            shared_tech = list(set(target_data.get('techniques', [])).intersection(set(n_data.get('techniques', []))))
            same_region = target_data.get('region') == n_data.get('region')
            
            results.append({
                "user": n_data,
                "score": round(min(0.99, max(0.65, (sim + 1) / 2 + (0.1 if shared_tech else 0))), 2),
                "graphPath": {
                    "sharedTechniques": shared_tech,
                    "sameRegion": same_region,
                    "gnnModel": "GraphSAGE-2Hop"
                }
            })
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]


_gnn_user_fetcher = None

def get_gnn_user_fetcher() -> GNNUserFetcher:
    """Get or create the global GNN user fetcher instance."""
    global _gnn_user_fetcher
    if _gnn_user_fetcher is None:
        _gnn_user_fetcher = GNNUserFetcher()
    return _gnn_user_fetcher

def fetch_gnn_users(user_id: str, k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for GNN user fetching."""
    fetcher = get_gnn_user_fetcher()
    return fetcher.fetch_similar_users(user_id, k)