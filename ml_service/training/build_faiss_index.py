"""
Build FAISS index for semantic search
Uses sentence-transformers (all-MiniLM-L6-v2)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import pickle

import faiss
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
EVAL_DIR = Path(__file__).parent.parent / "eval"
SEED = 42

def load_data():
    """Load data for indexing"""
    print("Loading data...")
    
    listings_df = pd.read_csv(DATA_DIR / "listings.csv")
    
    # Combine title and description for embedding
    listings_df['text'] = listings_df['title'] + ' ' + listings_df['description']
    
    # Load heritage corpus
    heritage_records = []
    heritage_path = DATA_DIR / "heritage_corpus.jsonl"
    if heritage_path.exists():
        with open(heritage_path, 'r', encoding='utf-8') as f:
            for line in f:
                heritage_records.append(json.loads(line))
    
    return listings_df, heritage_records

def build_index(texts, model_name='all-MiniLM-L6-v2'):
    """Build FAISS index from texts"""
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity for normalized vectors)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index, model, embeddings

def save_index(index, model, ids, index_name, metadata=None):
    """Save FAISS index and metadata"""
    print(f"Saving {index_name} index...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = MODELS_DIR / f"{index_name}.index"
    faiss.write_index(index, str(index_path))
    
    # Save model
    model_path = MODELS_DIR / f"{index_name}_model"
    model.save(str(model_path))
    
    # Save IDs
    ids_path = MODELS_DIR / f"{index_name}_ids.pkl"
    with open(ids_path, "wb") as f:
        pickle.dump(ids, f)
    
    # Save metadata
    if metadata:
        metadata_path = MODELS_DIR / f"{index_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Index saved to {index_path}")
    print(f"Model saved to {model_path}")
    print(f"IDs saved to {ids_path}")

def evaluate_index(index, embeddings, sample_queries=10):
    """Evaluate index with sample queries"""
    print("Evaluating index...")
    
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sample some queries
    n_queries = min(sample_queries, len(embeddings))
    query_indices = np.random.choice(len(embeddings), n_queries, replace=False)
    query_embeddings = embeddings[query_indices]
    
    # Search
    k = 10
    D, I = index.search(query_embeddings, k)
    
    # Calculate recall@k
    recall_count = 0
    for i, query_idx in enumerate(query_indices):
        if query_idx in I[i]:
            recall_count += 1
    
    recall_at_k = recall_count / n_queries
    
    metrics = {
        "recall_at_10": float(recall_at_k),
        "num_queries": n_queries,
        "k": k
    }
    
    eval_path = EVAL_DIR / "faiss_eval.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Recall@10: {recall_at_k:.4f}")
    print(f"Evaluation saved to {eval_path}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FAISS Index Builder")
    print("=" * 60)
    
    # Load data
    listings_df, heritage_records = load_data()
    
    # Build listings index
    print("\nBuilding listings index...")
    listings_texts = listings_df['text'].tolist()
    listings_index, model, listings_embeddings = build_index(listings_texts)
    
    listings_ids = listings_df['id'].tolist()
    listings_metadata = {
        "type": "listings",
        "num_items": len(listings_df),
        "model": "all-MiniLM-L6-v2",
        "created_at": datetime.now().isoformat()
    }
    
    save_index(listings_index, model, listings_ids, "faiss_listings", listings_metadata)
    evaluate_index(listings_index, listings_embeddings)
    
    # Build heritage index if records exist
    if heritage_records:
        print("\nBuilding heritage index...")
        heritage_texts = [f"{r['title']} {r['cultural_context']} {r['craft']}" for r in heritage_records]
        heritage_index, _, heritage_embeddings = build_index(heritage_texts)
        
        heritage_ids = [r['id'] for r in heritage_records]
        heritage_metadata = {
            "type": "heritage",
            "num_items": len(heritage_records),
            "model": "all-MiniLM-L6-v2",
            "created_at": datetime.now().isoformat()
        }
        
        save_index(heritage_index, model, heritage_ids, "faiss_heritage", heritage_metadata)
        evaluate_index(heritage_index, heritage_embeddings)
    
    print("\n" + "=" * 60)
    print("Index building complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
