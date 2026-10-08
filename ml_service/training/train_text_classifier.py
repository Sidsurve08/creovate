"""
Train text classifier for craft/category classification
Uses DistilBERT for fast training on CPU
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import os

import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
EVAL_DIR = Path(__file__).parent.parent / "eval"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def load_data(fast=False):
    """Load and prepare training data"""
    print("Loading data...")
    
    train_df = pd.read_csv(DATA_DIR / "listings_train.csv")
    val_df = pd.read_csv(DATA_DIR / "listings_val.csv")
    test_df = pd.read_csv(DATA_DIR / "listings_test.csv")
    
    if fast:
        # Subsample for fast training
        train_df = train_df.sample(n=min(500, len(train_df)), random_state=SEED)
        val_df = val_df.sample(n=min(100, len(val_df)), random_state=SEED)
        test_df = test_df.sample(n=min(100, len(test_df)), random_state=SEED)
        print(f"Fast mode: Using {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    # Combine title and description for text input
    train_df['text'] = train_df['title'] + ' ' + train_df['description']
    val_df['text'] = val_df['title'] + ' ' + val_df['description']
    test_df['text'] = test_df['title'] + ' ' + test_df['description']
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['category'])
    val_df['label'] = label_encoder.transform(val_df['category'])
    test_df['label'] = label_encoder.transform(test_df['category'])
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return train_df, val_df, test_df, label_encoder

def tokenize_data(texts, tokenizer, max_length=128):
    """Tokenize text data"""
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(train_df, val_df, label_encoder, fast=False):
    """Train DistilBERT classifier"""
    print("Initializing model...")
    
    # Load tokenizer and model
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=128
        )
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    train_dataset = train_dataset.remove_columns(['text', 'id', 'title', 'description', 'category', 'craft', 'materials', 'technique', 'region', 'state', 'language', 'dimensions', 'weight_g', 'complexity', 'handmade', 'is_rental', 'price_inr', 'rating', 'num_sales', 'image_path', 'creator_id', 'source'])
    val_dataset = val_dataset.remove_columns(['text', 'id', 'title', 'description', 'category', 'craft', 'materials', 'technique', 'region', 'state', 'language', 'dimensions', 'weight_g', 'complexity', 'handmade', 'is_rental', 'price_inr', 'rating', 'num_sales', 'image_path', 'creator_id', 'source'])
    
    train_dataset = train_dataset.rename_column('label', 'labels')
    val_dataset = val_dataset.rename_column('label', 'labels')
    
    train_dataset.set_format('torch')
    val_dataset.set_format('torch')
    
    # Training arguments
    if fast:
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=2,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=5e-5,
        )
    else:
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=3e-5,
        )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating...")
    eval_results = trainer.evaluate()
    
    return trainer, eval_results, tokenizer

def save_model(trainer, tokenizer, label_encoder, eval_results, fast=False):
    """Save model and artifacts"""
    print("Saving model...")
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_dir = MODELS_DIR / "category_distilbert"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    
    # Save label encoder
    import pickle
    with open(model_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    # Save configuration
    config = {
        "model_name": "distilbert-base-multilingual-cased",
        "num_labels": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "trained_at": datetime.now().isoformat(),
        "fast_mode": fast,
        "seed": SEED
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save evaluation results
    eval_results_path = EVAL_DIR / "text_classifier.json"
    with open(eval_results_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Model saved to {model_dir}")
    print(f"Evaluation results saved to {eval_results_path}")
    
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Train text classifier")
    parser.add_argument("--fast", action="store_true", help="Fast training with subsampled data")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Text Classifier Training")
    print("=" * 60)
    
    # Load data
    train_df, val_df, test_df, label_encoder = load_data(fast=args.fast)
    
    # Train model
    trainer, eval_results, tokenizer = train_model(train_df, val_df, label_encoder, fast=args.fast)
    
    # Save model
    model_dir = save_model(trainer, tokenizer, label_encoder, eval_results, fast=args.fast)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model: {model_dir}")
    print(f"Evaluation results: {eval_results}")
    print("=" * 60)

if __name__ == "__main__":
    main()
