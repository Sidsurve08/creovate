"""
Train price prediction model using XGBoost
Outputs min/suggested/max price with confidence intervals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent / "models"
EVAL_DIR = Path(__file__).parent.parent / "eval"
SEED = 42

np.random.seed(SEED)

def load_data(fast=False):
    """Load and prepare training data"""
    print("Loading data...")
    
    train_df = pd.read_csv(DATA_DIR / "listings_train.csv")
    val_df = pd.read_csv(DATA_DIR / "listings_val.csv")
    test_df = pd.read_csv(DATA_DIR / "listings_test.csv")
    
    if fast:
        train_df = train_df.sample(n=min(500, len(train_df)), random_state=SEED)
        val_df = val_df.sample(n=min(100, len(val_df)), random_state=SEED)
        test_df = test_df.sample(n=min(100, len(test_df)), random_state=SEED)
        print(f"Fast mode: Using {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
    
    return train_df, val_df, test_df

def prepare_features(df, label_encoders=None, fit=True):
    """Prepare features for training"""
    df = df.copy()
    
    # Encode categorical features
    categorical_cols = ['category', 'craft', 'region', 'technique']
    
    if label_encoders is None:
        label_encoders = {}
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            if fit:
                df[col + '_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen labels
                df[col + '_encoded'] = df[col].astype(str).map(
                    lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
                )
        else:
            if fit:
                df[col + '_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col + '_encoded'] = df[col].astype(str).map(
                    lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
                )
    
    # Feature columns
    feature_cols = ['complexity', 'weight_g', 'handmade', 'is_rental']
    for col in categorical_cols:
        if col + '_encoded' in df.columns:
            feature_cols.append(col + '_encoded')
    
    # Filter to available columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df['price_inr']
    
    return X, y, label_encoders, feature_cols

def train_model(train_df, val_df, fast=False):
    """Train XGBoost price model"""
    print("Training XGBoost model...")
    
    # Prepare features
    X_train, y_train, label_encoders, feature_cols = prepare_features(train_df, fit=True)
    X_val, y_val, _, _ = prepare_features(val_df, label_encoders=label_encoders, fit=False)
    
    # Train main model (median price prediction)
    model = xgb.XGBRegressor(
        n_estimators=100 if fast else 300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Train quantile models for min/max price
    model_min = xgb.XGBRegressor(
        n_estimators=100 if fast else 300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    model_min.fit(X_train, y_train * 0.8)  # 10th percentile approximation
    
    model_max = xgb.XGBRegressor(
        n_estimators=100 if fast else 300,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    model_max.fit(X_train, y_train * 1.2)  # 90th percentile approximation
    
    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, model_min, model_max, label_encoders, feature_cols, metrics

def save_model(model, model_min, model_max, label_encoders, feature_cols, metrics, fast=False):
    """Save model and artifacts"""
    print("Saving model...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_dir = MODELS_DIR / "price_xgb"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    model.save_model(str(model_dir / "price_model.json"))
    model_min.save_model(str(model_dir / "price_model_min.json"))
    model_max.save_model(str(model_dir / "price_model_max.json"))
    
    # Save label encoders
    with open(model_dir / "label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    
    # Save feature columns
    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    
    # Save configuration
    config = {
        "model_type": "XGBoostRegressor",
        "trained_at": datetime.now().isoformat(),
        "fast_mode": fast,
        "seed": SEED,
        "feature_cols": feature_cols
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save evaluation results
    eval_results_path = EVAL_DIR / "price_model.json"
    with open(eval_results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to {model_dir}")
    print(f"Evaluation results saved to {eval_results_path}")
    
    return model_dir

def main():
    parser = argparse.ArgumentParser(description="Train price prediction model")
    parser.add_argument("--fast", action="store_true", help="Fast training with subsampled data")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Price Prediction Model Training")
    print("=" * 60)
    
    # Load data
    train_df, val_df, test_df = load_data(fast=args.fast)
    
    # Train model
    model, model_min, model_max, label_encoders, feature_cols, metrics = train_model(train_df, val_df, fast=args.fast)
    
    # Save model
    model_dir = save_model(model, model_min, model_max, label_encoders, feature_cols, metrics, fast=args.fast)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model: {model_dir}")
    print(f"Metrics: {metrics}")
    print("=" * 60)

if __name__ == "__main__":
    main()
