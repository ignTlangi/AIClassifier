import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import sys
import os

def train_model(data_path, model_path, seed):
    # Load data
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train MLP with improved architecture
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),     # Reduced complexity
        activation='relu',               # ReLU activation for better gradient flow
        solver='adam',                   # Adam optimizer for better convergence
        alpha=0.001,                     # Increased L2 regularization
        batch_size=32,                   # Fixed batch size
        learning_rate='adaptive',        # Adaptive learning rate
        max_iter=1000,                   # Reduced iterations
        early_stopping=True,             # Enable early stopping
        validation_fraction=0.2,         # Increased validation set
        n_iter_no_change=20,            # More patience
        random_state=seed
    )
    
    # Train the model
    mlp.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_score = mlp.score(X_val_scaled, y_val)
    print(f"Validation accuracy: {val_score:.4f}")
    
    # Save model and scaler
    with open(model_path, 'wb') as f:
        pickle.dump((mlp, scaler), f)
    
    print("Model trained and saved successfully")

def predict(data_path, model_path):
    # Load model and scaler
    with open(model_path, 'rb') as f:
        mlp, scaler = pickle.load(f)
    
    # Load and scale data
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    X_scaled = scaler.transform(X)
    
    # Predict all instances
    predictions = mlp.predict(X_scaled)
    for pred in predictions:
        print(pred)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python mlp_classifier.py <data_path> <model_path> <seed> [--predict]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    seed = int(sys.argv[3])
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if len(sys.argv) > 4 and sys.argv[4] == "--predict":
        predict(data_path, model_path)
    else:
        train_model(data_path, model_path, seed) 