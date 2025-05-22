import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import sys
import os

def train_model(data_path, model_path):
    # Load data
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_scaled, y)
    
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
    
    # Predict
    prediction = mlp.predict(X_scaled)[0]
    print(prediction)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mlp_classifier.py <data_path> <model_path> [--predict]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if len(sys.argv) > 3 and sys.argv[3] == "--predict":
        predict(data_path, model_path)
    else:
        train_model(data_path, model_path) 