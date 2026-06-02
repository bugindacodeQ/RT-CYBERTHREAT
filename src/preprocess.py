import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess(data_path="data/raw/synthetic_traffic.csv"):
    df = pd.read_csv(data_path)
    
    # Feature engineering
    df['packet_size_ratio'] = df['bytes_sent'] / (df['packet_size'] + 1)
    df['flow_intensity'] = df['avg_packet_rate'] * df['duration']
    
    # Encode categorical
    le_protocol = LabelEncoder()
    df['protocol_enc'] = le_protocol.fit_transform(df['protocol'])
    
    # Select features for model
    feature_cols = ['src_port', 'dst_port', 'packet_size', 'duration', 
                   'bytes_sent', 'bytes_received', 'flow_count', 
                   'avg_packet_rate', 'packet_size_ratio', 'flow_intensity', 'protocol_enc']
    
    X = df[feature_cols]
    y = df['is_attack']
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    
    # Save processed
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(X_scaled, columns=feature_cols).to_csv("data/processed/processed_data.csv", index=False)
    
    return X_scaled, y, feature_cols

if __name__ == "__main__":
    X, y, cols = load_and_preprocess()
    print("Preprocessing complete. Shape:", X.shape)