import joblib
import numpy as np
import pandas as pd

def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return model, scaler, feature_cols

def predict_threat(input_dict: dict):
    """input_dict should contain the feature keys"""
    model, scaler, feature_cols = load_model()
    
    df = pd.DataFrame([input_dict])
    
    #  feature engineering 
    df['packet_size_ratio'] = df['bytes_sent'] / (df['packet_size'] + 1)
    df['flow_intensity'] = df['avg_packet_rate'] * df['duration']
    
    # Encode 
    protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
    df['protocol_enc'] = df['protocol'].map(protocol_map).fillna(0)
    
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
    
    return {
        "is_threat": bool(pred),
        "threat_probability": float(prob),
        "confidence": "High" if prob > 0.85 else "Medium" if prob > 0.6 else "Low"
    }