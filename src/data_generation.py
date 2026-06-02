import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_samples=100000, seed=42):
    np.random.seed(seed)
    
    # Features typical in network traffic / IDS datasets
    data = {
        'timestamp': [datetime.now() - timedelta(seconds=i) for i in range(n_samples)],
        'src_ip': [f"192.168.1.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'dst_ip': [f"10.0.0.{np.random.randint(1,255)}" for _ in range(n_samples)],
        'src_port': np.random.randint(1024, 65535, n_samples),
        'dst_port': np.random.randint(1, 65535, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'packet_size': np.random.normal(500, 200, n_samples).astype(int).clip(40, 1500),
        'duration': np.random.exponential(2, n_samples).clip(0.1, 300),
        'bytes_sent': np.random.lognormal(6, 1.5, n_samples).astype(int).clip(100, 1000000),
        'bytes_received': np.random.lognormal(5, 1.8, n_samples).astype(int).clip(50, 500000),
        'flow_count': np.random.poisson(5, n_samples),
        'avg_packet_rate': np.random.uniform(0.5, 500, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Label generation (0 = Normal, 1 = Attack)
    df['label'] = 0
    
    # Simulate attacks
    attack_mask = np.random.rand(n_samples) < 0.15  # ~15% attacks
    df.loc[attack_mask, 'label'] = 1
    
    # DDoS-like patterns
    ddos_mask = (attack_mask) & (np.random.rand(n_samples) < 0.4)
    df.loc[ddos_mask, 'packet_size'] = np.random.normal(60, 20, ddos_mask.sum()).astype(int)
    df.loc[ddos_mask, 'avg_packet_rate'] *= 8
    
    # Port scan / probing
    scan_mask = (attack_mask) & (np.random.rand(n_samples) < 0.35)
    df.loc[scan_mask, 'dst_port'] = np.random.choice(range(1, 1024), scan_mask.sum())
    
    # Add some noise to normal traffic
    df['is_attack'] = df['label']  # Binary for simplicity; can expand to multi-class later
    
    # Save
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/synthetic_traffic.csv", index=False)
    print(f"Generated {n_samples} samples. Attacks: {df['is_attack'].sum()}")
    return df

if __name__ == "__main__":
    generate_synthetic_data(150000)