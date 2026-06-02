import pandas as pd
import numpy as np
from datetime import datetime
import os


def generate_synthetic_data(n_samples=100000, seed=42):
    

    np.random.seed(seed)

    print(f"Generating {n_samples:,} samples...")

    base_time = pd.Timestamp.now()

    data = {
        "timestamp": pd.date_range(
            end=base_time,
            periods=n_samples,
            freq="s"
        ),
        "src_ip": [
            f"192.168.1.{np.random.randint(1, 255)}"
            for _ in range(n_samples)
        ],
        "dst_ip": [
            f"10.0.0.{np.random.randint(1, 255)}"
            for _ in range(n_samples)
        ],
        "src_port": np.random.randint(
            1024,
            65535,
            size=n_samples,
            dtype=np.int32
        ),
        "dst_port": np.random.randint(
            1,
            65535,
            size=n_samples,
            dtype=np.int32
        ),
        "protocol": np.random.choice(
            ["TCP", "UDP", "ICMP"],
            size=n_samples
        ),
        "packet_size": np.clip(
            np.random.normal(500, 200, size=n_samples),
            40,
            1500
        ).astype(np.int32),
        "duration": np.clip(
            np.random.exponential(2, size=n_samples),
            0.1,
            300
        ),
        "bytes_sent": np.clip(
            np.random.lognormal(6, 1.5, size=n_samples),
            100,
            1_000_000
        ).astype(np.int32),
        "bytes_received": np.clip(
            np.random.lognormal(5, 1.8, size=n_samples),
            50,
            500_000
        ).astype(np.int32),
        "flow_count": np.random.poisson(
            5,
            size=n_samples
        ).astype(np.int32),
        "avg_packet_rate": np.random.uniform(
            0.5,
            500,
            size=n_samples
        )
    }

    df = pd.DataFrame(data)


    integer_cols = [
        "src_port",
        "dst_port",
        "packet_size",
        "bytes_sent",
        "bytes_received",
        "flow_count"
    ]

    for col in integer_cols:
        df[col] = df[col].astype(np.int64)

   

    df["label"] = 0

    attack_mask = np.random.rand(n_samples) < 0.15
    df.loc[attack_mask, "label"] = 1

    

    ddos_mask = attack_mask & (
        np.random.rand(n_samples) < 0.40
    )

    ddos_sizes = np.clip(
        np.random.normal(60, 20, ddos_mask.sum()),
        40,
        150
    ).astype(np.int64)

    df.loc[ddos_mask, "packet_size"] = ddos_sizes

    df.loc[ddos_mask, "avg_packet_rate"] *= 8

   

    scan_mask = attack_mask & (
        np.random.rand(n_samples) < 0.35
    )

    scan_ports = np.random.randint(
        1,
        1024,
        size=scan_mask.sum(),
        dtype=np.int64
    )

    df.loc[scan_mask, "dst_port"] = scan_ports

   

    df["is_attack"] = df["label"]


    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        "synthetic_traffic.csv"
    )

    df.to_csv(output_file, index=False)

    print("\nDataset generation complete.")
    print(f"Rows: {len(df):,}")
    print(f"Attacks: {df['is_attack'].sum():,}")
    print(f"Saved to: {output_file}")

    return df


if __name__ == "__main__":
    generate_synthetic_data(150000)