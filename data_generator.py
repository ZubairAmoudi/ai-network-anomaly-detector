"""
Synthetic NetFlow Traffic Generator for AI Network Traffic Anomaly Detector.
Simulates realistic enterprise network traffic including normal baseline flows
and distinct cyberattack signatures (DDoS SYN Flood, UDP Amplification, Port Scan,
Data Exfiltration, and ICMP Flood).
"""

import os
import random
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import pandas as pd


INTERNAL_SUBNETS = ["10.0.1.", "10.0.2.", "10.0.10.", "192.168.1.", "172.16.0."]
SERVER_IPS = [
    "10.0.100.10",  # Web Server (HTTP/HTTPS)
    "10.0.100.11",  # API Gateway
    "10.0.100.20",  # DNS Server
    "10.0.100.30",  # Database Server
    "10.0.100.40",  # Auth / SSH Bastion
    "10.0.100.50",  # Internal File Server
]
EXTERNAL_BENIGN_SUBNETS = ["142.250.190.", "151.101.1.", "104.16.132.", "52.94.236.", "35.190.24."]
BOTNET_SUBNETS = ["185.220.101.", "45.155.205.", "91.240.118.", "194.26.192.", "103.145.13."]


def _random_ip(prefixes: list[str], rng: np.random.Generator) -> str:
    prefix = prefixes[int(rng.integers(0, len(prefixes)))]
    return f"{prefix}{int(rng.integers(2, 254))}"


def generate_network_traffic(
    n_samples: int = 5000,
    anomaly_ratio: float = 0.18,
    seed: Optional[int] = 42,
    start_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Generate realistic NetFlow v5/v9 style records with normal and anomalous patterns.

    Columns returned:
        - Timestamp: ISO timestamp of flow capture
        - IP_Source: Source IPv4 address
        - IP_Dest: Destination IPv4 address
        - Port: Destination port number (0-65535)
        - Protocol: Transport protocol (TCP, UDP, ICMP)
        - Packet_Size: Average packet payload size in bytes
        - Flow_Duration: Duration of flow in milliseconds (ms)
        - Packet_Count: Number of packets in the flow
        - Bytes_Per_Sec: Computed throughput rate
        - Packets_Per_Sec: Computed packet rate
        - TCP_Flags: Dominant TCP flag state (SYN, ACK, SYN-ACK, PSH-ACK, RST, NONE)
        - Attack_Type: Specific category (Normal, DDoS_SYN_Flood, UDP_Amplification,
                       Port_Scan, Data_Exfiltration, ICMP_Smurf)
        - Label: 'Normal' or 'Anomaly'
    """
    rng = np.random.default_rng(seed)
    if start_time is None:
        start_time = datetime.now() - timedelta(minutes=30)

    n_anomalies = int(round(n_samples * anomaly_ratio))
    n_normal = n_samples - n_anomalies

    records = []

    # 1. Generate Normal Traffic Flows
    normal_ports = [80, 443, 443, 443, 53, 22, 8080, 3306, 5432, 123, 993]
    for _ in range(n_normal):
        proto_choice = rng.choice(["TCP", "UDP", "ICMP"], p=[0.74, 0.23, 0.03])
        if proto_choice == "TCP":
            port = int(rng.choice([80, 443, 443, 443, 22, 8080, 3306, 5432, 993]))
            pkt_size = float(np.clip(rng.normal(loc=580, scale=220), 64, 1480))
            duration = float(np.clip(rng.lognormal(mean=5.8, sigma=1.1), 15.0, 25000.0))
            pkt_count = int(np.clip(rng.poisson(lam=18) + 2, 2, 180))
            tcp_flag = str(rng.choice(["ACK", "PSH-ACK", "SYN-ACK"], p=[0.45, 0.40, 0.15]))
        elif proto_choice == "UDP":
            port = int(rng.choice([53, 53, 123, 443, 500]))
            pkt_size = float(np.clip(rng.normal(loc=180, scale=75), 48, 600))
            duration = float(np.clip(rng.lognormal(mean=3.5, sigma=0.9), 2.0, 3500.0))
            pkt_count = int(np.clip(rng.poisson(lam=4) + 1, 1, 35))
            tcp_flag = "NONE"
        else:  # ICMP
            port = 0
            pkt_size = float(np.clip(rng.normal(loc=84, scale=15), 64, 128))
            duration = float(np.clip(rng.normal(loc=45, scale=20), 5.0, 250.0))
            pkt_count = int(rng.integers(1, 6))
            tcp_flag = "NONE"

        if rng.random() < 0.65:
            src_ip = _random_ip(INTERNAL_SUBNETS, rng)
            dst_ip = str(rng.choice(SERVER_IPS + [_random_ip(EXTERNAL_BENIGN_SUBNETS, rng)]))
        else:
            src_ip = _random_ip(EXTERNAL_BENIGN_SUBNETS, rng)
            dst_ip = str(rng.choice(SERVER_IPS))

        records.append(
            {
                "IP_Source": src_ip,
                "IP_Dest": dst_ip,
                "Port": port,
                "Protocol": proto_choice,
                "Packet_Size": round(pkt_size, 2),
                "Flow_Duration": round(duration, 2),
                "Packet_Count": pkt_count,
                "TCP_Flags": tcp_flag,
                "Attack_Type": "Normal",
                "Label": "Normal",
            }
        )

    # 2. Generate Anomalous Traffic Flows (5 distinct attack signatures)
    attack_types = [
        "DDoS_SYN_Flood",
        "UDP_Amplification",
        "Port_Scan",
        "Data_Exfiltration",
        "ICMP_Smurf",
    ]
    attack_weights = [0.35, 0.25, 0.20, 0.12, 0.08]

    for _ in range(n_anomalies):
        attack = str(rng.choice(attack_types, p=attack_weights))

        if attack == "DDoS_SYN_Flood":
            proto = "TCP"
            port = int(rng.choice([80, 443, 8080]))
            # Tiny SYN packets or malformed headers, ultra-short or bursts, huge packet count
            pkt_size = float(np.clip(rng.normal(loc=60, scale=8), 40, 96))
            duration = float(np.clip(rng.uniform(0.5, 25.0), 0.2, 40.0))
            pkt_count = int(rng.integers(250, 2200))
            tcp_flag = "SYN"
            src_ip = _random_ip(BOTNET_SUBNETS, rng)
            dst_ip = str(rng.choice(SERVER_IPS[:2]))

        elif attack == "UDP_Amplification":
            proto = "UDP"
            port = int(rng.choice([53, 123, 1900, 11211]))
            # Massive amplified UDP payload sizes
            pkt_size = float(np.clip(rng.normal(loc=4200, scale=950), 1800, 8192))
            duration = float(np.clip(rng.uniform(5.0, 180.0), 2.0, 250.0))
            pkt_count = int(rng.integers(120, 1500))
            tcp_flag = "NONE"
            src_ip = _random_ip(BOTNET_SUBNETS, rng)
            dst_ip = str(rng.choice(SERVER_IPS[:3]))

        elif attack == "Port_Scan":
            proto = "TCP"
            # Scanning unusual/high or sequential ports
            port = int(rng.integers(1, 65535))
            pkt_size = float(np.clip(rng.normal(loc=48, scale=6), 40, 68))
            duration = float(np.clip(rng.uniform(0.1, 4.5), 0.1, 8.0))
            pkt_count = int(rng.integers(1, 3))
            tcp_flag = str(rng.choice(["SYN", "RST"], p=[0.6, 0.4]))
            src_ip = _random_ip(BOTNET_SUBNETS, rng)
            dst_ip = str(rng.choice(SERVER_IPS))

        elif attack == "Data_Exfiltration":
            proto = "TCP"
            port = int(rng.choice([443, 8443, 4444, 9001, 22]))
            # Sustained high-volume outbound transfer with huge packets and long duration
            pkt_size = float(np.clip(rng.normal(loc=1440, scale=60), 1250, 1500))
            duration = float(np.clip(rng.uniform(35000.0, 120000.0), 28000.0, 150000.0))
            pkt_count = int(rng.integers(800, 5000))
            tcp_flag = "PSH-ACK"
            src_ip = str(rng.choice(SERVER_IPS[3:]))
            dst_ip = _random_ip(BOTNET_SUBNETS, rng)

        else:  # ICMP_Smurf
            proto = "ICMP"
            port = 0
            pkt_size = float(np.clip(rng.normal(loc=1350, scale=180), 900, 1500))
            duration = float(np.clip(rng.uniform(1.0, 35.0), 0.5, 50.0))
            pkt_count = int(rng.integers(300, 1800))
            tcp_flag = "NONE"
            src_ip = _random_ip(BOTNET_SUBNETS, rng)
            dst_ip = str(rng.choice(SERVER_IPS))

        records.append(
            {
                "IP_Source": src_ip,
                "IP_Dest": dst_ip,
                "Port": port,
                "Protocol": proto,
                "Packet_Size": round(pkt_size, 2),
                "Flow_Duration": round(duration, 2),
                "Packet_Count": pkt_count,
                "TCP_Flags": tcp_flag,
                "Attack_Type": attack,
                "Label": "Anomaly",
            }
        )

    # Shuffle records and assign sequential timestamps
    rng.shuffle(records)
    df = pd.DataFrame(records)

    time_deltas = np.cumsum(rng.exponential(scale=0.35, size=len(df)))
    timestamps = [
        (start_time + timedelta(seconds=float(dt))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        for dt in time_deltas
    ]
    df.insert(0, "Timestamp", timestamps)

    # Compute derived NetFlow rate metrics
    duration_sec = np.maximum(df["Flow_Duration"] / 1000.0, 0.001)
    df["Packets_Per_Sec"] = np.round(df["Packet_Count"] / duration_sec, 2)
    df["Bytes_Per_Sec"] = np.round((df["Packet_Size"] * df["Packet_Count"]) / duration_sec, 2)

    return df


def save_dataset(output_path: str = "data/network_traffic.csv", n_samples: int = 6000) -> pd.DataFrame:
    """Generate and persist synthetic NetFlow dataset to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_network_traffic(n_samples=n_samples, anomaly_ratio=0.18, seed=42)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    output_file = os.path.join("data", "network_traffic.csv")
    print(f"Generating synthetic NetFlow dataset -> {output_file} ...")
    dataset = save_dataset(output_path=output_file, n_samples=6000)
    counts = dataset["Label"].value_counts().to_dict()
    attack_counts = dataset["Attack_Type"].value_counts().to_dict()
    print(f"Successfully generated {len(dataset):,} NetFlow records.")
    print(f"Class distribution: {counts}")
    print(f"Traffic breakdown:  {attack_counts}")

