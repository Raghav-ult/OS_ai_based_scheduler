"""
ml/data_generator.py
---------------------
Generates a balanced synthetic workload dataset with 1,500+ samples.

Feature columns:
    avg_burst_time   : Mean CPU burst time across the process batch.
    burst_variance   : Variance of burst times (measures heterogeneity).
    cpu_usage_pct    : Estimated CPU utilization percentage.
    io_burst_ratio   : Fraction of burst time dominated by I/O events.
    process_count    : Number of concurrent processes in the batch.

Target column (workload_class):
    0 → CPU-bound  (heavy computation, low I/O)
    1 → I/O-bound  (light CPU, frequent I/O waits)
    2 → Mixed      (balanced CPU and I/O characteristics)

Output: data/workload_dataset.csv  (1,500 rows, balanced across 3 classes)
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 500          # 500 × 3 classes = 1,500 rows
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "workload_dataset.csv")


# ---------------------------------------------------------------------------
# Class-specific parameter profiles
# ---------------------------------------------------------------------------

CLASS_PROFILES = {
    # CPU-bound: long bursts, low variance, very high CPU usage, minimal I/O.
    0: {
        "label": "CPU-bound",
        "avg_burst_time":  {"low": 25,  "high": 80},
        "burst_variance":  {"low": 5,   "high": 50},
        "cpu_usage_pct":   {"low": 70,  "high": 98},
        "io_burst_ratio":  {"low": 0.0, "high": 0.15},
        "process_count":   {"low": 3,   "high": 15},
    },
    # I/O-bound: short bursts, moderate variance, low CPU, high I/O ratio.
    1: {
        "label": "I/O-bound",
        "avg_burst_time":  {"low": 2,   "high": 15},
        "burst_variance":  {"low": 1,   "high": 30},
        "cpu_usage_pct":   {"low": 10,  "high": 45},
        "io_burst_ratio":  {"low": 0.55,"high": 0.99},
        "process_count":   {"low": 5,   "high": 30},
    },
    # Mixed: medium bursts, higher variance, moderate CPU and I/O.
    2: {
        "label": "Mixed",
        "avg_burst_time":  {"low": 10,  "high": 40},
        "burst_variance":  {"low": 20,  "high": 150},
        "cpu_usage_pct":   {"low": 40,  "high": 75},
        "io_burst_ratio":  {"low": 0.20,"high": 0.55},
        "process_count":   {"low": 5,   "high": 25},
    },
}


# ---------------------------------------------------------------------------
# Generator function
# ---------------------------------------------------------------------------

def generate_dataset(
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
    output_path: str = OUTPUT_FILE,
    save: bool = True,
) -> pd.DataFrame:
    """
    Generate a balanced workload classification dataset.

    Args:
        samples_per_class : Number of rows to generate per class (default 500).
        seed              : NumPy random seed for reproducibility.
        output_path       : Filepath to write the CSV file to.
        save              : If True, write the CSV; otherwise return only the DataFrame.

    Returns:
        pandas DataFrame with feature columns and 'workload_class' target.
    """
    rng = np.random.default_rng(seed)
    all_rows = []

    for class_id, profile in CLASS_PROFILES.items():
        # Sample each feature from the class-specific uniform distribution,
        # then add small Gaussian noise to smooth boundaries between classes.
        n = samples_per_class

        avg_bt = rng.uniform(profile["avg_burst_time"]["low"],
                             profile["avg_burst_time"]["high"], n)
        bv = rng.uniform(profile["burst_variance"]["low"],
                         profile["burst_variance"]["high"], n)
        cpu = rng.uniform(profile["cpu_usage_pct"]["low"],
                          profile["cpu_usage_pct"]["high"], n)
        io = rng.uniform(profile["io_burst_ratio"]["low"],
                         profile["io_burst_ratio"]["high"], n)
        pc = rng.integers(profile["process_count"]["low"],
                          profile["process_count"]["high"] + 1, n)

        # Add structured noise (σ = 3% of range) for more realistic overlap.
        avg_bt += rng.normal(0, 1.5, n)
        cpu += rng.normal(0, 2.0, n)
        io += rng.normal(0, 0.02, n)

        # Clip to valid ranges.
        avg_bt = np.clip(avg_bt, 1, 100)
        cpu = np.clip(cpu, 1, 100)
        io = np.clip(io, 0.0, 1.0)
        bv = np.clip(bv, 0, 500)

        for i in range(n):
            all_rows.append({
                "avg_burst_time": round(float(avg_bt[i]), 3),
                "burst_variance": round(float(bv[i]), 3),
                "cpu_usage_pct": round(float(cpu[i]), 3),
                "io_burst_ratio": round(float(io[i]), 4),
                "process_count": int(pc[i]),
                "workload_class": class_id,
            })

    df = pd.DataFrame(all_rows)

    # Shuffle rows so classes aren't contiguous.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataGenerator] Dataset saved -> {os.path.abspath(output_path)}")
        print(f"[DataGenerator] Shape: {df.shape}")
        print(df["workload_class"].value_counts().rename({0: "CPU-bound", 1: "I/O-bound", 2: "Mixed"}))

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_dataset()
