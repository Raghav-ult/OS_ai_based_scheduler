"""
ml/predictor.py
----------------
Real-time inference module.

Responsibilities:
    1. Load the trained model pipeline from model.joblib (auto-trains if missing).
    2. Extract a feature vector from a batch of Process objects.
    3. Predict the workload class and return a confidence score.

No hard-coded if-else logic determines the class — the scikit-learn model
drives every prediction.
"""

import os
from typing import List, Tuple, Dict

import numpy as np
import joblib

from core.process import Process

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model.joblib")

FEATURE_COLS = [
    "avg_burst_time",
    "burst_variance",
    "cpu_usage_pct",
    "io_burst_ratio",
    "process_count",
]

CLASS_LABELS: Dict[int, str] = {
    0: "CPU-bound",
    1: "I/O-bound",
    2: "Mixed",
}


# ---------------------------------------------------------------------------
# Internal: auto-load or auto-train the model
# ---------------------------------------------------------------------------

def _load_model():
    """Load model.joblib; if absent, trigger training pipeline first."""
    if not os.path.exists(MODEL_PATH):
        print("[Predictor] model.joblib not found – training now …")
        from ml.train_model import train
        train()

    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as exc:
        raise RuntimeError(
            f"[Predictor] Failed to load model from {MODEL_PATH}: {exc}"
        ) from exc


# Lazy singleton – loaded once per Python session.
_MODEL = None


def get_model():
    """Return the cached model pipeline, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def reload_model():
    """Force a model reload (e.g., after retraining)."""
    global _MODEL
    _MODEL = _load_model()
    return _MODEL


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(processes: List[Process]) -> np.ndarray:
    """
    Compute a single feature vector from a batch of Process objects.

    Features extracted:
        avg_burst_time  : Mean burst time of all processes.
        burst_variance  : Statistical variance of burst times.
        cpu_usage_pct   : Estimated CPU utilisation (100 × (1 – avg_io_freq)).
        io_burst_ratio  : Mean I/O frequency across all processes.
        process_count   : Total number of processes in the batch.

    Args:
        processes: List of Process objects (at least one required).

    Returns:
        1-D numpy array of shape (5,) representing the feature vector.

    Raises:
        ValueError: If the process list is empty.
    """
    if not processes:
        raise ValueError("[Predictor] Cannot extract features from an empty process list.")

    burst_times = np.array([p.burst_time for p in processes], dtype=float)
    io_freqs = np.array([p.io_frequency for p in processes], dtype=float)

    avg_burst_time = float(np.mean(burst_times))
    burst_variance = float(np.var(burst_times))
    io_burst_ratio = float(np.mean(io_freqs))
    cpu_usage_pct = float(100.0 * (1.0 - io_burst_ratio))  # Inverse of I/O dominance.
    process_count = float(len(processes))

    return np.array([
        avg_burst_time,
        burst_variance,
        cpu_usage_pct,
        io_burst_ratio,
        process_count,
    ], dtype=float)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_workload(processes: List[Process]) -> Dict:
    """
    Predict the workload class for a batch of processes.

    Args:
        processes: List of Process objects representing the current batch.

    Returns:
        Dictionary containing:
            'class_id'      : int  (0=CPU-bound, 1=I/O-bound, 2=Mixed)
            'label'         : str  human-readable class name
            'confidence'    : float confidence of the top prediction (0–1)
            'probabilities' : dict mapping class label → probability
            'features'      : dict mapping feature name → extracted value
    """
    model = get_model()
    feature_vector = extract_features(processes)

    # Reshape to (1, n_features) for sklearn.
    X = feature_vector.reshape(1, -1)

    class_id = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]           # shape: (n_classes,)
    confidence = float(proba[class_id])

    label = CLASS_LABELS.get(class_id, f"Unknown({class_id})")

    prob_dict = {
        CLASS_LABELS[i]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    feature_dict = {
        name: round(float(val), 4)
        for name, val in zip(FEATURE_COLS, feature_vector)
    }

    return {
        "class_id": class_id,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "features": feature_dict,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    from core.simulation_engine import generate_processes

    sample_procs = generate_processes(count=10, seed=7)
    result = predict_workload(sample_procs)

    print("\n=== Predictor Demo ===")
    print(f"  Predicted Class : {result['label']} (id={result['class_id']})")
    print(f"  Confidence      : {result['confidence'] * 100:.1f}%")
    print(f"  Probabilities   : {result['probabilities']}")
    print(f"  Extracted Feats : {result['features']}")
