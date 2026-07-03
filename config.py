import os

CONFIG = {
    "target_digits": [0, 3, 7],
    "subsample_size": 5000,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "num_epochs": 30,
    "num_layers": 100,         # Keeps PGD unrolling deep enough to prove hull convergence
    "hidden_size": 24,
    "penalty": 0.5,
    # --- CHANGED: Strict convex hull feasibility tolerance ---
    "anomaly_threshold": 0.6, 
    "base_path": "./checkpoints",
    "model_name": "kds_production_baseline.pth"
}

def get_model_save_path():
    os.makedirs(CONFIG["base_path"], exist_ok=True)
    return os.path.join(CONFIG["base_path"], CONFIG["model_name"])