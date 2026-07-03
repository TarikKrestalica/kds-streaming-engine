import numpy as np
import torch
from dataset import UnitNormFilteredMNIST
from stream_engine import ProductionStreamSimulator

def run_leakage_audit():
    print("=" * 65)
    print("        KDS STREAM SYSTEM: DATA ISOLATION COMPLIANCE AUDIT       ")
    print("=" * 65)
    
    # 1. Instantiate the baseline training dataset (5000 points)
    print("[STEP 1] Generating Baseline Training Set...")
    train_dataset = UnitNormFilteredMNIST(
        target_digits=[0, 3, 7],
        subsample_size=5000,
        seed=42
    )
    
    # Extract the exact index tracking map used to carve out the 5000 baseline points
    trained_indices = train_dataset.chosen_indices
    print(f"  -> Total Baseline Training Points Loaded: {len(train_dataset.X)}")
    print(f"  -> Sample of internal training tracking indices: {trained_indices[:5]}")
    
    # Convert training data to a flattened numpy array for hashing/set comparisons
    X_train_np = train_dataset.X.numpy()
    
    # 2. Instantiate the Stream Simulator and feed it the indices to exclude
    print("\n[STEP 2] Generating Interleaved Evaluation Stream...")
    simulator = ProductionStreamSimulator(
        target_digits=[0, 3, 7],
        total_stream_samples=1000,
        target_ratio=0.85,
        seed=42
    )
    
    # PASS the exclusion list explicitly just like the pipeline does
    stream_X, stream_y, in_target = simulator.generate_interleaved_stream(
        trained_indices_to_exclude=trained_indices
    )
    
    print(f"  -> Total Stream Pipeline Volume: {len(stream_X)}")
    print(f"  -> Composition: In-Distribution ({in_target.sum()}) | Novel OOD ({(in_target==0).sum()})")
    
    # 3. Separate the stream into its two constituent parts for deep analysis
    X_stream_target = stream_X[in_target == 1]
    X_stream_novel = stream_X[in_target == 0]
    
    # 4. RUN CRITICAL MATRICES LOCK-MATCH CHECK
    print("\n[STEP 3] Executing Mathematical Leakage Tests...")
    
    # Convert rows to immutable tuple byte hashes for absolute set intersection checks
    train_set = set(tuple(row) for row in X_train_np)
    stream_target_set = set(tuple(row) for row in X_stream_target)
    stream_novel_set = set(tuple(row) for row in X_stream_novel)
    
    # Check 1: In-distribution stream vs. Baseline Training Set
    target_leakage = train_set.intersection(stream_target_set)
    
    # Check 2: Out-of-distribution stream vs. Baseline Training Set
    novel_leakage = train_set.intersection(stream_novel_set)
    
    # 5. Output Audit Results
    print("-" * 65)
    print(f"AUDIT RESULT 1 (In-Distribution Check): Found {len(target_leakage)} leaking samples.")
    print(f"AUDIT RESULT 2 (Novel OOD Check):      Found {len(novel_leakage)} leaking samples.")
    print("-" * 65)
    
    if len(target_leakage) == 0 and len(novel_leakage) == 0:
        print(" SUCCESS: Data Isolation Verified. 0.0% Overlap Detected.")
        print(" The stream consists purely of unobserved data instances.")
    else:
        print("CRITICAL CRASH ALERT: Data leakage discovered in current pool boundaries!")
    print("=" * 65)

if __name__ == "__main__":
    run_leakage_audit()