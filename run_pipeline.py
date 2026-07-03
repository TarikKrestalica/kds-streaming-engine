import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from config import CONFIG, get_model_save_path
from dataset import UnitNormFilteredMNIST 
from model import KDS
from loss import LocalDictionaryLoss
from stream_engine import ProductionStreamSimulator, IncrementalStreamEngine
from tqdm import tqdm as progress_bar

def run_data_isolation_audit(train_dataset, stream_X, in_target):
    """
    Production Pre-flight Audit: Ensures 0% data leakage between the baseline 
    training matrix and the streaming matrix before evaluation begins.
    """
    print("\n--- [AUDIT] Running Production Data Isolation Compliance Check ---")
    
    # Extract matrices as numpy arrays for efficient set operations
    X_train_np = train_dataset.X.numpy()
    X_stream_target = stream_X[in_target == 1]
    
    # Hash rows as immutable tuples for strict coordinate comparison
    train_set = set(tuple(row) for row in X_train_np)
    stream_target_set = set(tuple(row) for row in X_stream_target)
    
    # Calculate intersection
    leaking_samples = train_set.intersection(stream_target_set)
    overlap_count = len(leaking_samples)
    
    if overlap_count == 0:
        print("-> [AUDIT SUCCESS] 0.0% Data Overlap Detected. Training and Streaming pools are perfectly disjoint.")
        return True
    else:
        print(f"-> [CRITICAL AUDIT FAILURE] Data Leakage Detected! {overlap_count} training rows found in the stream.")
        return False


def run_system_pipeline():
    print("--- Phase 1: Baseline Dataset Ingestion ---")
    train_dataset = UnitNormFilteredMNIST(
        target_digits=CONFIG["target_digits"],
        subsample_size=CONFIG["subsample_size"]
    )
    
    trained_indices = train_dataset.chosen_indices
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True
    )
    
    net = KDS(
        num_layers=CONFIG["num_layers"],
        input_size=train_dataset.X.shape[1],
        hidden_size=CONFIG["hidden_size"],
        penalty=CONFIG["penalty"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = LocalDictionaryLoss(penalty=CONFIG["penalty"])
    optimizer = Adam(net.parameters(), lr=CONFIG["learning_rate"])
    
    with torch.no_grad():
        p = torch.randperm(len(train_dataset))[: net.hidden_size]
        net.W.data = train_dataset.X[p]
        net.step.fill_((net.W.data.svd()[1][0] ** -2).item())
    net = net.to(device)

    for epoch in progress_bar(range(CONFIG["num_epochs"])):
        shuffle = torch.randperm(len(train_dataset))
        data, labels = train_dataset[shuffle]
        
        batch_iterator = progress_bar(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for i in progress_bar(range(0, len(train_dataset), CONFIG["batch_size"]), disable=True):
            y = data[i : i + CONFIG["batch_size"]].to(device)
            x_hat = net.encode_accelerated(y)
            loss = criterion(net.W, y, x_hat)
            recon_loss, penalty_loss, total_loss = criterion.forward_detailed(net.W, y, x_hat)
            print(f"Total Loss: {total_loss}")
            print(f"Recon Loss: {recon_loss:.4f} || Penalty Loss: {penalty_loss:.4f}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
            optimizer.step()
            
            batch_iterator.set_postfix({
                "Recon": f"{recon_loss.item():.4f}",
                "Penalty": f"{penalty_loss.item():.4f}"
            })

        if epoch + 1 % 10 == 0:
            print(f"Epoch {epoch + 1}/{CONFIG["num_epochs"]}, Loss: {loss.item():.4f}")

    print("Training finished.")
            
    torch.save(net.state_dict(), get_model_save_path())
    print(f"Baseline saved successfully to: {get_model_save_path()}")
    
    print("\n--- Phase 3: Generating Disjoint Interleaved Stream Simulation ---")
    simulator = ProductionStreamSimulator(
        target_digits=CONFIG["target_digits"],
        total_stream_samples=1000,
        target_ratio=0.85
    )
    
    stream_X, stream_y, in_target = simulator.generate_interleaved_stream(
        trained_indices_to_exclude=trained_indices
    )
    
    print(f"Generated Stream Array Shape: {stream_X.shape}")
    print(f"Composition -> Target Samples: {in_target.sum()} | Novel Samples: {(in_target==0).sum()}")
    
    # --- ENFORCE AUDIT CHECK BEFORE PIPELINE PROGRESSION ---
    is_pipeline_safe = run_data_isolation_audit(train_dataset, stream_X, in_target)
    if not is_pipeline_safe:
        print("!!! HALTING PIPELINE EXECUTION DUE TO SECURITY AUDIT FAILURE !!!")
        sys.exit(1) # Crash out safely to prevent compromised evaluation logs
    # -------------------------------------------------------
    
    print("\n--- Phase 4: Live Processing Loop ---")
    streamer = IncrementalStreamEngine(model=net, threshold=CONFIG["anomaly_threshold"])
    
    for idx in range(50):
        vector = stream_X[idx]
        actual_digit = stream_y[idx]
        is_target = in_target[idx]
        
        did_expand, error, y_hat_vec = streamer.process_streaming_point(vector, is_target)
        status_msg = "PASSED" if not did_expand else "EXPANDED"
        print(f"Item {idx:02d} | True Digit: {actual_digit} | Target? {is_target} | Error: {error:.4f} | {status_msg}")
        
    print(f"\nFinal Adaptive Dictionary Size: {net.hidden_size} atoms.")

if __name__ == "__main__":
    run_system_pipeline()