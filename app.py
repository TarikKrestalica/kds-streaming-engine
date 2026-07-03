import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import CONFIG
from model import KDS
from stream_engine import IncrementalStreamEngine

app = FastAPI(title="KDS Incremental Streaming Engine")

# Global instances loaded in memory
model = None
streamer = None

class DataPoint(BaseModel):
    vector: list  # Expects a 784-dimensional flat list

@app.on_event("startup")
def load_infrastructure():
    global model, streamer
    input_size = 784 # MNIST dimensions
    
    # Initialize structural skeleton
    model = KDS(num_layers=CONFIG["num_layers"], input_size=input_size, hidden_size=CONFIG["hidden_size"])
    
    # Locate base weights (either local mount or pre-downloaded from Cloud Storage)
    checkpoint_path = os.path.join(CONFIG["base_path"], CONFIG["model_name"])
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Successfully synchronized baseline dictionary weights.")
        
    streamer = IncrementalStreamEngine(model=model, threshold=CONFIG["anomaly_threshold"])

@app.get("/health", methods=["GET", "HEAD"])
def health_check():
    # Crucial for GCP load balancer probes
    return {"status": "healthy", "dictionary_atoms": model.hidden_size}

@app.post("/predict")
def process_stream(payload: DataPoint):
    global model, streamer
    
    vec_array = np.array(payload.vector, dtype=np.float32)
    if vec_array.shape[0] != 784:
        raise HTTPException(status_code=400, detail="Invalid vector size. Input must be 784 components.")
    
    # Apply unit-norm scaling to the incoming request payload 
    norm = np.linalg.norm(vec_array)
    if norm > 0:
        vec_array = vec_array / norm
        
    # Convert to tensor and pipe into your engine
    tensor_input = torch.tensor(vec_array, dtype=torch.float32).unsqueeze(0)
    
    # Process point (Engine handles automatic expansion internally if error breaches threshold)
    did_expand, recon_error = streamer.process_streaming_point(tensor_input, is_in_target=1)
    
    return {
        "reconstruction_error": recon_error,
        "dictionary_expanded": did_expand,
        "current_atom_count": model.hidden_size
    }