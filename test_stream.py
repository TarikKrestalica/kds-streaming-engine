import requests
import numpy as np
from dataset import UnitNormFilteredMNIST
from stream_engine import ProductionStreamSimulator

print("Generating target validation disjoint dataset elements...")
simulator = ProductionStreamSimulator(target_digits=[0, 3, 7], total_stream_samples=100)
stream_X, stream_y, in_target = simulator.generate_interleaved_stream()

url = "http://localhost:8080/predict"

# Send the first 10 stream instances to your live Docker container containerized instance
for idx in range(10):
    vector_list = stream_X[idx].tolist()
    payload = {"vector": vector_list}
    
    # Post request payload over internal network port map
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        res_data = response.json()
        print(f"Sample {idx:02d} | Digit: {stream_y[idx]} | Target? {in_target[idx]} "
              f"| Recon Error: {res_data['reconstruction_error']:.4f} "
              f"| Atoms: {res_data['current_atom_count']} | Expanded? {res_data['dictionary_expanded']}")
    else:
        print(f"Error connecting to container endpoint: {response.text}")