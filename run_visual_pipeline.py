import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from torchvision.datasets import MNIST
import cvxpy as cp
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from config import CONFIG, get_model_save_path
from dataset import UnitNormFilteredMNIST
from model import KDS

# --- Embedded Data Stream Simulator ---
class ProductionStreamSimulator:
    def __init__(self, target_digits=[0, 3, 7], total_stream_samples=1000, target_ratio=0.5, seed=42):
        self.target_digits = target_digits
        self.total_stream_samples = total_stream_samples
        self.target_ratio = target_ratio
        self.rng = np.random.default_rng(seed)
        
        mnist_data = MNIST(root='./data', train=True, download=True)
        self.X_raw = mnist_data.data.numpy().reshape(60000, 784).astype(np.float32) / 255.0
        self.y_raw = mnist_data.targets.numpy().flatten()

    def generate_interleaved_stream(self, trained_indices_to_exclude=None):
        mask = np.isin(self.y_raw, self.target_digits)
        X_target_pool = self.X_raw[mask]
        y_target_pool = self.y_raw[mask]
        
        if trained_indices_to_exclude is not None:
            elements_to_keep_mask = np.ones(X_target_pool.shape[0], dtype=bool)
            elements_to_keep_mask[trained_indices_to_exclude] = False
            X_target_pool = X_target_pool[elements_to_keep_mask]
            y_target_pool = y_target_pool[elements_to_keep_mask]
            
        X_novel_pool = self.X_raw[~mask]
        y_novel_pool = self.y_raw[~mask]
        
        num_target = int(self.total_stream_samples * self.target_ratio)
        num_novel = self.total_stream_samples - num_target
        
        tgt_idx = self.rng.choice(len(X_target_pool), size=num_target, replace=False)
        novel_idx = self.rng.choice(len(X_novel_pool), size=num_novel, replace=False)
        
        X_t, y_t = X_target_pool[tgt_idx], y_target_pool[tgt_idx]
        X_n, y_n = X_novel_pool[novel_idx], y_novel_pool[novel_idx]
        
        X_t = X_t / np.linalg.norm(X_t, axis=1, keepdims=True)
        X_n = X_n / np.linalg.norm(X_n, axis=1, keepdims=True)
        
        total_subset = np.concatenate([X_t, X_n], axis=0)
        total_labels = np.concatenate([y_t, y_n])
        in_target = np.concatenate([np.ones(len(X_t)), np.zeros(len(X_n))]).astype(int)
        
        perm = self.rng.permutation(len(total_subset))
        return total_subset[perm], total_labels[perm], in_target[perm]


def update_visual_window(fig, ax, positions_2d, atoms_high_dim, title="", new_atom_idx=None, y_point=None, y_hat_point=None, current_stream_vector=None, image_zoom=0.45):
    ax.clear()
    m = positions_2d.shape[0]

    # 1. Plot Delaunay Triangulation Mesh lines
    if m >= 3:
        try:
            dt = Delaunay(positions_2d)
            ax.triplot(positions_2d[:, 0], positions_2d[:, 1],
                       dt.simplices, color='0.7', lw=1.2, zorder=1)
        except Exception:
            pass
            
    # 2. Render MNIST Dictionary Thumbnails over every node coordinate
    for node_idx in range(m):
        img = atoms_high_dim[node_idx].reshape(28, 28)
        imagebox = OffsetImage(img, zoom=image_zoom, cmap='gray')
        
        # Highlight newly appended dictionary additions with a Red frame
        if new_atom_idx is not None and node_idx == new_atom_idx:
            bbox_props = dict(edgecolor='red', linewidth=2.5, facecolor='none')
        else:
            bbox_props = dict(edgecolor='black', linewidth=0.8, facecolor='none')
            
        ab = AnnotationBbox(
            imagebox, 
            positions_2d[node_idx], 
            frameon=True, 
            bboxprops=bbox_props, 
            pad=0.1, 
            zorder=3
        )
        ax.add_artist(ab)

    # 3. Draw streaming element image (with Blue border frame) and its hull projection
    if y_point is not None and y_hat_point is not None:
        # If the high-dimensional vector is available, display it as a thumbnail instead of a blue star
        if current_stream_vector is not None:
            img_y = current_stream_vector.reshape(28, 28)
            imagebox_y = OffsetImage(img_y, zoom=image_zoom, cmap='gray')
            # Use a distinctive blue border line to signify that this is the streaming input
            bbox_props_y = dict(edgecolor='blue', linewidth=2.5, facecolor='none')
            
            ab_y = AnnotationBbox(
                imagebox_y, 
                y_point, 
                frameon=True, 
                bboxprops=bbox_props_y, 
                pad=0.1, 
                zorder=5
            )
            ax.add_artist(ab_y)
        else:
            # Fallback placeholder if vector array is not provided
            ax.scatter(y_point[0], y_point[1], s=160, c='blue', marker='*', zorder=5)

        # Draw closest anchor vector representation projection spot (Red Dot)
        ax.scatter(y_hat_point[0], y_hat_point[1], s=75, c='red', marker='o', zorder=5, label='Hull Projection (ŷ)')
        ax.plot([y_point[0], y_hat_point[0]], [y_point[1], y_hat_point[1]], 
                color='black', linestyle='--', lw=1.5, zorder=2)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    # Establish dynamic view windows encompassing both atoms and incoming targets
    all_points = positions_2d if y_point is None else np.vstack([positions_2d, y_point])
    pad_val = 0.15 * max(np.ptp(all_points, axis=0).max(), 1e-9)
    ax.set_xlim(all_points[:, 0].min() - pad_val, all_points[:, 0].max() + pad_val)
    ax.set_ylim(all_points[:, 1].min() - pad_val, all_points[:, 1].max() + pad_val)
    
    fig.canvas.draw()
    fig.canvas.flush_events()


def _solve_projection(y, A_cols):
    """Project y onto the convex hull of the columns of A_cols.
    A_cols: (784, k) — each column is one dictionary atom.
    Returns coefficient vector of length k, or None if the solver fails.
    """
    k = A_cols.shape[1]
    x = cp.Variable(k)
    prob = cp.Problem(cp.Minimize(cp.norm(y - A_cols @ x, 2)), [x >= 0, cp.sum(x) == 1])
    prob.solve()
    return x.value


def execute_visual_stream():
    train_dataset = UnitNormFilteredMNIST(target_digits=CONFIG["target_digits"], subsample_size=CONFIG["subsample_size"])
    
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_dataset.X.numpy())
    
    model = KDS(num_layers=CONFIG["num_layers"], input_size=train_dataset.X.shape[1], hidden_size=CONFIG["hidden_size"], penalty=CONFIG["penalty"])
    model.load_state_dict(torch.load(get_model_save_path()))
    
    # Extract baseline dictionary matrix [784, m] — model stores W as (m, 784), so transpose
    A_high_dim = model.W.data.cpu().numpy().T
    m = A_high_dim.shape[1]
    
    # Seeding tracking arrays matching the identity conditions of step 0
    Y_high_dim = A_high_dim.copy()                             
    Y_hat_high_dim = A_high_dim.copy()                         
    X_high_dim = np.eye(m)                                     
    
    simulator = ProductionStreamSimulator(target_digits=CONFIG["target_digits"], total_stream_samples=1000, target_ratio=0.85)
    stream_X, stream_y, in_target = simulator.generate_interleaved_stream(trained_indices_to_exclude=train_dataset.chosen_indices)
    
    plt.ion() 
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.show(block=False)
    
    epsilon = CONFIG["anomaly_threshold"]
    sparse_penalty = CONFIG["penalty"]
    
    for idx in range(50):
        current_data_point = stream_X[idx]

        # Project atoms to 2D *before* CVXPY so we can locate the containing simplex.
        # A_high_dim is (784, m); rows of A_high_dim.T are the atom positions for PCA.
        atoms_2d_pre = pca.transform(A_high_dim.T)
        y_point_2d = pca.transform(current_data_point.reshape(1, -1))[0]

        # 1. Delaunay-constrained CVXPY: only the 3 vertices of the containing triangle
        #    contribute to the reconstruction. Fall back to all atoms if outside the hull.
        sparse_code_x = np.zeros(m)
        in_hull = False
        if m >= 3:
            try:
                dt_pre = Delaunay(atoms_2d_pre)
                simplex_idx = dt_pre.find_simplex(y_point_2d)
                if simplex_idx >= 0:
                    in_hull = True
                    active_indices = dt_pre.simplices[simplex_idx]  # exactly 3 column indices
                    c = _solve_projection(current_data_point, A_high_dim[:, active_indices])
                    if c is not None:
                        sparse_code_x[active_indices] = c
                else:
                    c = _solve_projection(current_data_point, A_high_dim)
                    if c is not None:
                        sparse_code_x = c
            except Exception:
                c = _solve_projection(current_data_point, A_high_dim)
                if c is not None:
                    sparse_code_x = c
        else:
            c = _solve_projection(current_data_point, A_high_dim)
            if c is not None:
                sparse_code_x = c

        approx_y = A_high_dim @ sparse_code_x
        y_hat_point_2d = pca.transform(approx_y.reshape(1, -1))[0]

        # 2. Update tracking arrays
        Y_high_dim = np.column_stack((Y_high_dim, current_data_point))
        Y_hat_high_dim = np.column_stack((Y_hat_high_dim, approx_y))
        X_high_dim = np.column_stack((X_high_dim, sparse_code_x))
        
        reconstruction_error = np.linalg.norm(current_data_point - approx_y, ord=2)
        
        did_expand = False
        new_atom_index = None
        
        if reconstruction_error <= epsilon:
            status = "PASSED"
        else:
            status = "EXPANDED"
            did_expand = True
            
            # 3. Handle structure padding dimensions
            m = m + 1
            X_high_dim = np.vstack((X_high_dim, np.zeros((1, X_high_dim.shape[1]))))
            X_high_dim[-1, -1] = 1.0  
            
            atom_usage = np.sum(X_high_dim, axis=1)
            if np.any(atom_usage == 0):
                print(f"Iteration {idx}: Skipping update to avoid zero-support atoms.")
                continue
                
            ridge = 1e-8
            
            # 4. Gram Matrix Least-Squares Update Formula
            H = X_high_dim @ X_high_dim.T + sparse_penalty * np.diag(X_high_dim.sum(axis=1)) + ridge * np.eye(m)
            A_updated = (1 + sparse_penalty) * np.linalg.solve(H, X_high_dim @ Y_high_dim.T).T
            
            # Keep updated atoms correctly scaled on the unit-sphere surface
            A_updated /= np.linalg.norm(A_updated, axis=0, keepdims=True)
            A_high_dim = A_updated
            
            new_atom_index = m - 1
            
        # Transform updated atom vectors for 2D presentation
        atoms_2d = pca.transform(A_high_dim.T)
        
        hull_status = "Inside Hull" if in_hull else "Outside Hull"
        title_msg = (f"Point {idx} | Digit: {stream_y[idx]} | Atoms: {m} | {hull_status}\n"
                     f"Projection Error: {reconstruction_error:.4f} [{status}]")
        
        # Notice that we pass A_high_dim.T (which has shape m x 784) so we can map indices cleanly to positions
        update_visual_window(
            fig=fig, ax=ax, positions_2d=atoms_2d, atoms_high_dim=A_high_dim.T, title=title_msg,
            new_atom_idx=new_atom_index, y_point=y_point_2d, y_hat_point=y_hat_point_2d,
            current_stream_vector=current_data_point, image_zoom=0.45
        )
        time.sleep(1.2)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    execute_visual_stream()