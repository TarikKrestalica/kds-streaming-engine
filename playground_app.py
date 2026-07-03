import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import time

# Import your production-ready updates
from config import CONFIG, get_model_save_path
from dataset import UnitNormFilteredMNIST
from model import KDS
from loss import LocalDictionaryLoss
from stream_engine import ProductionStreamSimulator

# --- App Layout Configuration ---
st.set_page_config(layout="wide", page_title="KDS Trained Production Playground")
st.title("🔬 KDS Pre-Training & Streaming Engine Playground")
st.caption("Synchronized Evaluator Dashboard: Tracks dictionary atom expansion alongside reconstruction anomalies.")

# --- Sidebar Controls ---
st.sidebar.header("🎛️ 1. Pre-Training Parameters")
train_epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=300, value=10, step=1)
train_lr = st.sidebar.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 10e-3], value=float(CONFIG["learning_rate"]))
train_batch_size = st.sidebar.select_slider("Batch Size", options=[32, 64, 128, 256, 512, 1024], value=int(CONFIG["batch_size"]))
train_hidden_size = st.sidebar.slider("Hidden Size (Dictionary Atoms)", min_value=8, max_value=750, value=int(CONFIG["hidden_size"]), step=1)

scheduler_type = st.sidebar.selectbox("LR Scheduler", ["None", "StepLR", "CosineAnnealing", "ReduceOnPlateau"])
sched_step_size, sched_gamma, sched_tmax, sched_factor, sched_patience = 5, 0.5, 10, 0.5, 3
if scheduler_type == "StepLR":
    sched_step_size = st.sidebar.slider("Step Size (epochs)", min_value=1, max_value=20, value=5, step=1)
    sched_gamma = st.sidebar.slider("Decay Factor (γ)", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
elif scheduler_type == "CosineAnnealing":
    sched_tmax = st.sidebar.slider("T_max (epochs)", min_value=1, max_value=50, value=10, step=1)
elif scheduler_type == "ReduceOnPlateau":
    sched_factor = st.sidebar.slider("Reduction Factor", min_value=0.10, max_value=0.95, value=0.50, step=0.05)
    sched_patience = st.sidebar.slider("Patience (epochs)", min_value=1, max_value=10, value=3, step=1)

st.sidebar.header("🎛️ 2. Streaming Parameters")
anomaly_threshold = st.sidebar.slider("Anomaly Threshold (ε)", min_value=0.10, max_value=1.50, value=float(CONFIG["anomaly_threshold"]), step=0.05)
num_pgd_layers = st.sidebar.slider("PGD Unrolled Layers", min_value=10, max_value=300, value=int(CONFIG["num_layers"]), step=10)
penalty_lambda = st.sidebar.slider("Proximity Penalty (λ)", min_value=0.0, max_value=1.5, value=float(CONFIG["penalty"]), step=0.05)
stream_ratio = st.sidebar.slider("In-Distribution Ratio", min_value=0.10, max_value=1.0, value=0.80, step=0.05)
stream_speed = st.sidebar.slider("Stream Delay (Seconds)", min_value=0.1, max_value=3.0, value=0.4, step=0.1)

# --- BULLETPROOF STATE INITIALIZATION ---
if "initialized" not in st.session_state:
    train_ds = UnitNormFilteredMNIST(target_digits=CONFIG["target_digits"], subsample_size=int(CONFIG["subsample_size"]))
    st.session_state.train_dataset = train_ds
    
    pca_obj = PCA(n_components=2, random_state=42)
    pca_obj.fit(train_ds.X.numpy())
    st.session_state.pca = pca_obj
    
    # Initialize dictionary rows from training samples to mirror structured baseline initialization
    init_W = torch.tensor(train_ds.X[:int(CONFIG["hidden_size"])].clone(), dtype=torch.float32)
    
    st.session_state.kds_instance = KDS(
        num_layers=int(CONFIG["num_layers"]), 
        input_size=784, 
        hidden_size=int(CONFIG["hidden_size"]), 
        penalty=float(CONFIG["penalty"]),
        W=init_W
    )
    
    # App metric tracking caches
    st.session_state.is_trained = False
    st.session_state.step_idx = 0
    st.session_state.history_errors_cvx = []
    st.session_state.history_errors_kds = []
    st.session_state.history_atom_counts = []
    # Per-class error histories (step index + error value kept in sync)
    st.session_state.history_steps_in = []
    st.session_state.history_errors_cvx_in = []
    st.session_state.history_errors_kds_in = []
    st.session_state.history_steps_out = []
    st.session_state.history_errors_cvx_out = []
    st.session_state.history_errors_kds_out = []
    # Loss decomposition histories (streaming)
    st.session_state.history_recon_loss = []
    st.session_state.history_penalty_loss = []
    # Per-epoch training loss histories
    st.session_state.history_train_total_loss = []
    st.session_state.history_train_recon_loss = []
    st.session_state.history_train_penalty_loss = []
    st.session_state.initialized = True

# --- Layout Configuration Split ---
col1, col2 = st.columns([3, 2])

with col1:
    plot_placeholder = st.empty()

with col2:
    st.subheader("📈 System Metrics & Analytics")
    status_msg_box = st.empty()
    
    # Dual Column Metrics Grid
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        metric_cvx = st.empty()
        metric_kds = st.empty()
    with m_col2:
        metric_atoms = st.empty()  # Displays active atom dimensions live
        metric_recon_loss = st.empty()
        metric_penalty_loss = st.empty()
    
    chart_placeholder = st.empty()
    
    if not st.session_state.is_trained:
        status_msg_box.warning("⚠️ Model Bounding Dictionary is Untrained. Please execute 'Run Baseline Pre-Training' first.")

# --- Action Hooks ---
btn_train = st.sidebar.button("🚀 Run Baseline Pre-Training")
btn_stream = st.sidebar.button("📡 Execute Streaming Simulation")

# --- UI Helper Plot Engine ---
def render_playground_view(positions_2d, atoms_high_dim, y_2d, y_hat_2d, current_vec, new_atom_idx, title, atom_coeffs=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    m = positions_2d.shape[0]

    if m >= 3:
        try:
            dt = Delaunay(positions_2d)
            ax.triplot(positions_2d[:, 0], positions_2d[:, 1], dt.simplices, color='0.8', lw=1, zorder=1)
        except:
            pass

    ACTIVE_THRESH = 0.01
    for node_idx in range(m):
        img = atoms_high_dim[node_idx].reshape(28, 28)
        imagebox = OffsetImage(img, zoom=0.35, cmap='gray')

        # Priority: newly added atom (red) > active simplex atom (gold) > inactive (black)
        if new_atom_idx is not None and node_idx == new_atom_idx:
            edge_color, lw = 'red', 2.0
        elif (atom_coeffs is not None
              and node_idx < len(atom_coeffs)
              and atom_coeffs[node_idx] > ACTIVE_THRESH):
            # Thicker gold border for atoms with larger coefficient weight
            edge_color = 'goldenrod'
            lw = 1.0 + 3.0 * float(atom_coeffs[node_idx])
        else:
            edge_color, lw = 'black', 0.6

        bbox_props = dict(edgecolor=edge_color, linewidth=lw, facecolor='none')
        ab = AnnotationBbox(imagebox, positions_2d[node_idx], frameon=True, bboxprops=bbox_props, pad=0.05, zorder=3)
        ax.add_artist(ab)
        
    if y_2d is not None and y_hat_2d is not None:
        img_y = current_vec.reshape(28, 28)
        imagebox_y = OffsetImage(img_y, zoom=0.35, cmap='gray')
        ab_y = AnnotationBbox(imagebox_y, y_2d, frameon=True, bboxprops=dict(edgecolor='blue', linewidth=2.0, facecolor='none'), pad=0.05, zorder=5)
        ax.add_artist(ab_y)
        ax.scatter(y_hat_2d[0], y_hat_2d[1], s=50, c='red', marker='o', zorder=5)
        ax.plot([y_2d[0], y_hat_2d[0]], [y_2d[1], y_hat_2d[1]], color='black', linestyle='--', lw=1, zorder=2)
        
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    
    all_pts = positions_2d if y_2d is None else np.vstack([positions_2d, y_2d])
    pad = 0.15 * max(np.ptp(all_pts, axis=0).max(), 1e-9)
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    return fig


def _solve_projection(y, A_subset):
    """Solve min ||y - x @ A_subset||_2  s.t.  x >= 0, sum(x) == 1."""
    k = A_subset.shape[0]
    x = cp.Variable(k)
    prob = cp.Problem(cp.Minimize(cp.norm(y - x @ A_subset, 2)), [x >= 0, cp.sum(x) == 1])
    prob.solve()
    return x.value  # None when solver fails


# --- PHASE A: BASELINE PRE-TRAINING LOOP ---
if btn_train:
    st.session_state.is_trained = False
    # Clear streaming state and all histories so each training run starts clean
    for _k in ["stream_X", "stream_y", "stream_in_target"]:
        st.session_state.pop(_k, None)
    st.session_state.step_idx = 0
    for _k in ["history_errors_cvx", "history_errors_kds", "history_atom_counts",
                "history_steps_in", "history_errors_cvx_in", "history_errors_kds_in",
                "history_steps_out", "history_errors_cvx_out", "history_errors_kds_out",
                "history_recon_loss", "history_penalty_loss",
                "history_train_total_loss", "history_train_recon_loss", "history_train_penalty_loss"]:
        st.session_state[_k] = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    train_ds = st.session_state.train_dataset

    # Re-create KDS with the slider-selected hidden size so the user can experiment
    init_W = train_ds.X[:train_hidden_size].clone().float()
    net = KDS(
        num_layers=int(CONFIG["num_layers"]),
        input_size=784,
        hidden_size=train_hidden_size,
        penalty=float(penalty_lambda),
        W=init_W,
    )
    st.session_state.kds_instance = net

    # Mirror run_pipeline.py: seed W from random training samples, init Lipschitz step via SVD
    with torch.no_grad():
        p = torch.randperm(len(train_ds))[:int(net.hidden_size)]
        net.W.data = train_ds.X[p]
        net.step.fill_((net.W.data.svd()[1][0] ** -2).item())

    criterion = LocalDictionaryLoss(penalty=penalty_lambda)
    optimizer = torch.optim.Adam(net.parameters(), lr=train_lr)

    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sched_step_size, gamma=sched_gamma)
    elif scheduler_type == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sched_tmax)
    elif scheduler_type == "ReduceOnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=sched_factor, patience=sched_patience)
    else:
        scheduler = None

    net.train()

    batch_size = train_batch_size
    for epoch in range(train_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_penalty_loss = 0.0
        num_batches = 0

        # Manual shuffle matches run_pipeline.py exactly
        shuffle = torch.randperm(len(train_ds))
        data = train_ds.X[shuffle]

        for i in range(0, len(train_ds), batch_size):
            y = data[i : i + batch_size]

            # Use encode_accelerated directly (no redundant reconstruction pass)
            x_hat = net.encode_accelerated(y)
            recon_loss_t, penalty_loss_t, loss = criterion.forward_detailed(net.W, y, x_hat)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1e-4)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss_t.item()
            epoch_penalty_loss += (penalty_loss_t * penalty_lambda).item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_penalty_loss = epoch_penalty_loss / num_batches

        st.session_state.history_train_total_loss.append(avg_loss)
        st.session_state.history_train_recon_loss.append(avg_recon_loss)
        st.session_state.history_train_penalty_loss.append(avg_penalty_loss)

        if scheduler is not None:
            if scheduler_type == "ReduceOnPlateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.progress(int(((epoch + 1) / train_epochs) * 100))
        status_text.text(
            f"Epoch {epoch+1}/{train_epochs} | Total: {avg_loss:.4f} | "
            f"Recon: {avg_recon_loss:.4f} | Penalty: {avg_penalty_loss:.4f} | LR: {current_lr:.2e}"
        )

        # Live training loss chart
        fig_train, ax_train = plt.subplots(figsize=(5, 2.8))
        epochs_x = list(range(1, epoch + 2))
        ax_train.plot(epochs_x, st.session_state.history_train_total_loss,
                      color='black', lw=1.8, label='Total Loss')
        ax_train.plot(epochs_x, st.session_state.history_train_recon_loss,
                      color='teal', lw=1.5, linestyle='--', label='Recon Loss')
        ax_train.plot(epochs_x, st.session_state.history_train_penalty_loss,
                      color='crimson', lw=1.5, linestyle=':', label='Penalty Loss')
        ax_train.set_title("Training Loss Decomposition", fontsize=8, fontweight='bold')
        ax_train.set_xlabel("Epoch", fontsize=7)
        ax_train.legend(fontsize=6)
        ax_train.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        chart_placeholder.pyplot(fig_train)
        plt.close(fig_train)

    torch.save(net.state_dict(), get_model_save_path())
    st.session_state.is_trained = True
    status_msg_box.success(f"🎉 Pre-Training Complete! Checkpoint saved. Final Loss: {avg_loss:.4f}")

    A_trained = net.W.detach().numpy()
    atoms_2d = st.session_state.pca.transform(A_trained)
    fig = render_playground_view(atoms_2d, A_trained, None, None, None, None, "Optimized Baseline Dictionary Hull")
    plot_placeholder.pyplot(fig)
    plt.close(fig)

# --- PHASE B: LIVE INCREMENTAL STREAMING LOOP ---
if btn_stream:
    if not st.session_state.is_trained:
        st.error("Cannot execute data stream until baseline training is completed.")
    else:
        if "stream_X" not in st.session_state:
            simulator = ProductionStreamSimulator(target_digits=CONFIG["target_digits"], total_stream_samples=100, target_ratio=stream_ratio)
            stream_X, stream_y, in_target = simulator.generate_interleaved_stream(trained_indices_to_exclude=st.session_state.train_dataset.chosen_indices)
            st.session_state.stream_X = stream_X
            st.session_state.stream_y = stream_y
            st.session_state.stream_in_target = in_target
            
        while st.session_state.step_idx < len(st.session_state.stream_X):
            idx = st.session_state.step_idx
            current_data_point = st.session_state.stream_X[idx]
            true_digit = st.session_state.stream_y[idx]
            
            is_in_dist = bool(st.session_state.stream_in_target[idx])
            dist_tag = "✅ In-Distribution" if is_in_dist else "⚠️ Out-of-Distribution"

            A_current = st.session_state.kds_instance.W.detach().numpy()
            m = A_current.shape[0]

            # Project atoms and the current point into 2D before CVXPY so we can locate
            # the containing Delaunay simplex and restrict the solve to its 3 vertices.
            atoms_2d_pre = st.session_state.pca.transform(A_current)
            y_point_2d = st.session_state.pca.transform(current_data_point.reshape(1, -1))[0]

            # 1. Delaunay-constrained CVXPY:
            #    - If inside the hull: solve over exactly the 3 atoms of the containing triangle.
            #    - If outside the hull or m < 3: unconstrained solve over all atoms.
            atom_coeffs = np.zeros(m)
            in_hull = False
            if m >= 3:
                try:
                    dt_pre = Delaunay(atoms_2d_pre)
                    simplex_idx = dt_pre.find_simplex(y_point_2d)
                    if simplex_idx >= 0:
                        in_hull = True
                        active_indices = dt_pre.simplices[simplex_idx]  # exactly 3 vertex indices
                        c = _solve_projection(current_data_point, A_current[active_indices])
                        if c is not None:
                            atom_coeffs[active_indices] = c
                    else:
                        c = _solve_projection(current_data_point, A_current)
                        if c is not None:
                            atom_coeffs = c
                except Exception:
                    c = _solve_projection(current_data_point, A_current)
                    if c is not None:
                        atom_coeffs = c
            else:
                c = _solve_projection(current_data_point, A_current)
                if c is not None:
                    atom_coeffs = c

            hull_status = "Inside Hull" if in_hull else "Outside Hull"
            recon = atom_coeffs @ A_current
            recon_err_cvx = np.linalg.norm(current_data_point - recon, 2)
            y_hat_point_2d = st.session_state.pca.transform(recon.reshape(1, -1))[0]

            # 2. Sync slider hyperparameters into model buffers and run KDS forward pass
            st.session_state.kds_instance.num_layers.copy_(torch.tensor(int(num_pgd_layers)))
            st.session_state.kds_instance.penalty.copy_(torch.tensor(float(penalty_lambda)))
            st.session_state.kds_instance.eval()

            tensor_y = torch.tensor(current_data_point, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                y_hat_kds, x_hat = st.session_state.kds_instance(tensor_y)
                y_hat_kds_np = y_hat_kds.squeeze(0).numpy()
                stream_criterion = LocalDictionaryLoss(penalty=penalty_lambda)
                recon_loss, penalty_loss, total_loss = stream_criterion.forward_detailed(
                    st.session_state.kds_instance.W, tensor_y, x_hat
                )

            recon_err_kds = np.linalg.norm(current_data_point - y_hat_kds_np, 2)
            scaled_penalty_loss = penalty_loss * penalty_lambda
            st.session_state.history_recon_loss.append(recon_loss.item())
            st.session_state.history_penalty_loss.append(scaled_penalty_loss.item())

            # 3. Record errors — combined history for delta computation, split by true class
            prev_cvx = st.session_state.history_errors_cvx[-1] if st.session_state.history_errors_cvx else None
            prev_kds = st.session_state.history_errors_kds[-1] if st.session_state.history_errors_kds else None
            st.session_state.history_errors_cvx.append(recon_err_cvx)
            st.session_state.history_errors_kds.append(recon_err_kds)
            st.session_state.history_atom_counts.append(m)

            if is_in_dist:
                st.session_state.history_steps_in.append(idx)
                st.session_state.history_errors_cvx_in.append(recon_err_cvx)
                st.session_state.history_errors_kds_in.append(recon_err_kds)
            else:
                st.session_state.history_steps_out.append(idx)
                st.session_state.history_errors_cvx_out.append(recon_err_cvx)
                st.session_state.history_errors_kds_out.append(recon_err_kds)

            # 4. Expansion decision
            new_atom_index = None
            if recon_err_kds <= anomaly_threshold:
                classification = "PASSED"
            else:
                classification = "EXPANDED"
                new_atom = tensor_y.clone()
                st.session_state.kds_instance.W.data = torch.cat([st.session_state.kds_instance.W.data, new_atom], dim=0)
                st.session_state.kds_instance.W.data /= torch.norm(st.session_state.kds_instance.W.data, dim=1, keepdim=True)
                st.session_state.kds_instance.hidden_size += 1
                new_atom_index = st.session_state.kds_instance.W.shape[0] - 1

            # 5. Pad atom_coeffs for the newly added atom (coefficient = 0; not used in current reconstruction)
            if new_atom_index is not None:
                atom_coeffs = np.append(atom_coeffs, 0.0)

            # 6. Simplex visualisation
            atoms_2d = st.session_state.pca.transform(st.session_state.kds_instance.W.detach().numpy())
            title_msg = (f"Step {idx} | Digit {true_digit} | {dist_tag} | "
                         f"{hull_status} | Atoms: {st.session_state.kds_instance.W.shape[0]}")
            fig = render_playground_view(
                atoms_2d, st.session_state.kds_instance.W.detach().numpy(),
                y_point_2d, y_hat_point_2d, current_data_point,
                new_atom_index, title_msg, atom_coeffs=atom_coeffs
            )
            plot_placeholder.pyplot(fig)
            plt.close(fig)

            # 7. Dashboard metrics — delta shows change from previous step (inverse: lower = green)
            delta_cvx = f"{recon_err_cvx - prev_cvx:+.4f}" if prev_cvx is not None else None
            delta_kds = f"{recon_err_kds - prev_kds:+.4f}" if prev_kds is not None else None
            status_msg_box.markdown(
                f"**Step {idx}** | Digit `{true_digit}` | {dist_tag}  \n"
                f"**Model Decision:** `{classification}` | `{hull_status}`"
            )
            metric_cvx.metric("CVX Hull Error", f"{recon_err_cvx:.4f}", delta=delta_cvx, delta_color="inverse")
            metric_kds.metric("KDS Layer Error", f"{recon_err_kds:.4f}", delta=delta_kds, delta_color="inverse")
            metric_atoms.metric(
                "Active Atoms",
                f"{st.session_state.kds_instance.W.shape[0]}",
                "+1" if new_atom_index is not None else None
            )
            prev_recon = st.session_state.history_recon_loss[-2] if len(st.session_state.history_recon_loss) >= 2 else None
            prev_penalty = st.session_state.history_penalty_loss[-2] if len(st.session_state.history_penalty_loss) >= 2 else None
            metric_recon_loss.metric(
                "Recon Loss",
                f"{recon_loss.item():.4f}",
                delta=f"{recon_loss.item() - prev_recon:+.4f}" if prev_recon is not None else None,
                delta_color="inverse"
            )
            metric_penalty_loss.metric(
                "Penalty Loss (λ·b)",
                f"{scaled_penalty_loss.item():.4f}",
                delta=f"{scaled_penalty_loss.item() - prev_penalty:+.4f}" if prev_penalty is not None else None,
                delta_color="inverse"
            )

            # 8. Four-panel chart: in-distribution errors / OOD errors / atom growth / loss decomposition
            fig_chart, (ax_in, ax_out, ax_dim, ax_loss) = plt.subplots(4, 1, figsize=(5, 7.5))

            if st.session_state.history_steps_in:
                ax_in.plot(st.session_state.history_steps_in, st.session_state.history_errors_cvx_in,
                           color='steelblue', lw=1.5, marker='o', ms=3, label='CVX Error')
                ax_in.plot(st.session_state.history_steps_in, st.session_state.history_errors_kds_in,
                           color='cornflowerblue', lw=1.5, linestyle=':', marker='x', ms=3, label='KDS Error')
            ax_in.axhline(anomaly_threshold, color='red', linestyle='--', alpha=0.5, lw=1.0, label='ε')
            ax_in.set_title("In-Distribution Reconstruction Error", fontsize=8, fontweight='bold')
            ax_in.legend(fontsize=6)
            ax_in.grid(True, linestyle=':', alpha=0.5)

            if st.session_state.history_steps_out:
                ax_out.plot(st.session_state.history_steps_out, st.session_state.history_errors_cvx_out,
                            color='darkorange', lw=1.5, marker='o', ms=3, label='CVX Error')
                ax_out.plot(st.session_state.history_steps_out, st.session_state.history_errors_kds_out,
                            color='goldenrod', lw=1.5, linestyle=':', marker='x', ms=3, label='KDS Error')
            ax_out.axhline(anomaly_threshold, color='red', linestyle='--', alpha=0.5, lw=1.0, label='ε')
            ax_out.set_title("Out-of-Distribution Reconstruction Error", fontsize=8, fontweight='bold')
            ax_out.legend(fontsize=6)
            ax_out.grid(True, linestyle=':', alpha=0.5)

            ax_dim.plot(st.session_state.history_atom_counts, color='purple', lw=1.8, label='Dictionary Size (m)')
            ax_dim.set_title("Dictionary Growth (Basis Expansion)", fontsize=8, fontweight='bold')
            ax_dim.legend(fontsize=6)
            ax_dim.grid(True, linestyle=':', alpha=0.5)

            ax_loss.plot(st.session_state.history_recon_loss, color='teal', lw=1.5, marker='o', ms=3, label='Recon Loss')
            ax_loss.plot(st.session_state.history_penalty_loss, color='crimson', lw=1.5, linestyle=':', marker='x', ms=3, label='Penalty Loss')
            ax_loss.set_title("Loss Decomposition (KDS Criterion)", fontsize=8, fontweight='bold')
            ax_loss.set_xlabel("Stream Step", fontsize=7)
            ax_loss.legend(fontsize=6)
            ax_loss.grid(True, linestyle=':', alpha=0.5)

            plt.tight_layout()
            chart_placeholder.pyplot(fig_chart)
            plt.close(fig_chart)
            
            st.session_state.step_idx += 1
            time.sleep(stream_speed)

# --- Manual Reset Hook ---
if st.sidebar.button("🧹 Reset Playground Environment"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()