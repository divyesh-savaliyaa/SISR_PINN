import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'stix'   
mpl.rcParams['font.family'] = 'serif'
np.random.seed(42)
torch.manual_seed(42)

class StatePredictor(nn.Module):
    def __init__(self, hidden=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
    def forward(self, v, w, noise):
        x = torch.cat([v, w, noise], dim=1)
        return self.net(x)

def true_dynamics(v, w, noise,
                  a=0.05, b=1.0, c=2.0,
                  epsilon=0.005, sigma=0.03):
    dv = v * (a - v) * (v - 1) - w + sigma * noise
    dw = epsilon * (b * v - c * w)
    return dv, dw

def mse(a, b):
    return torch.mean((a - b) ** 2)

def nrmse_torch(true, pred):
    """Normalized RMSE for torch tensors."""
    rmse = torch.sqrt(torch.mean((pred - true)**2))
    return rmse / torch.std(true)

def nrmse_np(true, pred):
    """Normalized RMSE for numpy arrays."""
    return np.sqrt(np.mean((pred - true)**2)) / np.std(true)

def ic_loss(model, v0, w0, noise0, dt):
    pred = model(v0, w0, noise0)
    v_pred, w_pred = pred[:,0:1], pred[:,1:2]
    dv_true, dw_true = true_dynamics(v0, w0, noise0)
    v_true_next = v0 + dv_true * dt
    w_true_next = w0 + dw_true * dt
    return mse(v_pred, v_true_next) + mse(w_pred, w_true_next)

def load_csv(filename):
    df = pd.read_csv(filename)
    t     = df.time.values.astype(np.float32)
    v     = df.v.values   .astype(np.float32)[:,None]
    w     = df.w.values   .astype(np.float32)[:,None]
    noise = df.noise.values.astype(np.float32)[:,None]
    return t, v, w, noise

def train(csv_file,
                      epochs=5000,
                      lr=1e-3,
                      dt=0.1,
                      alpha_ic=1.0,
                      batch_size=512,
                      device='cpu'):
    
    t, v_np, w_np, n_np = load_csv(csv_file)
    v_t     = torch.from_numpy(v_np).to(device)
    w_t     = torch.from_numpy(w_np).to(device)
    noise_t = torch.from_numpy(n_np).to(device)
    
    dv_true, dw_true = true_dynamics(v_t, w_t, noise_t)
    v_next_true = v_t + dv_true * dt
    w_next_true = w_t + dw_true * dt
    
    model     = StatePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    v0, w0, noise0 = v_t[0:1], w_t[0:1], noise_t[0:1]
    n_samples = v_t.shape[0]

    records = []
    best_epoch = 0
    best_total_nrmse = float('inf')
    t_start = time.perf_counter()
    for ep in range(1, epochs+1):
        idx = torch.randperm(n_samples)[:batch_size]
        v_batch, w_batch, noise_batch = v_t[idx], w_t[idx], noise_t[idx]
        pred = model(v_batch, w_batch, noise_batch)
        v_pred, w_pred = pred[:,0:1], pred[:,1:2]

        loss_data   = mse(v_pred, v_next_true[idx]) + mse(w_pred, w_next_true[idx])
        loss_ic = ic_loss(model, v0, w0, noise0, dt)
        loss = loss_data + alpha_ic*loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred_all = model(v_t, w_t, noise_t)
            v_pred_all, w_pred_all = pred_all[:, :1], pred_all[:, 1:]
            nrmse_v_all = nrmse_torch(v_next_true, v_pred_all)
            nrmse_w_all = nrmse_torch(w_next_true, w_pred_all)
            total_nrmse = (nrmse_v_all + nrmse_w_all).item()
            records.append({'epoch': ep, 'nrmse': total_nrmse})
            if total_nrmse < best_total_nrmse:
                best_total_nrmse = total_nrmse
                best_epoch = ep
            model.train()

        if ep % 500 == 0:
            batch_nrmse_v = nrmse_torch(v_next_true[idx], v_pred).item()
            batch_nrmse_w = nrmse_torch(w_next_true[idx], w_pred).item()
            print(f"Epoch {ep:4d} | Data {loss_data:.2e} | IC {loss_ic:.2e} | "
                f"NRMSE v:{batch_nrmse_v:.3f}, w:{batch_nrmse_w:.3f}")

    print(f"\nTraining took {time.perf_counter()-t_start:.2f}s")
    df = pd.DataFrame(records).set_index('epoch')
    print("Training NRMSE (every epoch):\n", df.tail())

    print(f"\nLowest total NRMSE (v+w): {best_total_nrmse:.4f} at epoch {best_epoch}")


    if len(records) > 0:
        
        fig, ax = plt.subplots()
        ax.plot(df.index.to_numpy(), df['nrmse'].to_numpy(), lw=3,label='data')
        ax.set_xlim(0,5000)
        ax.set_ylim(0,0.2)
        ax.set_xticks(np.arange(0, 6000, 1000))
        ax.set_yticks(np.arange(0, 0.25, 0.05))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.set_xlabel('Epoch', fontsize=22)
        ax.set_ylabel('NRMSE', fontsize=22)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.tight_layout()
        ax.legend(loc='upper right', fontsize=18)
        plt.savefig('data.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()

    return model


def annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.985):
    
    ax.axvline(t_boundary_eps, ls=':', lw=1.2, c='k', zorder=3)

   
    x_left  = (t_eps[0] + t_boundary_eps) / 2.0
    x_right = (t_boundary_eps + t_eps[-1]) / 2.0

    
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(x_left,  y_frac, 'Training',
            ha='center', va='top',
            transform=trans, clip_on=False, zorder=5,
            fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.text(x_right, y_frac, 'Prediction',
            ha='center', va='top',
            transform=trans, clip_on=False, zorder=5,
            fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def predict_and_plot(model,
                     csv_file,
                     train_csv,
                     dt=0.1,
                     device='cpu',tail=3200):
    
    
    t,  v_np,  w_np,  n_np  = load_csv(csv_file)
    eps = 0.00025
    v_t, w_t, noise_t = map(lambda x: torch.from_numpy(x).to(device),
                            (v_np, w_np, n_np))
    t_train, *_ = load_csv(train_csv)
    t_boundary  = t_train[-1]
    t_eps = eps *t
    t_boundary_eps = eps * t_boundary
    
    dv_true, dw_true = true_dynamics(v_t, w_t, noise_t)
    v_next_true = (v_t + dv_true * dt).cpu().numpy()
    w_next_true = (w_t + dw_true * dt).cpu().numpy()

  
    with torch.no_grad():
        pred = model(v_t, w_t, noise_t).cpu().numpy()
    v_next_pred, w_next_pred = pred[:, 0:1], pred[:, 1:2]

    v_true_tail = v_next_true[-tail:]
    w_true_tail = w_next_true[-tail:]
    v_pred_tail = v_next_pred[-tail:]
    w_pred_tail = w_next_pred[-tail:]


    nrmse_v = nrmse_np(v_true_tail, v_pred_tail)
    nrmse_w = nrmse_np(w_true_tail, w_pred_tail)
    combined = nrmse_v + nrmse_w
    print(f"Prediction NRMSE → {combined:.3f}")

   
    fig, ax = plt.subplots()
    ax.plot(t_eps, v_next_true, label='True v')
    ax.plot(t_eps, v_next_pred, '--', label='Pred v')
    ax.grid(True)

    ax.set_xlim(0, 2)
    ax.set_ylim(-0.8, 1.6)
    ax.set_xticks(np.arange(0, 2.5, 0.5))
    plt.xticks(fontsize=22)
    y_ticks = np.linspace(-0.8, 1.6, 7)
    labels  = ['0' if np.isclose(t, 0.0, atol=1e-12) else f'{t:.1f}' for t in y_ticks]
    plt.yticks(y_ticks, labels, fontsize=22)


    ax.set_xlabel(r'$\varepsilon t$', fontsize=32)
    ax.set_ylabel('$v$', fontsize=32)
    ax.legend(loc='upper right')

    annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.83)  
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.tight_layout()
    
    plt.savefig('NASP_v.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.show()



    fig, ax = plt.subplots()
    ax.plot(t_eps, w_next_true, label='True w')
    ax.plot(t_eps, w_next_pred, '--', label='Pred w')
    ax.grid(True)

    ax.set_xlim(0, 2)
    ax.set_ylim(-0.05, 0.25)  
    ax.set_xticks(np.arange(0, 2.5, 0.5))
    plt.xticks(fontsize=22)
    y_ticks = np.linspace(-0.05, 0.25, 5)
    plt.yticks(y_ticks, [f"{tick:.2f}" for tick in y_ticks], fontsize=22)
    ax.set_xlabel(r'$\varepsilon t$', fontsize=32)
    ax.set_ylabel('$w$', fontsize=32)
    ax.legend(loc='upper right')

    annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.83)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.tight_layout()
    
    plt.savefig('NASP_w.eps', format='eps', dpi=300, bbox_inches='tight')  
    plt.show()


if __name__ == "__main__":
    t0 = time.perf_counter()

    train_csv = "../data/train_data.csv"
    full_csv  = "../data/test_data.csv"
    dt        = 0.1

    model = train(train_csv)

    
    predict_and_plot(model, full_csv, train_csv)