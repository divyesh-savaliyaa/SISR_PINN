import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
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
    dv = v*(a-v)*(v-1) - w + sigma*noise
    dw = epsilon*(b*v - c*w)
    return dv, dw


def mse(a, b):
    return torch.mean((a-b)**2)
def nrmse_torch(true, pred):
    
    rmse = torch.sqrt(torch.mean((pred - true)**2))
    return rmse / torch.std(true)

def nrmse_np(true, pred):
    
    return np.sqrt(np.mean((pred - true)**2)) / np.std(true)

def ic_loss(model, v0, w0, noise0, dt):
    pred = model(v0, w0, noise0)
    v_pred, w_pred = pred[:, 0:1], pred[:, 1:2]
    dv_true, dw_true = true_dynamics(v0, w0, noise0)
    v_true_next = v0 + dv_true * dt
    w_true_next = w0 + dw_true * dt
    return mse(v_pred, v_true_next) + mse(w_pred, w_true_next)


def load_csv(filename):
    df = pd.read_csv(filename)
    t     = df.time.values
    v     = df.v.values   [:,None].astype(np.float32)
    w     = df.w.values   [:,None].astype(np.float32)
    noise = df.noise.values[:,None].astype(np.float32)
    return t, v, w, noise


def train(csv_file,
                      epochs=5000,
                      lr=1e-3,
                      dt=0.1,
                      lambda_phys=1,
                      lambda_ic=1,
                      batch_size=512):
    
    t, v_np, w_np, n_np = load_csv(csv_file)
    v_t     = torch.from_numpy(v_np)
    w_t     = torch.from_numpy(w_np)
    noise_t = torch.from_numpy(n_np)

   
    dv_true_step, dw_true_step = true_dynamics(v_t, w_t, noise_t)
    v_next_true = v_t + dv_true_step * dt
    w_next_true = w_t + dw_true_step * dt

    model     = StatePredictor()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_samples = v_t.shape[0]
    v0, w0, noise0 = v_t[0:1], w_t[0:1], noise_t[0:1]
    records = []
    
    best_epoch = 0
    best_total_nrmse = float('inf')

    t_start = time.perf_counter()
    for ep in range(1, epochs+1):
        optimizer.zero_grad()

        
        idx = torch.randperm(n_samples)[:batch_size]
        v_batch     = v_t[idx]
        w_batch     = w_t[idx]
        noise_batch = noise_t[idx]

        
        pred_batch = model(v_batch, w_batch, noise_batch)
        v_pred = pred_batch[:,0:1]
        w_pred = pred_batch[:,1:2]

        
        v_next_true_b = v_next_true[idx]
        w_next_true_b = w_next_true[idx]

        #data loss
        L_data_v = mse(v_pred, v_next_true_b)
        L_data_w = mse(w_pred, w_next_true_b)
        L_data   = L_data_v + L_data_w

        #physics loss
        dv_est_b = (v_pred - v_batch) / dt
        dw_est_b = (w_pred - w_batch) / dt
        dv_phys_b, dw_phys_b = true_dynamics(v_batch, w_batch, noise_batch)
        L_phys_v = mse(dv_est_b, dv_phys_b)
        L_phys_w = mse(dw_est_b, dw_phys_b)
        L_phys   = L_phys_v + L_phys_w

        #initial condition loss
        L_ic = ic_loss(model, v0, w0, noise0, dt)

        #total loss
        loss = L_data + lambda_phys * L_phys + lambda_ic * L_ic
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            pred_all = model(v_t, w_t, noise_t)
            v_pred_all = pred_all[:, :1]
            w_pred_all = pred_all[:, 1:2]

            total_nrmse = (
                nrmse_torch(v_next_true, v_pred_all)
                + nrmse_torch(w_next_true, w_pred_all)
            ).item()

            records.append({'epoch': ep, 'nrmse': total_nrmse})

            if total_nrmse < best_total_nrmse:
                best_total_nrmse = total_nrmse
                best_epoch = ep
                
            model.train()

        if ep == 1 or ep % 500 == 0:
            batch_nrmse_v = nrmse_torch(v_next_true[idx], v_pred).item()
            batch_nrmse_w = nrmse_torch(w_next_true[idx], w_pred).item()
            print(f"Epoch {ep:4d} | Data {L_data.item():.2e} | Phy {L_phys.item():.2e} | Ic {L_ic.item():.2e}| NRMSE v:{batch_nrmse_v:.3f}, w:{batch_nrmse_w:.3f}")
            
            records.append({'epoch': ep, 'nrmse': batch_nrmse_v + batch_nrmse_w})
    print(f"\nTraining took {time.perf_counter()-t_start:.2f}s")
    df = pd.DataFrame(records).set_index('epoch')
    print("Training NRMSE every 500 epochs:\n", df)

    print(f"\nLowest total NRMSE (v+w): {best_total_nrmse:.6f} at epoch {best_epoch}")


    
    if not df.empty:
        fig, ax = plt.subplots()
        ax.plot(df.index.to_numpy(), df['nrmse'].to_numpy(), lw=3,label='data+Physics1')
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
        plt.savefig('Physics1.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()


    return model

def annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.9):
    
    ax.axvline(t_boundary_eps, ls=':', lw=1.2, c='k', zorder=3)

    x_left  = (t_eps[0] + t_boundary_eps) / 2.0
    x_right = (t_boundary_eps + t_eps[-1]) / 2.0
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(x_left,  y_frac, 'Training',
            ha='center', va='top', transform=trans, clip_on=False, zorder=5,
            fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.text(x_right, y_frac, 'Prediction',
            ha='center', va='top', transform=trans, clip_on=False, zorder=5,
            fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def predict_and_plot_eps(model, full_csv, train_csv, dt=0.1, eps_scale=2.5e-4):
    
    t_full, v_np, w_np, n_np = load_csv(full_csv)
    t_train, *_ = load_csv(train_csv)
    t_boundary = t_train[-1]

    
    v_t = torch.from_numpy(v_np)
    w_t = torch.from_numpy(w_np)
    n_t = torch.from_numpy(n_np)

    
    dv_true, dw_true = true_dynamics(v_t, w_t, n_t)
    v_next_true = (v_t + dv_true * dt).cpu().numpy().flatten()
    w_next_true = (w_t + dw_true * dt).cpu().numpy().flatten()

   
    with torch.no_grad():
        pred = model(v_t, w_t, n_t)
    v_next_pred = pred[:, 0:1].cpu().numpy().flatten()
    w_next_pred = pred[:, 1:2].cpu().numpy().flatten()

    
    t_eps = eps_scale * np.asarray(t_full, dtype=float)
    t_boundary_eps = eps_scale * float(t_boundary)

      
    idx_pred = np.where(np.asarray(t_full) > t_boundary)[0]
    if idx_pred.size == 0:
        print("No prediction region (t > t_boundary).")
    else:
        take = min(3200, idx_pred.size)
        tail_idx = idx_pred[-take:]
        nrmse_v_tail = nrmse_np(v_next_true[tail_idx], v_next_pred[tail_idx])
        nrmse_w_tail = nrmse_np(w_next_true[tail_idx], w_next_pred[tail_idx])
        combined_tail = nrmse_v_tail + nrmse_w_tail
        print(f"Predict NRMSE (last {take} pts after boundary) → "
              f"v:{nrmse_v_tail:.3f}, w:{nrmse_w_tail:.3f}, combined:{combined_tail:.3f}")


    
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
    ax.set_ylabel('$v$', fontsize=30)
    ax.legend(loc='upper right')


    annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.83)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    plt.tight_layout()
   
    #plt.savefig('NASP_1_v.eps', format='eps', dpi=300, bbox_inches='tight')
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
    
    #plt.savefig('NASP_1_w.eps', format='eps', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    train_csv = "Data/train_data.csv"
    full_csv = "Data/test_data.csv"

    model = train(train_csv)


    predict_and_plot_eps(model, full_csv,train_csv)
