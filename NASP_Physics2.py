import time
import math
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

def compute_barriers(a, sigma, w):
   
    if not torch.is_tensor(a):
        a = torch.tensor(a, dtype=w.dtype, device=w.device)
    else:
        a = a.to(w.dtype).to(w.device)
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=w.dtype, device=w.device)
    else:
        sigma = sigma.to(w.dtype).to(w.device)

    B = -(a + 1)
    C = a
    D = w
    p = C - B*B / 3
    q = 2*B**3/27 - (B*C)/3 + D
    Δ = (q/2)**2 + (p/3)**3

    sqrtΔ = torch.sqrt(Δ.clamp(min=0))
    u = torch.sign(q/2 + sqrtΔ) * ((q/2 + sqrtΔ).abs().pow(1/3))
    v = torch.sign(q/2 - sqrtΔ) * ((q/2 - sqrtΔ).abs().pow(1/3))
    t1 = u + v

    r = torch.sqrt((-p/3).clamp(min=0))
    arg = (-q) / (2*(r**3) + 1e-20)
    arg = arg.clamp(-1 + 1e-6, 1 - 1e-6)
    phi = torch.acos(arg)
    tA = 2 * r * torch.cos(phi/3)
    tB = 2 * r * torch.cos(phi/3 + 2*math.pi/3)

    v1 = t1 - B/3
    v2 = tA - B/3
    v3 = tB - B/3
    mask_three_real = (Δ < 0) 
    mask_pos = (Δ > 0).float()
    BIG = torch.tensor(1e6, device=w.device)
    v2_mod = v2 + mask_pos * BIG
    v3_mod = v3 + mask_pos * BIG

    stacked = torch.stack([v1, v2_mod, v3_mod], dim=0)
    roots, _ = torch.sort(stacked, dim=0)
    v_minus = roots[0]; v_zero = roots[1]; v_plus = roots[2]
   
    def U(x):  return 0.25*x**4 - ((a+1)/3)*x**3 + (a/2)*x**2 + x*w
    def U2(x): return 3*x**2 - 2*(a+1)*x + a

    U0 = U(v_zero)
    Um = U(v_minus)
    Up = U(v_plus)
    Δm = U0 - Um
    Δp = U0 - Up
    log_eps_m = ((2 * Δm) / (sigma**2)).clamp(min=-60.0, max=60.0)
    log_eps_p = ((2 * Δp) / (sigma**2)).clamp(min=-60.0, max=60.0)

    neg = (w < 0).float()
    log_eps = neg * log_eps_m + (1-neg) * log_eps_p

    return log_eps, Δm, Δp, mask_three_real

def gather_by_continuous_index(w_seq, idx):
    
    B,T,_ = w_seq.shape
    i0 = idx.floor().long().clamp(0, T-2)
    i1 = i0 + 1
    frac = (idx - i0.float()).unsqueeze(-1)
    w0 = w_seq[torch.arange(B), i0]
    w1 = w_seq[torch.arange(B), i1]
    return (1-frac)*w0 + frac*w1


def rollout(model, v0, w0, noise_seq, dt):
    B, T, _ = noise_seq.shape
    v, w = v0, w0
    vs, ws = [], []

    for t in range(T):
        n_t = noise_seq[:, t, :]               
        x   = torch.cat([v, w, n_t], dim=1)   
        dv_dw = model.dyn(x)                   
        dv, dw = dv_dw[:,0:1], dv_dw[:,1:2]

        v = v + dv * dt
        w = w + dw * dt

        vs.append(v)
        ws.append(w)

    v_seq = torch.stack(vs, dim=1)   
    w_seq = torch.stack(ws, dim=1)
    return v_seq, w_seq

class StatePredictor(nn.Module):
    def __init__(self, T, hidden=50,dt=0.1):
        super().__init__()
        self.T = T
        self.dt = dt
        self.dyn = nn.Sequential(
            nn.Linear(3,hidden), nn.Tanh(),
            nn.Linear(hidden,2)
        )
        self.index_head = nn.Linear(3,2)


    def forward(self, v, w, noise):
        
        x = torch.cat([v, w, noise], dim=1)        

        dv_dw = self.dyn(x)                          
        dv, dw = dv_dw[:,0:1], dv_dw[:,1:2]          

        v1 = v + dv * self.dt                             
        w1 = w + dw * self.dt                            

        raw_idx = self.index_head(x)                 
        idx = torch.sigmoid(raw_idx) * (self.T-1)
        return v1, w1, idx


def true_dynamics(v, w, noise, a=0.05, b=1.0, c=2.0, eps=0.005, sigma=0.03):
    dv = v*(a-v)*(v-1) - w + sigma*noise
    dw = eps*(b*v - c*w)
    return dv, dw


def mse(a, b):
    return torch.mean((a-b)**2)
def nrmse_torch(true, pred):
   
    rmse = torch.sqrt(torch.mean((pred - true)**2))
    return rmse / torch.std(true)

def nrmse_np(true, pred):
    
    return np.sqrt(np.mean((pred - true)**2)) / np.std(true)
def ic_loss(model, v0, w0, noise0, dt):
    v1_pred, w1_pred, _ = model(v0, w0, noise0)
    dv0, dw0 = true_dynamics(v0, w0, noise0)
    v1_true = v0 + dv0 * dt
    w1_true = w0 + dw0 * dt
    return mse(v1_pred, v1_true) + mse(w1_pred, w1_true)


def load_data(csv_file, T=None):
    df = pd.read_csv(csv_file)
    
    t = df['time'].values
    v = df['v'].values[:,None].astype(np.float32)
    w = df['w'].values[:,None].astype(np.float32)
    noise = df['noise'].values[:,None].astype(np.float32)
    

    if T is not None:
        total = len(df)
        B = total // T
        trimmed = B * T
        v_seq = v[:trimmed].reshape(B, T, 1)
        w_seq = w[:trimmed].reshape(B, T, 1)
        noise_seq = noise[:trimmed].reshape(B, T, 1)
        v_seq     = torch.from_numpy(v_seq)
        w_seq     = torch.from_numpy(w_seq)
        noise_seq = torch.from_numpy(noise_seq)
        return v_seq, w_seq, noise_seq

    return t, v, w, noise


def train(csv_file, T=200, epochs=5000, lr=1e-3, dt=0.1,
          alpha_ic=10, alpha_phys=1,
          a=0.05, sigma=0.03,
          batch_size=512):

    v_all, w_all, noise_all = load_data(csv_file, T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v_all, w_all, noise_all = v_all.to(device), w_all.to(device), noise_all.to(device)
    v0 = v_all[:, 0, 0].unsqueeze(1)
    w0 = w_all[:, 0, 0].unsqueeze(1)
    noise0 = noise_all[:, 0, 0].unsqueeze(1)

    model = StatePredictor(T).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)


    best_epoch = 0
    best_total_nrmse = float('inf')
    records = []

    t_start = time.perf_counter()
    for ep in range(1, epochs+1):
        opt.zero_grad()
        idx_batch = torch.randperm(v_all.size(0))[:batch_size]
        v_b, w_b, n_b = v_all[idx_batch], w_all[idx_batch], noise_all[idx_batch]
        
        v0_batch   = v_b[:, 0, 0].unsqueeze(1)  
        w0_batch   = w_b[:, 0, 0].unsqueeze(1)   
        noise_batch = n_b[:, 0, 0].unsqueeze(1) 
        v_seq_pred, w_seq_pred = rollout(model, v0_batch, w0_batch, n_b, dt)
       
        # Physical loss
        _, _, idx_pred = model(v0_batch, w0_batch, noise_batch)
        v1_pred = v_seq_pred[:, 0, :]   
        w1_pred = w_seq_pred[:, 0, :]   
        
        i_minus, i_plus = idx_pred[:,0], idx_pred[:,1]

  
        w_pred_minus = gather_by_continuous_index(w_seq_pred, i_minus)
        w_pred_plus  = gather_by_continuous_index(w_seq_pred, i_plus)

        w_true_minus = gather_by_continuous_index(w_b, i_minus).detach()
    
        _, Δm, Δp, mask = compute_barriers(a, sigma, w_true_minus.view(-1))

       
        log_eps_minus_pred, _, _, _ = compute_barriers(a, sigma, w_pred_minus.view(-1))
        log_eps_plus_pred,  _, _,_ = compute_barriers(a, sigma, w_pred_plus .view(-1))
        
        Phi_minus = 0.5 * sigma**2 * log_eps_minus_pred
        Phi_plus  = 0.5 * sigma**2 * log_eps_plus_pred

        loss_phys = (
            ((Phi_minus - Δm)[mask]).pow(2).mean() +
            ((Phi_plus  - Δp )[mask]).pow(2).mean()
        )
        
        # Data loss
        v_seq_pred, w_seq_pred = rollout(model, v0_batch, w0_batch, n_b, dt)
        v1_pred = v_seq_pred[:,0] 
        w1_pred = w_seq_pred[:,0]
        dv_true, dw_true = true_dynamics(v0_batch, w0_batch, n_b[:,0:1])
        v1_true = v0_batch + dv_true * dt
        w1_true = w0_batch + dw_true * dt

        loss_data = mse(v1_pred, v1_true) + mse(w1_pred, w1_true)

        # Initial condition loss
        v_ic_pred, w_ic_pred, _ = model(v0, w0, noise0)

        dv0, dw0 = true_dynamics(v0, w0, noise0)
        v0_true = v0 + dv0*dt
        w0_true = w0 + dw0*dt
        loss_ic = mse(v_ic_pred, v0_true) + mse(w_ic_pred, w0_true)

      

        # Total loss
        loss = loss_data + alpha_ic*loss_ic + alpha_phys*loss_phys
        loss.backward()
        opt.step()

       
            

           
        v0_all = v_all[:, 0, 0].unsqueeze(1)
        w0_all = w_all[:, 0, 0].unsqueeze(1)
        n0_all = noise_all[:, 0, 0].unsqueeze(1)
        dv_all, dw_all = true_dynamics(v0_all, w0_all, n0_all)
        v1_true_all = v0_all + dv_all * dt
        w1_true_all = w0_all + dw_all * dt

       
        v1_pred_all, w1_pred_all, _ = model(v0_all, w0_all, n0_all)

        nrmse_v_all = nrmse_torch(v1_true_all, v1_pred_all)
        nrmse_w_all = nrmse_torch(w1_true_all, w1_pred_all)
        total_nrmse = (nrmse_v_all + nrmse_w_all).item()
        records.append({'epoch': ep, 'nrmse': total_nrmse})

        if total_nrmse < best_total_nrmse:
            best_total_nrmse = total_nrmse
            best_epoch = ep


        if ep==1 or ep%500==0:
            batch_nrmse_v = nrmse_torch(v1_true, v1_pred).item()
            batch_nrmse_w = nrmse_torch(w1_true, w1_pred).item()
            print(f"Epoch {ep:4d} data {loss_data.item():.2e} | ic {loss_ic.item():.2e} | phys {loss_phys.item():.2e}| NRMSE v:{batch_nrmse_v:.3f}, w:{batch_nrmse_w:.3f}")
            records.append({'epoch': ep,
                            'nrmse': batch_nrmse_v + batch_nrmse_w})
    print(f"\nTraining took {time.perf_counter()-t_start:.2f}s")
    df = pd.DataFrame(records).set_index('epoch')
    print("Training NRMSE every 500 epochs:\n", df)
    print(f"\nLowest total NRMSE (v+w): {best_total_nrmse:.6f} at epoch {best_epoch}")

   
    if len(records) > 0:
        df = pd.DataFrame(records).set_index('epoch')
        fig, ax = plt.subplots()
        ax.plot(df.index.to_numpy(), df['nrmse'].to_numpy(), lw=3,label='data+Physics2')
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
        plt.savefig('Physics2.eps', format='eps', dpi=300, bbox_inches='tight')
        plt.show()

    return model
def annotate_regions(ax, t_eps, t_boundary_eps, y_frac=0.9):
    
    ax.axvline(t_boundary_eps, ls=':', lw=1.2, c='k', zorder=3)

    x_left  = (t_eps[0] + t_boundary_eps) / 2.0
    x_right = (t_boundary_eps + t_eps[-1]) / 2.0
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(x_left,  y_frac, 'Training',
            ha='center', va='top', transform=trans, clip_on=False,
            fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=5)
    ax.text(x_right, y_frac, 'Prediction',
            ha='center', va='top', transform=trans, clip_on=False,
            fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'), zorder=5)

def predict_and_plot_regions(model, train_csv, full_csv, dt=0.1, eps=2.5e-4):
   
    t_train, _, _, _ = load_data(train_csv)
    t_boundary = float(t_train[-1])

  
    t, v_np, w_np, noise_np = load_data(full_csv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v_t     = torch.from_numpy(v_np).to(device)
    w_t     = torch.from_numpy(w_np).to(device)
    noise_t = torch.from_numpy(noise_np).to(device)

    
    dv_true, dw_true = true_dynamics(v_t, w_t, noise_t)
    v_next_true = (v_t + dv_true * dt).cpu().numpy().flatten()
    w_next_true = (w_t + dw_true * dt).cpu().numpy().flatten()

    
    model.eval()
    with torch.no_grad():
        v1_pred, w1_pred, _ = model(v_t, w_t, noise_t)
    v_next_pred = v1_pred[:, 0:1].cpu().numpy().flatten()
    w_next_pred = w1_pred[:, 0:1].cpu().numpy().flatten()

        
    t_arr = np.asarray(t, dtype=float).flatten()
    pred_idx = np.where(t_arr > t_boundary)[0]
    if pred_idx.size == 0:
        print("No prediction region (t > t_boundary).")
    else:
        take = min(3200, pred_idx.size)
        tail_idx = pred_idx[-take:]
        nrmse_v_tail = nrmse_np(v_next_true[tail_idx], v_next_pred[tail_idx])
        nrmse_w_tail = nrmse_np(w_next_true[tail_idx], w_next_pred[tail_idx])
        combined_tail = nrmse_v_tail + nrmse_w_tail
        print(f"Predict NRMSE (last {take} pts) → "
              f"v:{nrmse_v_tail:.3f}, w:{nrmse_w_tail:.3f}, combined:{combined_tail:.3f}")


   
    t = np.asarray(t, dtype=float).flatten()
    t_eps = eps * t
    t_boundary_eps = eps * t_boundary

    
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
    
    #plt.savefig('NASP_2_v.eps', format='eps', dpi=300, bbox_inches='tight')
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
   
    #plt.savefig('NASP_2_w.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()


if __name__=='__main__':
    train_csv = 'Data/train_data.csv'  
    full_csv  = 'Data/test_data.csv'   

    model = train(train_csv)

    predict_and_plot_regions(model, train_csv, full_csv)
