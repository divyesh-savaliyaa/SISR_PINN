# A self-induced stochastic resonance problem:       A physics-informed neural network approach

Minimal **noise-augmented state predictor (NASP)** baselines for **one-step prediction** in the stochastic FitzHugh–Nagumo (FHN) model.

Scripts:
- `NASP_data.py` – Data + initial-condition (IC) loss  
- `NASP_Physics1.py` – Data + Physics1 (residual/derivative) loss  
- `NASP_Physics2.py` – Data + Physics2 (barrier-aware, Kramers-guided) loss  
- `NASP_hybrid.py` – Data + Physics1 + Physics2 (hybrid)

All scripts report **NRMSE** and produce **True vs NASP-predicted** plots.

---

## Repo layout


