````markdown
# A self-induced stochastic resonance problem: A physics-informed neural network approach

Minimal **noise-augmented state predictor (NASP)** baselines for **one-step prediction** in the stochastic FitzHugh–Nagumo (FHN) model.

---

## 📁 Repo Layout

```text
.
├── Data/
│   ├── train_data.csv
│   └── test_data.csv
├── NASP_data.py
├── NASP_Physics1.py
├── NASP_Physics2.py
├── NASP_hybrid.py
└── README.md
````

---

## ⚙️ Requirements

* Python 3.9+
* Install via:

```bash
pip install numpy pandas torch matplotlib
```

---

## 🚀 Quick Start

Run any variant:

```bash
python NASP_data.py
python NASP_Physics1.py
python NASP_Physics2.py
python NASP_hybrid.py
```

Defaults:

* `epochs = 5000`
* `batch_size = 512`
* `dt = 0.1`

Edit these at the top of each script for quicker runs.

---

## 🧠 Loss Functions

* **Data:** next-step prediction using Euler from `true_dynamics`
* **IC:** consistency on first step
* **Physics1:** derivative (residual) alignment
* **Physics2:** barrier-aware Kramers-style loss

Defaults in `true_dynamics`:

```python
a = 0.05
b = 1.0
c = 2.0
epsilon = 0.005
sigma = 0.03
```

---

## 📊 Outputs

Each script saves `.eps` figures:

* Training loss curves:
  `data.eps`, `Physics1.eps`, `Physics2.eps`, `Hybrid.eps`

* Time series:
  `NASP_v.eps`, `NASP_w.eps` (True vs NASP-predicted)

> 🔧 If `plt.savefig(...)` is commented, uncomment to write file.

---

## 🧪 Data Format

Expected CSV headers:

```
time,v,w,noise
```

Example rows:

```csv
time,v,w,noise
0.0,-0.12,0.01,0.54
0.1,-0.11,0.01,-0.24
```

Used in code as:

```python
train_csv = "Data/train_data.csv"
full_csv  = "Data/test_data.csv"
```

---

## 🧠 Evaluation

* NRMSE for predictions
* Plots showing **True vs NASP-predicted**
* Easily extensible to CV(σ) or ΔU± metrics

---


