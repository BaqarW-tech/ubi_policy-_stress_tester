# ⚖️ UBI Policy Stress-Tester

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

> **Interactive macro-micro simulation for Universal Basic Income policy analysis.**  
> Adjust transfers, tax rates, and behavioural elasticities to see real-time impacts on
> poverty, fiscal balance, and labour supply across 10,000 synthetic households.

---

## 📸 Features

| Feature | Description |
|---|---|
| **Income Distribution** | Before/after overlaid histograms across log-normal population |
| **Budget Constraint** | Classic labour-leisure diagram with UBI floor annotation |
| **Fiscal Balance** | Gross cost vs. incremental tax revenue vs. net position |
| **Poverty Waterfall** | Poverty rate change decomposition |
| **Sensitivity Heatmap** | 14×14 sweep of ε_h × UBI amount → net fiscal cost |
| **Scenario Comparison** | Save and compare multiple policy configurations |
| **Export** | JSON report + household-level micro-data CSV |

---

## 🏗️ Model

### Population
- **N = 10,000** synthetic households (configurable up to 20k)
- Wages ~ **LogNormal(ln 3500, 0.70)** — calibrated to middle-income country
- Baseline participation rate: **75 %**

### Behavioural Response
$$h_i^{\text{new}} = h_i^{\text{base}} \cdot \left(1 + \varepsilon_h \cdot \frac{\Delta w}{w}\right)$$

$$\Delta p_i = \varepsilon_p \cdot \frac{T}{w_i h_i}$$

### Fiscal Accounting
$$\text{Net Cost} = N \cdot T \cdot 12 - \Delta(\tau \cdot Y^L)$$

### Limitations
- Partial equilibrium — no GE wage feedback
- No savings, inflation, or multi-period dynamics
- Household heterogeneity limited to wages

---

## 🚀 Local Setup

```bash
git clone https://github.com/BaqarW-tech/ubi-policy-stress-tester
cd ubi-policy-stress-tester
pip install -r requirements.txt
streamlit run app.py
```

Requires **Python 3.11**. Set in Streamlit Cloud via **Settings → Python version → 3.11**.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, set branch `main`, main file `app.py`
4. Set Python version to **3.11** in Advanced Settings
5. Click **Deploy**

---

## 📂 File Structure

```
ubi_simulator/
├── app.py            ← Main Streamlit application
├── requirements.txt  ← Dependencies (Streamlit, NumPy, Pandas, Plotly)
└── README.md         ← This file
```

---

## 🔬 Relevant Economics

- **Income effect**: UBI raises non-labour income → workers demand more leisure → participation falls
- **Substitution effect**: Higher tax to fund UBI lowers net wage → hours worked fall
- **Fiscal sustainability**: Net cost depends on how much of gross UBI is recovered via incremental tax revenue from the new rate applied to a (potentially smaller) tax base
- **Cost-effectiveness**: Cost-per-poverty-exit metric allows cross-scenario efficiency comparison

---

## 👤 Author

**Muhammad Baqar Wagan** — MA Economics | KSA Data Analytics Portfolio  
GitHub: [BaqarW-tech](https://github.com/BaqarW-tech)
