# Latent Space Geometry and Sparse Matrix Completion for Collaborative Filtering

An exploratory machine learning study comparing **deterministic**, **probabilistic**, and **propensity-debiased** matrix completion models on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) benchmark ($n = 943$ users, $m = 1{,}682$ movies, $100{,}000$ ratings, $93.7\%$ sparsity). Beyond predictive accuracy and Top-$K$ ranking quality, this project investigates **what unsupervised latent representations actually encode** through leakage-free linear probing, geometric alignment, and counterfactual debiasing under *Missing Not At Random* (MNAR) exposure bias.

---

## Key Highlights & Findings

- **Exposure Bias (MNAR) is Massive:** Movie popularity (training rating count) correlates strongly with average rating (Spearman $r = 0.500, p = 4.73 \times 10^{-107}$), meaning missing ratings are not missing at random.
- **Imputation Warps Geometry & Hurts Accuracy:** Filling missing entries with item means (`ImputedPCA`, test RMSE $0.9720$) performs worse than a simple additive bias baseline ($0.9355$) and forces the latent space to encode **observation density** ($\log$-popularity $R^2 = 0.614$ vs. $0.081\text{–}0.161$ for sparse models), yielding near-zero distance correlation ($\rho \in [-0.034, 0.028]$) with sparse latent spaces.
- **Additive Biases Unlock Latent Capacity:** Separating user/item scalar biases ($b_u, b_i$) from inner-product interactions improves Matrix Factorization test RMSE from $0.9258$ (`use_biases=False`) to **$0.9082$** (`use_biases=True`) and triples catalog-wide `NDCG@10` ($0.0134 \to 0.0401$).
- **RMSE vs. NLL Divergence in Probabilistic Models:** In sparse PPCA, the noise variance $\sigma^2$ plays a dual role — acting as user-level posterior shrinkage in point predictions ($\sigma^2 = 2.5$ minimizes RMSE at $0.9144$) while governing predictive Gaussian calibration ($\sigma^2 = 0.85$ minimizes CV NLL at $1.3733$ vs. $1.5666$ at $\sigma^2=2.5$).
- **Exact Posterior vs. Variational Approximation:** Sparse PPCA (EM with full $k \times k$ posterior covariance $S_u$) outperforms a matched-assumption Linear VAE (diagonal covariance $q_\phi(\mathbf{z}_u)$) in both RMSE ($0.9144$ vs. $0.9306$) and NLL ($1.5591$ vs. $1.5653$), while a nonlinear `Tanh` decoder overfits under $93.7\%$ sparsity ($0.9384$).
- **Sparse Models Converge to a Shared Semantic Manifold:** Matrix Factorization, PPCA, and Linear VAE exhibit strong pairwise distance correlation ($\rho = 0.707\text{–}0.843$) and Procrustes alignment ($0.705$ between PPCA and Linear VAE), recovering unseen demographic and content attributes (movie genre mean ROC-AUC $0.682$, release year $R^2 = 0.178$, user gender AUC $0.686$).
- **Inverse Propensity Weighting (IPW) Trade-off:** Reweighting the MF objective by $w_{ui} \propto \hat{P}(O_{ui}=1)^{-\alpha}$ lowers RMSE on long-tail items ($\le 20$ ratings: $0.9946 \to 0.9916$ at $\alpha=0.50$) and removes the correlation between item embedding norm $\|\mathbf{w}_i\|_2$ and popularity ($r = +0.256 \to -0.150$), at the cost of higher error on head items and higher IPS-weighted variance.

---

## Mathematical Formulations & Model Families

All models operate in a **transductive** setting: each user $u \in \{0,\dots,n-1\}$ and item $i \in \{0,\dots,m-1\}$ is represented solely by its ID without side features during training, predicting ratings $R_{ui} \in [1, 5]$ from observed pairs $\Omega \subset \{0,\dots,n-1\} \times \{0,\dots,m-1\}$.

| Model | Predictive Equation $\hat{R}_{ui}$ | Objective / Generative Assumption | Optimization & Complexity / Epoch |
| :--- | :--- | :--- | :--- |
| **Additive Baseline** | $\mu + b_u + b_i$ | $\min_{b} \sum_{(u,i)\in\Omega}(r_{ui}-\mu-b_u-b_i)^2 + \lambda_u\|b_u\|_2^2 + \lambda_i\|b_i\|_2^2$ | Exact Alternating Least Squares ($\mathcal{O}(|\Omega| + n + m)$) |
| **Imputed PCA** | $\bar{R}_i + \mathbf{z}_u^\top \mathbf{w}_i$ | $\min_{\mathrm{rank}(\hat{X})\le k} \|\tilde{R} - \mathbf{1}\bar{\mathbf{R}}^\top - ZW^\top\|_F^2$ ($\tilde{R}_{ui}=\bar{R}_i$ if $(u,i)\notin\Omega$) | Dense Truncated SVD ($\mathcal{O}(n m \min(n,m))$, single pass) |
| **Matrix Factorization (MF)** | $\mu + b_u + b_i + \mathbf{z}_u^\top \mathbf{w}_i$ | $\min_{b,Z,W}\sum_{(u,i)\in\Omega}(r_{ui}-\hat{R}_{ui})^2 + \lambda_b(\|b_u\|_2^2+\|b_i\|_2^2) + \lambda_z(\|Z\|_F^2+\|W\|_F^2)$ | PyTorch Mini-batch Adam on $\Omega$ ($\mathcal{O}(|\Omega|\cdot k)$) |
| **Sparse PPCA** | $\mu_i + \mathbf{m}_u^\top \mathbf{w}_i$ | $\mathbf{z}_u \sim \mathcal{N}(\mathbf{0}, I_k)$, $R_{ui}\mid\mathbf{z}_u \sim \mathcal{N}(\mu_i + \mathbf{z}_u^\top \mathbf{w}_i, \sigma^2)$ | Exact Sparse EM ($\mathcal{O}(n k^3 + |\Omega|k^2)$) |
| **Transductive $\beta$-VAE** | $\mu_i + \mathbf{m}_u^\top \mathbf{w}_i$ (Lin.) / $\mathbf{w}_{2,i}^\top\tanh(W_1\mathbf{m}_u+\mathbf{c}_1)+\mu_i$ (NL) | $\max_{\phi,\theta}\sum_u \mathbb{E}_{q_\phi}[-\frac{1}{2\sigma^2}\sum_{i\in\Omega_u}(r_{ui}-g_\theta(\mathbf{z}_u)_i)^2] - \beta\,\mathrm{KL}(q_\phi(\mathbf{z}_u)\|\mathcal{N}(\mathbf{0},I_k))$ | Reparameterization + Adam ($\mathcal{O}(n m k)$ dense mask) |
| **IPW-MF** | $\mu + b_u + b_i + \mathbf{z}_u^\top \mathbf{w}_i$ | $\min_{b,Z,W}\sum_{(u,i)\in\Omega} w_{ui}(r_{ui}-\hat{R}_{ui})^2 + \text{reg}$, with $w_{ui}\propto \hat{P}(O_{ui}=1)^{-\alpha}$ | Propensity-weighted PyTorch Mini-batch ($\mathcal{O}(|\Omega|\cdot k)$) |

### Key Mechanism Insights
1. **Why `ImputedPCA` fails under high sparsity:** Setting missing cells to $\bar{R}_i$ creates zero residuals across $93.7\%$ of $\tilde{R}$. SVD then spends its top components explaining variance concentrated in high-degree rows/columns (popularity), and larger $k$ rapidly memorizes sparse noise.
2. **Why `PPCA` separates RMSE and NLL optima:** In the E-step, posterior precision $S_u^{-1} = I_k + \frac{1}{\sigma^2}\sum_{i\in\Omega_u}\mathbf{w}_i\mathbf{w}_i^\top$ scales shrinkage with $\sigma^2$. A large $\sigma^2$ stabilizes sparse users (lowering RMSE), whereas the predictive variance $\sigma^2 + \mathbf{w}_i^\top S_u \mathbf{w}_i$ enters the Gaussian log-likelihood directly, favoring $\sigma^2 \approx 0.85$ (matching empirical squared error).
3. **Why `All-Unseen` and `Test-Only` Top-10 rankings diverge:** Ranking only observed test items (`Test-Only`) measures local preference ordering and tracks RMSE. Ranking the entire unseen catalog (`All-Unseen`, where hits are test items with $r_{ui} \ge 4$) exposes models without bias shrinkage (PPCA, VAE) to unregularized high sample means on 1–2 rating niche films, dropping `NDCG@10` from $0.0401$ (regularized MF) to $0.0007\text{–}0.0010$.

---

## Experimental Protocol & Results

### Validation & Uncertainty Quantification
- **Train/Test Split:** $80\%$ train ($80{,}696$ ratings) / $20\%$ test ($19{,}304$ ratings), stratified per user with each item's first appearance forced into train to avoid cold-start unidentifiability.
- **Model Selection:** 5-fold user-stratified cross-validation strictly inside `train_df` (with item coverage protection); test set evaluated once per model family after retraining on full `train_df`.
- **Confidence Intervals:** $1{,}000$-resample percentile bootstrap ($95\%$ CI) on test predictions clipped to $[1.0, 5.0]$.

### Consolidated Test Results

| Model | Test RMSE [95% CI] | Test MAE [95% CI] | Test NLL [95% CI] | NDCG@10 (All-Unseen) | HitRate@10 (All-Unseen) | NDCG@10 (Test-Only) | Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.1 Additive Baseline** ($\lambda_u=5, \lambda_i=1$) | $0.9355\,[0.9259, 0.9451]$ | $0.7382\,[0.7305, 0.7463]$ | — | $0.0181$ | $0.1765$ | $0.8390$ | $0.01$ |
| **1.2 Imputed PCA** ($k=8$) | $0.9720\,[0.9620, 0.9814]$ | $0.7690\,[0.7609, 0.7773]$ | — | $0.0045$ | $0.0194$ | $0.8492$ | $0.16$ |
| **1.3 MF (No Biases)** ($k=5, \lambda_z=0.1$) | $0.9258\,[0.9168, 0.9359]$ | $0.7286\,[0.7208, 0.7366]$ | — | $0.0134$ | $0.1119$ | $0.8509$ | $0.91$ |
| **1.3 MF (With Biases)** ($k=40, \lambda_z=0.1, \lambda_b=0.05$) | **$0.9082\,[0.8989, 0.9179]$** | **$0.7133\,[0.7059, 0.7212]$** | — | **$0.0401$** | **$0.2745$** | **$0.8629$** | $3.14$ |
| **2.1 Sparse PPCA (EM)** ($k=20, \sigma^2=2.5$) | $0.9144\,[0.9047, 0.9243]$ | $0.7169\,[0.7094, 0.7250]$ | **$1.5591\,[1.5558, 1.5626]$** | $0.0010$ | $0.0075$ | $0.8606$ | $15.95$ |
| **2.2 VAE Linear** ($k=20, \sigma^2=2.5, \beta=1.0$) | $0.9306\,[0.9212, 0.9406]$ | $0.7322\,[0.7247, 0.7401]$ | $1.5653\,[1.5619, 1.5688]$ | $0.0007$ | $0.0032$ | $0.8543$ | $2.82$ |
| **2.2 VAE Nonlinear (`Tanh`)** ($h=64, \beta=1.0$) | $0.9384\,[0.9285, 0.9485]$ | $0.7353\,[0.7275, 0.7432]$ | $1.5644\,[1.5609, 1.5680]$ | $0.0007$ | $0.0075$ | $0.8493$ | $3.20$ |

*(Note: PPCA selected by CV NLL achieves $\text{CV NLL} = 1.3733$ at $k=2, \sigma^2=0.85$, confirming the variance calibration trade-off.)*

### Latent Space Probing & Geometric Alignment ($k = 10$, 5-Fold CV)

| Model | User Age ($R^2$) | User Gender (AUC) | Movie Year ($R^2$) | Movie Log-Pop ($R^2$) | Movie Genres Mean (AUC) | Dist. Corr. vs. PPCA ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Imputed PCA** | $0.025 \pm 0.044$ | $0.641 \pm 0.029$ | $0.115 \pm 0.037$ | **$0.614 \pm 0.024$** | $0.710 \pm 0.061$ | $-0.016$ |
| **Matrix Factorization** | $0.028 \pm 0.037$ | $0.659 \pm 0.011$ | $0.103 \pm 0.023$ | $0.081 \pm 0.017$ | $0.671 \pm 0.062$ | $0.724$ |
| **Sparse PPCA** | **$0.088 \pm 0.022$** | **$0.686 \pm 0.027$** | **$0.178 \pm 0.018$** | $0.128 \pm 0.028$ | **$0.682 \pm 0.061$** | $1.000$ |
| **Linear VAE** | $0.051 \pm 0.035$ | $0.641 \pm 0.021$ | $0.154 \pm 0.019$ | $0.161 \pm 0.033$ | $0.655 \pm 0.051$ | **$0.843$** |

### Propensity Debiasing (`IPWMatrixFactorization`, $w_{ui} \propto \hat{P}(O_{ui}=1)^{-\alpha}$)

| $\alpha$ | Test RMSE (Std) | $\text{RMSE}_{\text{IPS}}$ | Tail RMSE ($\le 20$) | Mid RMSE ($21\text{–}80$) | Head RMSE ($>80$) | Spearman($\|\mathbf{w}_i\|_2$, pop) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $0.00$ | **$0.9082$** | **$0.9857$** | $0.9946$ | **$0.9339$** | **$0.8876$** | $+0.256$ |
| $0.25$ | $0.9136$ | $0.9917$ | $0.9938$ | $0.9379$ | $0.8943$ | $+0.049$ |
| $0.50$ | $0.9206$ | $0.9991$ | **$0.9916$** | $0.9423$ | $0.9036$ | $-0.150$ |
| $0.75$ | $0.9291$ | $1.0041$ | $0.9918$ | $0.9470$ | $0.9146$ | $-0.269$ |
| $1.00$ | $0.9376$ | $1.0093$ | $0.9968$ | $0.9525$ | $0.9249$ | $-0.345$ |

---

## Repository Structure & Visual Artifacts

```text
.
├── data/                                  # MovieLens 100K (auto-downloaded on first run)
├── figures/                               # High-resolution plots (300 DPI)
│   ├── fig1_eda_movielens.png             # Rating distribution, Pareto long-tail, MNAR exposure bias
│   ├── fig2_baseline_analysis.png         # L2 regularization surface and learned user/item biases
│   ├── fig3_imputed_pca_curve.png         # Train vs. CV RMSE curve across PCA rank k
│   ├── fig4_mf_hyperparams_biases.png     # MF hyperparameter sweeps with vs. without biases
│   ├── fig5_ppca_rmse_vs_nll.png          # PPCA heatmaps showing RMSE vs. NLL optima divergence
│   ├── fig6_vae_convergence_and_sensitivity.png # ELBO convergence & beta-VAE U-shaped sensitivity
│   ├── fig6b_runtime_comparison.png       # Wall-clock training and inference time comparison
│   ├── fig7_latent_projections_2d.png     # Movie 2D PCA & t-SNE projections (popularity & genres)
│   ├── fig7b_user_latent_projections_2d.png # User 2D PCA & t-SNE projections (age & gender)
│   ├── fig8_latent_space_comparisons.png  # Pairwise distance correlation, Jaccard@10, Procrustes
│   └── fig9_stage3_ipw_analysis.png       # IPW stratified RMSE, IPS error, and norm-pop correlation
├── results/                               # Exported CV tables and summary CSVs
├── unsupervised_assignment_2s2026.ipynb   # Complete executable notebook with all experiments
├── pyproject.toml                         # Project dependencies
├── uv.lock                                # Locked package versions for exact reproducibility
├── .python-version                        # Python 3.13
└── README.md
```

---

## How to Reproduce & Run

This project uses [`uv`](https://docs.astral.sh/uv/) for deterministic environment management (`SEED = 42` is fixed globally across `random`, `numpy`, and `torch`).

```bash
# 1. Create the virtual environment and install exact locked dependencies
uv sync

# 2. Launch Jupyter Lab to inspect or run cells interactively
uv run --with jupyter jupyter lab

# 3. Or execute the entire notebook headlessly from start to finish
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=5400 unsupervised_assignment_2s2026.ipynb
```

Running the notebook automatically downloads MovieLens 100K into `data/`, executes all cross-validation sweeps and bootstrap evaluations, exports CSV tables to `results/`, and saves all figures to `figures/`.
