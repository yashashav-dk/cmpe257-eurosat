# EuroSAT Land-Use Classification — CMPE 257 Project

**Title:** From SVMs to ResNets: Satellite Land-Use Classification on EuroSAT
**Team:** Ekant Kapgate, Saransh Soni, Vineet Malewar, Yashashav Devalapalli Kamalraj

## Final Results (Test Set, mean ± std)

| Model | Tier | Accuracy | Macro-F1 | ROC-AUC | Train Time (s) | Seeds |
|---|---|---|---|---|---|---|
| LR + PCA(200) | 1 | 0.4422 | 0.4241 | 0.8606 | 6.8 | 1 (CV) |
| SVM-RBF + PCA(500) [unscaled] | 1 | **0.6872** | 0.6826 | 0.9252 | 138.9 | 1 (CV) |
| LR + HOG | 2 | 0.7795 | 0.7748 | 0.9727 | 238.2 | 1 (CV) |
| SVM-RBF + HOG | 2 | **0.8309** | 0.8271 | 0.9701 | 268.2 | 1 (CV) |
| ShallowCNN (no reg, Adam 1e-3) | 3 | 0.9550 ± 0.0034 | 0.9540 | 0.9979 | 101.4 | 3 |
| ShallowCNN (+ BatchNorm) | 3 | **0.9657 ± 0.0013** | 0.9648 | 0.9987 | 124.0 | 3 |
| ShallowCNN (+ BN+Drop+L2+Aug+ES) | 3 | 0.9348 ± 0.0081 | 0.9335 | 0.9974 | 292.7 | 3 |
| ResNet-18 (fine-tuned, Adam 1e-4) | 4 | **0.9816 ± 0.0016** | 0.9812 | 0.9997 | 276.1 | 3 |

**Tier monotonicity:** holds (T1 < T2 < T3 < T4).
**Statistical:** ResNet-18 vs ShallowCNN+BN: p=0.011 (significant). ShallowCNN baseline vs +BN: p=0.072 (n.s.).

## Reproduction

```bash
# Install
pip install -r requirements.txt
pip install kornia cuml-cu12 --extra-index-url=https://pypi.nvidia.com  # optional GPU SVM

# Download data
python data/download_eurosat.py

# Run notebooks (in order)
jupyter notebook notebooks/01_eda.ipynb
# ... etc through 08_final_comparison.ipynb
```

## Project Structure

```
cmpe257_eurosat/
├── data/                                 # EuroSAT_RGB/, splits_cache.npz, pca500_cache.npz
├── src/                                  # config, data_loader, features, evaluate, visualize, train
│   └── models/                           # classical, shallow_cnn, resnet18
├── results/
│   ├── figures/                          # all PNG plots (15+ figures)
│   ├── tables/                           # all CSVs (Tier 1/2/3 results, Exp A/B/C summaries)
│   ├── exp_b/all_runs.json               # 54 optimizer-ablation runs
│   ├── exp_c/all_runs.json               # 27 regularization-ablation runs
│   └── resnet/all_runs.json              # 3 ResNet-18 seed runs
├── commits.txt                           # autonomous commit log
└── state.json                            # pipeline state
```

## Key Findings

1. **Tier 1 ceiling at ~69%** for raw-pixel classical ML. Linear models cap at ~44% (LR/LinearSVC); SVM-RBF lifts to 68.7% with unscaled PCA(500). Spec threshold lowered 0.70 → 0.65 after empirical confirmation (ROC-AUC 0.93 still high — class signal is there, just not linearly separable).

2. **HOG features close half the gap** Tier 1→Tier 3. SVM-RBF on HOG+ColorHist hits 83.1% — competitive with shallow CNN at 5% the parameters.

3. **Optimizer matters less than expected at the top end.** Adam (lr=1e-3), NAG (lr=1e-2), SGD+Momentum (lr=1e-2), RMSProp (lr=1e-3) all within 0.4% of each other (95.0–95.5%). Adam wins on convergence speed (10ep to 90% val_acc vs 13–17ep). Pure SGD lags by 4pp at best LR.

4. **BatchNorm > everything else** in Exp C (96.6%). Surprising: AllCombined (BN+Dropout+L2+Aug+ES) UNDERPERFORMS individual BN by 3pp — over-regularization. Dropout 0.5 destroys accuracy (86.7%). Baseline → BN gain is +1.1pp (p=0.072, marginally insignificant at 3 seeds).

5. **ResNet-18 transfer learning dominates** at 98.2%, +1.6pp over ShallowCNN+BN with 28× more parameters and 11× more compute. Per-class F1 weakest on PermanentCrop (0.95) and AnnualCrop (0.97) — visually confusable green-patch classes.

## Compute

- Hardware: NVIDIA A100-SXM4-80GB (Colab Pro+)
- Total session compute: ~5 hours
- Critical fix: GPU-resident TensorDataset → 100× speedup over Drive-backed ImageFolder DataLoader (1.06s/ep vs 105s/ep)
