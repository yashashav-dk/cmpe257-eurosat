# From SVMs to ResNets: Satellite Land-Use Classification on EuroSAT

**Authors:** Ekant Kapgate, Saransh Soni, Vineet Malewar, Yashashav Devalapalli Kamalraj
**Course:** CMPE 257 — Machine Learning, San José State University

## Abstract

We benchmark four model tiers on the EuroSAT RGB land-use classification task (10 classes, 27,000 64×64 satellite images): (T1) classical linear/kernel models on PCA-reduced raw pixels, (T2) classical models on HOG + color-histogram features, (T3) a 391K-parameter shallow CNN with optimizer and regularization ablations, and (T4) a fine-tuned ImageNet-pretrained ResNet-18. Test accuracy scales monotonically across tiers from 0.687 (T1) → 0.831 (T2) → 0.966 (T3 + BatchNorm) → 0.982 (T4). We additionally report a 6×3-grid optimizer ablation (54 runs) and a 9×3-grid regularization ablation (27 runs) on the shallow CNN. Adam (lr=1e-3) wins on convergence speed; BatchNorm alone delivers the highest single-technique improvement; combining all regularization techniques HURTS performance through over-regularization.

## 1. Introduction
- Motivation: rapid scalable land-use classification for EO data.
- RQ1: How does test accuracy scale across model tiers?
- RQ2: Which optimizer best trains the shallow CNN baseline?
- RQ3: Which regularization technique best closes the train/test gap?

## 2. Related Work
- Classical SVM/HOG benchmarks on remote sensing (Helber et al. 2019).
- Optimizer comparisons (Wilson et al. 2017, Choi et al. 2019).
- Regularization in CNNs (Ioffe & Szegedy 2015, Srivastava et al. 2014).

## 3. Methodology

### 3.1 Dataset & Splits
EuroSAT RGB (Helber et al., Zenodo doi:10.5281/zenodo.7711810). Stratified 70/15/15 split, seed=42. EUROSAT mean=[0.345, 0.381, 0.408], std=[0.203, 0.137, 0.116] computed on training set.

### 3.2 Tier 1 — Pixel + PCA Classical
Flatten 64·64·3=12288 → StandardScaler (pre-PCA) → PCA{50,100,200,500} → optionally StandardScaler (post-PCA, for LR/LinearSVC) → classifier. cuML GPU SVC enables full 4×4 grid search in seconds.

### 3.3 Tier 2 — HOG + Color Histograms
HOG (cv2: 64×64, blocks 16×16, stride 8, cells 8×8, 9 bins) → 1764-dim. + per-channel L1-normalized 32-bin color histograms → 96-dim. Concat → 1860-dim → StandardScaler → LR / cuML SVC-RBF.

### 3.4 Tier 3 — ShallowCNN (391K params)
4 conv blocks (3→32→64→128→256), 3×3 kernels, GAP, FC(256→10). Toggleable BatchNorm and Dropout2d. GPU-resident TensorDataset, batch_size=64.

### 3.5 Tier 4 — ResNet-18 Transfer
torchvision ResNet-18 (ImageNet1K_V1 weights), FC replaced with Linear(512, 10), full fine-tune. Inputs upsampled 64→224, ImageNet normalization, full augmentation. Adam lr=1e-4, AMP on, early stopping (patience 10) over 50 epochs.

### 3.6 Experiment B — Optimizer Ablation
6 optimizers × 3 LRs × 3 seeds = 54 runs. ShallowCNN baseline (no BN/dropout/aug/wd), 100 epochs, no early stopping. Optimizers: SGD, SGD+Momentum(0.9), NAG, Adam(β=(0.9, 0.999)), Adagrad, RMSProp(α=0.99). LRs: {1e-4, 1e-3, 1e-2}.

### 3.7 Experiment C — Regularization Ablation
9 configs × 3 seeds = 27 runs. Optimizer fixed to Adam lr=1e-3 (Exp B winner). Configs: Baseline / L2(λ∈{1e-4, 1e-3}) / Dropout(p∈{0.3, 0.5}) / BatchNorm / DataAug / EarlyStop / AllCombined.

### 3.8 Evaluation
All models evaluated identically: accuracy, macro-precision/recall/F1, per-class F1, 10×10 confusion matrix, OvR macro ROC-AUC, train wall-clock time, train/test gap, mean ± std over 3 seeds (CNN tiers).

## 4. Results

### 4.1 Exp A — Tier Comparison (RQ1)
[See: results/tables/exp_a_final_comparison.csv and results/figures/exp_a_tier_comparison.png]

Test acc by tier: 0.4422 (LR+PCA) → 0.6872 (SVM-RBF+PCA) → 0.7795 (LR+HOG) → 0.8309 (SVM-RBF+HOG) → 0.9550 (CNN baseline) → 0.9657 (CNN+BN) → 0.9816 (ResNet-18). Monotonic improvement validates tier hierarchy. ResNet-18 vs CNN+BN: p=0.011 (significant). CNN baseline vs CNN+BN: p=0.072 (marginal at 3 seeds).

### 4.2 Exp B — Optimizer Ablation (RQ2)
Best LR per optimizer (mean ± std test acc, 3 seeds):
- Adam (1e-3):         0.9550 ± 0.0042   conv@90 = 10 ep
- NAG (1e-2):          0.9547 ± 0.0063   conv@90 = 13 ep
- SGD+Momentum (1e-2): 0.9546 ± 0.0066   conv@90 = 14 ep
- RMSProp (1e-3):      0.9519 ± 0.0036   conv@90 = 14 ep
- Adagrad (1e-2):      0.9472 ± 0.0024   conv@90 = 21 ep
- SGD (1e-2):          0.9165 ± 0.0021   conv@90 = 79 ep

Adam wins on speed; top-4 optimizers tied within 0.3pp accuracy. Pure SGD lags 4pp at best LR. Adam diverged once at lr=1e-2 (seed 456).

### 4.3 Exp C — Regularization Ablation (RQ3)
Mean test acc / overfitting gap (train_acc_final − test_acc), 3 seeds:
- 1_Baseline:       0.9550 / +0.035
- 2a_L2(1e-4):      0.9612 / +0.031
- 2b_L2(1e-3):      0.9542 / +0.011
- 3a_Dropout(0.3):  0.9527 / -0.011
- 3b_Dropout(0.5):  0.8669 / -0.015 (over-reg)
- 4_BatchNorm:      **0.9657** / +0.028
- 5_DataAug:        0.9641 / +0.023
- 6_EarlyStop:      0.9550 / +0.032
- 7_AllCombined:    0.9348 / -0.056 (over-reg)

BatchNorm alone is the strongest single technique. Combining everything OVER-regularizes (−3pp vs BN alone). Dropout 0.5 destroys learning capacity. L2 1e-3 closes the train/test gap to 0.011 but doesn't improve test acc.

### 4.4 Per-Class Analysis
ResNet-18 per-class F1: best on SeaLake (0.999), Residential (0.996), Forest (0.994); weakest on PermanentCrop (0.954) and AnnualCrop (0.969). Top-2 confused pairs (off-diagonal): AnnualCrop↔PermanentCrop, HerbaceousVegetation↔Pasture — all spectrally similar green-patch crop classes.

## 5. Discussion
- **RQ1**: Yes, tiers are monotonically ordered. Largest jump is T2→T3 (+13pp), where learned features replace handcrafted ones.
- **RQ2**: At the top end, optimizer choice contributes <1pp variance. Adam's main advantage is convergence speed, not asymptotic accuracy. Wilson et al. (2017) hypothesis (SGD+Momentum > Adam) is NOT confirmed on EuroSAT — they tied within noise.
- **RQ3**: BatchNorm dominates. Combining regularizers without retuning hurts. Suggests EuroSAT (clean, well-balanced) does not need aggressive regularization; the ShallowCNN already generalizes well from Adam alone.

## 6. Conclusion & Future Work
Pretrained ResNet-18 wins (98.2%) but ShallowCNN+BN (96.6%) closes 89% of the gap with 1/28 the parameters and 1/11 the compute. Practical recommendation: for resource-constrained deployment, BN-equipped shallow CNNs are competitive with deep transfer learning. Future: multispectral (13-channel) inputs, stronger augmentation, distilled smaller models.

## References
[20+ references; see proposal]
