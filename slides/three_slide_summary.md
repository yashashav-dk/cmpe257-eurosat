---
marp: true
theme: default
paginate: true
---

# Problem Statement

**Satellite land-use classification on EuroSAT RGB**
10 classes · 27,000 Sentinel-2 patches (64×64) · CC-BY-4.0

**The gap in prior work**
Most published EuroSAT results are point comparisons:
one architecture, one recipe, one seed → hard to attribute *which* decision drove the gain.

**Three research questions**
- **RQ1** How does accuracy scale across model tiers (raw pixels → handcrafted features → shallow CNN → deep transfer)? Where does the marginal return diminish?
- **RQ2** Among 6 optimizers, which converges fastest and which generalizes best? Does Wilson et al. (2017) "SGD beats Adam" hold here?
- **RQ3** Among 5 regularizers, which most closes the train/test gap on a shallow CNN? How do they compose?

---

# Proposed Solution: Controlled Tier Hierarchy + Two Ablations

**Single fixed seed-42 stratified 70/15/15 split. Test set touched once per model.**

| Tier | Feature extractor φ | Classifier g |
|---|---|---|
| **T1** | PCA(k∈{50,100,200,500}) on standardized pixels | LR / LinearSVC / RBF-SVC (cuML) |
| **T2** | HOG (1764-D) ⊕ color histogram (96-D) = 1860-D | LR / RBF-SVC (cuML) |
| **T3** | ShallowCNN (391K params, 4 conv blocks + GAP) | end-to-end |
| **T4** | ResNet-18 ImageNet-pretrained, end-to-end fine-tuned | end-to-end |

**Ablations on T3 (isolate optimization vs regularization)**
- **Exp B** — 6 optimizers × 3 LRs × 3 seeds = **54 runs**, 100 ep each, no ES
- **Exp C** — 9 reg configs × 3 seeds = **27 runs**, Adam lr=1e-3 fixed

**Engineering enabler:** GPU-resident `TensorDataset` → 100× speedup (1.06 s/ep vs 105 s/ep). Without this, Exp B = 158 hr A100 → infeasible.

---

# Results & Conclusion

**RQ1 — Tier monotonicity holds. Diminishing returns above T3.**

| Tier | Best model | Test acc | Macro-F1 |
|---|---|---|---|
| T1 | SVM-RBF + PCA(500) | 0.6872 | 0.6826 |
| T2 | SVM-RBF + HOG | 0.8309 | 0.8271 |
| T3 | ShallowCNN + BatchNorm | 0.9657 ± 0.0013 | 0.9648 |
| **T4** | **ResNet-18 (3 seeds)** | **0.9816 ± 0.0016** | **0.9812** |

Largest jump T2→T3 (**+12.4pp**, learned features). T3→T4 only **+1.6pp** at 28× params.

**RQ2** — Top-4 optimizers (Adam, NAG, SGD+Mom, RMSProp) tie within 0.3pp. Adam wins on **speed** (10 ep to 90% val), not asymptote. Wilson et al. claim *not* confirmed at this scale.

**RQ3** — **BatchNorm alone** is the strongest single regularizer (0.9657). Stacking all five degrades by **−3pp** → over-regularization on a clean dataset.

**Per-class bottleneck:** four crop-family classes (PermanentCrop, AnnualCrop, HerbaceousVegetation, Pasture) consistently hardest across all tiers — RGB ceiling, multispectral likely needed.

> **Conclusion.** Representation learning (T2→T3) is where the win is; depth and optimizer choice are diminishing returns once a competent CNN + BatchNorm is in place.
