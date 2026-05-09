# Speaker Notes — Single Speaker, 3 minutes

Target: ~450–500 words at 160 wpm. Pause briefly between slides for transitions.

---

## Slide 1 — Problem Statement (~45 seconds, ~110 words)

Good morning. Our project asks a simple question: when a satellite land-use classifier gets a percentage point better, where does that gain actually come from?

We work with **EuroSAT** — twenty-seven thousand Sentinel-2 RGB patches across ten land-use classes. The samples on screen are real images from the dataset: crop fields, forests, highways, water bodies, all sixty-four by sixty-four pixels.

Most published EuroSAT results are point comparisons — one architecture, one recipe, one seed. That makes it hard to attribute *what* drove the gain. So we built a controlled four-tier hierarchy and asked three research questions: how accuracy scales, which optimizer wins, and which regularizer wins. Same data, same split, same protocol everywhere.

---

## Slide 2 — Proposed Solution (~75 seconds, ~185 words)

Slide two — the method.

Four tiers, increasing in capacity. **Tier one** is PCA on standardized raw pixels, fed to a classical SVM. **Tier two** swaps the feature extractor for HOG plus a per-channel color histogram — an eighteen-sixty-dimensional handcrafted descriptor — still with a classical SVM. **Tier three** is a three-hundred-ninety-one-thousand-parameter shallow CNN, learned end-to-end. **Tier four** fine-tunes an ImageNet-pretrained ResNet-eighteen. All four tiers share the same seed-forty-two stratified split, and the test set is touched once per model.

Inside tier three we run two ablations. **Experiment B** sweeps six optimizers across three learning rates and three seeds — fifty-four total runs, a hundred epochs each. **Experiment C** sweeps nine regularization configurations across three seeds — twenty-seven runs.

One engineering note: the hundred-x speedup at the bottom is what made experiment B tractable. By loading the entire dataset onto GPU as a TensorDataset once, per-epoch time drops from a hundred and five seconds to one second flat. Without it, experiment B alone would have taken a hundred and fifty-eight hours.

---

## Slide 3 — Results & Conclusion (~80 seconds, ~200 words)

Slide three — the numbers.

Accuracy climbs monotonically across the tiers. Tier one: **sixty-nine percent**. Tier two: **eighty-three**. Tier three: **ninety-six point six**. Tier four: **ninety-eight point two**.

The big jump is tier two to tier three — **fourteen points** — at the handcrafted-to-learned-features boundary. From tier three to tier four, only **one-point-six points**, despite twenty-eight times the parameters. Diminishing returns above the shallow CNN.

On the optimizer ablation, the top four — Adam, NAG, SGD with momentum, and RMSProp — are tied within zero-point-three points. The Wilson et al. claim that SGD with momentum dominates Adam is **not confirmed** at this scale. Adam wins on convergence speed — ten epochs to ninety percent validation — but not on final accuracy.

On regularization, **BatchNorm alone is the strongest single technique**. Stacking all five — BatchNorm, Dropout, L2, augmentation, early stopping — actually degrades accuracy by three points. Over-regularization on a clean dataset.

Per-class, four crop-family classes sit consistently at the bottom — pointing to a fundamental RGB ceiling that multispectral inputs would likely fix.

**Conclusion: representation learning is where the gain lives.** Once you have a competent CNN with BatchNorm, depth and optimizer choice barely matter.

Thank you.

---

## Delivery tips

- **Pace:** 160 wpm. Keep moving. Don't dwell.
- **Numbers:** speak the percentages crisply (six-eight-seven-two = "sixty-eight seventy-two"). Round in delivery (0.6872 → "sixty-nine percent").
- **Transitions:** pause one beat at slide change, say "Slide two —" or "Slide three —" then keep going.
- **Don't read the slide.** The audience can read; you add the *why*.
- **Land the conclusion.** Pause, deliver the one-liner, then "thank you."
- **If running long:** drop the per-class crop sentence on slide 3 (saves ~10 sec) before cutting anything else.
- **If running short:** add one sentence on slide 2 about *why* a tiered design isolates the contribution of each modeling decision (the controlled-comparison motivation).
