# Self-Supervised Pretraining for 3D Medical-Image Classification
### A label-efficiency study
---

## TL;DR

I built and trained (on a 1-CPU-core sandbox, no GPU) an end-to-end pipeline that:

1. Pretrains a 3D CNN encoder with a **masked-volume reconstruction** pretext task (no labels), in the spirit of MAE (He et al., 2022), on a synthetic dataset shaped like MedMNIST3D/OrganMNIST3D.
2. Compares **from-scratch training** vs. **fine-tuning the pretrained encoder** vs. **freezing it and only training a linear head**, across five label budgets (5%–100% of the training set).
3. Finds that SSL pretraining **roughly doubles accuracy at 5% labels** (0.26 → 0.44) — the clinically relevant regime, since annotated whole-body MRI is expensive — but that the **frozen** representation is a poor linear classifier (peaks at 0.46 vs. 0.84 for a fully-labeled from-scratch model).
4. Diagnoses that gap, runs a targeted ablation grounded in the MAE literature to test one hypothesis, gets an inconclusive result, and explains why that's actually consistent with what's published about reconstruction-based vs. contrastive SSL objectives.

Everything below is real output from real runs, not a proposal. Numbers, plots, and code are all in this folder.

## Why this project, and an upfront caveat

The JD leans hard on self-supervised learning (masked image modeling, contrastive, JEPA), ViT/multimodal foundation models, and *designing and diagnosing experiments independently* rather than only running them. I wanted something I could actually execute, not just describe.

**Caveat on data:** this sandbox has no route to Zenodo, where the real MedMNIST3D/OrganMNIST3D volumes are hosted, so I couldn't download the real dataset here. I generated a synthetic substitute with the same problem shape — single-channel 3D volumes, multi-class, corrupted with a simulated MRI bias field and acquisition noise (`data.py`) — so the *methodology* is real and tested, even though the *data* is synthetic. Swapping in the real dataset is a one-line change:

```python
# instead of:
train_ds, val_ds, test_ds = make_splits(...)
# use:
import medmnist
train_ds = medmnist.OrganMNIST3D(split="train", download=True)
```

**Caveat on architecture:** the JD's SSL bullet points toward ViT-based MAE. A ViT3D (patchify + multi-head self-attention over hundreds of tokens) was too slow to train to convergence on a single CPU core in the time I had, so I implemented the same *pretext task* (reconstruct masked regions from context) on a **3D CNN encoder-decoder** instead — a context-encoder / convolutional-MAE hybrid. The masking strategy, the "loss only on masked voxels" rule, and the downstream linear-probe/fine-tune protocol are the same as MAE; only the backbone differs. I'd frame this honestly in the interview as a compute-driven substitution, not a design choice I'd default to with real hardware.

## Data

`SyntheticOrgan3D` (`data.py`) generates 24×24×24 single-channel volumes on the fly (no disk I/O), across 6 classes defined by geometric structure (sphere, cube, ellipsoid, cylinder, cross, ring), each with:
- random position, scale, and full 3D rotation
- a simulated multiplicative bias field (smooth low-frequency intensity inhomogeneity, the classic MRI artifact)
- additive Gaussian acquisition noise

Splits: 900 train / 300 val / 400 test, seeded for reproducibility.

## Method

**Architecture** (`models.py`): a small 3D CNN encoder (3 conv+BN+GELU downsampling blocks, 24³→3³, 1→16→32→64 channels, global-average-pooled to a 128-d embedding) with a mirrored transposed-conv decoder for the reconstruction pretext task, and a 2-layer MLP head for classification.

**SSL pretext task**: randomly zero out non-overlapping cubic patches of the input volume (patch size 4, 50% of volume masked by default), feed the masked volume through the encoder-decoder, and compute MSE loss **only on the masked voxels** — exactly MAE's loss masking, adapted to a conv backbone. No labels are used anywhere in this stage.

**Downstream protocol**: for each of 3 conditions × 5 label fractions {5%, 10%, 25%, 50%, 100%}, train a classifier, select the best checkpoint by validation accuracy, and report held-out test accuracy / macro-F1.
- `scratch` — randomly initialized encoder, trained end-to-end (the baseline)
- `ssl_frozen` — SSL-pretrained encoder frozen, only the classifier head trains (a linear/shallow probe)
- `ssl_finetuned` — SSL-pretrained encoder + head, both trained end-to-end

## Results

| Label % | n labeled | From scratch | SSL frozen (probe) | SSL fine-tuned |
|---|---|---|---|---|
| 5%  | 48  | 0.263 | 0.188 | **0.440** |
| 10% | 90  | 0.568 | 0.260 | 0.590 |
| 25% | 228 | 0.673 | 0.380 | **0.765** |
| 50% | 450 | 0.743 | 0.415 | **0.785** |
| 100%| 900 | 0.840 | 0.463 | 0.838 |

*(test accuracy; macro-F1 tracks the same pattern — full numbers in `outputs/results.json`)*

![Label efficiency](outputs/label_efficiency.png)

**Reading this:** fine-tuning the SSL-pretrained encoder gives its biggest win exactly where labels are scarcest — nearly 2x the from-scratch accuracy at 5% labels — and stays ahead through 50%. By 100% labels the gap closes, which is expected: with enough supervision, the from-scratch model has enough signal to learn good features on its own. This is the practically important story for a clinic , where annotated whole-body MRI is expensive and pretraining on the much larger pool of *unannotated* scans is exactly the lever available.

## The more interesting part: diagnosing the frozen-probe gap

The frozen linear probe is bad — it never breaks 0.47, while the same pretrained encoder gets to 0.84 test accuracy once fine-tuned. That's a real result worth explaining rather than hiding.

**Hypothesis:** with only 50% masking and small 4-voxel patches, the encoder can solve reconstruction via local interpolation from neighboring unmasked voxels, without ever needing a global, class-discriminative representation. This is precisely why He et al.'s MAE paper uses a high 75% mask ratio — lower ratios make the pretext task too easy and let the model shortcut around learning global structure.

**Test:** I retrained pretraining with mask ratio 0.75 and larger, block-structured 6-voxel patches (`ssl_frozen__v2` below), then re-ran the frozen probe.

| Label % | SSL frozen, v1 (mask=0.5, patch=4) | SSL frozen, v2 (mask=0.75, patch=6) |
|---|---|---|
| 5%  | 0.188 | 0.230 |
| 10% | 0.260 | 0.218 |
| 25% | 0.380 | 0.348 |
| 50% | 0.415 | 0.378 |
| 100%| 0.463 | 0.458 |

**Result: inconclusive — the fix didn't help** (v2 trained for fewer epochs, 6 vs. 10, so it's not a fully fair comparison, but the trend is flat at best). Two honest readings:

1. This may simply need more capacity/training than a 3-stage CNN and a handful of epochs on 900 volumes can deliver — I'd want to rule that out before concluding the hypothesis is wrong.
2. It may be reproducing a **documented, not incidental, property of reconstruction-based SSL**: He et al. themselves report that MAE underperforms contrastive methods (SimCLR, MoCo) and self-distillation methods (DINO) on *linear probing*, precisely because reconstruction losses reward reconstructing pixels/voxels, not reward separating classes linearly — while fine-tuning recovers (and often exceeds) contrastive methods because gradient descent can reshape the representation using label signal. Our fine-tuned numbers (0.84, matching the from-scratch ceiling, but with much better low-label performance) are consistent with that story.

If I were continuing this the next experiment isn't "mask ratio again" — it's **adding a contrastive or self-distillation term (SimCLR-style or DINO-style) alongside the reconstruction loss**, since that's the documented fix for weak linear separability, and evaluating with a kNN probe as a cheaper, non-parametric alternative to a trained linear head to rule out probe-optimization noise as a confound.

![PCA of frozen embeddings](outputs/pca_embeddings.png)

The PCA of the frozen embeddings (colors = true class, never seen during pretraining) visually corroborates this: there's some structure (spheres and cubes trend toward the bottom, cross/ellipsoid trend toward the top) but heavy overlap — consistent with a representation that's *useful as an initialization* but not *linearly decodable*.

## A second, smaller diagnostic: confusion matrix

![Confusion matrices](outputs/confusion_matrices.png)

Both models' main error mode is **ellipsoid ↔ cylinder confusion**. This is explainable rather than a bug: both are smooth, elongated, convex shapes, and under random 3D rotation their silhouettes at 24³ resolution genuinely overlap for a range of aspect ratios. In a real context, this is the same category of failure I'd expect between anatomically similar structures (e.g. adjacent organ boundaries, or lesions with similar morphology) — worth calling out specifically because "which classes get confused, and why" is usually more informative than the aggregate accuracy number.

## Honest limitations

- **Compute**: 1 CPU core, no GPU. Model is small (≈0.4M params), training runs are short (10–15 epochs), and the mask-ratio ablation used fewer epochs than the main run for time reasons — a real comparison would match training budgets exactly.
- **Synthetic data**: geometric shapes are a much easier, lower-dimensional problem than real anatomy; I'd expect the *qualitative* label-efficiency story to transfer but not the specific numbers.
- **CNN, not ViT**: the backbone substitution described above is a real deviation from what the JD asks for; I'm treating it as an honest constraint, not something to gloss over.
- **No held-out clinical validation, no multi-task heads, no text/lab fusion**: this is a single-organ, single-modality toy problem. Scaling it to clinical's actual foundation-model direction is the next section.

## How this would scale to clinic's actual problem

- **Backbone**: swap the CNN for a 3D ViT (or a hybrid conv-stem + ViT, common for volumetric MAE) once GPU compute is available; patch-based tokenization is a natural fit for whole-body MRI's large volumes.
- **SSL objective**: given the linear-probe finding above, I'd pair masked-volume reconstruction with a contrastive or self-distillation term (JEPA-style or DINO-style) rather than relying on reconstruction alone, especially for any use case that wants frozen-embedding transfer (fast triage models, similarity search across scans) rather than full fine-tuning per task.
- **Multimodal fusion**: extend the encoder to accept structured/lab data and radiology text as auxiliary tokens or a separate fusion head, per the JD's "imaging, radiology, lab, and longitudinal data" language.
- **Multi-task heads**: one pretrained backbone, multiple lightweight heads (organ classification, segmentation, biomarker regression) — this is where the label-efficiency result matters most operationally, since it means the expensive backbone pretraining is shared across every downstream clinical task.
- **Validation rigor**: real deployment needs clinical sign-off, documented validation studies, and likely FDA 510(k)-relevant benchmarking — the "diagnose failure modes independently" muscle exercised above (confusion-matrix analysis, ablation design, honest reporting of a negative result) is the same muscle that validation work requires.

## Repo structure

```
ssl/
├── config.py              # shared experiment config
├── data.py                # synthetic 3D dataset (on-the-fly generation)
├── models.py               # 3D CNN encoder/decoder, classifier head, masking utility
├── train.py                # SSL pretraining loop + downstream training/eval
├── step1_pretrain.py        # CLI: SSL pretrain, checkpoints encoder
├── step2_downstream.py      # CLI: downstream training for one condition, all label fractions
├── step3_pca.py             # frozen-embedding PCA
├── step4_plots.py           # generates all figures
├── checkpoints/             # saved encoder weights (v1: mask=0.5/patch=4, v2: mask=0.75/patch=6)
└── outputs/
    ├── results.json         # full numeric results (all conditions, all fractions)
    ├── label_efficiency.png
    ├── ssl_pretrain_loss.png
    ├── pca_embeddings.png
    └── confusion_matrices.png
```

To reproduce: `python3 step1_pretrain.py 0.5 4 v1` → `python3 step2_downstream.py scratch v1` (and `ssl_frozen` / `ssl_finetuned`) → `python3 step3_pca.py` → `python3 step4_plots.py`.
