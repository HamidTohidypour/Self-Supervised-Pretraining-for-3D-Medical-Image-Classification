"""
Synthetic 3D volumetric dataset, generated on-the-fly (no disk I/O, no network).

Why synthetic: this sandbox cannot reach Zenodo, where the real MedMNIST3D /
OrganMNIST3D volumes are hosted, so a literal drop-in of that dataset isn't
possible here. Instead we generate a dataset with the same *shape* of problem
that OrganMNIST3D poses: single-channel 3D volumes, multiple classes defined
by geometric structure, corrupted by realistic-ish acquisition noise and
intensity inhomogeneity (a simplified bias field, common in MRI). Every
volume is generated deterministically from an integer index + global seed, so
the dataset is reproducible and doesn't need to be materialized in memory.

Swapping in the real data later is a single change: replace `SyntheticOrgan3D`
with `medmnist.OrganMNIST3D(split=..., download=True)` (see README.md).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["sphere", "cube", "ellipsoid", "cylinder", "cross", "ring"]
NUM_CLASSES = len(CLASS_NAMES)


def _grid(size):
    ax = np.linspace(-1, 1, size)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    return x, y, z


def _rotate(x, y, z, rng):
    """Apply a random 3D rotation (composed of 3 random-angle axis rotations)."""
    for _ in range(3):
        axis = rng.integers(0, 3)
        theta = rng.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        if axis == 0:
            y, z = c * y - s * z, s * y + c * z
        elif axis == 1:
            x, z = c * x + s * z, -s * x + c * z
        else:
            x, y = c * x - s * y, s * x + c * y
    return x, y, z


def _make_shape(label, size, rng):
    x, y, z = _grid(size)
    cx, cy, cz = rng.uniform(-0.15, 0.15, 3)
    x, y, z = x - cx, y - cy, z - cz
    x, y, z = _rotate(x, y, z, rng)
    scale = rng.uniform(0.55, 0.85)

    if label == 0:  # sphere
        mask = (x**2 + y**2 + z**2) < scale**2
    elif label == 1:  # cube
        s = scale * 0.8
        mask = (np.abs(x) < s) & (np.abs(y) < s) & (np.abs(z) < s)
    elif label == 2:  # ellipsoid
        a, b, c = scale, scale * rng.uniform(0.4, 0.6), scale * rng.uniform(0.7, 0.9)
        mask = (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 < 1
    elif label == 3:  # cylinder
        r = scale * 0.55
        h = scale * 0.9
        mask = (x**2 + y**2 < r**2) & (np.abs(z) < h)
    elif label == 4:  # cross / plus
        s = scale * 0.28
        long_ = scale * 0.9
        mask = (
            ((np.abs(x) < long_) & (np.abs(y) < s) & (np.abs(z) < s))
            | ((np.abs(y) < long_) & (np.abs(x) < s) & (np.abs(z) < s))
            | ((np.abs(z) < long_) & (np.abs(x) < s) & (np.abs(y) < s))
        )
    elif label == 5:  # ring / torus-like (sphere minus inner sphere)
        r_out = scale
        r_in = scale * rng.uniform(0.45, 0.65)
        rad = x**2 + y**2 + z**2
        mask = (rad < r_out**2) & (rad > r_in**2)
    else:
        raise ValueError(label)
    return mask


def _bias_field(size, rng):
    """Smooth low-frequency multiplicative field, mimicking MRI intensity
    inhomogeneity (bias field)."""
    x, y, z = _grid(size)
    freq = rng.uniform(0.5, 1.5, 3)
    phase = rng.uniform(0, 2 * np.pi, 3)
    field = (
        np.sin(freq[0] * np.pi * x + phase[0])
        + np.sin(freq[1] * np.pi * y + phase[1])
        + np.sin(freq[2] * np.pi * z + phase[2])
    )
    field = 1.0 + 0.15 * (field / 3.0)
    return field


def generate_volume(index, size=24, seed=0):
    """Deterministically generate one (volume, label) pair from an integer index."""
    rng = np.random.default_rng(seed * 1_000_003 + index)
    label = int(index % NUM_CLASSES)
    mask = _make_shape(label, size, rng)

    fg_intensity = rng.uniform(0.65, 1.0)
    bg_intensity = rng.uniform(0.0, 0.12)
    vol = np.where(mask, fg_intensity, bg_intensity).astype(np.float32)

    vol *= _bias_field(size, rng).astype(np.float32)
    vol += rng.normal(0, 0.05, size=vol.shape).astype(np.float32)  # thermal/acquisition noise
    vol = np.clip(vol, 0, None)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    return vol.astype(np.float32), label


class SyntheticOrgan3D(Dataset):
    """On-the-fly synthetic 3D shape-classification dataset.

    Args:
        indices: list/array of global sample indices this split owns.
        size: volume side length (volumes are size^3, single channel).
        seed: global seed controlling the whole dataset's randomness.
    """

    def __init__(self, indices, size=24, seed=0):
        self.indices = list(indices)
        self.size = size
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        vol, label = generate_volume(self.indices[i], size=self.size, seed=self.seed)
        vol = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
        return vol, label


def make_splits(n_train=3000, n_val=500, n_test=500, size=24, seed=0):
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    return (
        SyntheticOrgan3D(train_idx, size=size, seed=seed),
        SyntheticOrgan3D(val_idx, size=size, seed=seed),
        SyntheticOrgan3D(test_idx, size=size, seed=seed),
    )


def label_subset(dataset, fraction, seed=0):
    """Return a Subset with a stratified fraction of the labels (simulates a
    clinical setting where only a fraction of scans have annotation)."""
    from torch.utils.data import Subset

    n = len(dataset)
    rng = np.random.default_rng(seed)
    per_class = {}
    for i in range(n):
        _, label = generate_volume(dataset.indices[i], size=dataset.size, seed=dataset.seed)
        per_class.setdefault(label, []).append(i)
    keep = []
    for label, idxs in per_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * fraction)))
        keep.extend(idxs[:k].tolist())
    keep.sort()
    return Subset(dataset, keep)
