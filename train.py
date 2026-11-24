import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MaskedAutoencoder3D, EncoderClassifier, random_cube_mask


def pretrain_ssl(train_ds, epochs=15, batch_size=16, lr=1e-3, patch=4, mask_ratio=0.5,
                  base=16, embed_dim=128, log_every=5, verbose=True):
    """Masked-volume reconstruction pretraining (no labels used)."""
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = MaskedAutoencoder3D(base=base, embed_dim=embed_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for vol, _label in loader:
            x_masked, mask = random_cube_mask(vol, patch=patch, mask_ratio=mask_ratio)
            recon = model(x_masked)
            loss = ((recon - vol) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()
        avg = running / max(1, n_batches)
        history.append({"epoch": epoch + 1, "recon_loss": avg, "elapsed_s": time.time() - t0})
        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1):
            print(f"  [SSL pretrain] epoch {epoch+1}/{epochs}  masked-voxel MSE={avg:.4f}  ({time.time()-t0:.0f}s elapsed)")
    return model.encoder, history


def train_classifier(train_ds, val_ds, num_classes, encoder=None, freeze_encoder=False,
                      epochs=25, batch_size=16, lr=1e-3, embed_dim=128, base=16, verbose=False):
    """Train a classifier head, optionally on top of a (pretrained) encoder.
    If encoder is None, a freshly-initialized encoder is trained end-to-end
    (this is the 'from scratch' baseline)."""
    from models import Encoder3D

    if encoder is None:
        encoder = Encoder3D(base=base, embed_dim=embed_dim)
        freeze_encoder = False  # can't freeze a randomly-initialized encoder, nothing to leverage
    else:
        encoder = copy.deepcopy(encoder)

    model = EncoderClassifier(encoder, num_classes=num_classes, embed_dim=embed_dim, freeze_encoder=freeze_encoder)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        for vol, label in train_loader:
            logits = model(vol)
            loss = crit(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for vol, label in val_loader:
                logits = model(vol)
                pred = logits.argmax(1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_acc = correct / max(1, total)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"    epoch {epoch+1}/{epochs}  val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


@torch.no_grad()
def evaluate(model, test_ds, batch_size=64):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    for vol, label in loader:
        logits = model(vol)
        pred = logits.argmax(1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        all_preds.extend(pred.tolist())
        all_labels.extend(label.tolist())
    acc = correct / max(1, total)
    return acc, all_preds, all_labels
