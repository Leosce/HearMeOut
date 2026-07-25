"""
model.py
────────
Contains:
  • LipNet       — 3D Conv + Bi-GRU (original architecture, recommended)
  • AttentionNet — 3D Conv + Transformer (experimental)
  • Training utilities: train_one_epoch, validate, train, plot_loss
  • Checkpoint helpers

Changes in this version for fresh training:
  - LipNet GRU dropout = 0.3
  - Conv Dropout3d = 0.3
  - Random train/val split helper fixes speaker/file ordering bias
  - ReduceLROnPlateau uses factor=0.5, patience=3
  - No checkpoint is loaded by default; this file trains from scratch
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import CONFIG as _cfg
from CONFIG import DEVICE, VOCAB_SIZE, MODEL_PATH, EPOCHS, BATCH_SIZE, LR, MODEL_TYPE,idx_to_char
from data_preprocessing import LipDataset, collate_fn, ctc_decode


# ── LipNet (3D Conv + Bi-GRU) ─────────────────────────────────────────────────

class LipNet(nn.Module):
    """
    LipNet-style model.
    Input  : (B, T, H=46, W=100, C=1) — channel-last, as returned by LipDataset
    Output : (B, T', vocab_size)

    dropout controls the GRU layer dropout only.
    dropout controls the GRU layer dropout.
    Conv Dropout3d is set to 0.3 for fresh training.
    """
    def __init__(self, vocab_size: int = VOCAB_SIZE, dropout: float = 0.3):
        super().__init__()

        # 3D Conv front-end — spatial dropout fixed at 0.3
        self.conv1 = nn.Conv3d(1,  32, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.drop1 = nn.Dropout3d(0.3)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2), bias=False)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.drop2 = nn.Dropout3d(0.3)

        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn3   = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.drop3 = nn.Dropout3d(0.3)

        # After 3 × (1,2,2) pool: H: 46→23→11→5, W: 100→50→25→12
        self._cnn_feat = 96 * 5 * 12

        # Bi-GRU back-end — temporal dropout controlled by `dropout` param
        self.gru1  = nn.GRU(self._cnn_feat, 256, bidirectional=True, batch_first=True)
        self.ln1   = nn.LayerNorm(512)
        self.drop4 = nn.Dropout(dropout)

        self.gru2  = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.ln2   = nn.LayerNorm(512)
        self.drop5 = nn.Dropout(dropout)

        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, C=1)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)

        x = self.drop1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, -1)

        x, _ = self.gru1(x)
        x = self.drop4(F.relu(self.ln1(x)))

        x, _ = self.gru2(x)
        x = self.drop5(F.relu(self.ln2(x)))

        return self.fc(x)  # (B, T, vocab_size)


# ── AttentionNet (3D Conv + Transformer) ──────────────────────────────────────

class AttentionNet(nn.Module):
    """Transformer variant — experimental."""
    def __init__(self, vocab_size: int = VOCAB_SIZE, dropout: float = 0.25):
        super().__init__()

        self.conv1 = nn.Conv3d(1,  32, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False)
        self.bn1   = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2), bias=False)
        self.bn2   = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(64, 96, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn3   = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        d_model = 96 * 5 * 12

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=1024,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 4, 1, 2, 3)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, T, -1)

        x = self.transformer(x)
        return self.fc(x)


# ── Weight initialisation ─────────────────────────────────────────────────────

def initialize_weights(m):
    if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
    elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ── Dataset split helper ─────────────────────────────────────────────────────

import copy
import random

def make_random_split_datasets(seed=42, val_ratio=0.1):
    """
    Build a true random train/val split from the existing LipDataset samples.

    Important:
    LipDataset keeps labels in self.cache, so if we mix samples from the old
    train and val datasets, we must also merge their caches.
    """
    old_train = LipDataset(split="train")
    old_val = LipDataset(split="val")

    # Combine samples from both old splits
    all_samples = list(old_train.samples) + list(old_val.samples)

    # Merge caches so every sample key exists
    full_cache = {}

    if hasattr(old_train, "cache"):
        full_cache.update(old_train.cache)

    if hasattr(old_val, "cache"):
        full_cache.update(old_val.cache)

    # Shuffle before new split
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    split_idx = int((1.0 - val_ratio) * len(all_samples))

    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Copy the dataset object so paths/config stay the same
    train_ds = copy.copy(old_train)
    val_ds = copy.copy(old_train)

    train_ds.samples = train_samples
    val_ds.samples = val_samples

    # Critical fix: both datasets need the full merged cache
    train_ds.cache = full_cache
    val_ds.cache = full_cache

    print(f"Total samples: {len(all_samples)}")
    print(f"Train samples: {len(train_ds.samples)}")
    print(f"Val samples  : {len(val_ds.samples)}")

    print("\nFirst 10 train samples:")
    for sample in train_ds.samples[:10]:
        print(sample)

    print("\nFirst 10 val samples:")
    for sample in val_ds.samples[:10]:
        print(sample)

    return train_ds, val_ds


# ── Training utilities ────────────────────────────────────────────────────────

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(model, optimizer, loss_fn, loader):
    model.train()
    total_loss = 0.0

    for i, (frames, labels, video_lengths, label_lengths) in enumerate(loader):
        frames = frames.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(frames)  # (B, T, V)
        B, T, V = logits.shape

        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)  # (T, B, V)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)
        target_lengths = label_lengths.to(DEVICE, non_blocking=True)

        loss = loss_fn(log_probs, labels, input_lengths, target_lengths)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (i + 1)


def labels_to_text(labels, label_lengths):
    """
    Convert padded label tensor back to text.
    labels shape: (B, L_max)
    """
    texts = []

    labels = labels.cpu()
    label_lengths = label_lengths.cpu()

    for i in range(labels.size(0)):
        length = int(label_lengths[i].item())
        seq = labels[i, :length].tolist()

        text = "".join(idx_to_char[idx] for idx in seq if idx != 0)
        texts.append(text.strip())

    return texts


def edit_distance(a, b):
    """
    Edit distance between two lists.
    For CER: list of characters.
    For WER: list of words.
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i

    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


def calculate_metrics(preds, targets):
    sentence_correct = 0
    total_words = 0
    correct_words = 0

    total_cer = 0.0
    total_wer = 0.0

    n = len(preds)

    for pred, target in zip(preds, targets):
        pred = pred.strip()
        target = target.strip()

        if pred == target:
            sentence_correct += 1

        pred_words = pred.split()
        target_words = target.split()

        # Word accuracy by position
        min_len = min(len(pred_words), len(target_words))
        correct_words += sum(
            1 for i in range(min_len)
            if pred_words[i] == target_words[i]
        )
        total_words += len(target_words)

        # WER
        if len(target_words) > 0:
            total_wer += edit_distance(pred_words, target_words) / len(target_words)

        # CER
        pred_chars = list(pred.replace(" ", ""))
        target_chars = list(target.replace(" ", ""))

        if len(target_chars) > 0:
            total_cer += edit_distance(pred_chars, target_chars) / len(target_chars)

    metrics = {
        "sentence_acc": sentence_correct / n if n > 0 else 0.0,
        "word_acc": correct_words / total_words if total_words > 0 else 0.0,
        "wer": total_wer / n if n > 0 else 0.0,
        "cer": total_cer / n if n > 0 else 0.0,
    }

    return metrics

@torch.no_grad()
def validate(model, loss_fn, loader):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []

    last_decoded = []

    for i, (frames, labels, video_lengths, label_lengths) in enumerate(loader):
        frames = frames.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(frames)
        B, T, V = logits.shape

        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)
        target_lengths = label_lengths.to(DEVICE, non_blocking=True)

        loss = loss_fn(log_probs, labels, input_lengths, target_lengths)
        total_loss += loss.item()

        pred_ids = torch.argmax(logits, dim=-1).cpu()
        decoded = ctc_decode(pred_ids)
        targets = labels_to_text(labels.cpu(), label_lengths.cpu())

        all_preds.extend([p.strip() for p in decoded])
        all_targets.extend([t.strip() for t in targets])

        last_decoded = decoded

    avg_loss = total_loss / (i + 1)
    metrics = calculate_metrics(all_preds, all_targets)

    return avg_loss, last_decoded, metrics


def save_checkpoint(path, model, optimizer, epoch, val_loss, speaker):
    """Save model + optimizer state with metadata."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "speaker": speaker,
    }, path)


def load_checkpoint(path, model, optimizer=None, device=DEVICE, load_optimizer=True):
    """
    Load checkpoint.

    For the continue-from-best experiment, use:
        load_checkpoint(path, model, optimizer=None, load_optimizer=False)

    That loads weights only and keeps your fresh lower-LR optimizer clean.
    """
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and load_optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        speaker = ckpt.get("speaker", "unknown")
        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", None)
    else:
        model.load_state_dict(ckpt)
        speaker, epoch, val_loss = "unknown", "?", None

    print("✓ Loaded checkpoint")
    print(f"  Path      : {path}")
    print(f"  Speaker   : {speaker}")
    print(f"  Epoch     : {epoch}")
    print(f"  Val loss  : {val_loss}")
    print(f"  Optimizer : {'loaded' if optimizer is not None and load_optimizer else 'fresh / not loaded'}")

    return {
        "speaker": speaker,
        "epoch": epoch,
        "val_loss": val_loss,
    }


def train(model, optimizer, loss_fn, scheduler, train_loader, val_loader,
          epochs=None, speaker=None, initial_best_vloss=float("inf"),
          best_model_name="best_model.pt"):
    n_epochs = epochs if epochs is not None else _cfg.EPOCHS
    speaker_name = speaker if speaker is not None else getattr(_cfg, "SPEAKER", "unknown")

    os.makedirs(MODEL_PATH, exist_ok=True)

    history = {"epochs": [], "train_loss": [], "val_loss": [], "lr": []}
    best_record = {"epoch": [], "loss": [], "predictions": []}
    best_vloss = initial_best_vloss

    print(f"Training for {n_epochs} epochs  |  speaker={speaker_name}  |  device={DEVICE}")
    print(f"Starting LR: {get_lr(optimizer):.2e}")
    print(f"Current best val to beat: {best_vloss:.4f}" if best_vloss < float("inf") else "Current best val to beat: none")

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
        val_loss, decoded, metrics = validate(model, loss_fn, val_loader)
        scheduler.step(val_loss)

        current_lr = get_lr(optimizer)
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        sample = decoded[0] if decoded else ""

        print(
            f"Epoch {epoch:>3}/{n_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"sent_acc={metrics['sentence_acc']:.3f}  "
            f"word_acc={metrics['word_acc']:.3f}  "
            f"WER={metrics['wer']:.3f}  "
            f"CER={metrics['cer']:.3f}  "
            f"lr={current_lr:.2e}  "
            f"sample='{sample}'"
        )

        if val_loss < best_vloss:
            best_vloss = val_loss
            best_record["epoch"].append(epoch)
            best_record["loss"].append(val_loss)
            best_record["predictions"].append(decoded)

            save_path = os.path.join(MODEL_PATH, best_model_name)
            save_checkpoint(save_path, model, optimizer, epoch, val_loss, speaker_name)
            print(f"  ✓ New best saved: {save_path} (val={val_loss:.4f})")

        # Save latest checkpoint every epoch so GPU work is never lost
        latest_path = os.path.join(MODEL_PATH, "latest_continue_checkpoint.pt")
        save_checkpoint(latest_path, model, optimizer, epoch, val_loss, speaker_name)

    return history, best_record


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_loss(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    epochs = history["epochs"]

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(epochs, history["val_loss"], label="val")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epoch")

    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(MODEL_PATH, "loss_curve_continue.png")
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to: {save_path}")
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device : {DEVICE}")

    # Fresh training: make a random train/val split before DataLoader creation.
    # This fixes ordered speaker/file split bias.
    train_ds, val_ds = make_random_split_datasets(seed=42, val_ratio=0.10)

    num_workers = getattr(_cfg, "NUM_WORKERS", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = (LipNet() if MODEL_TYPE == "lipnet" else AttentionNet()).to(DEVICE)
    model.apply(initialize_weights)

    print(f"Model      : {MODEL_TYPE}")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    speaker_name = "+".join(getattr(_cfg, "SPEAKERS", [getattr(_cfg, "SPEAKER", "unknown")]))

    print("\nMode: fresh training from scratch")
    history, best = train(
        model,
        optimizer,
        loss_fn,
        scheduler,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        speaker=speaker_name + "_scratch_random_split_dropout03",
        initial_best_vloss=float("inf"),
        best_model_name="best_model.pt",
    )

    plot_loss(history)
