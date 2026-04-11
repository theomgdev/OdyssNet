import os
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from odyssnet import OdyssNet, OdyssNetTrainer, TrainingHistory, set_seed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-2

# Architecture
NUM_NEURONS = 10
EMBED_NEURONS = 4   # Neurons that receive patch input (first N neurons)
NUM_CLASSES = 10    # Output classes

# Patch strategy: divide 28×28 image into GRID_SIZE×GRID_SIZE non-overlapping patches
IMAGE_SIZE = 28
GRID_SIZE = 4
PATCH_SIZE = IMAGE_SIZE // GRID_SIZE    # 7 pixels per side
PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE  # 49 pixels per patch (embed input dim)
NUM_PATCHES = GRID_SIZE * GRID_SIZE     # 16 patches total
THINKING_RATIO = 1                      # Thinking steps per patch (1 = inject only, 2 = inject + 1 free step, ...)
THINKING_STEPS = NUM_PATCHES * THINKING_RATIO

# DataLoader
NUM_WORKERS = min(4, os.cpu_count() or 1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_spiral_indices(rows: int, cols: int) -> list[int]:
    """Return flat patch indices in clockwise inward spiral order."""
    indices = []
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1

    while top <= bottom and left <= right:
        for i in range(left, right + 1):        # right
            indices.append(top * cols + i)
        top += 1

        for i in range(top, bottom + 1):        # down
            indices.append(i * cols + right)
        right -= 1

        if top <= bottom:
            for i in range(right, left - 1, -1):  # left
                indices.append(bottom * cols + i)
            bottom -= 1

        if left <= right:
            for i in range(bottom, top - 1, -1):  # up
                indices.append(i * cols + left)
            left += 1

    return indices


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_spiral_patches(images: torch.Tensor, spiral: list[int]) -> torch.Tensor:
    """
    Extract and spiral-reorder non-overlapping patches from a batch of images.

    Args:
        images: (B, 1, H, W) image batch.
        spiral: Patch visit order (flat indices).

    Returns:
        (B, NUM_PATCHES, PATCH_PIXELS) tensor.
    """
    b = images.size(0)
    patches = images.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    # (B, 1, GRID_SIZE, GRID_SIZE, PATCH_SIZE, PATCH_SIZE) → (B, NUM_PATCHES, PATCH_PIXELS)
    patches = patches.contiguous().view(b, NUM_PATCHES, PATCH_PIXELS)
    return patches[:, spiral, :]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("OdyssNet: MNIST Record Challenge")
    print(
        f"  Strategy : {NUM_PATCHES} spiral patches "
        f"({PATCH_SIZE}x{PATCH_SIZE}={PATCH_PIXELS} px) → "
        f"Embed({EMBED_NEURONS}) → Core({NUM_NEURONS}) → Decoder({NUM_CLASSES})"
    )
    set_seed(SEED)

    # GPU optimisations
    use_compile = False
    if DEVICE == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        use_compile = hasattr(torch, 'compile')
        if use_compile:
            print("  torch.compile enabled.")

    # Model
    input_ids = list(range(EMBED_NEURONS))
    output_ids = list(range(NUM_NEURONS))

    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[PATCH_PIXELS, NUM_CLASSES],
        vocab_mode='continuous',
        weight_init='micro_quiet_warm',
        gate='none',
    )

    if use_compile:
        model = torch.compile(model)

    total_params = model.get_num_params()
    print(f"  Params   : {total_params} (target: < 500)\n")

    # Data
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    loader_kwargs = dict(batch_size=BATCH_SIZE, pin_memory=(DEVICE == 'cuda'), num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Trainer
    trainer = OdyssNetTrainer(model, device=DEVICE, lr=LR)
    trainer.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Training for {NUM_EPOCHS} epochs | batch {BATCH_SIZE} | lr {LR} | device {DEVICE}")

    history = TrainingHistory()
    spiral = get_spiral_indices(GRID_SIZE, GRID_SIZE)
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            seq = extract_spiral_patches(images, spiral)
            total_loss += trainer.train_batch(seq, targets, thinking_steps=THINKING_STEPS)

        avg_loss = total_loss / len(train_loader)

        # --- Evaluate ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                seq = extract_spiral_patches(images, spiral)
                preds = trainer.predict(seq, thinking_steps=THINKING_STEPS)
                correct += (preds.argmax(1) == targets).sum().item()
                total += targets.size(0)

        acc = 100.0 * correct / total

        elapsed = time.time() - start_time
        eta = (elapsed / (epoch + 1)) * (NUM_EPOCHS - epoch - 1)

        history.record(loss=avg_loss, accuracy=acc)
        print(
            f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | "
            f"Loss {avg_loss:.4f} | "
            f"Acc {acc:5.2f}% | "
            f"Elapsed {format_time(elapsed)} | "
            f"ETA {format_time(eta)}"
        )

    history.plot(title=f"MNIST Record ({total_params} params) — {NUM_PATCHES}-patch spiral")


if __name__ == "__main__":
    main()
