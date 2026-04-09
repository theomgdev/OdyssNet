import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import os
import time
import warnings

from odyssnet import OdyssNet, OdyssNetTrainer, TrainingHistory, set_seed

# --- Configuration ---
SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-2
MIN_LR = 1e-7
WARMUP_EPOCHS = 3
NUM_NEURONS = 10
GRID_SIZE = 4
THINKING_STEPS = GRID_SIZE * GRID_SIZE  # Total patches in spiral order
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_spiral_indices(rows: int, cols: int):
    """
    Algorithmically calculates the clockwise inward spiral visit order for any grid size.
    Returns a list of flat indices.
    """
    indices = []
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1

    while top <= bottom and left <= right:
        # Move Right
        for i in range(left, right + 1):
            indices.append(top * cols + i)
        top += 1

        # Move Down
        for i in range(top, bottom + 1):
            indices.append(i * cols + right)
        right -= 1

        if top <= bottom:
            # Move Left
            for i in range(right, left - 1, -1):
                indices.append(bottom * cols + i)
            bottom -= 1

        if left <= right:
            # Move Up
            for i in range(bottom, top - 1, -1):
                indices.append(i * cols + left)
            left += 1

    return indices


def main():
    # Silence internal PyTorch scheduler warnings during sequential transitions
    warnings.filterwarnings("ignore", message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary")

    print("OdyssNet: MNIST RECORD CHALLENGE (Spiral-Fed 4x4 Patch Model)")
    print(f"Strategy: 16 Spiral Patches (7x7=49 pixels) -> Embed(4 Neurons) -> Core({NUM_NEURONS}) -> Decoder(10 Classes)")
    set_seed(SEED)

    if DEVICE == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        # Try to compile model for speed if available
        model_compile = hasattr(torch, 'compile')
        if model_compile:
            print("OdyssNet: torch.compile enabled for speed.")
    else:
        model_compile = False

    # Strategy: 16 Patches -> Embed(4) -> Core(10) -> Output Decoder (10)
    input_ids = [0, 1, 2, 3]  # Map to first 4 neurons
    output_ids = list(range(NUM_NEURONS))  # Decoder reads from all neurons

    # vocab_size = [v_in, v_out]
    # v_in = 49 pixels (from each 7x7 patch)
    # v_out = 10 classes
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[49, 10],
        vocab_mode='continuous',
        weight_init='micro_quiet_warm',
        gate='none'
    )

    # Speed up core with torch.compile if on PyTorch 2.0+
    if 'model_compile' in locals() and model_compile:
        model = torch.compile(model)

    total_params = model.get_num_params()
    print(f"Total Params: {total_params} (Goal: < 500)")

    # Data Preparation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

    steps_per_epoch = len(train_loader)

    trainer = OdyssNetTrainer(
        model,
        device=DEVICE, lr=LR,
    )

    # Scheduler: Warmup + Cosine Annealing (batch-wise precise)
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    total_steps = NUM_EPOCHS * steps_per_epoch

    scheduler1 = LinearLR(trainer.optimizer, start_factor=MIN_LR / LR, end_factor=1.0, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(trainer.optimizer, T_max=total_steps - warmup_steps, eta_min=MIN_LR)
    scheduler = SequentialLR(trainer.optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainer.loss_fn = loss_fn

    print(f"Training with Batch Size: {BATCH_SIZE} for {NUM_EPOCHS} Epochs...")
    history = TrainingHistory()
    start_time = time.time()

    SPIRAL = get_spiral_indices(GRID_SIZE, GRID_SIZE)

    # Processing Loop

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            batch_size = data.size(0)
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            # Extract 4x4 grid of 7x7 patches: (B, 1, 28, 28) -> (B, 16, 49)
            patches = data.unfold(2, 7, 7).unfold(3, 7, 7)       # (B, 1, 4, 4, 7, 7)
            patches = patches.contiguous().view(batch_size, 16, 49)  # (B, 16, 49)

            # Reorder patches: edges first, center last (spiral inward)
            seq_input = patches[:, SPIRAL, :]  # (B, 16, 49)

            loss = trainer.train_batch(seq_input, target, thinking_steps=THINKING_STEPS)
            total_loss += loss
            scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                batch_size = data.size(0)
                data = data.to(DEVICE)
                target = target.to(DEVICE)

                # Same patch extraction and spiral reordering
                patches = data.unfold(2, 7, 7).unfold(3, 7, 7)
                patches = patches.contiguous().view(batch_size, 16, 49)
                seq_input = patches[:, SPIRAL, :]

                preds = trainer.predict(seq_input, thinking_steps=THINKING_STEPS)
                correct += (preds.argmax(1) == target).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # Calculate time metrics
        elapsed = time.time() - start_time
        avg_time_per_epoch = elapsed / (epoch + 1)
        remaining_epochs = NUM_EPOCHS - (epoch + 1)
        eta_seconds = remaining_epochs * avg_time_per_epoch

        def format_time(seconds):
            m, s = divmod(int(seconds), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        history.record(loss=avg_loss, accuracy=acc, lr=current_lr)

        print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | Loss {avg_loss:.4f} | Acc {acc:5.2f}% | "
              f"LR {current_lr:.2e} | Elapsed {format_time(elapsed)} | ETA {format_time(eta_seconds)}")

    history.plot(title="MNIST Record (480 Params) Training")

if __name__ == "__main__":
    main()
