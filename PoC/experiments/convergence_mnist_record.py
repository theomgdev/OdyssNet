import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import time
import warnings


# Disable BNB for this experiment to rule out quantization noise and use pure dynamics
os.environ["NO_BNB"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from odyssnet import OdyssNet, OdyssNetTrainer, set_seed


def _spiral_order_4x4():
    """
    Returns patch indices in clockwise inward spiral order for a 4x4 grid.
    Edges first (12 patches), center last (4 patches).

    Grid layout:          Spiral visit order:
     0  1  2  3            1  2  3  4
     4  5  6  7           12 13 14  5
     8  9 10 11           11 16 15  6
    12 13 14 15           10  9  8  7
    """
    return [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4, 5, 6, 10, 9]


def main():
    print("OdyssNet 2.2: MNIST RECORD CHALLENGE (Spiral-Fed 4x4 Patch Model)")
    print("Strategy: 16 Spiral Patches (7x7=49 pixels) -> Embed(4 Neurons) -> Core(10) -> Decoder(10 Classes)")
    set_seed(42)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if DEVICE == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        # Try to compile model for speed if available
        if hasattr(torch, 'compile'):
            try:
                model_compile = True
                print("OdyssNet: torch.compile enabled for speed.")
            except:
                model_compile = False
        else:
            model_compile = False

    # Strategy: 16 Patches (49 pixels) -> Embed(4 Neurons) -> Core(10 Neurons) -> Output Decoder (10 Neurons -> 10 Classes)
    NUM_NEURONS = 10
    input_ids = [0, 1, 2, 3]  # 49 pixels projected to 4 neurons
    output_ids = list(range(10))  # Decoder reads from all 10 neurons

    # vocab_size = [v_in, v_out]
    # v_in = 49 pixels (from each 7x7 patch)
    # v_out = 10 classes
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[49, 10],   # [49 pixels -> 4 neurons, 10 neurons -> decoder]
        vocab_mode='continuous',
        weight_init='micro_quiet_8bit'
    )

    # Speed up core with torch.compile if on PyTorch 2.0+
    if 'model_compile' in locals() and model_compile:
        model = torch.compile(model)

    total_params = model.get_num_params()
    print(f"Total Params: {total_params} (Goal: < 500)")

    # Data Preparation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

    NUM_EPOCHS = 100
    steps_per_epoch = len(train_loader)

    trainer = OdyssNetTrainer(
        model,
        device=DEVICE, lr=5e-3,
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainer.loss_fn = loss_fn

    print(f"Training with Batch Size: {BATCH_SIZE} for {NUM_EPOCHS} Epochs...")
    start_time = time.time()

    # 16 patches in spiral order -> 16 sequential steps
    TOTAL_STEPS = 16
    SPIRAL = _spiral_order_4x4()

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

            loss = trainer.train_batch(seq_input, target, thinking_steps=TOTAL_STEPS)
            total_loss += loss

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

                preds = trainer.predict(seq_input, thinking_steps=TOTAL_STEPS)
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

        print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | Loss {avg_loss:.4f} | Acc {acc:5.2f}% | "
              f"LR {current_lr:.2e} | Elapsed {format_time(elapsed)} | ETA {format_time(eta_seconds)}")

if __name__ == "__main__":
    main()
