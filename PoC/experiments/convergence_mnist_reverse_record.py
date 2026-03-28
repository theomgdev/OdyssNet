"""
OdyssNet 2.0 - Inverse MNIST Reverse Record Experiment

This experiment tests the network's ability to GENERATE images from labels,
reversing the normal classification direction.

Normal Classification: Image (28×28) -> Digit Label (0-9)
Inverse Generation:    Digit Label (0-9) -> Image (28×28)

The network learns to store visual patterns in its recurrent dynamics and
reconstruct full MNIST images from a single scalar input (digit/10.0).

Architecture:
- 12 neurons total (2 input, 6 output, 4 hidden)
- 21 total thinking steps: 5 warmup + 16 output steps
- Patches are tiled together to form complete 28×28 images
- Tests: Pattern storage, sequential generation, inverse mapping

This is a "reverse" of the standard Record experiment, proving the network
can learn bidirectional mappings between concepts (digits) and percepts (images).
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import time
import glob
import re

# --- Environment Setup ---
import warnings
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")
os.environ["NO_BNB"] = "1"  # Disable bitsandbytes for pure dynamics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'tmp', 'reverse_record_samples')
SAVE_EVERY_EPOCHS = 5

sys.path.append(PROJECT_ROOT)
from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGradConfig, set_seed

def format_time(seconds):
    """Formats seconds into HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def save_samples(model, trainer, device, total_thinking_steps, warmup_steps, output_steps, epoch, folder=SAMPLE_DIR):
    """
    Generate and save sample images for digits 0-9.
    
    Process:
    1. Feed digit labels (0.0 to 0.9) as input
    2. Run total thinking steps (warmup + output steps)
    3. Ignore warmup step outputs, keep only output steps
    4. Concatenate patches: (10 digits × 16 steps × 49 pixels) -> (10, 784)
    5. Reshape to 28×28 images and save visualization
    
    The saved images show how well the network learned to reconstruct
    MNIST digits from their labels alone - a true "reverse" of classification.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(folder, exist_ok=True)
    model.eval()
    
    # Generate 0-9: Create scalar inputs (0.0, 0.1, ..., 0.9) for each digit
    digits = torch.arange(10, dtype=torch.float32, device=device).view(10, 1, 1) / 10.0
    
    with torch.no_grad():
        # Predict full sequence, then drop initial warmup-only thought steps.
        preds_full = trainer.predict(digits, thinking_steps=total_thinking_steps, full_sequence=True)
        preds = preds_full[:, warmup_steps:warmup_steps + output_steps, :]
        
        # Reshape patches: (10, 16, 49) -> (10, 4, 4, 7, 7) -> (10, 28, 28)
        # Interpret 16 outputs as a 4×4 grid of 7×7 patches and tile them
        # back into a full 28×28 image for visualization.
        images = preds.reshape(10, 4, 4, 7, 7)              # (B, patch_row, patch_col, h, w)
        images = images.permute(0, 1, 3, 2, 4).reshape(10, 28, 28)  # (B, H, W)
        images = (images + 1.0) / 2.0  # Denormalize from (-1, 1) to (0, 1)
        images = images.cpu().numpy()
        
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(str(i))
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(folder, f"epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    """
    INVERSE MNIST REVERSE RECORD: Digit-to-Image Generation
    
    Architecture:
    - Input: Single digit (0-9) encoded as scalar value (digit/10.0)
    - Processing: 21 total steps through 12-neuron core (5 warmup + 16 output)
    - Output: 16 patches × 49 pixels = 784 pixels (28×28 image)
    
    The network learns to "hallucinate" MNIST digits from their label alone,
    reversing the normal classification direction (image -> digit).
    This tests the network's ability to store and reconstruct visual patterns.
    """
    set_seed(42)
    
    print("=" * 70)
    print("OdyssNet 2.0: INVERSE MNIST REVERSE RECORD (Generation Record)")
    print("=" * 70)
    print("Task: Generate 28×28 MNIST images from digit labels (0-9)")
    print("Direction: Digit (scalar) -> Image (784 pixels)")
    print("Architecture: 12 neurons | 5 warmup thoughts + 16 output thoughts | 16 patches × 49 pixels")
    print("=" * 70)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if DEVICE == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        
    # Model Configuration
    # 12 neurons total:
    #   - Neurons 0-1: Input (receives digit label as scalar)
    #   - Neurons 2-7: Output (6 neurons decoded together into 49 pixels per step)
    #   - Neurons 8-11: Hidden (internal processing)
    NUM_NEURONS = 12
    input_ids = [0, 1]
    output_ids = [2, 3, 4, 5, 6, 7]
    
    # Model Initialization
    # vocab_size=[1, 49]:
    #   - Input vocab: 1 value (digit/10.0 scalar)
    #   - Output vocab: 49 values (7×7 pixel patch decoded from all 6 output neurons per step)
    # Runtime uses 21 internal steps: first 5 warmup-only, last 16 supervised outputs.
    # Supervised output: 16 patches × 49 pixels = 784 pixels (28×28 image)
    # micro_quiet_8bit is kept intentionally for this tiny-core stability profile.
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[1, 49],   # 1 scalar input -> 49 pixel outputs
        vocab_mode='continuous',
        activation=['tanh', 'tanh', 'tanh'],
        weight_init='micro_quiet_8bit'
    )
    model = model.compile()
    
    total_params = model.get_num_params()
    print(f"Total Params: {total_params} (Goal: < 500)")
    
    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to (-1, 1)
    ])
    
    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    
    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    
    NUM_EPOCHS = 100
    steps_per_epoch = len(train_loader)
    
    scheduler_config = dict(
        warmup_steps=5 * steps_per_epoch,
        max_steps=NUM_EPOCHS * steps_per_epoch,
        min_lr_ratio=1e-3
    )
    
    trainer = OdyssNetTrainer(
        model, 
        device=DEVICE,
        chaos_config=ChaosGradConfig.tiny_network(lr=2e-3),
        scheduler_config=scheduler_config,
        use_temporal_scheduler=True,
        loss_fn=nn.MSELoss()
    )

    print(f"Training initialized. Batch size: {BATCH_SIZE} | Optimizer steps: {NUM_EPOCHS * steps_per_epoch}")
    print("Training strategy: 5 warmup thought steps, then 16 output steps for 16 image patches")
    print(f"Sample save cadence: every {SAVE_EVERY_EPOCHS} epochs (+epoch 1 and final epoch)")
    print(f"Sample directory: {SAMPLE_DIR}")
    print(f"Each patch is a 7×7 tile; 16 patches tile together to form 28×28 image")
    print("=" * 70)
    start_time = time.time()
    WARMUP_STEPS = 5
    OUTPUT_STEPS = 16
    TOTAL_THINKING_STEPS = WARMUP_STEPS + OUTPUT_STEPS
    
    try:
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            
            for data, target in train_loader:
                # data: (B, 1, 28, 28) normalized MNIST images
                # target: (B,) digit labels (0-9)
                batch_size = data.size(0)
                
                # Reshape image into 16 spatial patches of 49 pixels each (non-overlapping 7×7 tiles)
                # This matches the 16 supervised output-step structure (4×4 grid of 7×7 tiles)
                patches = data.unfold(2, 7, 7).unfold(3, 7, 7)  # (B, 1, 4, 4, 7, 7)
                patches = patches.contiguous().view(batch_size, 1, 16, 49)  # (B, 1, 16, 49)
                target_images = patches.squeeze(1).to(DEVICE, non_blocking=True)  # (B, 16, 49)
                
                # Encode digit label as scalar (0.0 to 0.9) for input neurons
                digit_inputs = target.view(batch_size, 1, 1).float().to(DEVICE, non_blocking=True) / 10.0
                
                # Train: network first thinks for warmup steps, then emits 16 supervised frames.
                loss = trainer.train_batch(
                    digit_inputs,
                    target_images,
                    thinking_steps=TOTAL_THINKING_STEPS,
                    full_sequence=True,
                    output_transform=lambda y: y[:, WARMUP_STEPS:WARMUP_STEPS + OUTPUT_STEPS, :]
                )
                total_loss += loss
                
            avg_loss = total_loss / len(train_loader)
            
            # Save generation samples periodically
            should_save = ((epoch + 1) % SAVE_EVERY_EPOCHS == 0) or (epoch == 0) or (epoch + 1 == NUM_EPOCHS)
            if should_save:
                try:
                    save_path = save_samples(
                        model,
                        trainer,
                        DEVICE,
                        TOTAL_THINKING_STEPS,
                        WARMUP_STEPS,
                        OUTPUT_STEPS,
                        epoch,
                        folder=SAMPLE_DIR,
                    )
                    print(f"[Samples] Saved {save_path}")
                except Exception as e:
                    print(f"[Samples] Save failed at epoch {epoch+1}: {e}")
            
            # Calculate metrics
            current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else trainer.optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            avg_time_per_epoch = elapsed / (epoch + 1)
            remaining_epochs = NUM_EPOCHS - (epoch + 1)
            eta_seconds = remaining_epochs * avg_time_per_epoch
            
            print(f"Epoch {epoch+1:4d}/{NUM_EPOCHS} | Loss {avg_loss:.6f} | "
                  f"LR {current_lr:.2e} | Elapsed {format_time(elapsed)} | ETA {format_time(eta_seconds)}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        if 'epoch' in locals():
            try:
                save_path = save_samples(
                    model,
                    trainer,
                    DEVICE,
                    TOTAL_THINKING_STEPS,
                    WARMUP_STEPS,
                    OUTPUT_STEPS,
                    epoch,
                    folder=SAMPLE_DIR,
                )
                print(f"[Samples] Saved interrupt snapshot: {save_path}")
            except Exception as e:
                print(f"[Samples] Interrupt snapshot failed: {e}")

    # Build final stitched summary from all saved epochs
    build_epoch_summary(SAMPLE_DIR)

def build_epoch_summary(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not os.path.exists(folder):
        return
        
    files = glob.glob(os.path.join(folder, "epoch_*.png"))

    # Keep chronological order by epoch number, not filesystem timestamp.
    def _epoch_from_name(path):
        base = os.path.basename(path)
        m = re.search(r"epoch_(\d+)\.png$", base)
        return int(m.group(1)) if m else float('inf')

    files = sorted(files, key=_epoch_from_name)
    if not files:
        print(f"No epoch samples found in {folder}; summary not generated.")
        return
        
    print(f"\nTraining Complete. Samples saved to {folder}/")
    print(f"Generated {len(files)} sample images during training.")

    # Stitch all epoch images into a single summary grid.
    n = len(files)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 2.4 * rows))

    # Normalize axes shape for both 1-row and multi-row cases.
    if rows == 1:
        axes = [axes]

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        if idx < n:
            file = files[idx]
            img = plt.imread(file)
            name = os.path.basename(file)
            epoch_label = name.replace("epoch_", "Epoch ").replace(".png", "")
            ax.imshow(img)
            ax.set_title(epoch_label, fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{folder}/_summary.png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary image saved to {folder}/_summary.png")
    print(f"Samples retained in {folder}")

if __name__ == "__main__":
    main()
