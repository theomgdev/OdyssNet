import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import time
import warnings

# Suppress the specific PyTorch warning about scheduler step order
# This is a known artifact when using GradScaler (AMP) in the first epoch
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

# Disable BNB for this experiment to rule out quantization noise and use pure dynamics
os.environ["NO_BNB"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGradConfig, set_seed

def main():
    print("OdyssNet 2.0: MNIST RECORD CHALLENGE (Elite 480-Param Model)")
    print("Strategy: 10 Sequential Chunks (79 pixels) -> Embed(3 Neurons) -> Core(10) -> Decoder(10 Classes)")
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
        
    # Strategy: 10 Chunks (79 pixels) -> Embed(3 Neurons) -> Core(10 Neurons) -> Output Decoder (10 Neurons -> 10 Classes)
    NUM_NEURONS = 10
    input_ids = [0, 1, 2] # 79 pixels will be projected to 3 neurons
    output_ids = list(range(10)) # Decoder reads from all 10 neurons
    
    # vocab_size = [v_in, v_out]
    # v_in = 79 pixels (from each chunk)
    # v_out = 10 classes
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[79, 10],   # [79 pixels -> 3 neurons, 10 neurons -> decoder]
        vocab_mode='continuous',
        weight_init='micro_quiet_8bit'
    )
    
    # Speed up core with torch.compile if on PyTorch 2.0+
    if 'model_compile' in locals() and model_compile:
        model = torch.compile(model)
    
    total_params = model.get_num_params()
    print(f"Total Params: {total_params} (Goal: < 1000)")
    
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
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    # Hyperparameters for the Elite 480 model
    BATCH_SIZE = 16 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)
    
    NUM_EPOCHS = 100
    steps_per_epoch = len(train_loader)
    
    scheduler_config = dict(
        warmup_steps=10 * steps_per_epoch,
        max_steps=NUM_EPOCHS * steps_per_epoch,
        min_lr_ratio=1e-6 / 1e-2
    )
    
    trainer = OdyssNetTrainer(
        model, 
        device=DEVICE,
        chaos_config=ChaosGradConfig.tiny_network(lr=1e-2),
        scheduler_config=scheduler_config,
        use_temporal_scheduler=True
    )

    loss_fn = nn.CrossEntropyLoss()
    trainer.loss_fn = loss_fn
    
    print(f"Training with Batch Size: {BATCH_SIZE} for {NUM_EPOCHS} Epochs...")
    start_time = time.time()
    
    # 10 chunks of 79 pixels -> 10 steps
    TOTAL_STEPS = 10 
    
    # Processing Loop

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_size = data.size(0)
            data = data.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            
            # Flatten and pad to 790 (10 chunks * 79)
            pixels = data.view(batch_size, 784)
            padded_pixels = torch.cat([pixels, torch.zeros(batch_size, 6, device=DEVICE)], dim=1)
            seq_input = padded_pixels.view(batch_size, 10, 79) # (B, 10, 79)
            
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
                
                # Flatten and pad to 790 (10 chunks * 79)
                pixels = data.view(batch_size, 784)
                padded_pixels = torch.cat([pixels, torch.zeros(batch_size, 6, device=DEVICE)], dim=1)
                seq_input = padded_pixels.view(batch_size, 10, 79)
                
                preds = trainer.predict(seq_input, thinking_steps=TOTAL_STEPS)
                correct += (preds.argmax(1) == target).sum().item()
                total += target.size(0)
        
        acc = 100.0 * correct / total
        current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else trainer.optimizer.param_groups[0]['lr']
        
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
