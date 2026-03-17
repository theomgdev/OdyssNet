import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
import time

# Ensure library path is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig

def main():
    print("RealNet 2.0: DARWINIAN REGENERATION EXPERIMENT (The Phoenix Effect)...")
    print("Hypothesis: Reviving weak synapses with random initialization improves learning capacity.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if DEVICE == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    # CONFIG: Pure MNIST
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    NUM_NEURONS = INPUT_SIZE + OUTPUT_SIZE
    
    input_ids = list(range(784))
    output_ids = list(range(784, 794))
    
    # Match EXACTLY with convergence_mnist.py (Default: Tanh, Orthogonal)
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        dropout_rate=0.1,
        # Default activation is 'tanh', weight_init is 'orthogonal'
        device=DEVICE
    )
    
    # Compile model
    model = model.compile()
    
    trainer = RealNetTrainer(model, device=DEVICE,
                             chaos_config=ChaosGradConfig.default(lr=1e-4))
    loss_fn = nn.MSELoss()
    trainer.loss_fn = loss_fn
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    SUBSET_SIZE = 10000
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    test_subset = Subset(test_dataset, range(1000))
    
    # Optimization: Pin Memory & Workers (Same as baseline)
    kwargs = {'num_workers': 4, 'pin_memory': True} if DEVICE == 'cuda' else {}
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, **kwargs)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, **kwargs)
    
    NUM_EPOCHS = 50 
    THINKING_STEPS = 10
    REGENERATE_THRESHOLD = 0.01 
    
    print(f"Total Connections: {model.W.numel()}")
    print(f"Regeneration Threshold: abs(W) < {REGENERATE_THRESHOLD}")
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            inputs_val = data.view(data.size(0), -1).to(DEVICE, non_blocking=True)
            
            # Efficient target creation
            targets_val = torch.ones(data.size(0), 10, device=DEVICE) * -1.0
            targets_val.scatter_(1, target.view(-1, 1).to(DEVICE), 1.0)
                
            loss = trainer.train_batch(inputs_val, targets_val, thinking_steps=THINKING_STEPS)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        
        # --- THE PHOENIX MOMENT (Regeneration) ---
        # Revive weak connections
        revived, total = trainer.regenerate_synapses(threshold=REGENERATE_THRESHOLD)
        revive_pct = (revived / total) * 100.0
        
        # Validation
        model.eval()
        correct = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), -1).to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)
                
                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes == target).sum().item()
                total_samples += target.size(0)
        
        acc = 100.0 * correct / total_samples
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Acc {acc:.2f}% | Revived: {revived}/{total} ({revive_pct:.2f}%)")

if __name__ == "__main__":
    main()
