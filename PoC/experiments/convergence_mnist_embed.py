
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig, TemporalSchedulerConfig

def main():
    print("RealNet 2.0: EMBEDDED MNIST CHALLENGE (8k Params)")
    print("Strategy: 784 Pixels -> Proj(10) -> RNN(10) -> Decode(10)")
    
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Enable TF32 for Speed
    if DEVICE == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        
    # CONFIGURATION
    # We set VOCAB_SIZE = 784 to allow the 'continuous' projection layer to accept 784-dim input vectors.
    INPUT_DIM = 784
    NUM_NEURONS = 10
    
    print(f"Neurons: {NUM_NEURONS}")
    print(f"Projected Input: {INPUT_DIM} -> {NUM_NEURONS}")
    
    input_ids = list(range(NUM_NEURONS))
    output_ids = list(range(NUM_NEURONS))
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[INPUT_DIM, 10],   # [784, 10] -> Input 784, Output 10
        dropout_rate=0.0,
        vocab_mode='continuous'       # Uses nn.Linear for projection
    )
    
    # Calculate Params
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    
    trainer = RealNetTrainer(model, device=DEVICE, synaptic_noise=0.0,
                             chaos_config=ChaosGradConfig.default(lr=1e-3))
    loss_fn = nn.CrossEntropyLoss()
    trainer.loss_fn = loss_fn
    
    # Prepare Data
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if DEVICE == 'cuda' else {}
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, **kwargs)
    
    NUM_EPOCHS = 100
    THINKING_STEPS = 10
    
    print("Training...")
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Input: (Batch, 1, 784)
            # Cast to float for projection layer
            inputs_val = data.view(data.size(0), 1, -1).to(DEVICE, non_blocking=True).float()
            target = target.to(DEVICE, non_blocking=True)
            
            loss = trainer.train_batch(
                inputs_val, 
                target, 
                thinking_steps=THINKING_STEPS
            )
            total_loss += loss
            
        avg_loss = total_loss / len(train_loader)
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), 1, -1).to(DEVICE, non_blocking=True).float()
                target = target.to(DEVICE, non_blocking=True)
                
                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes == target).sum().item()
                total += target.size(0)
                
        acc = 100.0 * correct / total
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Test Acc {acc:.2f}%")

if __name__ == "__main__":
    main()
