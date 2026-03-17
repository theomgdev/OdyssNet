import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig

def main():
    print("RealNet 2.0: TINY EXPERIMENT (7x7 Input)...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # EXPERIMENTAL CONFIG: "Tiny RealNet"
    # 28x28 resized to 7x7 = 49 Pixels (Input)
    # 10 Classes (Output)
    # 0 Hidden Neurons.
    # Total: 59 Neurons.
    # Params: 59*59 = 3,481.
    
    # Goal: Observe behavior under extreme parameter constraints.
    
    INPUT_SIZE = 49
    OUTPUT_SIZE = 10
    NUM_NEURONS = INPUT_SIZE + OUTPUT_SIZE
    
    print(f"Neurons: {NUM_NEURONS} (49 In + 10 Out + 0 Hidden)")
    print(f"Params: {NUM_NEURONS*NUM_NEURONS} (~3.5k)")
    
    input_ids = list(range(49))
    output_ids = list(range(49, 59))
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        dropout_rate=0.05, 
        device=DEVICE
    )
    
    transform = transforms.Compose([
        transforms.Resize((7, 7)), # Downscaling for experimental constraints
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    SUBSET_SIZE = 5000 
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    
    test_subset = Subset(test_dataset, range(1000))
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    trainer = RealNetTrainer(model, device=DEVICE,
                             chaos_config=ChaosGradConfig.default(lr=1e-3))
    loss_fn = nn.MSELoss()
    trainer.loss_fn = loss_fn
    
    NUM_EPOCHS = 50 
    THINKING_STEPS = 15
    
    print("Training Tiny RealNet...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            inputs_val = data.view(data.size(0), -1).to(DEVICE)
            targets_val = torch.ones(data.size(0), 10, device=DEVICE) * -1.0
            for i, label in enumerate(target):
                targets_val[i, label] = 1.0
                
            loss = trainer.train_batch(inputs_val, targets_val, thinking_steps=THINKING_STEPS)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), -1).to(DEVICE)
                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes.cpu() == target).sum().item()
                total += target.size(0)
        
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Test Acc {acc:.2f}%")

if __name__ == "__main__":
    main()
