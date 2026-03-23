import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig, TemporalSchedulerConfig

def main():
    print("RealNet 2.0: SCALED EXPERIMENT (14x14 Input)...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # SCALED ZERO-HIDDEN CONFIG
    # 28x28 resized to 14x14 = 196 Pixels (Input)
    # 10 Classes (Output)
    # 0 Hidden Neurons.
    # Total: 206 Neurons.
    # Params: 206*206 = ~42k.
    
    # Goal: Medium efficiency test.
    
    INPUT_SIZE = 196
    OUTPUT_SIZE = 10
    NUM_NEURONS = INPUT_SIZE + OUTPUT_SIZE
    
    print(f"Neurons: {NUM_NEURONS} (196 In + 10 Out + 0 Hidden)")
    print(f"Params: {NUM_NEURONS*NUM_NEURONS} (~42k)")
    
    input_ids = list(range(196))
    output_ids = list(range(196, 206))
    
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=input_ids, 
        output_ids=output_ids, 
        pulse_mode=True, 
        dropout_rate=0.0, 
        device=DEVICE
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
    
    SUBSET_SIZE = 5000 
    train_subset = Subset(train_dataset, range(SUBSET_SIZE))
    test_subset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    trainer = RealNetTrainer(model, device=DEVICE,
                             chaos_config=ChaosGradConfig.default(lr=2e-4))
    loss_fn = nn.MSELoss()
    trainer.loss_fn = loss_fn
    
    NUM_EPOCHS = 100
    THINKING_STEPS = 15
    
    print("Training Scaled RealNet...")
    
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
