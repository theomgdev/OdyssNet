import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from odyssnet import OdyssNet, OdyssNetTrainer, TrainingHistory, set_seed

def main():
    print("OdyssNet: PURE MNIST CHALLENGE (28x28 Raw Input)...")
    set_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Performance Tuning
    if DEVICE == 'cuda':
        # Enable TF32 for significantly faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        print(f"CUDA Enabled. Device: {torch.cuda.get_device_name(0)}")

    # PURE ZERO-HIDDEN CONFIG
    # 28x28 = 784 Pixels (Input)
    # 10 Classes (Output)
    # 0 Hidden Neurons.
    # Total: 794 Neurons. (Zero Buffer Layers)

    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    NUM_NEURONS = INPUT_SIZE + OUTPUT_SIZE

    print(f"Neurons: {NUM_NEURONS} (784 In + 10 Out + 0 Hidden)")

    input_ids = list(range(784))
    output_ids = list(range(784, 794))

    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        pulse_mode=True,
        device=DEVICE
    )

    print(f"Params: {model.get_num_params()} (~630k)")

    # Compile for speed (PyTorch 2.0+)
    model = model.compile()

    trainer = OdyssNetTrainer(model, device=DEVICE, lr=1e-4)

    # NO RESIZE used. Pure 28x28.
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)

    # Optimization: Pin Memory & Workers
    kwargs = {'num_workers': 4, 'pin_memory': True} if DEVICE == 'cuda' else {}
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, **kwargs)

    NUM_EPOCHS = 100
    THINKING_STEPS = 10

    print("Training...")
    history = TrainingHistory()
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            inputs_val = data.view(data.size(0), -1).to(DEVICE, non_blocking=True)
            targets_val = torch.ones(data.size(0), 10, device=DEVICE) * -0.90
            targets_val.scatter_(1, target.view(-1, 1).to(DEVICE), 0.90)

            loss = trainer.train_batch(inputs_val, targets_val, thinking_steps=THINKING_STEPS)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)

        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                inputs_val = data.view(data.size(0), -1).to(DEVICE, non_blocking=True)
                target = target.to(DEVICE, non_blocking=True)

                preds = trainer.predict(inputs_val, thinking_steps=THINKING_STEPS)
                predicted_classes = torch.argmax(preds, dim=1)
                correct += (predicted_classes == target).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total

        elapsed = time.time() - start_time
        fps = ((epoch + 1) * len(train_dataset)) / elapsed

        history.record(loss=avg_loss, accuracy=acc)
        print(f"Epoch {epoch+1}: Loss {avg_loss:.4f} | Test Acc {acc:.2f}% | FPS: {fps:.1f}")

    history.plot(title="MNIST Zero-Hidden Training")

if __name__ == "__main__":
    main()
