
import torch
import torch.nn as nn
import sys
import os
import math

# Adjust path to import odyssnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGradConfig

def generate_sine_data(batch_size, steps, device):
    """
    Generates batch of data.
    Input: Frequency multiplier (random between 0.1 and 0.5) (Batch, 1)
    Target: Sine wave sequence (Batch, Steps, 1)
    """
    # Random frequencies: shape (Batch, 1)
    frequencies = torch.rand(batch_size, 1, device=device) * 0.4 + 0.1
    
    # Time steps: shape (1, Steps)
    t = torch.arange(1, steps + 1, device=device).float().unsqueeze(0)
    
    # Target Sequence: sin(w * t)
    # We need shape (Batch, Steps) then unsqueeze to (Batch, Steps, 1)
    
    # frequencies: (Batch, 1)
    # t: (1, Steps)
    # matmul? No, broadcast. (Batch, 1) * (1, Steps) -> (Batch, Steps)
    
    targets = torch.sin(frequencies * t).unsqueeze(2) # (Batch, Steps, 1)
    
    return frequencies, targets

def main():
    print("OdyssNet Experiment: The Harmonic Oscillator (Sine Wave Generator)")
    print("Objective: One continuous input sets the frequency. The network must oscillate at that frequency for N steps.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    INPUT_IDS = [0] # Input 0: Frequency Control
    OUTPUT_IDS = [1] # Output 1: Sine Wave Output
    STEPS = 30 # Length of the wave
    BATCH_SIZE = 128
    EPOCHS = 10000
    
    # Initialize Model
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=INPUT_IDS,
        output_ids=OUTPUT_IDS,
        pulse_mode=False, # Continuous Control (VCO Mode)
        device=DEVICE
    )
    
    # Initialize Trainer
    trainer = OdyssNetTrainer(model, device=DEVICE,
                             chaos_config=ChaosGradConfig.default(lr=5e-5))
    
    print(f"Model: {NUM_NEURONS} Neurons. Thinking for {STEPS} steps.")
    
    # TRAINING LOOP
    for epoch in range(EPOCHS):
        inputs, targets = generate_sine_data(BATCH_SIZE, STEPS, DEVICE)
        
        # Train on specific batch
        # Inputs: (Batch, 1) -> OdyssNet converts to (Batch, N)
        # Targets: (Batch, Steps, 1) -> Trainer compares with Permuted Sequence output
        loss = trainer.train_batch(inputs, targets, thinking_steps=STEPS, full_sequence=True)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss:.6f}")
            
    # TEST
    print("\nTraining Complete. Testing Generator...")
    
    # Test Frequencies
    test_freqs = torch.tensor([[0.15], [0.45]], device=DEVICE) # Low and High freq
    
    # Predict Full Sequence
    with torch.no_grad():
        # Input (Batch, 1) -> Predict (Batch, Steps, 1)
        predictions = trainer.predict(test_freqs, thinking_steps=STEPS, full_sequence=True)
    
    # Print comparison
    print("\nFrequency 0.15 (Slow Wave):")
    for t in range(0, STEPS, 5):
        target = math.sin((t+1) * 0.15)
        pred = predictions[0, t, 0].item()
        print(f"  t={t+1}: Target {target:.4f} | OdyssNet {pred:.4f}")

    print("\nFrequency 0.45 (Fast Wave):")
    for t in range(0, STEPS, 5):
        target = math.sin((t+1) * 0.45)
        pred = predictions[1, t, 0].item()
        print(f"  t={t+1}: Target {target:.4f} | OdyssNet {pred:.4f}")

if __name__ == "__main__":
    main()
