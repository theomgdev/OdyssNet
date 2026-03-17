import torch
import torch.nn as nn
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print("RealNet 2.0: The Impossible XOR (Zero-Hidden)...")
    set_seed(42) # Reproducibility
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2 Inputs, 1 Output, 0 Hidden = 3 Neurons (True Zero-Hidden)
    NUM_NEURONS = 3
    INPUT_IDS = [0, 1]
    OUTPUT_ID = [2]
    
    print(f"Neurons: {NUM_NEURONS} (9 Parameters)")
    
    # CRITICAL CONFIG FOR TINY NETWORKS:
    # 1. dropout_rate=0.0 (Every neuron is vital)
    # 2. activation='gelu' (Flow allows better gradient flow in small circuits)
    # 3. weight_init='xavier_uniform' (Higher variance needed for signal propagation in small nets)
    model = RealNet(
        num_neurons=NUM_NEURONS, 
        input_ids=INPUT_IDS, 
        output_ids=OUTPUT_ID, 
        pulse_mode=True, 
        device=DEVICE,
        dropout_rate=0.0,
        activation='gelu',
        weight_init='xavier_uniform'
    )

    trainer = RealNetTrainer(model, device=DEVICE, synaptic_noise=0.0,
                             chaos_config=ChaosGradConfig.tiny_network(lr=0.01))
    
    # XOR Data
    data = [
        (-1.0, -1.0, -1.0),
        (-1.0,  1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0, -1.0),
    ]
    
    # Batching (Create a dataset)
    inputs_list = []
    targets_list = []
    for _ in range(50): # 200 samples total
        for row in data:
            inputs_list.append([row[0], row[1]])
            targets_list.append([row[2]])
            
    inputs_val = torch.tensor(inputs_list, device=DEVICE)
    targets_val = torch.tensor(targets_list, device=DEVICE)

    print("Training...")
    # 5 Thinking steps to allow chaotic resonance to find the XOR pattern
    history = trainer.fit(inputs_val, targets_val, epochs=100, batch_size=16, thinking_steps=5)

    print(f"Final Loss: {history[-1]:.6f}")

    print("\nVerifying Truth Table:")
    print(f"{'A':>6} {'B':>6} | {'XOR (Pred)':>12} | {'Logic'}")
    print("-" * 40)
    
    test_data = torch.tensor([[-1.0,-1.0], [-1.0,1.0], [1.0,-1.0], [1.0,1.0]], device=DEVICE)
    preds = trainer.predict(test_data, thinking_steps=5)
    
    success = True
    for i in range(4):
        a = test_data[i][0].item()
        b = test_data[i][1].item()
        out = preds[i].item()
        
        # Logic check (-1 is 0, 1 is 1)
        expected = -1.0 if (a * b) > 0 else 1.0 
        logic_pred = "0" if out < 0 else "1"
        logic_target = "0" if expected < 0 else "1"
        
        status = "OK" if logic_pred == logic_target else "FAIL"
        if status == "FAIL": success = False
        
        print(f"{a:>6.1f} {b:>6.1f} | {out:>12.4f} | {logic_pred} (Target: {logic_target}) {status}")

    if success:
        print("\nResult: CONVERGED. XOR Solved with Zero Hidden Layers.")
    else:
        print("\nResult: FAILED. (Try different seed or more epochs)")

if __name__ == "__main__":
    main()
