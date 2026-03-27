import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGradConfig

def main():
    print("OdyssNet 2.0: The Atomic Identity...")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ATOMIC UNIT OF CHAOS
    # 1 Input, 1 Output. 
    # Minimum possible configuration for OdyssNet.
    
    NUM_NEURONS = 2
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    # CRITICAL CONFIG FOR TINY NETWORKS:
    # (Every neuron is vital)
    model = OdyssNet(
        num_neurons=NUM_NEURONS, 
        input_ids=[INPUT_ID], 
        output_ids=[OUTPUT_ID], 
        pulse_mode=True, 
        device=DEVICE
    )
    trainer = OdyssNetTrainer(model, device=DEVICE,
                             chaos_config=ChaosGradConfig.tiny_network(lr=0.01))
    
    # Data
    inputs_val = torch.randint(0, 2, (100, 1)).float() * 2 - 1 
    targets_val = inputs_val

    print("Training...")
    trainer.fit(inputs_val, targets_val, epochs=100, batch_size=32, thinking_steps=50)

    print("\nTest Result:")
    test_inputs = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    preds = trainer.predict(test_inputs, thinking_steps=50)
    
    for i in range(len(test_inputs)):
        print(f"In: {test_inputs[i].item()} -> Out: {preds[i].item():.4f}")

if __name__ == "__main__":
    main()
