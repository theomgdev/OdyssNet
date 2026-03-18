
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig

def generate_stopwatch_data(batch_size, seq_len, device):
    """
    Generates data for 'Stopwatch'.
    Input: (Batch, Seq, 1). Value at t=0 indicates duration.
    Target: (Batch, Seq). Pulse at t=duration.
    """
    # (Batch, Seq, Features=1)
    inputs = torch.zeros(batch_size, seq_len, 1, device=device)
    # (Batch, Seq) but we need (Batch, Seq, OutputDim) for MSE
    targets = torch.zeros(batch_size, seq_len, 1, device=device)
    
    for i in range(batch_size):
        # Determine duration
        duration = int(torch.randint(5, seq_len - 5, (1,)).item())
        
        # Input Value: Normalized duration
        input_val = duration / 20.0
        inputs[i, 0, 0] = input_val
        
        # Target: Pulse at t = duration
        targets[i, duration, 0] = 1.0
        # Soft targets
        if duration > 0: targets[i, duration-1, 0] = 0.5
        if duration < seq_len-1: targets[i, duration+1, 0] = 0.5
        
    return inputs, targets

def main():
    print("RealNet Experiment: The Stopwatch (Internal Clock)")
    print("Objective: Input tells 'Wait X steps'. Network must wait in silence and fire at exactly t=X.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    SEQ_LEN = 30
    BATCH_SIZE = 64
    EPOCHS = 3000
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        device=DEVICE,
        dropout_rate=0.0 # overfitting to a mental mechanic is desired
    )
    
    trainer = RealNetTrainer(model, device=DEVICE, synaptic_noise=0.0,
                             chaos_config=ChaosGradConfig.default(lr=1e-3))
    
    print("Training...")
    
    # TRAINING LOOP
    for epoch in range(EPOCHS):
        # Generate new random batch every epoch
        inputs, targets = generate_stopwatch_data(BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # Train on sequence
        loss = trainer.train_batch(inputs, targets, thinking_steps=SEQ_LEN, full_sequence=True)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss:.6f}")
            
    # TEST
    print("\nTesting Stopwatch...")
    
    durations = [10, 20]
    test_batch = len(durations)
    
    x_test = torch.zeros(test_batch, SEQ_LEN, 1, device=DEVICE)
    for i, d in enumerate(durations):
        x_test[i, 0, 0] = d / 20.0
        
    with torch.no_grad():
        # Predict Full Sequence
        # Returns (Batch, Steps, OutputDim)
        preds = trainer.predict(x_test, thinking_steps=SEQ_LEN, full_sequence=True)
        # We need (Steps, Batch) for the printing logic below, or just adjust logic.
        # Let's adjust logic to use preds[i, t, 0]
    
    for i, d in enumerate(durations):
        print(f"\nTarget Timer: {d} steps (Input val: {d/20.0:.2f})")
        peak_t = -1
        max_val = -100
        
        for t in range(SEQ_LEN):
            val = preds[i, t, 0].item()
            bar = "█" * int(val * 10)
            if val > max_val:
                max_val = val
                peak_t = t
                
            mark = ""
            if t == d: mark = "🎯 TARGET"
            
            # Reduces noise in print
            if val > 0.1 or t == d: 
                print(f"t={t:02d} | Out: {val:.4f} {bar} {mark}")
                
        print(f"Result: Peak at t={peak_t} (Error: {peak_t - d})")

if __name__ == "__main__":
    main()
