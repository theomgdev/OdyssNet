
import torch
import torch.nn as nn
import sys
import os

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig

def generate_latch_data(batch_size, seq_len, device):
    """
    Generates data for 'Catch & Hold'.
    Input: (Batch, Seq, 1). -1.0 usually, 1.0 at trigger.
    Target: (Batch, Seq, 1). -1.0 before trigger, 1.0 after trigger.
    """
    inputs = torch.ones(batch_size, seq_len, 1, device=device) * -1.0
    targets = torch.ones(batch_size, seq_len, 1, device=device) * -1.0
    
    for i in range(batch_size):
        # Trigger happens somewhere between step 2 and seq_len-5
        trigger = torch.randint(2, seq_len - 5, (1,)).item()
        
        # Pulse input at trigger
        inputs[i, trigger, 0] = 1.0
        
        # Target becomes 1 AFTER trigger (inclusive)
        targets[i, trigger:, 0] = 1.0
        
    return inputs, targets

def main():
    print("RealNet Experiment: Catch & Hold (The Latch)")
    print("Objective: Wait for a pulse using chaos. Once received, hold the state output at 1.0 forever.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 32
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    SEQ_LEN = 20
    BATCH_SIZE = 64
    EPOCHS = 2000
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        device=DEVICE,
        dropout_rate=0.0 # overfitting to a mental mechanic is desired
    )
    
    trainer = RealNetTrainer(model, device=DEVICE, synaptic_noise=0.0,
                             chaos_config=ChaosGradConfig.aggressive(lr=0.01))
    
    print("Training...")
    
    # TRAINING LOOP
    for epoch in range(EPOCHS):
        inputs, targets = generate_latch_data(BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # inputs: (Batch, Seq, 1) -> RealNet converts to (Batch, Seq, N) via prepare_input
        # targets: (Batch, Seq, 1)
        loss = trainer.train_batch(inputs, targets, thinking_steps=SEQ_LEN, full_sequence=True)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss:.6f}")
            
    # TEST
    print("\nTraining Complete. Testing Latch Mechanism...")
    
    # Test case: Trigger at t=5
    test_trigger = 5
    test_input = torch.ones(1, SEQ_LEN, 1, device=DEVICE) * -1.0
    test_input[0, test_trigger, 0] = 1.0
    
    with torch.no_grad():
        # Predict Full Sequence
        # Returns (Batch, Steps, OutputDim)
        preds = trainer.predict(test_input, thinking_steps=SEQ_LEN, full_sequence=True)
        # We want (Steps) for printing
        # preds[0, :, 0] shape is (Steps)
        
    print(f"Trigger sent at t={test_trigger}")
    for t in range(SEQ_LEN):
        val = preds[0, t, 0].item()
        status = "OFF" if val < 0.0 else "ON "
        visual = "🔴" if val < 0.0 else "🟢"
        if t == test_trigger: visual = "⚡ TRIGGER!"
        
        print(f"t={t:02d} | Out: {val:.4f} | {status} {visual}")

if __name__ == "__main__":
    main()
