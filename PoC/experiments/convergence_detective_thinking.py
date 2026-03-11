
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from realnet import RealNet, RealNetTrainer, ChaosGradConfig

def generate_dilated_data(batch_size, logic_len, gap, device):
    """
    Generates 'dilated' binary sequence.
    logic_len: How many actual bits of info.
    gap: How many empty steps between bits.
    Total steps = logic_len * (gap + 1)
    
    Values:
    - Input: -1.0 (Bit 0), 1.0 (Bit 1), 0.0 (Silence/Gap)
    - Target: -1.0 (No Alarm), 1.0 (Alarm)
    """
    # 1. Generate Raw Sequence (Logic Layer)
    # 0 or 1
    raw_bits = torch.randint(0, 2, (batch_size, logic_len), device=device)
    
    # Calculate total physical steps
    total_steps = logic_len * (gap + 1)
    
    # 2. Create Physical Tensors (Time Layer)
    # Inputs start at 0.0 (Silence)
    inputs = torch.zeros(batch_size, total_steps, 1, device=device)
    # Targets start at -1.0 (No Alarm)
    targets = torch.ones(batch_size, total_steps, 1, device=device) * -1.0
    
    # 3. Fill and Calculate Logic
    for i in range(batch_size):
        prev_bit = -1 # Start with neutral/invalid
        
        for step in range(logic_len):
            bit_val = raw_bits[i, step].item() # 0 or 1
            
            # Place bit at the start of the block
            t_real = step * (gap + 1)
            
            # Map 0 -> -1.0, 1 -> 1.0
            phys_val = 1.0 if bit_val == 1 else -1.0
            inputs[i, t_real, 0] = phys_val
            
            # Logic: If Current=1 AND Prev=1 -> Alarm MATCH
            # Alarm should sound from arrival until end of gap
            if bit_val == 1 and prev_bit == 1:
                 targets[i, t_real : t_real + gap + 1, 0] = 1.0
            
            prev_bit = bit_val
            
    return inputs, targets

def main():
    print("RealNet Experiment: The Thinking Detective 🕵️‍♂️")
    print("Objective: Watch a stream of bits. BUT... you have time to think between bits.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    GAP = 3 # 3 steps of silence between every bit
    LOGIC_LEN = 10 # 10 bits of information
    SEQ_LEN = LOGIC_LEN * (GAP + 1)
    
    BATCH_SIZE = 128
    EPOCHS = 2000
    
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    model = RealNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        dropout_rate=0.0, # Focused thinking
        device=DEVICE
    )
    
    trainer = RealNetTrainer(model, device=DEVICE, synaptic_noise=0.0,
                             chaos_config=ChaosGradConfig.default(lr=1e-3))
    
    print(f"Logic Steps: {LOGIC_LEN} | Thinking Gap: {GAP} | Total Physical Steps: {SEQ_LEN}")

    # TRAINING
    for epoch in range(EPOCHS):
        inputs, targets = generate_dilated_data(BATCH_SIZE, LOGIC_LEN, GAP, DEVICE)
        
        # inputs: (Batch, Seq, 1)
        # targets: (Batch, Seq, 1)
        
        # Use full_sequence=True to train on every step (Silence/Hold phases)
        loss = trainer.train_batch(inputs, targets, thinking_steps=SEQ_LEN, full_sequence=True)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss:.6f}")

    # TEST
    print("\nTesting Thinking Detective...")
    
    # Custom Test: 0 1 0 1 1 0
    # Expected Match at index 4 (the second 1)
    test_bits = [0, 1, 0, 1, 1, 0]
    total_len = len(test_bits) * (GAP + 1)
    
    x_test = torch.zeros(1, total_len, 1, device=DEVICE)
    
    # Construct input manually
    print(f"\nTimeline Analysis (GAP={GAP}):")
    print(f"{'Time':<5} | {'Input':<5} | {'Output':<8} | {'Status'}")
    print("-" * 40)
    
    for i, bit in enumerate(test_bits):
        t_real = i * (GAP + 1)
        phys_val = 1.0 if bit == 1 else -1.0
        x_test[0, t_real, 0] = phys_val
        
    with torch.no_grad():
         # Predict full sequence (Batch, Steps, Out)
         preds = trainer.predict(x_test, thinking_steps=total_len, full_sequence=True)
         
    # Visualization
    prev_bit = -1
    for i, bit in enumerate(test_bits):
        t_start = i * (GAP + 1)
        
        # Check logic
        is_match = (bit == 1 and prev_bit == 1)
        expected = "SHOULD FIRE" if is_match else ""
        
        # Print the burst for this bit
        for t in range(t_start, t_start + GAP + 1):
            val = preds[0, t, 0].item()
            
            # Display Input
            if t == t_start:
                inp_display = str(bit)
            else:
                inp_display = "."
            
            # Display Output
            # -1 is OFF, 1 is ON
            alert = "🚨" if val > 0.0 else ""
            
            if t == t_start:
                 print(f"{t:<5} | {inp_display:<5} | {val:.4f} {alert} | {expected}")
            else:
                 print(f"{t:<5} | {inp_display:<5} | {val:.4f} {alert} | (Thinking...)")
                 
        prev_bit = bit

if __name__ == "__main__":
    main()
