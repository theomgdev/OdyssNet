import torch
import torch.nn as nn
from odyssnet import OdyssNet, OdyssNetTrainer, TrainingHistory, set_seed

def generate_adder_data(batch_size, seq_len, delay_1, delay_2, device):
    """
    Generates sequential data for adding two numbers.
    Input:
        t=delay_1:  Value A
        t=delay_2:  Value B
        others:     0
    Target:
        Scalar: A + B
    """
    
    # 1. Generate random numbers A and B
    # Range -0.5 to 0.5 so sum is between -1 and 1
    val_a = torch.rand(batch_size, 1, device=device) - 0.5
    val_b = torch.rand(batch_size, 1, device=device) - 0.5
    target_sum = val_a + val_b # (Batch, 1)
    
    # 2. Construct Input Sequence (Batch, Steps, 1)
    # The new Trainer/prepare_input supports 3D input directly.
    inputs = torch.zeros(batch_size, seq_len, 1, device=device)
    
    # Place pulses
    inputs[:, delay_1, 0] = val_a[:, 0]
    inputs[:, delay_2, 0] = val_b[:, 0]
    
    return inputs, target_sum

def main():
    print("OdyssNet Experiment: The Delayed Adder (Algorithmic Logic)")
    print("Objective: Remember Number A. Wait. Receive Number B. Output A+B.")
    set_seed(42)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_NEURONS = 128
    INPUT_ID = 0
    OUTPUT_ID = 1
    
    SEQ_LEN = 15
    DELAY_1 = 2
    DELAY_2 = 8
    
    BATCH_SIZE = 1024
    EPOCHS = 2000
    
    # Initialize Model
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=[INPUT_ID],
        output_ids=[OUTPUT_ID],
        device=DEVICE
    )
    
    trainer = OdyssNetTrainer(model, device=DEVICE, lr=1e-4)
    
    print(f"Structure: Pulse A at t={DELAY_1}. Pulse B at t={DELAY_2}. Target at t={SEQ_LEN-1}.")
    
    # TRAINING
    history = TrainingHistory()
    for epoch in range(EPOCHS):
        # Generate Data
        inputs_seq, targets = generate_adder_data(BATCH_SIZE, SEQ_LEN, DELAY_1, DELAY_2, DEVICE)

        # Train using Trainer
        # Trainer will use 'final_state' by default because full_sequence=False
        loss = trainer.train_batch(inputs_seq, targets, thinking_steps=SEQ_LEN)
        history.record(loss=loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss {loss:.6f}")
            
    # TEST
    print("\nTesting...")
    
    test_a = [-0.3, 0.5, 0.1, -0.4]
    test_b = [0.1, 0.2, -0.1, -0.4]
    
    # Custom Test Batch
    batch_size = len(test_a)
    x_test = torch.zeros(batch_size, SEQ_LEN, 1, device=DEVICE)
    for i in range(batch_size):
        x_test[i, DELAY_1, 0] = test_a[i]
        x_test[i, DELAY_2, 0] = test_b[i]
        
    with torch.no_grad():
        # Predict using Trainer (returns final state output by default)
        preds = trainer.predict(x_test, thinking_steps=SEQ_LEN)
        
    for i in range(batch_size):
        tgt = test_a[i] + test_b[i]
        p = preds[i, 0].item()
        print(f" {test_a[i]} + {test_b[i]} = {tgt:.2f} | OdyssNet: {p:.4f} (Diff: {abs(tgt-p):.4f})")

    history.plot(title="Delayed Adder Convergence")

if __name__ == "__main__":
    main()
