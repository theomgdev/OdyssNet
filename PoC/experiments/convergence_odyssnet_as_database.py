
import torch
import torch.nn as nn
import sys
import os
import random

# Adjust path to import odyssnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGradConfig

def generate_db_data(batch_size, num_keys, seq_per_op, num_ops, device, think_steps=5):
    """
    Generates a complex CRUD sequence for OdyssNet Database.
    Returns: inputs, targets, mask
    Mask is 1.0 only for READ operations after think_steps.
    """
    total_steps = num_ops * seq_per_op
    input_dim = 3 + num_keys + 1
    
    inputs = torch.zeros(batch_size, total_steps, input_dim, device=device)
    targets = torch.zeros(batch_size, total_steps, 1, device=device)
    mask = torch.zeros(batch_size, total_steps, 1, device=device)
    
    memories = [{} for _ in range(batch_size)]
    
    for b in range(batch_size):
        for op in range(num_ops):
            t_start = op * seq_per_op
            t_end = t_start + seq_per_op
            
            r = random.random()
            if r < 0.1: cmd = "WRITE"
            elif r < 0.9: cmd = "READ"
            else: cmd = "DELETE"
                
            k_idx = random.randint(0, num_keys - 1)
            
            if cmd == "WRITE":
                val = random.uniform(-1.0, 1.0)
                memories[b][k_idx] = val
                
                inputs[b, t_start, 0] = 1.0 
                inputs[b, t_start, 3 + k_idx] = 1.0 
                inputs[b, t_start, 3 + num_keys] = val 
                targets[b, t_start : t_end, 0] = val
                
            elif cmd == "READ":
                inputs[b, t_start, 1] = 1.0 
                inputs[b, t_start, 3 + k_idx] = 1.0 
                
                val = memories[b].get(k_idx, 0.0)
                targets[b, t_start : t_end, 0] = val
                mask[b, t_start + think_steps : t_end, 0] = 1.0
                
            elif cmd == "DELETE":
                inputs[b, t_start, 2] = 1.0 
                inputs[b, t_start, 3 + k_idx] = 1.0 
                
                if k_idx in memories[b]:
                    del memories[b][k_idx]
                
                targets[b, t_start : t_end, 0] = 0.0
                    
    return inputs, targets, mask

def main():
    print("🚀 OdyssNet Experiment: NEURAL DATABASE (Implicit CRUD)")
    print("Objective: Prove that OdyssNet hidden state can act as an Adressable, Updatable Memory Table.")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {DEVICE}")

    # CONFIGURATION
    NUM_KEYS = 4 
    SEQ_PER_OP = 8 
    NUM_OPS = 12 
    NUM_NEURONS = 256

    RAW_INPUT_DIM = 3 + NUM_KEYS + 1
    DECODED_OUTPUT_DIM = 1

    PROJ_INPUT_NEURONS = 128
    DECODER_OUTPUT_NEURONS = 128

    input_ids = list(range(PROJ_INPUT_NEURONS))
    output_start = PROJ_INPUT_NEURONS
    output_ids = list(range(output_start, output_start + DECODER_OUTPUT_NEURONS))
    
    model = OdyssNet(
        num_neurons=NUM_NEURONS,
        input_ids=input_ids,
        output_ids=output_ids,
        device=DEVICE,
        vocab_size=[RAW_INPUT_DIM, DECODED_OUTPUT_DIM],
        vocab_mode='continuous'
    )
    
    trainer = OdyssNetTrainer(model, device=DEVICE,
                             chaos_config=ChaosGradConfig.default(lr=1e-3))

    print(
        f"Config: {NUM_KEYS} Keys | {NUM_OPS} Ops per Batch | Core: {NUM_NEURONS} Neurons "
        f"| Input: {RAW_INPUT_DIM} -> Proj({PROJ_INPUT_NEURONS}) "
        f"| Output: Decode({DECODER_OUTPUT_NEURONS}) -> {DECODED_OUTPUT_DIM}"
    )
    print("Training...")
    
    EPOCHS = 1000
    BATCH_SIZE = 64
    total_steps = NUM_OPS * SEQ_PER_OP
    THINK_STEPS = 5
    
    for epoch in range(EPOCHS):
        inputs, targets, mask = generate_db_data(BATCH_SIZE, NUM_KEYS, SEQ_PER_OP, NUM_OPS, DEVICE, think_steps=THINK_STEPS)
        
        # TRAIN using Library with Mask support
        loss = trainer.train_batch(inputs, targets, thinking_steps=total_steps, full_sequence=True, mask=mask)
        
        if epoch % 100 == 0:
            with torch.no_grad():
                # Get predictions for validation
                preds = trainer.predict(inputs, thinking_steps=total_steps, full_sequence=True)
                
                # Check shapes and types
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds, device=DEVICE)
                
                # Accuracy only for masked (meaningful) steps
                # targets is already a tensor on DEVICE from generate_db_data
                diff = torch.abs(preds - targets)
                
                accurate = (diff < 0.1).float() * mask
                acc = (accurate.sum() / mask.sum()) * 100
                print(f"Epoch {epoch}: Loss {loss:.6f} | Acc {acc.item():.2f}%")
                
            if loss < 0.0005: 
                print("Convergence Reached!")
                break
                
    # TEST RUN (Simulation)
    print("\n🔍 Simulation Run (The Neural Database in Action):")
    test_ops = 8
    inputs, targets, mask = generate_db_data(1, NUM_KEYS, SEQ_PER_OP, test_ops, DEVICE, think_steps=THINK_STEPS)
    
    with torch.no_grad():
        preds = trainer.predict(inputs, thinking_steps=(test_ops * SEQ_PER_OP), full_sequence=True)
        
    print(f"{'Step':<5} | {'Command':<8} | {'Key':<5} | {'Val_In':<8} | {'Target':<8} | {'OdyssNet':<8} | {'Status'}")
    print("-" * 80)
    
    for op in range(test_ops):
        t_start = op * SEQ_PER_OP
        row = inputs[0, t_start]
        cmd = ""
        if row[0] > 0.5: cmd = "WRITE"
        elif row[1] > 0.5: cmd = "READ"
        elif row[2] > 0.5: cmd = "DELETE"
        
        key = -1
        for k in range(NUM_KEYS):
            if row[3+k] > 0.5: key = k
            
        val_in = row[3 + NUM_KEYS].item()
        
        for t in range(t_start, t_start + SEQ_PER_OP):
            target = targets[0, t, 0].item()
            pred = preds[0, t, 0].item()
            m_val = mask[0, t, 0].item()
            
            diff = abs(target - pred)
            if m_val > 0.5:
                status = "✅" if diff < 0.1 else "❌"
            else:
                status = ".." # Masked (thinking or non-READ op)

            if t > t_start: 
                c_display = f"({t-t_start})"
                k_display = "..."
                v_display = ""
            else:
                c_display = cmd
                k_display = f"K{key}"
                v_display = f"{val_in:.4f}"
                
            print(f"{t:<5} | {c_display:<8} | {k_display:<5} | {v_display:<8} | {target:<8.4f} | {pred:<8.4f} | {status}")

if __name__ == "__main__":
    main()
