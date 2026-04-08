import torch
from odyssnet import OdyssNet, OdyssNetTrainer, TrainingHistory, set_seed

def main():
    print("OdyssNet: The Atomic Identity...")
    set_seed(42)
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
    trainer = OdyssNetTrainer(model, device=DEVICE, lr=1e-4)

    # Data
    inputs_val = torch.randint(0, 2, (100, 1)).float() * 2 - 1
    targets_val = inputs_val

    print("Training...")
    history = TrainingHistory()
    loss_list = trainer.fit(inputs_val, targets_val, epochs=50, batch_size=32, thinking_steps=50)
    for loss in loss_list:
        history.record(loss=loss)

    print("\nTest Result:")
    test_inputs = torch.tensor([[1.0], [-1.0]], device=DEVICE)
    preds = trainer.predict(test_inputs, thinking_steps=50)

    for i in range(len(test_inputs)):
        print(f"In: {test_inputs[i].item()} -> Out: {preds[i].item():.4f}")

    history.plot(title="Identity Convergence")

if __name__ == "__main__":
    main()
