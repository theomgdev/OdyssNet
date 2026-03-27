import os
import random
import sys

import numpy as np
import torch

# Adjust path to import realnet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from realnet import ChaosGradConfig, RealNet, RealNetTrainer, save_checkpoint, transplant_weights


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_two_pulse_batch(batch_size, seq_len, delay_a, delay_b, task, device):
    """
    Input sequence receives two scalar pulses on one input channel.
    Task output is read from final step on one output neuron.
    """
    inputs = torch.zeros(batch_size, seq_len, 1, device=device)

    a = (torch.rand(batch_size, 1, device=device) * 1.8) - 0.9
    b = (torch.rand(batch_size, 1, device=device) * 1.8) - 0.9

    inputs[:, delay_a, 0] = a[:, 0]
    inputs[:, delay_b, 0] = b[:, 0]

    if task == "add":
        target = a + b
    elif task == "mul":
        target = a * b
    else:
        raise ValueError(f"Unknown task: {task}")

    return inputs, target


def train_single_task(trainer, task, epochs, batch_size, seq_len, delay_a, delay_b, log_every=100):
    losses = []
    for epoch in range(epochs):
        x, y = generate_two_pulse_batch(batch_size, seq_len, delay_a, delay_b, task=task, device=trainer.device)
        loss = trainer.train_batch(x, y, thinking_steps=seq_len)
        losses.append(loss)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[{task}] Epoch {epoch:4d} | loss={loss:.6f}")

    return losses


def train_dual_multiplication(trainer_a, trainer_b, epochs, batch_size, seq_len, delay_a, delay_b, log_every=50):
    losses_a = []
    losses_b = []

    for epoch in range(epochs):
        x, y = generate_two_pulse_batch(batch_size, seq_len, delay_a, delay_b, task="mul", device=trainer_a.device)

        loss_a = trainer_a.train_batch(x, y, thinking_steps=seq_len)
        loss_b = trainer_b.train_batch(x, y, thinking_steps=seq_len)

        losses_a.append(loss_a)
        losses_b.append(loss_b)

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(
                f"[mul] Epoch {epoch:4d} | transplanted={loss_a:.6f} | scratch={loss_b:.6f}"
            )

    return losses_a, losses_b


def first_epoch_below(losses, threshold):
    for i, value in enumerate(losses):
        if value <= threshold:
            return i
    return -1


def evaluate_examples(trainer, seq_len, delay_a, delay_b, pairs):
    x = torch.zeros(len(pairs), seq_len, 1, device=trainer.device)
    y = torch.zeros(len(pairs), 1, device=trainer.device)

    for i, (a, b) in enumerate(pairs):
        x[i, delay_a, 0] = a
        x[i, delay_b, 0] = b
        y[i, 0] = a * b

    with torch.no_grad():
        pred = trainer.predict(x, thinking_steps=seq_len)

    mae = torch.mean(torch.abs(pred - y)).item()
    return pred[:, 0].detach().cpu().numpy().tolist(), y[:, 0].detach().cpu().numpy().tolist(), mae


def main():
    print("RealNet Experiment: Skill Transfer (Add -> Multiply)")
    print("Objective: Learn addition in small net, transplant to larger net, compare multiplication convergence vs scratch.")

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    seq_len = 14
    delay_a = 2
    delay_b = 8

    small_neurons = 24
    large_neurons = 96

    add_epochs = 1200
    mul_epochs = 700
    batch_size = 256
    lr = 1e-3

    print("\nStep 1/3: Train SMALL model on ADD")
    small_model = RealNet(
        num_neurons=small_neurons,
        input_ids=[0],
        output_ids=[1],
        device=device,
    )
    small_trainer = RealNetTrainer(
        small_model,
        device=device,
        chaos_config=ChaosGradConfig.default(lr=lr),
    )
    add_losses = train_single_task(
        small_trainer,
        task="add",
        epochs=add_epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        delay_a=delay_a,
        delay_b=delay_b,
        log_every=150,
    )

    ckpt_dir = os.path.join(os.path.dirname(__file__), "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "skill_transfer_add_small.pth")
    save_checkpoint(
        model=small_model,
        optimizer=small_trainer.optimizer,
        epoch=add_epochs,
        loss=add_losses[-1],
        path=ckpt_path,
        trainer_state=small_trainer.state_dict(),
    )
    print(f"Saved ADD checkpoint: {ckpt_path}")

    print("\nStep 2/3: Build LARGE models (transplanted vs scratch)")
    set_seed(777)
    scratch_model = RealNet(
        num_neurons=large_neurons,
        input_ids=[0],
        output_ids=[1],
        device=device,
    )

    set_seed(777)
    transfer_model = RealNet(
        num_neurons=large_neurons,
        input_ids=[0],
        output_ids=[1],
        device=device,
    )

    transplant_stats = transplant_weights(transfer_model, ckpt_path, device=device, verbose=True)

    scratch_trainer = RealNetTrainer(
        scratch_model,
        device=device,
        chaos_config=ChaosGradConfig.default(lr=lr),
    )
    transfer_trainer = RealNetTrainer(
        transfer_model,
        device=device,
        chaos_config=ChaosGradConfig.default(lr=lr),
    )

    print("\nStep 3/3: Train both LARGE models on MULTIPLY")
    transfer_losses, scratch_losses = train_dual_multiplication(
        transfer_trainer,
        scratch_trainer,
        epochs=mul_epochs,
        batch_size=batch_size,
        seq_len=seq_len,
        delay_a=delay_a,
        delay_b=delay_b,
        log_every=100,
    )

    threshold = 0.02
    hit_transfer = first_epoch_below(transfer_losses, threshold)
    hit_scratch = first_epoch_below(scratch_losses, threshold)

    avg_transfer = float(np.mean(transfer_losses))
    avg_scratch = float(np.mean(scratch_losses))

    pairs = [(-0.8, -0.7), (-0.8, 0.6), (-0.4, 0.9), (0.5, 0.5), (0.9, -0.3), (0.2, 0.7)]
    pred_t, tgt, mae_t = evaluate_examples(transfer_trainer, seq_len, delay_a, delay_b, pairs)
    pred_s, _, mae_s = evaluate_examples(scratch_trainer, seq_len, delay_a, delay_b, pairs)

    print("\n================ SUMMARY ================")
    print(f"Small ADD final loss: {add_losses[-1]:.6f}")
    print(
        f"Transplant copied: {transplant_stats['transplanted_params']}/{transplant_stats['total_params']} "
        f"({100.0 * transplant_stats['transplanted_params'] / transplant_stats['total_params']:.1f}%)"
    )
    print(f"MULTIPLY avg loss | transplanted={avg_transfer:.6f} | scratch={avg_scratch:.6f}")
    print(f"MULTIPLY final loss | transplanted={transfer_losses[-1]:.6f} | scratch={scratch_losses[-1]:.6f}")
    print(f"First epoch loss<={threshold:.3f} | transplanted={hit_transfer} | scratch={hit_scratch}")
    print(f"Test MAE | transplanted={mae_t:.6f} | scratch={mae_s:.6f}")

    print("\nExample predictions (target= a*b):")
    for i, (a, b) in enumerate(pairs):
        print(
            f"a={a:+.2f}, b={b:+.2f}, target={tgt[i]:+.4f} | "
            f"transferred={pred_t[i]:+.4f} | scratch={pred_s[i]:+.4f}"
        )

    print("\nClaim check:")
    if transfer_losses[-1] < scratch_losses[-1] and avg_transfer < avg_scratch:
        print("Transplanted model converged faster/better than scratch on multiplication.")
    else:
        print("No clear transfer win in this run. Try multi-seed for robust conclusion.")


if __name__ == "__main__":
    main()
