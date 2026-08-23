"""
The Hive: many bodies, one memory.

Every bee is the same network. Bodies never touch: separate hidden states,
separate attention caches, separate forward passes. The only thing a colony
shares is the chaos core -- and the Hebbian trace that lives on it, which the
library pools as the batch mean and hands back to every body on the next call.

The task is a ring over SYMBOLS nodes, drawn fresh for every episode so the
answer cannot sit in the weights. Each bee is shown exactly one edge of it and
nothing else. Then every body's hidden state and attention cache is wiped --
after the wipe no bee holds anything privately -- and one query symbol is put
to each bee. One echo step walks one edge:

    hop 1   an edge another bee was shown        -> shared memory
    hop 2   two edges, from two different bees   -> the colony composes them

Alone, a bee holds one edge and can go no further; that is the control, run on
the same weights with the same inputs, apart instead of together.
"""

import torch
import torch.nn as nn
from odyssnet import OdyssNet, OdyssNetTrainer, TrainingHistory, set_seed

SYMBOLS = 8              # ring nodes; also the colony size -- one edge per bee
NEURONS = 128
HOPS = 3                 # echo steps after the query lands
STUDY_STEPS = 2          # symbol, then symbol: the whole of a bee's experience
COLONIES = 8             # colonies per optimizer step (gradient accumulation)
TRAIN_STEPS = 350
EVAL_EPISODES = 40

HEBB_STATE = ("t_hebb_state_W", "t_hebb_state_mem",
              "s_hebb_state_W", "s_hebb_state_mem")


def build(device, hebb="both", attn=1):
    return OdyssNet(
        num_neurons=NEURONS,
        input_ids=list(range(SYMBOLS)),      # one neuron per symbol, in and out
        output_ids=list(range(SYMBOLS)),
        pulse_mode=True,
        hebb_type=hebb,
        hebb_res="neuron",
        attn_heads=attn or None,
        attn_head_dim=8 if attn else None,
        device=device,
    )


# --------------------------------------------------------------------------- #
# The colony's shared memory                                                   #
# --------------------------------------------------------------------------- #

def memory_of(model):
    """The colony's memory: the Hebbian trace, as plain tensors."""
    return {name: getattr(model, name).detach().clone()
            for name in HEBB_STATE if getattr(model, name, None) is not None}


def install(model, memory):
    with torch.no_grad():
        for name, value in memory.items():
            getattr(model, name).copy_(value)


def wipe_bodies(model, bees, device):
    """Erase every private carrier and keep the synapses."""
    model.state = torch.zeros(bees, NEURONS, device=device)
    if model.attn is not None:
        model.attn.reset()


def edge_frames(src, dst, device):
    """Bee i is shown symbol src[i], then symbol dst[i]. That is all it sees."""
    x = torch.zeros(len(src), STUDY_STEPS, NEURONS, device=device)
    for i, (a, b) in enumerate(zip(src, dst)):
        x[i, 0, a] = 1.0
        x[i, 1, b] = 1.0
    return x


def query_frames(symbols, device):
    x = torch.zeros(len(symbols), 1, NEURONS, device=device)
    for i, s in enumerate(symbols):
        x[i, 0, s] = 1.0
    return x


@torch.no_grad()
def forage(model, src, dst, device):
    """One study pass. Returns what the colony now remembers together."""
    bees = len(src)
    model.reset_state(bees)
    model(edge_frames(src, dst, device), steps=STUDY_STEPS,
          current_state=torch.zeros(bees, NEURONS, device=device))
    return memory_of(model)


@torch.no_grad()
def ask(model, memory, symbols, device):
    """Wipe the bodies, install the memory, walk. Returns (B, HOPS+1, SYMBOLS).

    `current_state` is passed explicitly on purpose: forward re-runs
    `reset_state` when the batch size changes, and that would zero the very
    memory this call is here to read.
    """
    bees = len(symbols)
    wipe_bodies(model, bees, device)
    install(model, memory)
    out, _ = model(query_frames(symbols, device), steps=HOPS + 1,
                   current_state=torch.zeros(bees, NEURONS, device=device))
    return out[:, :, list(range(SYMBOLS))]


# --------------------------------------------------------------------------- #
# Episodes                                                                     #
# --------------------------------------------------------------------------- #

def episode(gen, device):
    """A fresh ring, one edge per bee, one query per bee."""
    order = torch.randperm(SYMBOLS, generator=gen).tolist()
    src = order
    dst = order[1:] + order[:1]
    next_of = dict(zip(src, dst))

    queries = torch.randint(0, SYMBOLS, (SYMBOLS,), generator=gen).tolist()
    walk = torch.zeros(SYMBOLS, HOPS + 1, dtype=torch.long, device=device)
    for i, start in enumerate(queries):
        node = start
        for hop in range(HOPS + 1):
            walk[i, hop] = node
            node = next_of[node]
    return src, dst, queries, walk, next_of


def train(model, device, steps=TRAIN_STEPS, log=None):
    """Only the reading is trained: the study pass runs under no_grad, so the
    write is the architecture's own plasticity and never sees a gradient."""
    trainer = OdyssNetTrainer(model, device=device)
    trainer.loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(1234)

    for step in range(steps):
        for _ in range(COLONIES):
            src, dst, queries, walk, _ = episode(gen, device)
            forage(model, src, dst, device)
            wipe_bodies(model, SYMBOLS, device)
            loss = trainer.train_batch(
                query_frames(queries, device), walk, thinking_steps=HOPS + 1,
                full_sequence=True, keep_state=True,
                gradient_accumulation_steps=COLONIES,
                output_transform=lambda p: p.transpose(1, 2),
            )
        if log is not None:
            log.record(loss=loss)
        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  loss {loss:.4f}", flush=True)
    return trainer


# --------------------------------------------------------------------------- #
# The measurements                                                             #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def measure(model, device, episodes):
    """Together, apart, and with no memory at all -- same weights, same inputs."""
    model.eval()
    together = torch.zeros(HOPS + 1)
    apart = torch.zeros(HOPS + 1)
    blank = torch.zeros(HOPS + 1)
    empty = {name: torch.zeros_like(value)
             for name, value in memory_of(model).items()}
    asked = 0

    for src, dst, queries, walk, _ in episodes:
        colony = forage(model, src, dst, device)
        together += (ask(model, colony, queries, device).argmax(2).cpu()
                     == walk.cpu()).float().sum(0)
        blank += (ask(model, empty, queries, device).argmax(2).cpu()
                  == walk.cpu()).float().sum(0)
        for i in range(SYMBOLS):               # the same bee, living alone
            solo = forage(model, [src[i]], [dst[i]], device)
            answer = ask(model, solo, [queries[i]], device).argmax(2).cpu()[0]
            apart += (answer == walk[i].cpu()).float()
        asked += SYMBOLS

    model.train()
    return together / asked, apart / asked, blank / asked


@torch.no_grad()
def intervene(model, device, episodes):
    """Move one bee's edge; watch a different bee's answer follow it."""
    followed = unchanged_apart = trials = 0
    gen = torch.Generator().manual_seed(5)

    for src, dst, _, _, _ in episodes:
        j = int(torch.randint(0, SYMBOLS, (1,), generator=gen))
        step = 1 + int(torch.randint(0, SYMBOLS - 1, (1,), generator=gen))
        moved = list(dst)
        moved[j] = (dst[j] + step) % SYMBOLS
        if moved[j] == dst[j] or moved[j] == src[j]:
            continue
        listener = (j + 1) % SYMBOLS           # a different body does the asking

        before = ask(model, forage(model, src, dst, device),
                     [src[j]] * SYMBOLS, device).argmax(2)[listener, 1].item()
        after = ask(model, forage(model, src, moved, device),
                    [src[j]] * SYMBOLS, device).argmax(2)[listener, 1].item()
        followed += int(before == dst[j] and after == moved[j])

        solo_before = ask(model, forage(model, [src[listener]], [dst[listener]], device),
                          [src[j]], device)
        solo_after = ask(model, forage(model, [src[listener]], [moved[listener]], device),
                         [src[j]], device)
        unchanged_apart += int(torch.equal(solo_before, solo_after))
        trials += 1

    return followed / trials, unchanged_apart / trials


@torch.no_grad()
def wrong_colony(model, device, episodes):
    """Install a different colony's memory: the answers must follow it, not us."""
    theirs = ours = asked = 0
    for k, (src, dst, queries, walk, _) in enumerate(episodes):
        other = episodes[(k + 1) % len(episodes)]
        elsewhere = forage(model, other[0], other[1], device)
        answer = ask(model, elsewhere, queries, device).argmax(2).cpu()

        other_next = other[4]
        theirs += sum(int(answer[i, 1] == other_next[q]) for i, q in enumerate(queries))
        ours += (answer[:, 1] == walk[:, 1].cpu()).sum().item()
        asked += SYMBOLS
    return theirs / asked, ours / asked


@torch.no_grad()
def pooling_is_not_a_trick(model, device, episode_data):
    """Running the colony as one batch, or every bee separately and pooling the
    memories afterwards, must give the same memory."""
    src, dst = episode_data[0], episode_data[1]
    batched = forage(model, src, dst, device)
    apart = [forage(model, [src[i]], [dst[i]], device) for i in range(SYMBOLS)]
    pooled = {name: torch.stack([m[name] for m in apart]).mean(0)
              for name in batched}
    worst = max((pooled[n] - batched[n]).abs().max().item() for n in batched)
    scale = max(batched[n].abs().max().item() for n in batched)
    return worst, scale


# --------------------------------------------------------------------------- #

def main():
    print("OdyssNet: The Hive Mind")
    set_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build(device)
    print(f"A colony of {SYMBOLS} bees sharing one {model.get_num_params()}-parameter "
          f"brain, on {device}.")
    print(f"Each bee is shown one edge of a {SYMBOLS}-node ring, drawn fresh every "
          f"episode. Chance is {1 / SYMBOLS:.3f}.\n")

    print("Training the reading (the write is never trained):")
    history = TrainingHistory()
    train(model, device, log=history)

    eval_gen = torch.Generator().manual_seed(99)
    episodes = [episode(eval_gen, device) for _ in range(EVAL_EPISODES)]

    together, apart, blank = measure(model, device, episodes)
    print(f"\nRecall on rings never seen in training, "
          f"{EVAL_EPISODES * SYMBOLS} queries per column:")
    print(f"  {'':<26}" + "".join(f"{'hop ' + str(h):>10}" for h in range(HOPS + 1)))
    for label, row in (("together (one memory)", together),
                       ("apart (one bee alone)", apart),
                       ("together, memory blank", blank)):
        print(f"  {label:<26}" + "".join(f"{v:>10.3f}" for v in row))
    print("  hop 0 is the query itself, hop 1 is another bee's edge, "
          "hop 2 needs two bees' edges.")

    followed, unchanged = intervene(model, device, episodes)
    print("\nOne bee's edge is moved:")
    print(f"  a different bee's answer follows the change   {followed:.3f}")
    print(f"  that bee, run alone, answers identically      {unchanged:.3f}")

    theirs, ours = wrong_colony(model, device, episodes)
    print("\nAnother colony's memory installed:")
    print(f"  answers match the installed ring              {theirs:.3f}")
    print(f"  answers match the ring we asked about         {ours:.3f}")

    worst, scale = pooling_is_not_a_trick(model, device, episodes[0])
    print("\nBees run together, or run apart and pooled afterwards:")
    print(f"  largest difference in the memory              {worst:.3e} "
          f"(memory scale {scale:.3e})")

    # A body that did not exist while the colony foraged.
    src, dst, queries, walk, _ = episodes[0]
    colony = forage(model, src, dst, device)
    newborn = build(device)
    newborn.load_state_dict(model.state_dict())     # the genome and the memory
    newborn.eval()
    born_answer = ask(newborn, colony, queries, device).argmax(2).cpu()
    hatched = (born_answer[:, 1] == walk[:, 1].cpu()).float().mean().item()
    print("\nA bee built after the foraging, never run before:")
    print(f"  hop 1 on the colony's memory                  {hatched:.3f}")

    print("\nAblation, the same run with plasticity off:")
    set_seed(42)
    plain = build(device, hebb=None)
    train(plain, device, steps=TRAIN_STEPS // 2)
    plain_together, _, _ = measure(plain, device, episodes[:10])
    print(f"  hebb_type=None, together, hop 1               "
          f"{plain_together[1]:.3f} (chance {1 / SYMBOLS:.3f})")

    print("\nOne walk, in full:")
    print("  the ring        " + " -> ".join(str(s) for s in src) + f" -> {src[0]}")
    print(f"  bee {SYMBOLS - 1} was shown  {src[-1]} -> {dst[-1]}, and nothing else")
    trail = " -> ".join(str(int(s)) for s in born_answer[SYMBOLS - 1, :3])
    truth = " -> ".join(str(int(s)) for s in walk[SYMBOLS - 1, :3])
    print(f"  asked to walk from {queries[SYMBOLS - 1]}: {trail}   "
          f"(the ring says {truth})")

    print("\nVERDICT")
    proven = (together[1] > 0.9 and together[2] > 0.9
              and apart[1] < 0.35 and blank[1] < 0.35
              and plain_together[1] < 0.35)
    if proven:
        print("  The colony recalls edges no member observed, and composes edges")
        print("  held by different members. Apart, on the same weights and the")
        print("  same inputs, none of it survives. The shared plastic trace on")
        print("  the chaos core is the only channel between the bodies.")
    else:
        print("  Not reproduced on this run -- read the table above.")

    history.plot(title="Hive Mind Convergence")


if __name__ == "__main__":
    main()
