"""
OdyssNet-Diffusion — a denoiser that remembers its own trajectory.

Every image diffusion model in use today calls a memoryless network once per
timestep. The UNet at step t knows nothing of what it computed at step t+1; all
continuity is smuggled through the noisy image itself. OdyssNet does not have to
work that way, because diffusion is already a loop over time and OdyssNet is a
network whose depth *is* time.

The mapping is native, not bolted on. With `pulse_mode=False` and a 3-D input of
K frames run for K*E steps, `forward` resolves `ratio = E` and:

    frame k injected at step k*E          one denoising timestep
    E echo steps through the N x N core   temporal depth in place of UNet depth
    output collected at step (k+1)*E - 1  the epsilon for that timestep
    h_t crosses every frame boundary      the denoiser remembers its trajectory
    attention writes once per frame       attention along the reverse process
    the plastic trace spans all K*E steps a per-generation fast weight

So the whole reverse trajectory is one differentiable forward pass, and
`train_batch(..., full_sequence=True)` against the trainer's default MSELoss is
the entire training call.

There is no VAE. `vocab_size=[F_in, P]` with `vocab_mode='continuous'` makes
OdyssNet's own `proj` and `output_decoder` the encoder and decoder, trained
end-to-end by the diffusion objective, so every learned parameter lives inside
the model. Conditioning is fixed-basis -- a sinusoidal timestep embedding and a
class one-hot with a null slot for classifier-free guidance -- so nothing
learned sits outside the network either.

The model predicts x_0, and on this architecture that is not a preference. The
answer is read off `n_out` neurons, so whatever the network outputs is a
rank-`n_out` view of a P-dimensional image, and the parameterisation decides
whether that rank is enough. Epsilon is white noise -- isotropic, full rank,
incompressible -- so a rank-192 view of it keeps 192/784 of the variance and
pins the achievable MSE at 0.755 however long training runs. A 573k-parameter
run measured 0.791: saturated, not undertrained. Natural images are low rank,
and the same 192 directions carry all but 3.4% of MNIST's variance. Measured at
3 minutes per arm, as a fraction of the do-nothing predictor on the same frozen
grid:

    x_0     10.2%      falls 0.215 -> 0.048 from pure noise to nearly clean
    v       56.3%      x_0-like at high t, epsilon-like at low t, so it
                       inherits the rank problem over half the range
    eps     79.2%      flat across every timestep, at the bound

`--sweep size` carries epsilon arms at four widths, and they behave as the rank
argument says they must -- always above the bound, monotone in `n_out`, and
closing on it with training. All four
sample at chance:

    n_out    bound 1 - n_out/P    measured
       96                0.878       0.896  (2,200 steps)
      144                0.816       0.845  (2,115 steps)
      192                0.755       0.791  (3,003 steps)
      288                0.633       0.681  (3,643 steps)

Usage
-----
    python -u experiment_diffusion.py --mode smoke
    python -u experiment_diffusion.py --mode train --minutes 15
    python -u experiment_diffusion.py --mode sweep --sweep memory --minutes 4
    python -u experiment_diffusion.py --mode sweep --sweep memory --max-steps 600 --minutes 25
    python -u experiment_diffusion.py --mode sweep --sweep depth --minutes 3
    python -u experiment_diffusion.py --mode sample --tag base --cfg 3.0
    python -u experiment_diffusion.py --dataset cifar10 --neurons 768

What the memory is worth
------------------------
Measured, not claimed. `--sweep memory` at equal wall clock -- 3 minutes per arm,
MNIST, seed 54321, guidance 2.0, and the Frechet distance taken in a fixed
classifier's feature space:

    arm                    val MSE   fidelity   frechet     params
    trajectory              0.0967      83.6%    19.921    573,376
    traj_attn               0.1250      83.0%    30.423    901,184
    traj_noise_shared       0.1762      83.4%    18.912    573,376
    traj_full               0.1980      19.6%   169.113    902,720
    traj_hebb_spatial       0.2037      11.6%   202.660    574,912
    traj_hebb_temporal      0.2059      10.8%   214.899    574,912
    traj_hebb_both          0.2142      16.8%   149.040    576,448
    independent             0.3054      10.4%   152.321    573,376

`independent` is the control that matters: the same frames, the same targets and
the same gradient budget, issued as K separate calls so the denoiser begins every
frame with nothing -- which is what a UNet sampler does. Carrying the trajectory
instead takes conditioning fidelity from chance to 83.6% and the Frechet
distance from 152.3 to 19.9 at an identical parameter count. That is what this
file was written to test, and it survived its control.

The cleanest form of the same measurement needs no second training run at all.
`--mode eval` samples one checkpoint twice, with the carry on and with it wiped
between denoising steps -- identical weights, identical guidance, one line of
difference at inference:

    carried            fidelity 91.4%   frechet  9.8
    wiped each step    fidelity 53.6%   frechet 40.8

That run also reports sampling-batch sensitivity, because the plastic buffer is
a batch mean and a batch generated together would share one memory. With
plasticity off, which is the default, there is nothing to share and fidelity
holds between 90.8% and 92.8% from batch 10 to batch 100. Turn `--hebb` on and
the question becomes live again, which is why the probe is printed rather than
argued.

The rest is worth reading for what it costs.

`traj_noise_shared` is the arm the validation grid decides. One epsilon per
trajectory leaves two frames enough to recover x_0 by linear algebra, so the
model can learn an inversion instead of a denoiser, and an inversion cannot
follow it into sampling, where the frames come from its own predictions. Scored
on the iid grid it sits at 0.1762 against `trajectory`'s 0.0967 -- the shortcut
does not survive contact with independent noise, which is the theory holding.
Its samples are a different story: 83.4% fidelity and the best Frechet in the
table on this seed, against 54.2% and 76.8 on seed 42. `--traj-noise iid` is the
default because it has never been worse on the sample columns, which are the
ones that decide; the loss column is reported beside them so an arm that trades
one for the other stays visible.

`traj_attn` is behind `trajectory` on all three columns here for 57% more
parameters, and it reached only 862 steps in the same three minutes. On one seed
at 500 samples that is not a separation, and per parameter it is a loss, so
attention is available and is not the default. Data harder than MNIST is the
case for `--attn-heads 4`, and that case is not measured here.

`traj_hebb_temporal` is behind on every column, and `traj_full` does not recover
what attention alone had. Plasticity is also slow: at equal wall clock rather
than equal gradients the plastic arms reach roughly 6% of the plain arm's step
count and attention roughly 28%, because the retained trace grows with the step
count, the batch and the neuron count together. On either budget, plasticity
loses on this task -- which is what `hebb_type=None` costing nothing is for.

A note on what the batch means here: the plastic buffer is a batch mean, so a
batch of images generated together share one plastic memory -- the hive mind
applied to generation. Sample quality can therefore depend on the sampling batch
size, which is why `--sample-batch` exists and why eval reports it.

What temporal depth is worth
----------------------------
The K frames and the E echo steps between them multiply into the same compute,
so `--sweep depth` holds K*E = 64 fixed and asks which of the two the budget
should buy. Only this architecture can ask it: on a UNet the number of denoising
steps and the depth spent inside one are different resources, and here they are
the same one. MNIST, seed 54321, x_0, guidance 2.0, 3 minutes per arm:

    arm      frames  echo   val MSE   fidelity   frechet    steps
    k32_e2       32     2    0.0860      73.0%    13.059    3,593
    k16_e4       16     4    0.0914      83.6%    15.968    3,800
    k8_e8         8     8    0.0947      88.0%    13.963    3,932
    k4_e16        4    16    0.0999      95.0%    15.657    4,004

Conditioning fidelity climbs monotonically with echo depth at identical compute,
while the Frechet distance stays flat across all four -- 13.1 to 16.0, in no
order -- so what improves is the conditioning rather than the sample
distribution narrowing onto a few modes. Held-out loss runs the other way, and
that is the denoising grid rather than the model: fewer frames means coarser
timesteps, so each one is a harder prediction. The deepest arm also samples in
four denoising steps instead of thirty-two, which is the cheapest inference in
the table by a factor of eight.

Two things to hold against it. The arms are equal wall clock rather than equal
gradients, and the step counts spread 11% in the deepest arm's favour. And the
ranked table crowns k32_e2, because `RANK_KEY` is Frechet and Frechet is the one
column that does not separate here -- read the fidelity column for this sweep.
The default stays 16 x 4 on one seed of evidence; trading denoising resolution
for echo depth is a change worth a second seed first.

What width is worth
-------------------
`--sweep size` at the same budget, the x_0 arms:

    arm    neurons  n_out     params   val MSE   fidelity   frechet
    n256       256     96    221,152    0.1107      75.6%    27.633
    n384       384    144    380,880    0.0959      83.0%    21.165
    n512       512    192    573,376    0.0918      85.8%    14.732
    n768       768    288  1,056,672    0.0866      87.6%    11.523

Returns are still positive at a million parameters and already shallow: 4.8x the
parameters of n256 buys twelve points of fidelity. `n_out` scales with the width
in this grid, so the curve mixes capacity with output rank -- which is the pair
the epsilon arms above separate, since those move only rank and stay at chance
whatever the width.
"""

import sys

# Keep emoji-rich console output from crashing legacy Windows code pages.
# line_buffering=True is not optional: reconfigure() rebuilds the TextIOWrapper
# and would otherwise discard `python -u`'s write-through, leaving a long
# training run's progress invisible until the process exits.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from odyssnet import (OdyssNet, OdyssNetTrainer, load_checkpoint,
                      save_checkpoint, set_seed)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "ckpt")
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(HERE, "..", "..", "out", "diffusion_samples")

# Every run is scored against the do-nothing predictor -- output zero -- on the
# same frozen validation grid. For epsilon that is exactly 1.0 by construction;
# for x_0 it is the image variance, which is far smaller. Measuring it rather
# than assuming it is what keeps the parameterisations comparable.

# Samples drawn to score a run. The Frechet estimate is a covariance over
# Scorer.FEATURES dimensions, so a hundred images would leave it rank-starved.
MEASURE_SAMPLES = 500

DATASETS = {
    #        channels, side, classes
    "mnist":   (1, 28, 10),
    "cifar10": (3, 32, 10),
}


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Cfg:
    # data
    dataset: str = "mnist"
    train_images: int = -1          # -1 = all
    val_images: int = 2048

    # diffusion
    timesteps: int = 1000           # T, the continuous schedule
    frames: int = 16                # K, denoising steps actually visited
    echo: int = 4                   # E, echo steps per denoising step
    predict: str = "x0"             # x0 | eps | v
    carry: str = "trajectory"       # trajectory | independent
    traj_noise: str = "iid"         # iid | shared
    class_dropout: float = 0.1      # for classifier-free guidance

    # architecture
    neurons: int = 512
    n_in: int = 192
    n_out: int = 192
    t_embed: int = 32
    activation: tuple = ("none", "tanh", "tanh")
    weight_init: tuple = ("quiet", "resonant", "quiet", "zero")
    gates: tuple = ("none", "none", "identity")
    hebb_type: str = ""             # "" = off
    hebb_res: str = "neuron"
    dropout: float = 0.0

    # attention
    attn_heads: int = 0
    attn_kv_heads: int = 1
    attn_head_dim: int = 0
    attn_window: int = 256
    attn_write: str = "token"       # token = one KV entry per denoising step
    attn_read: str = "step"
    attn_rope: bool = True
    attn_qk_norm: bool = True
    attn_dropout: float = 0.0

    # optimization
    batch: int = 64
    lr: float = None                # None = zero-config ChaosGrad
    grad_ckpt: bool = False
    compile: bool = False

    # sampling
    sampler: str = "ddim"           # ddim | ddpm
    eta: float = 0.0
    # 3.0 rather than 2.0: it leads on both sample columns at every seed and on
    # both datasets measured. Past it fidelity still climbs while the Frechet
    # distance turns, which is guidance trading variety for class-purity.
    cfg_scale: float = 3.0
    sample_batch: int = 64

    # run control
    minutes: float = 0.0            # 0 = until Ctrl-C
    max_steps: int = 0
    tag: str = "base"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def io_ids(self):
        return (list(range(self.n_in)),
                list(range(self.n_in, self.n_in + self.n_out)))

    def shape(self):
        """(channels, side, classes) for the selected dataset."""
        return DATASETS[self.dataset]

    def pixels(self):
        c, s, _ = self.shape()
        return c * s * s

    def feature_width(self):
        """Width of one frame: the noisy image, the clock, and the class."""
        _, _, classes = self.shape()
        return self.pixels() + self.t_embed + classes + 1

    def steps(self):
        return self.frames * self.echo


# --------------------------------------------------------------------------- #
# The forward process                                                          #
# --------------------------------------------------------------------------- #

class Schedule:
    """A cosine noise schedule and the K-step grid the model actually visits.

    Cosine rather than linear because at 28x28 a linear schedule spends most of
    its budget on timesteps that are already pure noise (Nichol & Dhariwal).

    The training grid *is* the sampling grid. Training on the same strided
    timesteps the sampler will walk keeps the trajectory the model learns and
    the trajectory it is asked to produce the same shape, so what is left is
    ordinary exposure bias rather than a wholesale mismatch.
    """

    def __init__(self, cfg, device):
        self.T = cfg.timesteps
        self.K = cfg.frames
        self.device = device

        t = torch.linspace(0, 1, self.T + 1, dtype=torch.float64, device=device)
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = (f / f[0]).clamp(1e-9, 1.0)

        self.alpha_bar = alpha_bar[1:].float()          # (T,), index t-1 .. t
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_1mab = (1.0 - self.alpha_bar).sqrt()

        # The visited grid, noisiest first: t_0 > t_1 > ... > t_{K-1}, and a
        # trailing -1 standing for the clean image the last step lands on.
        grid = torch.linspace(self.T - 1, 0, self.K, device=device).round().long()
        self.grid = grid
        self.prev = torch.cat([grid[1:], torch.tensor([-1], device=device)])

    def ab(self, t):
        """alpha_bar at integer timesteps, with alpha_bar(-1) = 1 (clean)."""
        return torch.where(t < 0, torch.ones_like(t, dtype=torch.float32),
                           self.alpha_bar[t.clamp(min=0)])

    def q_sample(self, x0, t, eps):
        """x_t = sqrt(ab) x_0 + sqrt(1-ab) eps, broadcasting over the frame axis."""
        ab = self.ab(t).unsqueeze(-1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps

    def target(self, x0, eps, t, predict):
        """What the network is asked to output at timestep t.

        The choice is not cosmetic here, it is the difference between a model
        that works and one that cannot. OdyssNet reads its answer off `n_out`
        neurons, so the output is a rank-`n_out` view of a `P`-dimensional
        image, and the parameterisation decides whether that rank is enough.

        Epsilon is white noise: isotropic, full rank, incompressible. A rank-192
        view of it keeps 192/784 of the variance, which pins the achievable MSE
        at 0.755 no matter how long the model trains -- and a 573k-parameter run
        measured 0.791, within 5% of that floor. The network was saturated, not
        untrained.

        x_0 is a natural image, and those are low rank: the same 192 directions
        carry all but 3.4% of MNIST's variance. Same bottleneck, a fortieth of
        the loss. `--predict x0` is the default for that reason.
        """
        if predict == "x0":
            return x0
        if predict == "eps":
            return eps
        # v = sqrt(ab) eps - sqrt(1-ab) x_0: epsilon-like at low t, x_0-like at
        # high t, so it inherits epsilon's rank problem over part of the range.
        ab = self.ab(t).unsqueeze(-1)
        return ab.sqrt() * eps - (1.0 - ab).sqrt() * x0

    def to_eps(self, out, x_t, t, predict):
        """Read a prediction back as epsilon, whichever parameterisation made it."""
        if predict == "eps":
            return out
        ab = self.ab(t).unsqueeze(-1)
        if predict == "x0":
            return (x_t - ab.sqrt() * out) / (1.0 - ab).clamp(min=1e-8).sqrt()
        return ab.sqrt() * out + (1.0 - ab).sqrt() * x_t

    def to_x0(self, eps, x_t, t):
        ab = self.ab(t).unsqueeze(-1)
        return (x_t - (1.0 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)

    def read_x0(self, out, x_t, t, predict):
        """A prediction read as x_0, without a detour through epsilon.

        At the clean end sqrt(1-ab) is ~0.01, so converting an x_0 prediction to
        epsilon and back multiplies and then divides by a hundred for no reason.
        """
        if predict == "x0":
            return out
        return self.to_x0(self.to_eps(out, x_t, t, predict), x_t, t)


def timestep_embedding(t, dim):
    """Sinusoidal clock, no parameters. `t` is (...,) integer; returns (..., dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    ang = t.float().unsqueeze(-1) * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def class_vector(labels, classes, drop_mask=None):
    """One-hot over classes plus a null slot.

    The null slot is an explicit unconditional token rather than an all-zero
    vector, so "no class" is a signal the projection can learn to read instead
    of being the average of every class.
    """
    v = torch.zeros(*labels.shape, classes + 1, device=labels.device)
    v.scatter_(-1, labels.unsqueeze(-1).clamp(min=0), 1.0)
    if drop_mask is not None:
        v[drop_mask] = 0.0
        v[..., classes] = torch.where(drop_mask, torch.ones_like(v[..., classes]),
                                      v[..., classes])
    return v


def null_class_vector(shape, classes, device):
    v = torch.zeros(*shape, classes + 1, device=device)
    v[..., classes] = 1.0
    return v


def make_frames(x_t, t, cls_vec, cfg):
    """Assemble (B, K, F_in): the noisy image, the clock, and the class."""
    return torch.cat([x_t, timestep_embedding(t, cfg.t_embed), cls_vec], dim=-1)


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #

def load_images(cfg):
    """Flat images in [-1, 1] plus labels, held in RAM.

    Small enough at both tiers (MNIST 60k x 784, CIFAR 50k x 3072) that a
    DataLoader would only add per-batch overhead to a step that is already
    kernel-launch-bound.
    """
    from torchvision import datasets, transforms

    root = os.path.normpath(DATA_DIR)
    tf = transforms.ToTensor()
    ctor = datasets.MNIST if cfg.dataset == "mnist" else datasets.CIFAR10

    out = []
    for train in (True, False):
        ds = ctor(root=root, train=train, download=True, transform=tf)
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
        out.append((x.reshape(len(ds), -1) * 2.0 - 1.0, y))

    (xtr, ytr), (xva, yva) = out
    if cfg.train_images > 0:
        xtr, ytr = xtr[:cfg.train_images], ytr[:cfg.train_images]
    if cfg.val_images > 0:
        xva, yva = xva[:cfg.val_images], yva[:cfg.val_images]
    return xtr, ytr, xva, yva


class Batches:
    """Shuffled epochs over the training images, on the target device."""

    def __init__(self, x, y, batch, device, seed):
        self.x, self.y = x.to(device), y.to(device)
        self.batch, self.device = batch, device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.order, self.pos, self.epochs = None, 0, 0
        self._reshuffle()

    def _reshuffle(self):
        self.order = torch.randperm(len(self.x), generator=self.gen).to(self.device)
        self.pos = 0

    def next(self):
        if self.pos + self.batch > len(self.order):
            self._reshuffle()
            self.epochs += 1
        idx = self.order[self.pos:self.pos + self.batch]
        self.pos += self.batch
        return self.x[idx], self.y[idx]


def trajectory_batch(x0, labels, sched, cfg, gen=None):
    """One reverse trajectory per image: (frames, targets).

    `'iid'` draws a fresh epsilon per frame, so each frame is an independent
    view of the same image at a different signal-to-noise ratio and carried
    state has evidence to accumulate. It is the default because it measured
    better where it counts.

    `'shared'` draws one epsilon per image and derives every frame from it in
    closed form. That is the trajectory a perfect DDIM sampler walks, which is
    why it looks like the right choice, and it carries a risk: two frames of a
    shared epsilon trajectory determine x_0 by linear algebra, so the model can
    learn an inversion instead of a denoiser. Scored on the iid grid, which is
    the one `Validator` holds for every arm, that is what it looks like: 0.1762
    against iid's 0.0967 at equal wall clock, seed 54321. What did not replicate
    is the sample half:

        seed 42     shared fid 54.2% frechet 76.845   iid fid 78.4% frechet 40.821
        seed 54321  shared fid 83.4% frechet 18.912   iid fid 83.6% frechet 19.921

    Both rows are equal wall clock, so what disagrees is the seed and not the
    budget. `iid` is the default because it has never been worse on those two
    columns, not on a settled gap. `--sweep memory` reports loss and Frechet
    separately, because an arm that improves on one while worsening on the other
    is the signature to watch for.
    """
    B, P = x0.shape
    K = cfg.frames
    _, _, classes = cfg.shape()
    t = sched.grid.unsqueeze(0).expand(B, K)

    if cfg.traj_noise == "shared":
        eps = torch.randn(B, 1, P, device=x0.device, generator=gen).expand(B, K, P)
    else:
        eps = torch.randn(B, K, P, device=x0.device, generator=gen)

    x_wide = x0.unsqueeze(1).expand(B, K, P)
    x_t = sched.q_sample(x_wide, t, eps)
    target = sched.target(x_wide, eps, t, cfg.predict)

    drop = torch.rand(B, device=x0.device, generator=gen) < cfg.class_dropout
    cls = class_vector(labels.unsqueeze(1).expand(B, K), classes,
                       drop.unsqueeze(1).expand(B, K))
    return make_frames(x_t, t, cls, cfg), target


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

def build(cfg):
    input_ids, output_ids = cfg.io_ids()
    model = OdyssNet(
        num_neurons=cfg.neurons,
        input_ids=input_ids,
        output_ids=output_ids,
        device=cfg.device,
        # A 3-D input is only read as a sequence of frames when injection is
        # continuous; with pulse_mode the whole trajectory would land at t=0.
        pulse_mode=False,
        activation=list(cfg.activation),
        weight_init=list(cfg.weight_init),
        gate=list(cfg.gates),
        hebb_type=cfg.hebb_type or None,
        hebb_res=cfg.hebb_res,
        attn_heads=cfg.attn_heads or None,
        attn_kv_heads=cfg.attn_kv_heads,
        attn_head_dim=cfg.attn_head_dim or None,
        attn_window=cfg.attn_window,
        attn_write=cfg.attn_write,
        attn_read=cfg.attn_read,
        attn_rope=cfg.attn_rope,
        attn_qk_norm=cfg.attn_qk_norm,
        attn_dropout=cfg.attn_dropout,
        dropout_rate=cfg.dropout,
        gradient_checkpointing=cfg.grad_ckpt,
        # OdyssNet's own projections are the encoder and decoder; there is no
        # autoencoder outside the model. `activation[0]` stays 'none' because
        # the decoder's output passes through it, and epsilon must not be
        # squashed.
        vocab_size=[cfg.feature_width(), cfg.pixels()],
        vocab_mode="continuous",
    )
    if cfg.compile:
        model.forward = torch.compile(model.forward)

    trainer = OdyssNetTrainer(model, lr=cfg.lr, device=cfg.device)
    return model, trainer


def snapshot_state(model):
    """Capture every recurrent carrier: hidden state, plastic trace, KV cache."""
    snap = {"state": model.state.clone()}
    if model.hebb_type is not None:
        for name in ("t_hebb_state_W", "t_hebb_state_mem",
                     "s_hebb_state_W", "s_hebb_state_mem"):
            buf = getattr(model, name, None)
            if buf is not None:
                snap[name] = buf.clone()
    if getattr(model, "attn", None) is not None:
        snap["attn"] = model.attn.snapshot()
    return snap


def restore_state(model, snap):
    with torch.no_grad():
        model.state = snap["state"].clone()
        for name, value in snap.items():
            if name.endswith("_hebb_state_W") or name.endswith("_hebb_state_mem"):
                getattr(model, name).copy_(value)
        if "attn" in snap and getattr(model, "attn", None) is not None:
            model.attn.restore(snap["attn"])


def wipe(model, batch):
    """A fresh generation: no state, no cache, no plastic memory carried in."""
    model.reset_state(batch)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def train_step(trainer, frames, target, cfg):
    """One optimizer step over a batch of trajectories.

    The two arms differ in exactly one thing: whether the denoiser is allowed to
    remember the previous frame.

    `trajectory` runs all K frames in a single call, so hidden state, the
    attention cache and the plastic trace cross every frame boundary.

    `independent` issues the same K frames as K separate calls with
    `keep_state=False`. `reset_state` zeroes state, cache and plastic buffers
    alike, so each frame is denoised with no memory of the last one -- a
    memoryless denoiser, which is what every UNet sampler is. The target count,
    the step count and the gradient budget are identical to `trajectory`;
    only the memory differs, which is what makes the comparison readable.
    """
    if cfg.carry == "trajectory":
        return trainer.train_batch(frames, target,
                                   thinking_steps=cfg.steps(),
                                   full_sequence=True)

    # The trainer already reports the un-normalised loss, so averaging the K
    # calls gives the same quantity the trajectory arm's single call reports:
    # the mean squared error over every frame.
    total = 0.0
    for k in range(cfg.frames):
        total += trainer.train_batch(
            frames[:, k:k + 1], target[:, k:k + 1],
            thinking_steps=cfg.echo,
            full_sequence=True,
            gradient_accumulation_steps=cfg.frames,
        )
    return total / cfg.frames


class Validator:
    """Held-out epsilon-MSE on a frozen noise/timestep grid.

    Deterministic on purpose: the same images, the same epsilon and the same
    class dropout pattern every call, so two runs are comparable to the last
    digit and a sweep arm's number means something.

    The grid is always iid, whatever `--traj-noise` trained the model. A shared
    epsilon determines x_0 from any two frames, so a grid built that way is the
    shared arm's own training distribution and a foreign one for every other
    arm -- it scores an inversion as though it were a denoiser. Holding the grid
    at iid is what makes the column comparable across arms.
    """

    def __init__(self, x, y, sched, cfg):
        self.sched, self.cfg = sched, cfg
        self.batch = min(cfg.batch, len(x))
        self.chunks = []

        gen = torch.Generator(device="cpu").manual_seed(1234)
        _, _, classes = cfg.shape()
        K, P = cfg.frames, cfg.pixels()

        for i in range(0, len(x) - self.batch + 1, self.batch):
            x0 = x[i:i + self.batch].to(cfg.device)
            labels = y[i:i + self.batch].to(cfg.device)
            B = x0.shape[0]
            t = sched.grid.unsqueeze(0).expand(B, K)
            eps = torch.randn(B, K, P, generator=gen).to(cfg.device)
            x_wide = x0.unsqueeze(1).expand(B, K, P)
            cls = class_vector(labels.unsqueeze(1).expand(B, K), classes)
            self.chunks.append((
                make_frames(sched.q_sample(x_wide, t, eps), t, cls, cfg),
                sched.target(x_wide, eps, t, cfg.predict),
            ))

        # The do-nothing predictor on this exact grid: the bar every arm clears
        # or fails to, whatever the parameterisation.
        self.trivial = sum(tg.pow(2).mean().item() for _, tg in self.chunks)             / max(len(self.chunks), 1)

    @torch.no_grad()
    def score(self, model):
        snap = snapshot_state(model)
        model.eval()
        total, n = 0.0, 0
        try:
            for frames, target in self.chunks:
                B = frames.shape[0]
                wipe(model, B)
                out, _ = model(frames, steps=self.cfg.steps(),
                               current_state=torch.zeros(B, model.num_neurons,
                                                         device=model.device))
                total += F.mse_loss(out.float(), target).item()
                n += 1
        finally:
            model.train()
            restore_state(model, snap)
        return total / max(n, 1)


# --------------------------------------------------------------------------- #
# Sampling                                                                     #
# --------------------------------------------------------------------------- #

class Carrier:
    """One sampling run's memory: hidden state, KV cache, plastic trace.

    Classifier-free guidance needs a conditional and an unconditional pass per
    denoising step, and the two must not read each other's memory. When
    plasticity is off that is free -- state and cache rows are independent, so
    the two branches ride in one 2B batch. With plasticity on the buffer is a
    module-level batch mean, which would pool the branches and let the second
    call read the first; there the branches are run separately and their whole
    carrier is swapped in and out around each pass.
    """

    def __init__(self, model, batch):
        wipe(model, batch)
        self.snap = snapshot_state(model)

    def run(self, model, frames, steps):
        restore_state(model, self.snap)
        out, h = model(frames, steps=steps, current_state=self.snap["state"])
        self.snap = snapshot_state(model)
        self.snap["state"] = h.detach()
        return out


@torch.no_grad()
def sample(model, sched, cfg, labels, carry=True, progress=False):
    """Walk the reverse trajectory. Returns images in [-1, 1]."""
    model.eval()
    device = cfg.device
    B, P = labels.shape[0], cfg.pixels()
    _, _, classes = cfg.shape()
    guided = cfg.cfg_scale != 1.0
    split = guided and model.hebb_type is not None

    x = torch.randn(B, P, device=device)
    cond = class_vector(labels, classes)
    null = null_class_vector((B,), classes, device)

    if split:
        carriers = (Carrier(model, B), Carrier(model, B))
    elif guided:
        carriers = (Carrier(model, 2 * B),)
    else:
        carriers = (Carrier(model, B),)

    for k in range(cfg.frames):
        t = sched.grid[k].expand(B)
        t_prev = sched.prev[k].expand(B)

        if split:
            e_c = carriers[0].run(model, make_frames(x, t, cond, cfg).unsqueeze(1),
                                  cfg.echo)[:, 0]
            e_u = carriers[1].run(model, make_frames(x, t, null, cfg).unsqueeze(1),
                                  cfg.echo)[:, 0]
        elif guided:
            both = torch.cat([make_frames(x, t, cond, cfg),
                              make_frames(x, t, null, cfg)], dim=0).unsqueeze(1)
            out = carriers[0].run(model, both, cfg.echo)[:, 0]
            e_c, e_u = out[:B], out[B:]
        else:
            e_c = carriers[0].run(model, make_frames(x, t, cond, cfg).unsqueeze(1),
                                  cfg.echo)[:, 0]
            e_u = e_c

        # A memoryless denoiser is the control: rebuild the carriers so the next
        # step starts from nothing, exactly as a UNet sampler does.
        if not carry:
            carriers = tuple(Carrier(model, c.snap["state"].shape[0]) for c in carriers)

        # Guidance is applied in whatever space the model predicts. epsilon is
        # affine in x_0 at fixed (x_t, t), so guiding either one and converting
        # gives the same result -- and guiding the model's own output keeps the
        # guidance term off the 1/sqrt(1-ab) amplifier.
        pred = e_c if not guided else e_u + cfg.cfg_scale * (e_c - e_u)

        # Clamping the predicted clean image is what keeps the first steps
        # usable: alpha_bar at t=T-1 is ~1e-9, so an unclamped x_0 estimate there
        # is divided by ~3e-5 and explodes.
        x0 = sched.read_x0(pred.float(), x, t, cfg.predict).clamp(-1.0, 1.0)

        ab, ab_prev = sched.ab(t).unsqueeze(-1), sched.ab(t_prev).unsqueeze(-1)
        # Epsilon consistent with the clamped x_0, which is what both samplers
        # below are written against.
        eps = (x - ab.sqrt() * x0) / (1 - ab).clamp(min=1e-8).sqrt()
        if cfg.sampler == "ddim":
            sigma = cfg.eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()
            sigma = torch.nan_to_num(sigma, nan=0.0)
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_xt
            if cfg.eta > 0 and k < cfg.frames - 1:
                x = x + sigma * torch.randn_like(x)
        else:
            alpha = (ab / ab_prev).clamp(max=1.0)
            beta = 1.0 - alpha
            mean = (x - beta / (1 - ab).clamp(min=1e-8).sqrt() * eps) / alpha.sqrt()
            x = mean
            if k < cfg.frames - 1:
                var = beta * (1 - ab_prev) / (1 - ab).clamp(min=1e-8)
                x = x + var.clamp(min=0).sqrt() * torch.randn_like(x)

        if progress:
            print(f"   step {k + 1:2d}/{cfg.frames}  t={int(t[0]):<4} "
                  f"|x| {x.abs().mean():.3f}", flush=True)

    model.train()
    return x.clamp(-1.0, 1.0)


# --------------------------------------------------------------------------- #
# Measurement                                                                  #
# --------------------------------------------------------------------------- #

class Scorer(nn.Module):
    """A small convnet used as a measuring instrument, never as part of the model.

    It is trained once on real images, cached, and then asked two questions of a
    batch of samples: does an image asked to be a `c` read as a `c` (conditioning
    fidelity), and how far is the distribution of its features from the
    distribution of real features.

    That second number is a Frechet distance in this classifier's feature space.
    It is not FID -- FID is defined against InceptionV3 pool3 features and is not
    comparable across feature extractors -- so it is reported under its own name
    and only ever compared between arms of the same sweep. Its parameters are
    excluded from every count this file prints.
    """

    FEATURES = 64

    def __init__(self, channels, side, classes):
        super().__init__()
        self.side, self.channels = side, channels
        self.body = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, self.FEATURES), nn.ReLU(),
        )
        self.head = nn.Linear(self.FEATURES, classes)

    def features(self, flat):
        x = flat.reshape(-1, self.channels, self.side, self.side)
        return self.body(x)

    def forward(self, flat):
        return self.head(self.features(flat))


def scorer_path(cfg):
    return os.path.join(CKPT_DIR, f"diffusion_scorer_{cfg.dataset}.pth")


def get_scorer(cfg, xtr, ytr, xva, yva):
    """Train the instrument once, then reuse it for every arm and every run."""
    c, side, classes = cfg.shape()
    net = Scorer(c, side, classes).to(cfg.device)
    path = scorer_path(cfg)

    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=cfg.device))
        net.eval()
        return net

    print("📏 training the scorer (once, then cached) ...", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    x, y = xtr.to(cfg.device), ytr.to(cfg.device)
    for epoch in range(3):
        perm = torch.randperm(len(x), device=cfg.device)
        for i in range(0, len(x) - 255, 256):
            idx = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(net(x[idx]), y[idx]).backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        acc = (net(xva.to(cfg.device)).argmax(1) == yva.to(cfg.device)).float().mean()
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save(net.state_dict(), path)
    print(f"📏 scorer ready — {acc * 100:.2f}% on held-out real images "
          f"({sum(p.numel() for p in net.parameters()):,} params, not counted "
          f"toward the model)", flush=True)
    return net


def _sym_sqrt(m):
    """Square root of a symmetric PSD matrix, via eigendecomposition.

    scipy.linalg.sqrtm is the usual route and scipy is not a dependency here;
    both covariance operands are symmetric PSD, so eigh is exact and cheaper.
    """
    vals, vecs = torch.linalg.eigh(m.double())
    return (vecs * vals.clamp(min=0).sqrt()) @ vecs.T


def frechet_distance(f_real, f_fake):
    f_real, f_fake = f_real.double(), f_fake.double()
    mu_r, mu_f = f_real.mean(0), f_fake.mean(0)
    cov_r = torch.cov(f_real.T)
    cov_f = torch.cov(f_fake.T)
    root = _sym_sqrt(cov_r)
    cross = _sym_sqrt(root @ cov_f @ root)
    return float((mu_r - mu_f).pow(2).sum() + torch.trace(cov_r + cov_f - 2 * cross))


@torch.no_grad()
def measure_samples(model, sched, cfg, scorer, x_real, count, carry=True):
    """Generate `count` images with known labels and score them."""
    _, _, classes = cfg.shape()
    imgs, wanted = [], []
    made = 0
    while made < count:
        n = min(cfg.sample_batch, count - made)
        labels = torch.arange(made, made + n, device=cfg.device) % classes
        imgs.append(sample(model, sched, cfg, labels, carry=carry))
        wanted.append(labels)
        made += n

    imgs, wanted = torch.cat(imgs), torch.cat(wanted)
    logits = scorer(imgs)
    fidelity = (logits.argmax(1) == wanted).float().mean().item()
    confidence = logits.softmax(1).max(1).values.mean().item()
    fd = frechet_distance(scorer.features(x_real[:len(imgs)].to(cfg.device)),
                          scorer.features(imgs))
    return {"fidelity": fidelity, "confidence": confidence, "frechet": fd,
            "images": imgs}


def save_grid(images, cfg, path, rows=None):
    """Write a sample grid. Lazily imported so a headless run never needs it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, side, classes = cfg.shape()
    n = len(images)
    rows = rows or max(1, n // classes)
    cols = math.ceil(n / rows)
    grid = ((images.reshape(n, c, side, side).float().cpu() + 1.0) / 2.0).clamp(0, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9))
    for i, ax in enumerate(list(axes.flat) if n > 1 else [axes]):
        ax.axis("off")
        if i < n:
            ax.imshow(grid[i, 0], cmap="gray") if c == 1 else \
                ax.imshow(grid[i].permute(1, 2, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Advisories — what a setting costs, before it costs it                        #
# --------------------------------------------------------------------------- #

def memory_advisory(cfg, model):
    steps = cfg.steps()
    if model.hebb_type is not None:
        paths = 2 if model.hebb_type == "both" else 1
        row_gb = (26 if paths == 1 else 35) * steps * cfg.neurons * 4 / 1e9
        trace_gb = row_gb * cfg.batch
        print(f"🧬 plasticity {model.hebb_type}/{model.hebb_res} | {steps} steps "
              f"x batch {cfg.batch} x {cfg.neurons} neurons | trace kept for "
              f"backward {trace_gb:.2f} GB")
        if trace_gb > 2.0:
            print(f"⚠️  that trace alone is {trace_gb:.1f} GB. It is linear in the "
                  f"batch and in the step count — try --batch "
                  f"{max(1, int(2.0 / row_gb))}, fewer --frames, or --grad-ckpt.")

    if model.attn is not None:
        writes = steps if cfg.attn_write == "step" else cfg.frames
        gb = model.attn.training_cache_bytes(cfg.batch, writes) / 1e9
        print(f"👁️  attention {model.attn.heads}x{model.attn.head_dim} "
              f"(kv {model.attn.kv_heads}, window {model.attn.window}) | "
              f"write {cfg.attn_write} / read {cfg.attn_read} | "
              f"{writes} writes per call | cache kept for backward {gb:.2f} GB")

    if cfg.carry == "independent" and (model.attn is not None
                                       or model.hebb_type is not None):
        print("ℹ️  --carry independent resets state, cache and plastic buffers "
              "every frame, so attention and plasticity have nothing to carry. "
              "That is the control arm working as intended.")


def describe(cfg, model):
    c, side, classes = cfg.shape()
    print(f"\n🧠 {model.get_num_params():,} trainable params | {cfg.neurons} neurons "
          f"| in {cfg.n_in} / out {cfg.n_out} | {cfg.frames} frames x {cfg.echo} echo "
          f"= {cfg.steps()} steps | batch {cfg.batch} | "
          f"lr {'auto' if cfg.lr is None else cfg.lr}")
    print(f"   {cfg.dataset} {c}x{side}x{side} = {cfg.pixels()} px | feature width "
          f"{cfg.feature_width()} | predict {cfg.predict} | carry {cfg.carry} | "
          f"noise {cfg.traj_noise}"
          + (f" | hebb {cfg.hebb_type}/{cfg.hebb_res}" if cfg.hebb_type else "")
          + (f" | attn {cfg.attn_heads}h" if cfg.attn_heads else ""))
    memory_advisory(cfg, model)


# --------------------------------------------------------------------------- #
# Checkpoints                                                                  #
# --------------------------------------------------------------------------- #

ARCH_FIELDS = ("dataset", "neurons", "n_in", "n_out", "t_embed",
               "predict", "activation", "weight_init", "gates",
               "hebb_type", "hebb_res", "attn_heads", "attn_kv_heads",
               "attn_head_dim", "attn_window", "attn_write", "attn_read",
               "attn_rope", "attn_qk_norm")

# The denoising grid. None of these fixes a tensor shape -- they set how many
# frames are visited and how long the schedule is -- so a resume may change
# them, and they follow the checkpoint only when the command line stays quiet.
GRID_FIELDS = ("frames", "echo", "timesteps")


def ckpt_paths(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    stem = os.path.join(CKPT_DIR, f"diffusion_odyss_{cfg.tag}")
    return stem + "_latest.pth", stem + "_best.pth"


def guard_overwrite(cfg, overwrite, resume):
    latest, _ = ckpt_paths(cfg)
    if os.path.exists(latest) and not (overwrite or resume):
        raise SystemExit(
            f"\n✋ Refusing to overwrite {os.path.basename(latest)}.\n"
            f"   --resume     continue that run\n"
            f"   --tag NAME   start a separate one\n"
            f"   --overwrite  discard it and start over\n")


def adopt_saved_arch(cfg, path, grid_from_cli=()):
    """Re-adopt the architecture a checkpoint was trained with.

    `ARCH_FIELDS` fixes tensor shapes, so those always come from the
    checkpoint -- the state dict would not load otherwise. `GRID_FIELDS` does
    not, and a field the command line named explicitly stays as the caller
    asked, so a resume can move the denoising grid without retraining.
    """
    if not os.path.exists(path):
        raise SystemExit(f"\n✋ No checkpoint at {path}.\n"
                         f"   Train one first, or pass a different --tag.\n")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # save_checkpoint merges extra_data into the top level rather than nesting it.
    saved = payload.get("cfg") or {}
    adopt = ARCH_FIELDS + tuple(f for f in GRID_FIELDS if f not in grid_from_cli)
    changed = {f: saved[f] for f in adopt
               if f in saved and saved[f] != getattr(cfg, f)}
    for f in ("activation", "weight_init", "gates"):
        if f in changed:
            changed[f] = tuple(changed[f])
    if changed:
        print(f"🔧 adopting saved architecture: "
              f"{', '.join(f'{k}={v}' for k, v in changed.items())}")
    return replace(cfg, **changed)


# --------------------------------------------------------------------------- #
# Training session                                                             #
# --------------------------------------------------------------------------- #

def _save(path, model, trainer, cfg, step, val, best):
    payload = asdict(cfg)
    for f in ("activation", "weight_init", "gates"):
        payload[f] = list(payload[f])
    save_checkpoint(model, trainer.optimizer, step, val, path,
                    extra_data={"cfg": payload, "step": step, "best_val": best},
                    trainer_state=trainer.state_dict())


def run_session(cfg, data, quiet=False, eval_every=200, log_every=50,
                sample_every=0, resume=None, measure_at_end=True):
    """Train until the budget runs out. Returns the run's metrics."""
    xtr, ytr, xva, yva, scorer = data
    set_seed(cfg.seed)

    model, trainer = build(cfg)
    sched = Schedule(cfg, cfg.device)
    validator = Validator(xva, yva, sched, cfg)
    batches = Batches(xtr, ytr, cfg.batch, cfg.device, cfg.seed)

    step, best = 0, float("inf")
    if resume and os.path.exists(resume):
        # `lr` is a param-group key, so loading the optimizer state would
        # otherwise put the checkpoint's rate back over the one asked for.
        # `cfg.lr` is None under zero-config, which leaves the estimate alone.
        info = load_checkpoint(model, trainer.optimizer, resume,
                               device=cfg.device, trainer=trainer, lr=cfg.lr)
        step, best = info.get("step", 0), info.get("best_val", float("inf"))
        print(f"📂 resumed {os.path.basename(resume)} at step {step:,} "
              f"(best val {best:.4f})")

    if not quiet:
        describe(cfg, model)
        print()

    latest, best_path = ckpt_paths(cfg)
    deadline = time.time() + cfg.minutes * 60 if cfg.minutes > 0 else None
    started, window, val = time.time(), [], float("nan")
    interrupted = False

    try:
        while True:
            if deadline and time.time() > deadline:
                break
            if cfg.max_steps and step >= cfg.max_steps:
                break

            x0, labels = batches.next()
            frames, target = trajectory_batch(x0, labels, sched, cfg)
            window.append(train_step(trainer, frames, target, cfg))
            step += 1

            if log_every and step % log_every == 0 and not quiet:
                avg = sum(window) / len(window)
                window = []
                el = time.time() - started
                print(f"step {step:>7,} | loss {avg:6.4f} | "
                      f"lr {trainer._current_lr():.2e} | "
                      f"{step * cfg.batch / max(el, 1e-6):7,.0f} img/s | "
                      f"epoch {batches.epochs} | {el / 60:5.1f}m", flush=True)

            if eval_every and step % eval_every == 0:
                val = validator.score(model)
                if not quiet:
                    mark = "  🏆" if val < best else ""
                    print(f"   ↳ VAL  {cfg.predict}-MSE {val:.4f}  "
                          f"({val / validator.trivial:5.1%} of trivial "
                          f"{validator.trivial:.3f}){mark}", flush=True)
                if val < best:
                    best = val
                    _save(best_path, model, trainer, cfg, step, val, best)
                if sample_every and (step // eval_every) % sample_every == 0:
                    _, _, classes = cfg.shape()
                    labels = torch.arange(2 * classes, device=cfg.device) % classes
                    imgs = sample(model, sched, cfg, labels)
                    path = save_grid(imgs, cfg,
                                     os.path.join(OUT_DIR, cfg.tag,
                                                  f"step_{step:07d}.png"), rows=2)
                    print(f"   ↳ 🖼️  {os.path.relpath(path, os.getcwd())}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️  interrupted — saving", flush=True)

    if math.isnan(val):
        val = validator.score(model)
        best = min(best, val)
    _save(latest, model, trainer, cfg, step, val, best)

    metrics = {"step": step, "val_mse": val, "best_val": best,
               "trivial": validator.trivial,
               "minutes": (time.time() - started) / 60, "epochs": batches.epochs,
               "params": model.get_num_params(), "interrupted": interrupted}

    if measure_at_end and scorer is not None:
        # A Frechet distance over 64 features needs more than a hundred samples
        # before its covariance means anything; the grid only shows the first
        # hundred of them.
        got = measure_samples(model, sched, cfg, scorer, xva, MEASURE_SAMPLES)
        metrics.update({k: got[k] for k in ("fidelity", "confidence", "frechet")})
        metrics["sample_grid"] = save_grid(
            got["images"][:100], cfg,
            os.path.join(OUT_DIR, cfg.tag, "final.png"), rows=10)

    return model, trainer, sched, metrics


# --------------------------------------------------------------------------- #
# Sweeps — the arms that decide what this file is allowed to claim             #
# --------------------------------------------------------------------------- #

SWEEPS = {
    # Does carrying the trajectory help? `independent` is the matched control:
    # same frames, same targets, same gradient budget, no memory. The rest add
    # one mechanism at a time on top of `trajectory`, because a win has to be
    # attributable to something.
    "memory": {
        "independent":        {"carry": "independent"},
        "trajectory":         {"carry": "trajectory"},
        "traj_noise_shared":  {"carry": "trajectory", "traj_noise": "shared"},
        "traj_attn":          {"carry": "trajectory", "attn_heads": 4},
        "traj_hebb_temporal": {"carry": "trajectory", "hebb_type": "temporal"},
        "traj_hebb_spatial":  {"carry": "trajectory", "hebb_type": "spatial"},
        "traj_hebb_both":     {"carry": "trajectory", "hebb_type": "both"},
        "traj_full":          {"carry": "trajectory", "attn_heads": 4,
                               "hebb_type": "temporal"},
    },
    # At a fixed step budget, is temporal depth worth more than denoising
    # resolution? No other architecture can be asked this.
    "depth": {
        "k32_e2": {"frames": 32, "echo": 2},
        "k16_e4": {"frames": 16, "echo": 4},
        "k8_e8":  {"frames": 8, "echo": 8},
        "k4_e16": {"frames": 4, "echo": 16},
    },
    "size": {
        "n256": {"neurons": 256, "n_in": 96, "n_out": 96},
        "n384": {"neurons": 384, "n_in": 144, "n_out": 144},
        "n512": {"neurons": 512, "n_in": 192, "n_out": 192},
        "n768": {"neurons": 768, "n_in": 288, "n_out": 288},
        # The same widths under epsilon, where the answer is white noise. Their
        # losses sit just above 1 - n_out/P and fall with n_out exactly as that
        # bound does -- 0.900, 0.852, 0.806, 0.710 at 1200 steps against bounds
        # of 0.878, 0.816, 0.755, 0.633 -- while every arm samples at chance.
        # That is the rank argument measured rather than asserted.
        "n256_eps": {"neurons": 256, "n_in": 96, "n_out": 96, "predict": "eps"},
        "n384_eps": {"neurons": 384, "n_in": 144, "n_out": 144, "predict": "eps"},
        "n512_eps": {"neurons": 512, "n_in": 192, "n_out": 192, "predict": "eps"},
        "n768_eps": {"neurons": 768, "n_in": 288, "n_out": 288, "predict": "eps"},
    },
    "predict": {"x0": {"predict": "x0"}, "eps": {"predict": "eps"},
                "v": {"predict": "v"}},
}

RANK_KEY = "frechet"


def run_sweep(cfg, data, name, arms=None, minutes=3.0):
    grid = SWEEPS[name]
    chosen = arms or list(grid)
    unknown = [a for a in chosen if a not in grid]
    if unknown:
        raise SystemExit(f"\n✋ Unknown arm(s) for --sweep {name}: "
                         f"{', '.join(unknown)}\n   Available: "
                         f"{', '.join(grid)}\n")

    print(f"\n{'=' * 92}")
    # Which budget actually binds decides what the table means. Equal wall clock
    # asks which arm is the better use of a GPU-minute; equal steps asks which
    # mechanism is better per gradient. An arm carrying a plastic trace runs an
    # order of magnitude fewer steps per minute, so the two questions have
    # different answers and both are worth having.
    bound = (f"{cfg.max_steps:,} steps each, equal gradient budget"
             if cfg.max_steps else f"{minutes:g} min each, equal wall clock")
    print(f"🔬 SWEEP {name} — {len(chosen)} arms x {bound}")
    print(f"{'=' * 92}")

    results = {}
    for i, arm in enumerate(chosen, 1):
        arm_cfg = replace(cfg, minutes=minutes, tag=f"sweep_{name}_{arm}",
                          **grid[arm])
        print(f"\n🏁 [{i}/{len(chosen)}] {arm} — "
              f"{', '.join(f'{k}={v}' for k, v in grid[arm].items())}")
        _, _, _, metrics = run_session(arm_cfg, data, quiet=True,
                                       eval_every=200, log_every=0)
        results[arm] = metrics
        print(f"   val {metrics['val_mse']:.4f} | fidelity "
              f"{metrics.get('fidelity', float('nan')) * 100:5.1f}% | frechet "
              f"{metrics.get('frechet', float('nan')):7.3f} | "
              f"{metrics['step']:,} steps | {metrics['params']:,} params")

    path = os.path.join(CKPT_DIR, f"sweep_diffusion_{name}_results.json")
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"sweep": name, "minutes": minutes,
                   "max_steps": cfg.max_steps,
                   "base": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in asdict(cfg).items()},
                   "results": results}, fh, indent=2)

    order = sorted(results, key=lambda a: results[a].get(RANK_KEY, float("inf")))
    print(f"\n{'=' * 92}")
    print(f"{'arm':<22}{'val MSE':>10}{'fidelity':>11}{'confidence':>12}"
          f"{'frechet':>10}{'steps':>10}{'params':>11}")
    print(f"{'-' * 92}")
    for arm in order:
        m = results[arm]
        print(f"{arm:<22}{m['val_mse']:>10.4f}"
              f"{m.get('fidelity', float('nan')) * 100:>10.1f}%"
              f"{m.get('confidence', float('nan')):>12.3f}"
              f"{m.get('frechet', float('nan')):>10.3f}"
              f"{m['step']:>10,}{m['params']:>11,}")
    print(f"{'=' * 92}")
    print(f"🥇 {order[0]} leads on {RANK_KEY} (lower is better)")
    print(f"📄 {path}")
    return results


# --------------------------------------------------------------------------- #
# Smoke — the self-test, since test_all.py only globs convergence_*.py         #
# --------------------------------------------------------------------------- #

def run_smoke(cfg, data):
    print(f"\n{'=' * 78}")
    print("🔥 SMOKE TEST")
    print(f"{'=' * 78}")

    base = replace(cfg, neurons=128, n_in=48, n_out=48, frames=4, echo=2,
                   batch=32, max_steps=150, minutes=0.0, tag="smoke")
    sched = Schedule(base, base.device)
    failures = []

    def check(name, ok, detail=""):
        print(f"{'✅' if ok else '❌'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    # 1. The shape contract the whole design rests on: K frames in, K
    #    predictions out, with E echo steps of temporal depth between them.
    set_seed(base.seed)
    model, trainer = build(base)
    x0, labels = data[0][:8].to(base.device), data[1][:8].to(base.device)
    frames, target = trajectory_batch(x0, labels, sched, base)
    out, _ = model(frames, steps=base.steps())
    check("frame contract", tuple(out.shape) == (8, base.frames, base.pixels()),
          f"{tuple(frames.shape)} -> {tuple(out.shape)} over {base.steps()} steps")

    # 2. Learning at all, measured against the do-nothing predictor on the same
    #    frozen grid. A run that cannot beat outputting zero has learned nothing.
    for variant, over in (("plain", {}), ("attention", {"attn_heads": 2}),
                          ("plastic", {"hebb_type": "temporal"}),
                          ("independent", {"carry": "independent"}),
                          ("eps-pred", {"predict": "eps"})):
        arm = replace(base, **over)
        set_seed(arm.seed)
        model, trainer = build(arm)
        val = Validator(data[2][:256], data[3][:256], sched, arm)
        batches = Batches(data[0][:4096], data[1][:4096], arm.batch,
                          arm.device, arm.seed)
        for _ in range(arm.max_steps):
            bx, by = batches.next()
            f, t = trajectory_batch(bx, by, sched, arm)
            loss = train_step(trainer, f, t, arm)
        score = val.score(model)
        check(f"learns ({variant})", score < val.trivial,
              f"val {score:.4f} < trivial {val.trivial:.3f}")

    # 3. A rollout produces finite images, guided and unguided, with and
    #    without the trajectory memory.
    # The default is guided, so the arm that differs is the unguided one --
    # `cfg_scale=1.0` takes the branch where the conditional pass is the answer.
    for label, over, carry in (("ddim", {}, True), ("ddim no-carry", {}, False),
                               ("ddpm", {"sampler": "ddpm"}, True),
                               ("unguided", {"cfg_scale": 1.0}, True)):
        arm = replace(base, **over)
        imgs = sample(model, sched, arm, torch.arange(10, device=arm.device))
        check(f"samples ({label})",
              imgs.shape == (10, arm.pixels()) and torch.isfinite(imgs).all().item(),
              f"{tuple(imgs.shape)} in [{imgs.min():.2f}, {imgs.max():.2f}]")

    # 4. Checkpoint round-trip through the library's own functions.
    latest, _ = ckpt_paths(base)
    _save(latest, model, trainer, base, 1, 0.5, 0.5)
    set_seed(base.seed + 1)
    fresh, fresh_trainer = build(base)
    load_checkpoint(fresh, fresh_trainer.optimizer, latest, device=base.device,
                    trainer=fresh_trainer)
    same = torch.allclose(fresh.W, model.W) and torch.allclose(
        fresh.output_decoder.weight, model.output_decoder.weight)
    check("checkpoint round-trip", same, os.path.basename(latest))

    # 5. The invariant the whole library is built on.
    check("W diagonal pinned to zero",
          float(model.W.diagonal().abs().max()) == 0.0)

    print(f"{'=' * 78}")
    if failures:
        print(f"💥 {len(failures)} failed: {', '.join(failures)}")
        return False
    print("🎉 all checks passed")
    return True


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

EPILOG = """
examples:

  first thing to run -- the self-test, a couple of minutes on CPU, no GPU needed
    python -u experiment_diffusion.py --mode smoke

  train the default configuration and write a sample grid every 2nd validation
    python -u experiment_diffusion.py --mode train --minutes 15 --sample-every 2

  keep going where that left off, or fork a second run under its own name
    python -u experiment_diffusion.py --mode train --minutes 15 --resume
    python -u experiment_diffusion.py --mode train --minutes 15 --tag wide --neurons 768 --n-in 288 --n-out 288

  attention is optional. It measured roughly a wash on MNIST -- slightly better
  Frechet, slightly worse conditioning, +57% parameters -- so it is not the
  default. Reach for it when the data is harder than MNIST:
    python -u experiment_diffusion.py --mode train --tag attn --attn-heads 4 --minutes 20

  generate from a trained tag; the guidance scale is the usual quality/variety dial
    python -u experiment_diffusion.py --mode sample --tag base --cfg 3.0
    python -u experiment_diffusion.py --mode sample --tag base --cfg 1.0 --sampler ddpm
    python -u experiment_diffusion.py --mode sample --tag base --frames 32 --eta 0.2

  score a checkpoint: conditioning fidelity, Frechet, sampling-batch sensitivity,
  and what the trajectory memory is worth at sampling time
    python -u experiment_diffusion.py --mode eval --tag base

  the measurements this file's claims rest on. Equal wall clock asks which arm
  is the better use of a GPU-minute; equal steps asks which mechanism is better
  per gradient. A plastic arm runs an order of magnitude fewer steps per minute,
  so the two questions have different answers and both are worth having:
    python -u experiment_diffusion.py --mode sweep --sweep memory --minutes 4
    python -u experiment_diffusion.py --mode sweep --sweep memory --max-steps 600 --minutes 25

  narrow a sweep to the arms you care about
    python -u experiment_diffusion.py --mode sweep --sweep memory --arms independent,trajectory --minutes 5

  the other three grids: is temporal depth worth more than denoising resolution
  at a fixed step budget, how the model scales, and why x_0 is the default
    python -u experiment_diffusion.py --mode sweep --sweep depth --minutes 4
    python -u experiment_diffusion.py --mode sweep --sweep size --minutes 4
    python -u experiment_diffusion.py --mode sweep --sweep predict --minutes 4

  plasticity, which pays for itself in memory and step rate -- the advisory
  prints the cost before it is paid
    python -u experiment_diffusion.py --mode train --tag plastic --hebb temporal --batch 32

  the harder tier. CIFAR-10 is 3072 pixels, so the projections and the core both
  grow; expect colourful structure, not photographs, at this parameter count
    python -u experiment_diffusion.py --dataset cifar10 --neurons 768 --n-in 288 --n-out 288 --minutes 30

  running out of VRAM: shrink the batch first, then the frame count, then turn on
  checkpointing -- the retained trace is linear in all three
    python -u experiment_diffusion.py --mode train --batch 16 --frames 8 --grad-ckpt

  a byte-for-byte reproducible curve needs ChaosGrad's fixed-rate mode
    python -u experiment_diffusion.py --mode train --lr 1e-3 --max-steps 2000 --seed 123
"""


def parse_args():
    d = Cfg()
    p = argparse.ArgumentParser(
        prog="experiment_diffusion.py", description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("mode")
    g.add_argument("--mode", default="train",
                   choices=["train", "sample", "sweep", "smoke", "eval"])
    g.add_argument("--sweep", default="memory", choices=sorted(SWEEPS))
    g.add_argument("--arms", default=None, metavar="A,B",
                   help="comma-separated subset of the sweep's arms")

    g = p.add_argument_group("data")
    g.add_argument("--dataset", default=d.dataset, choices=sorted(DATASETS))
    g.add_argument("--train-images", type=int, default=d.train_images,
                   help="-1 = all (default: %(default)s)")
    g.add_argument("--val-images", type=int, default=d.val_images)

    g = p.add_argument_group("diffusion")
    # These three carry no tensor shape, so a resume is free to change them.
    # Left unset they follow the checkpoint, which is why the default is a
    # sentinel rather than the value.
    g.add_argument("--timesteps", type=int, default=None,
                   help=f"T, the continuous schedule (default: {d.timesteps})")
    g.add_argument("--frames", type=int, default=None,
                   help=f"K, denoising steps visited (default: {d.frames})")
    g.add_argument("--echo", type=int, default=None,
                   help=f"E, echo steps per denoising step (default: {d.echo})")
    g.add_argument("--predict", default=d.predict, choices=["x0", "eps", "v"])
    g.add_argument("--carry", default=d.carry,
                   choices=["trajectory", "independent"],
                   help="trajectory keeps state/cache/trace across frames; "
                        "independent is the matched memoryless control")
    g.add_argument("--traj-noise", default=d.traj_noise, choices=["iid", "shared"])
    g.add_argument("--class-dropout", type=float, default=d.class_dropout)

    g = p.add_argument_group("architecture")
    g.add_argument("--neurons", type=int, default=d.neurons)
    g.add_argument("--n-in", type=int, default=d.n_in)
    g.add_argument("--n-out", type=int, default=d.n_out)
    g.add_argument("--t-embed", type=int, default=d.t_embed)
    g.add_argument("--activation", default=",".join(d.activation),
                   help="ENC,CORE,MEM (default: %(default)s)")
    g.add_argument("--weight-init", default=",".join(d.weight_init),
                   help="ENC,CORE,MEM,GATE (default: %(default)s)")
    g.add_argument("--gates", default=",".join(d.gates),
                   help="IN,CORE,MEM (default: %(default)s)")
    g.add_argument("--hebb", default="none",
                   choices=["none", "temporal", "spatial", "both"])
    g.add_argument("--hebb-res", default=d.hebb_res, choices=["global", "neuron"])
    g.add_argument("--dropout", type=float, default=d.dropout)

    g = p.add_argument_group("attention")
    g.add_argument("--attn-heads", type=int, default=d.attn_heads,
                   help="0 builds no attention module (default: %(default)s)")
    g.add_argument("--attn-kv-heads", type=int, default=d.attn_kv_heads)
    g.add_argument("--attn-head-dim", type=int, default=d.attn_head_dim,
                   help="0 derives it from the neuron count")
    g.add_argument("--attn-window", type=int, default=d.attn_window)
    g.add_argument("--attn-write", default=d.attn_write, choices=["token", "step"],
                   help="token = one cache entry per denoising step")
    g.add_argument("--attn-read", default=d.attn_read, choices=["token", "step"])
    g.add_argument("--attn-rope", action=argparse.BooleanOptionalAction,
                   default=d.attn_rope)
    g.add_argument("--attn-qk-norm", action=argparse.BooleanOptionalAction,
                   default=d.attn_qk_norm)
    g.add_argument("--attn-dropout", type=float, default=d.attn_dropout)

    g = p.add_argument_group("optimization")
    g.add_argument("--batch", type=int, default=d.batch)
    g.add_argument("--lr", default="auto",
                   help="auto = zero-config ChaosGrad; a float pins fixed-rate "
                        "mode (default: %(default)s)")
    g.add_argument("--grad-ckpt", action="store_true")
    g.add_argument("--compile", action="store_true")

    g = p.add_argument_group("sampling")
    g.add_argument("--sampler", default=d.sampler, choices=["ddim", "ddpm"])
    g.add_argument("--eta", type=float, default=d.eta)
    g.add_argument("--cfg", type=float, default=d.cfg_scale, dest="cfg_scale",
                   help="classifier-free guidance scale; 1.0 disables it")
    g.add_argument("--sample-batch", type=int, default=d.sample_batch)
    g.add_argument("--sample-count", type=int, default=MEASURE_SAMPLES)

    g = p.add_argument_group("run control")
    g.add_argument("--minutes", type=float, default=d.minutes,
                   help="0 = until Ctrl-C (default: %(default)s)")
    g.add_argument("--max-steps", type=int, default=d.max_steps)
    g.add_argument("--tag", default=d.tag)
    g.add_argument("--resume", action="store_true")
    g.add_argument("--resume-best", action="store_true")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--device", default=d.device)

    g = p.add_argument_group("logging")
    g.add_argument("--eval-every", type=int, default=200)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--sample-every", type=int, default=0,
                   help="grid every N validations; 0 = never (default: %(default)s)")

    a = p.parse_args()
    if a.resume_best:
        a.resume = True

    # Which grid fields the command line actually named, before the sentinels
    # are resolved -- `adopt_saved_arch` leaves exactly these alone.
    a.grid_from_cli = {f for f in GRID_FIELDS if getattr(a, f) is not None}
    for f in GRID_FIELDS:
        if getattr(a, f) is None:
            setattr(a, f, getattr(d, f))

    # Everything that can be rejected is rejected here, before the images are
    # loaded and before CUDA is initialised.
    for name in ("neurons", "n_in", "n_out", "frames", "echo", "timesteps",
                 "batch", "t_embed", "val_images", "sample_batch"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if a.n_in + a.n_out > a.neurons:
        p.error(f"--n-in + --n-out ({a.n_in + a.n_out}) exceeds "
                f"--neurons ({a.neurons})")
    if a.t_embed % 2:
        p.error("--t-embed must be even")
    if a.frames > a.timesteps:
        p.error("--frames cannot exceed --timesteps")
    if not 0.0 <= a.class_dropout < 1.0:
        p.error("--class-dropout must be in [0, 1)")
    if a.attn_heads and a.attn_heads % a.attn_kv_heads:
        p.error("--attn-kv-heads must divide --attn-heads")
    for name, parts in (("activation", 3), ("weight-init", 4), ("gates", 3)):
        got = getattr(a, name.replace("-", "_")).split(",")
        if len(got) != parts:
            p.error(f"--{name} needs {parts} comma-separated entries, got {len(got)}")
    if str(a.lr) != "auto":
        try:
            float(a.lr)
        except ValueError:
            p.error("--lr must be 'auto' or a float")
    if a.attn_heads == 0 and any(
            getattr(a, f) != getattr(d, f)
            for f in ("attn_kv_heads", "attn_head_dim", "attn_window")):
        print("ℹ️  attention flags ignored — --attn-heads is 0")
    return a


def cfg_from_args(a):
    return Cfg(
        dataset=a.dataset, train_images=a.train_images, val_images=a.val_images,
        timesteps=a.timesteps, frames=a.frames, echo=a.echo, predict=a.predict,
        carry=a.carry, traj_noise=a.traj_noise, class_dropout=a.class_dropout,
        neurons=a.neurons, n_in=a.n_in, n_out=a.n_out, t_embed=a.t_embed,
        activation=tuple(a.activation.split(",")),
        weight_init=tuple(a.weight_init.split(",")),
        gates=tuple(a.gates.split(",")),
        hebb_type="" if a.hebb == "none" else a.hebb, hebb_res=a.hebb_res,
        dropout=a.dropout,
        attn_heads=a.attn_heads, attn_kv_heads=a.attn_kv_heads,
        attn_head_dim=a.attn_head_dim, attn_window=a.attn_window,
        attn_write=a.attn_write, attn_read=a.attn_read, attn_rope=a.attn_rope,
        attn_qk_norm=a.attn_qk_norm, attn_dropout=a.attn_dropout,
        batch=a.batch, lr=None if str(a.lr) == "auto" else float(a.lr),
        grad_ckpt=a.grad_ckpt, compile=a.compile,
        sampler=a.sampler, eta=a.eta, cfg_scale=a.cfg_scale,
        sample_batch=a.sample_batch,
        minutes=a.minutes, max_steps=a.max_steps, tag=a.tag, seed=a.seed,
        device=a.device,
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    a = parse_args()
    cfg = cfg_from_args(a)
    set_seed(cfg.seed)

    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("🚀 OdyssNet-Diffusion — the denoiser remembers its own trajectory")
    print(f"   mode {a.mode} | {cfg.dataset} | device {cfg.device} | seed {cfg.seed}")

    latest, best = ckpt_paths(cfg)
    if a.mode in ("sample", "eval"):
        cfg = adopt_saved_arch(cfg, best if os.path.exists(best) else latest,
                               a.grid_from_cli)
    elif a.mode == "train":
        if a.resume:
            cfg = adopt_saved_arch(cfg, best if a.resume_best else latest,
                                   a.grid_from_cli)
        else:
            guard_overwrite(cfg, a.overwrite, a.resume)

    print(f"🔤 loading {cfg.dataset} ...", flush=True)
    xtr, ytr, xva, yva = load_images(cfg)
    print(f"💾 {len(xtr):,} train / {len(xva):,} val images, "
          f"{cfg.pixels()} px each")

    scorer = None if a.mode == "smoke" else get_scorer(cfg, xtr, ytr, xva, yva)
    data = (xtr, ytr, xva, yva, scorer)

    if a.mode == "smoke":
        sys.exit(0 if run_smoke(cfg, data) else 1)

    if a.mode == "sweep":
        arms = a.arms.split(",") if a.arms else None
        run_sweep(cfg, data, a.sweep, arms, a.minutes or 3.0)
        return

    if a.mode in ("sample", "eval"):
        set_seed(cfg.seed)
        model, trainer = build(cfg)
        path = best if os.path.exists(best) else latest
        load_checkpoint(model, trainer.optimizer, path, device=cfg.device,
                        trainer=trainer)
        print(f"📂 {os.path.basename(path)}")
        describe(cfg, model)
        sched = Schedule(cfg, cfg.device)

        got = measure_samples(model, sched, cfg, scorer, xva, a.sample_count)
        shown = min(len(got["images"]), 100)
        grid = save_grid(got["images"][:shown], cfg,
                         os.path.join(OUT_DIR, cfg.tag, f"{a.mode}.png"),
                         rows=max(1, shown // cfg.shape()[2]))
        print(f"\n📊 cfg {cfg.cfg_scale:g} | {cfg.sampler} | "
              f"conditioning fidelity {got['fidelity'] * 100:.1f}% | "
              f"confidence {got['confidence']:.3f} | "
              f"frechet {got['frechet']:.3f}")
        print(f"🖼️  {os.path.relpath(grid, os.getcwd())}")

        if a.mode == "eval":
            # The plastic buffer is a batch mean, so a batch generated together
            # shares one memory. Whether that matters is measured, not assumed.
            print("\n   sampling batch sensitivity (the buffer is a batch mean):")
            for sb in (10, 25, 50, 100):
                if sb > a.sample_count:
                    continue
                probe = measure_samples(model, sched, replace(cfg, sample_batch=sb),
                                        scorer, xva, a.sample_count)
                print(f"     batch {sb:>4} | fidelity {probe['fidelity'] * 100:5.1f}% "
                      f"| frechet {probe['frechet']:7.3f}")
            print("\n   trajectory memory at sampling time, same weights:")
            for label, carry in (("carried", True), ("wiped each step", False)):
                probe = measure_samples(model, sched, cfg, scorer, xva,
                                        a.sample_count, carry=carry)
                print(f"     {label:<16} | fidelity {probe['fidelity'] * 100:5.1f}% "
                      f"| frechet {probe['frechet']:7.3f}")
        return

    guard = latest if a.resume and not a.resume_best else (
        best if a.resume_best else None)
    model, trainer, sched, metrics = run_session(
        cfg, data, eval_every=a.eval_every, log_every=a.log_every,
        sample_every=a.sample_every, resume=guard)

    print(f"\n{'=' * 78}")
    print("📊 FINAL")
    print(f"{'=' * 78}")
    print(f"   {metrics['step']:,} steps over {metrics['minutes']:.1f} min "
          f"({metrics['epochs']} epochs) | {metrics['params']:,} params")
    print(f"   val {cfg.predict}-MSE {metrics['val_mse']:.4f} "
          f"(best {metrics['best_val']:.4f}, trivial {metrics['trivial']:.3f} — "
          f"{metrics['best_val'] / metrics['trivial']:.1%} of it)")
    if "fidelity" in metrics:
        print(f"   conditioning fidelity {metrics['fidelity'] * 100:.1f}% | "
              f"confidence {metrics['confidence']:.3f} | "
              f"frechet {metrics['frechet']:.3f}")
        print(f"🖼️  {os.path.relpath(metrics['sample_grid'], os.getcwd())}")


if __name__ == "__main__":
    main()
