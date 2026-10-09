# TD3 Agent for BipedalWalker (`rldurham/Walker`)

A reinforcement learning coursework project implementing **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** from scratch in PyTorch to solve the `rldurham/Walker` continuous-control environment, in both its **standard (easy)** and **hardcore** variants.

Each variant has its own self-contained Jupyter notebook.

| Notebook | Environment | Episodes |
|---|---|---|
| `zzjw57-agent-code.ipynb` | `rldurham/Walker` (standard) | 1,000 |
| `zzjw57-agent-code-hardcore.ipynb` | `rldurham/Walker` with `hardcore=True` | 2,000 |

---

## Algorithm

The agent follows Fujimoto et al. (2018), *[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)*, with the [official implementation](https://github.com/sfujim/TD3) used as a reference for architecture and hyperparameters.

The three core TD3 ingredients are all implemented:

- **Clipped double-Q learning:** twin critics, with the TD target built from the minimum of the two target critics.
- **Delayed policy updates:** the actor and target networks are updated once every 2 critic updates.
- **Target policy smoothing:** clipped Gaussian noise is added to the target actor's action when computing the TD target.

### Implementation overview

- **Replay buffer:** a `deque`-based buffer (capacity 500k) with uniform random sampling.
- **Actor:** MLP with three 256-unit hidden layers, `LayerNorm` + ReLU, and a `tanh` output to match the `[-1, 1]` action bounds.
- **Critic:** two independent Q-networks, optimised jointly.
- **Agent:** handles action selection with exploration noise, the update step, and episode rollouts.
- **Training schedule:** a random-action warm-up phase, then gradient updates run after every episode, with step-decayed learning rates.

---

## Changes beyond vanilla TD3

Both notebooks add some stabilisation tweaks on top of the original algorithm.

**Shared by both notebooks**

- `LayerNorm` in all actor and critic hidden layers.
- Gradient-norm clipping (max norm 1.0) for actor and critic.
- Linearly decaying exploration noise.
- Gradient updates are batched after each episode (200 updates per episode) rather than after every environment step.
- Bootstrapping through time-limit truncation: only true termination zeroes the bootstrap term.
- Reward clipping at a minimum of -10 (`gym.wrappers.ClipReward`), plus asymmetric reward scaling (×0.5 for rewards ≤ -10, ×3 otherwise).
- Step-wise learning-rate decay during training.

**Differences between the two notebooks**

| | Easy | Hardcore |
|---|---|---|
| Critic size | 512-512-256 | 256-256-256 |
| Critic loss | MSE | Huber (smooth L1) |
| Actor loss | `-Q1` | `-Q1`, switching to **TD3+BC** (normalised Q + behavioural-cloning term) after 500k steps |
| Warm-up steps | 5,000 | 15,000 |
| Exploration noise | 0.2 → 0.01 over 400k steps | 0.2 → 0.005 over 1M steps |
| LR schedule (3e-4 start) | 1e-4 at episode 700, 3e-5 at 900 | 1e-4 at episode 1500, 3e-5 at 1800, 1e-5 at 1900 |
| Training length | 1,000 episodes | 2,000 episodes |

### Key hyperparameters (shared)

| Parameter | Value |
|---|---|
| Discount `gamma` | 0.99 |
| Soft-update rate `tau` | 0.005 |
| Batch size | 256 |
| Replay buffer size | 500,000 |
| Target policy noise / clip | 0.2 / 0.5 |
| Policy delay | 2 |
| Optimiser | Adam, lr 3e-4 |
| Random seed | 42 |

---

## Getting started

### Requirements

- Python 3.9+
- `swig` (needed to build Box2D)
- `rldurham`, which pulls in `gymnasium`, `box2d`, `pygame`, `moviepy` and others
- PyTorch and NumPy

### Installation

The first cell of each notebook installs the dependencies:

```bash
pip install swig
pip install rldurham
```

The code is set to run on **CPU only** (`CUDA_VISIBLE_DEVICES=""`), since the Walker environment is not compute-bound.

### Running

1. Open a notebook in Jupyter, JupyterLab, or Google Colab.
2. Run all cells in order. Training starts in the final cell.
3. Progress is printed every 10 episodes along with a live reward plot (`rld.InfoTracker`).

---

## Outputs

Running a notebook produces:

| Output | Easy | Hardcore |
|---|---|---|
| Training videos (recorded every 100 episodes) | `videos/` | `videos-hardcore/` |
| Training log | `logs/zzjw57-agent-log.txt` | `logs-hardcore/zzjw57-agent-log-hardcore.txt` |

---

## Repository structure

```
.
├── zzjw57-agent-code.ipynb            # TD3 agent, standard Walker
├── zzjw57-agent-code-hardcore.ipynb   # TD3 agent, hardcore Walker
├── videos/                            # (generated) training videos, easy
├── videos-hardcore/                   # (generated) training videos, hardcore
├── logs/                              # (generated) training log, easy
├── logs-hardcore/                     # (generated) training log, hardcore
└── README.md
```

---

## References

- S. Fujimoto, H. van Hoof, D. Meger. *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- Official TD3 implementation: [sfujim/TD3](https://github.com/sfujim/TD3)
- S. Fujimoto, S. Gu. *A Minimalist Approach to Offline Reinforcement Learning (TD3+BC).* NeurIPS 2021. [arXiv:2106.06860](https://arxiv.org/abs/2106.06860)
- [Gymnasium](https://gymnasium.farama.org/) and `rldurham`

---
