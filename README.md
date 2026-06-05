# ternary-active-inference

**Active Inference with ternary {-1, 0, +1} actions: perceive, infer, act, repeat.**

This crate implements Karl Friston's Active Inference framework where agents select actions from {-1, 0, +1} by minimizing Expected Free Energy. The three-way action space maps naturally to "do negative thing", "do nothing", and "do positive thing" — which covers most real-world control scenarios.

---

## The Core Idea

Active Inference says: an agent should act to minimize its *expected surprise* (technically, its variational free energy). This single principle unifies:

1. **Perception** — Update beliefs to match observations (minimize prediction error)
2. **Action** — Change the world to match predictions (minimize surprise)
3. **Learning** — Update the model to improve future predictions

With ternary actions, the Expected Free Energy (EFE) for each action {-1, 0, +1} is:

```
EFE(a) = -H[q(o|a)] + KL[q(s|o,a) || q(s)]
         ────────────   ─────────────────────
         seek novelty    stay close to prior
```

The action with lowest EFE wins. The 0 action (do nothing) is preferred when both novelty-seeking and prior-maintenance are balanced — this is mathematically identical to thermostat deadband behavior.

---

## Architecture

```
         ┌──────────────────────────────────────┐
         │         GenerativeModel               │
         │  states → observations → likelihood   │
         └────────────┬─────────────────────────┘
                      │
         ┌────────────▼─────────────────────────┐
         │      VariationalBayes                  │
         │  observation → posterior over states   │
         └────────────┬─────────────────────────┘
                      │
         ┌────────────▼─────────────────────────┐
         │     ExpectedFreeEnergy                 │
         │  for each action a ∈ {-1, 0, +1}:     │
         │    compute EFE(a) = novelty + KL       │
         └────────────┬─────────────────────────┘
                      │
         ┌────────────▼─────────────────────────┐
         │      PolicySelection                   │
         │  argmin EFE(a) → chosen action         │
         └────────────┬─────────────────────────┘
                      │
         ┌────────────▼─────────────────────────┐
         │   PerceptionActionLoop                 │
         │  observe → infer → act → update        │
         └──────────────────────────────────────┘
```

### Key Types

- **`GenerativeModel`** — P(observation | state) with ternary-valued states
- **`VariationalBayes`** — Approximate Bayesian inference: update posterior from observations
- **`ExpectedFreeEnergy`** — Compute EFE for each ternary action
- **`PolicySelection`** — Choose action = argmin EFE
- **`PrecisionWeighting`** — Attention-like precision for different observation modalities
- **`PerceptionActionLoop`** — Full cycle: observe → infer → plan → act → update

---

## Quick Start

```rust
use ternary_active_inference::{PerceptionActionLoop, GenerativeModel};

let mut agent = PerceptionActionLoop::new(GenerativeModel::default());

// Agent observes ternary signal, infers state, selects action
let observation = 1; // +1 observation
let action = agent.step(observation);
// action ∈ {-1, 0, +1}
```

---

## Why Ternary Actions?

Most real decisions have three options:

| Domain | -1 | 0 | +1 |
|--------|----|---|-----|
| Trading | Sell | Hold | Buy |
| Thermostat | Cool | Idle | Heat |
| Steering | Left | Straight | Right |
| Review | Reject | Abstain | Accept |

The 0 action ("do nothing") is the most important — it's the agent's resting state. Active inference naturally spends most time in the 0 state because EFE is lowest when observations match predictions. Action is triggered only when predictions are wrong enough to justify the energy cost.

---

## Ecosystem

- **ternary-free-energy** — Lower-level FEP computations: entropy, KL divergence, surprise
- **ternary-belief** — Belief propagation for inference on ternary factor graphs
- **ternary-thermostat** — Concrete thermostat implementation using ternary PID
- **ternary-pid** — PID controller with ternary output

## License

MIT
