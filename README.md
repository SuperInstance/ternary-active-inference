# ternary-active-inference

Active Inference over ternary action spaces `{-1, 0, +1}` for the SuperInstance ecosystem.

## Components

- `GenerativeModel` — encodes P(o|s), P(s'|s,a), P(s)
- `VariationalBayes` — Bayesian posterior update
- `ExpectedFreeEnergy` — EFE G(a) for ternary action selection
- `PolicySelection` — softmax policy over –G
- `PrecisionWeighting` — γ precision scaling of EFE
- `PerceptionActionLoop` — full perception–action cycle

## Usage

```rust
use ternary_active_inference::{GenerativeModel, PerceptionActionLoop};

let model = GenerativeModel::uniform(4, 2);
let mut loop_ = PerceptionActionLoop::new(model, 1.0);
let action = loop_.step(&[-1, 0]); // returns -1, 0, or +1
```
