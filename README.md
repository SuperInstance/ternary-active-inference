# ternary-active-inference

**Active Inference over ternary action spaces {-1, 0, +1}.**

## What problem does this solve?

Imagine a robot that can only push left, stay still, or push right — no continuous torques, no high-dimensional joint spaces, just three discrete motor commands. How should it choose among them? Karl Friston's active inference reframes control as *inference*: the agent doesn't "decide" what to do; it infers which action most minimizes its expected surprise. This crate implements the full perception-action loop — variational Bayesian state estimation, expected free energy computation, and precision-weighted softmax policy selection — for agents whose action space is the ternary set ℤ₃ = {-1, 0, +1}.

If you are building controllers for discrete ternary systems (ternary logic gates, three-position switches, or coarse-grained motor primitives) and want a principled, neurobiologically-motivated alternative to reinforcement learning, start here.

## Mathematical foundations

### Variational Free Energy

The agent maintains a generative model with likelihood P(o|s), transition P(s′|s,a), and prior P(s). Given an observation o, the posterior Q(s) is updated via Bayes' rule:

```
Q(s) ∝ P(o|s) P(s)
```

The variational free energy bounds the log-evidence:

```
F = E_Q[ln Q(s) - ln P(o,s)] = KL[Q(s) || P(s|o)] - ln P(o)
```

Minimizing F tightens the posterior and reduces surprise.

### Expected Free Energy of an Action

For each action a, the expected free energy G(a) trades off epistemic value (information gain) against pragmatic cost (deviation from prior preferences):

```
predicted(s′|a) = Σ_s Q(s) P(s′|s,a)

G(a) = -H[predicted(s′|a)] + KL[predicted(s′|a) || P(s)]
```

- **Epistemic term** `-H[·]`: negative entropy. High entropy means the action is exploratory — it leads to states about which the agent is maximally uncertain. Minimizing G therefore *prefers* informative actions.
- **Pragmatic term** `KL[·||·]`: divergence from the prior. If the prior encodes goal states, this term penalizes actions that drive the agent away from its preferences.

### Precision and Policy Selection

Actions are sampled from a softmax over precision-weighted negative free energy:

```
π(a) = softmax(-γ · G(a))
```

where γ is an inverse-temperature (precision) parameter. The precision itself adapts online:

```
γ ← 1 / E_π[|G(π)|]
```

This implements the *precision-engineering* step in active inference: when expected free energies are large and uncertain, the agent acts more stochastically; as evidence accumulates, it sharpens its policy.

### Shannon Entropy

```
H[p] = -Σ_i p_i ln p_i
```

Used in the epistemic value term. Entropy is maximized (≈1.0986 nats, or ≈1.585 bits for a ternary distribution) for the uniform distribution and zero for deterministic beliefs.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    GenerativeModel                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  P(o|s)     │  │ P(s′|s,a)   │  │      P(s)           │  │
│  │  likelihood │  │  transition │  │      prior          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │ observations
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                 VariationalBayes                              │
│          Q(s) ← normalize( P(o|s) · prior )                  │
└────────────────────┬────────────────────────────────────────┘
                     │ posterior
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ExpectedFreeEnergy                               │
│    G(a) = -H[ predicted(s′|a) ] + KL[ predicted || prior ]   │
└────────────────────┬────────────────────────────────────────┘
                     │ G(a) for a ∈ {-1,0,+1}
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PrecisionWeighting                               │
│              γ · G(a)  (adaptive precision)                  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               PolicySelection                                 │
│         π(a) = softmax(-γ · G(a)) → argmax or sample         │
└────────────────────┬────────────────────────────────────────┘
                     │ selected action ∈ {-1, 0, +1}
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PerceptionActionLoop                             │
│         perceive(obs) → act() → step(obs)                    │
└─────────────────────────────────────────────────────────────┘
```

## Getting Started

Add the crate to your project:

```bash
cargo add ternary-active-inference
```

Then run a minimal perception-action loop:

```rust
use ternary_active_inference::{GenerativeModel, PerceptionActionLoop};

fn main() {
    // 3 hidden states, 1 observation dimension
    let likelihood = vec![
        vec![[0.8, 0.1, 0.1]], // state 0 emits -1
        vec![[0.1, 0.8, 0.1]], // state 1 emits  0
        vec![[0.1, 0.1, 0.8]], // state 2 emits +1
    ];
    let transition = vec![
        vec![vec![0.7, 0.2, 0.1], vec![0.5, 0.3, 0.2], vec![0.3, 0.4, 0.3]],
        vec![vec![0.33, 0.34, 0.33]; 3], // neutral action
        vec![vec![0.1, 0.2, 0.7], vec![0.2, 0.3, 0.5], vec![0.1, 0.2, 0.7]],
    ];
    let prior = vec![1.0 / 3.0; 3];
    let model = GenerativeModel::new(3, 1, likelihood, transition, prior);

    let mut agent = PerceptionActionLoop::new(model, 1.0);

    // Observe -1, then choose an action
    let action = agent.step(&[-1i8]);
    println!("Belief entropy: {:.4}", agent.belief_entropy());
    println!("Selected action: {}", action);
}
```

Compile and run:

```bash
cargo run --example my_agent
```

## Running the Tests

The test suite exercises every conceptual component of the active inference loop:

```bash
cargo test
```

| Test | What it verifies |
|------|------------------|
| `test_generative_model_construction` | The generative model is instantiated with the correct state/action dimensions. |
| `test_vb_posterior_normalizes` | After Bayesian updating, the posterior Q(s) sums to 1 (proper probability distribution). |
| `test_vb_posterior_updates_correctly` | Observing -1 shifts the highest posterior mass onto state 0, as dictated by the likelihood. |
| `test_kl_divergence_identical` | KL[Q\|\|Q] = 0, confirming the divergence vanishes when distributions match. |
| `test_kl_divergence_nonneg` | KL[Q\|\|P] ≥ 0, a fundamental property of relative entropy. |
| `test_efe_returns_scalar` | `ExpectedFreeEnergy::compute` yields a finite scalar for every action. |
| `test_efe_all_actions_length` | The EFE vector has exactly 3 entries, one per ternary action. |
| `test_policy_distribution_sums_to_one` | The softmax policy π(a) is a normalized probability distribution. |
| `test_policy_selects_lowest_efe` | With high precision (low temperature), the greedy policy selects the action with minimal G(a). |
| `test_policy_select_ternary_in_range` | The selected action is always a valid ternary value (-1, 0, or +1). |
| `test_precision_weighting_scales` | `PrecisionWeighting` correctly multiplies the EFE vector by γ. |
| `test_precision_update` | Adaptive precision update produces a finite, positive γ. |
| `test_perception_action_loop_returns_ternary` | The full `step()` pipeline returns a valid ternary action after perception. |
| `test_belief_entropy_decreases_with_strong_evidence` | Repeated consistent observations reduce posterior entropy, reflecting growing certainty. |

## Related Crates

- [`ternary-free-energy`](https://crates.io/crates/ternary-free-energy) — Core FEP primitives: entropy, KL divergence, variational free energy, and surprise tracking for ternary distributions.
- [`ternary-belief`](https://crates.io/crates/ternary-belief) — Loopy belief propagation on ternary factor graphs; useful when your generative model has structured conditional independencies.
- [`ternary-inference`](https://crates.io/crates/ternary-inference) — General-purpose probabilistic inference routines for ternary state spaces.
- [`ternary-bayesian`](https://crates.io/crates/ternary-bayesian) — Bayesian networks and posterior updating specialized for ℤ₃ variables.

## License

MIT
