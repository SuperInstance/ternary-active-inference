//! Active Inference over ternary action spaces {-1, 0, +1}.

pub type TernaryVal = i8;
pub const TERNARY_VALS: [TernaryVal; 3] = [-1, 0, 1];

fn ternary_idx(v: TernaryVal) -> usize {
    (v + 1) as usize
}

fn normalize(v: &mut Vec<f64>) {
    let sum: f64 = v.iter().sum();
    if sum > 1e-12 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

fn softmax(v: &[f64], beta: f64) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = v.iter().map(|&x| (beta * x - beta * max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Encodes P(o|s), P(s'|s,a), P(s).
pub struct GenerativeModel {
    pub n_states: usize,
    pub n_obs_dims: usize,
    /// likelihood[s][obs_dim] = [P(o=-1|s), P(o=0|s), P(o=+1|s)]
    pub likelihood: Vec<Vec<[f64; 3]>>,
    /// transition[action_idx][from_state] = distribution over next states
    pub transition: Vec<Vec<Vec<f64>>>,
    /// prior P(s)
    pub prior: Vec<f64>,
    /// always 3 for ternary
    pub n_actions: usize,
}

impl GenerativeModel {
    pub fn new(
        n_states: usize,
        n_obs_dims: usize,
        likelihood: Vec<Vec<[f64; 3]>>,
        transition: Vec<Vec<Vec<f64>>>,
        prior: Vec<f64>,
    ) -> Self {
        assert_eq!(likelihood.len(), n_states);
        assert_eq!(transition.len(), 3, "need 3 ternary actions");
        assert_eq!(prior.len(), n_states);
        Self {
            n_states,
            n_obs_dims,
            likelihood,
            transition,
            prior,
            n_actions: 3,
        }
    }

    pub fn uniform(n_states: usize, n_obs_dims: usize) -> Self {
        let likelihood = vec![vec![[1.0 / 3.0; 3]; n_obs_dims]; n_states];
        let trans_row = vec![1.0 / n_states as f64; n_states];
        let transition = vec![vec![trans_row.clone(); n_states]; 3];
        let prior = vec![1.0 / n_states as f64; n_states];
        Self::new(n_states, n_obs_dims, likelihood, transition, prior)
    }
}

/// Updates posterior Q(s) given observations via Bayesian inference.
pub struct VariationalBayes;

impl VariationalBayes {
    pub fn update_posterior(
        &self,
        model: &GenerativeModel,
        prior: &[f64],
        obs: &[TernaryVal],
    ) -> Vec<f64> {
        let mut posterior = prior.to_vec();
        for (dim, &o) in obs.iter().enumerate() {
            let oi = ternary_idx(o);
            for s in 0..model.n_states {
                posterior[s] *= model.likelihood[s][dim][oi];
            }
        }
        normalize(&mut posterior);
        posterior
    }

    /// KL divergence KL[Q||P] for distributions over states.
    pub fn kl_divergence(q: &[f64], p: &[f64]) -> f64 {
        q.iter()
            .zip(p.iter())
            .filter(|(&qi, &pi)| qi > 1e-12 && pi > 1e-12)
            .map(|(&qi, &pi)| qi * (qi / pi).ln())
            .sum()
    }
}

/// Computes Expected Free Energy G(a) for each ternary action.
pub struct ExpectedFreeEnergy;

impl ExpectedFreeEnergy {
    /// G(a) = epistemic_value + pragmatic_cost
    /// epistemic: entropy of predicted next-state distribution
    /// pragmatic: KL[Q(s_next|a) || P(s)]
    pub fn compute(
        &self,
        model: &GenerativeModel,
        posterior: &[f64],
        action_idx: usize,
    ) -> f64 {
        let trans = &model.transition[action_idx];
        // Predicted next state: sum_s Q(s) * P(s'|s,a)
        let mut predicted = vec![0.0f64; model.n_states];
        for s in 0..model.n_states {
            for s_next in 0..model.n_states {
                predicted[s_next] += posterior[s] * trans[s][s_next];
            }
        }

        // Epistemic value: entropy H(predicted)
        let entropy: f64 = predicted
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum();

        // Pragmatic cost: KL[predicted || prior]
        let kl = VariationalBayes::kl_divergence(&predicted, &model.prior);

        // G = -epistemic + pragmatic (higher entropy reduces G, KL increases G)
        -entropy + kl
    }

    pub fn all_actions(&self, model: &GenerativeModel, posterior: &[f64]) -> Vec<f64> {
        (0..model.n_actions)
            .map(|a| self.compute(model, posterior, a))
            .collect()
    }
}

/// Selects policy (action) via softmax over -G(a).
pub struct PolicySelection {
    pub precision: f64,
}

impl PolicySelection {
    pub fn new(precision: f64) -> Self {
        Self { precision }
    }

    pub fn action_distribution(&self, efe: &[f64]) -> Vec<f64> {
        // softmax(-precision * G)
        softmax(&efe.iter().map(|&g| -g).collect::<Vec<_>>(), self.precision)
    }

    pub fn select_action(&self, efe: &[f64]) -> usize {
        let probs = self.action_distribution(efe);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    pub fn select_ternary(&self, efe: &[f64]) -> TernaryVal {
        TERNARY_VALS[self.select_action(efe)]
    }
}

/// Scales EFE by a precision (inverse-temperature) parameter γ.
pub struct PrecisionWeighting {
    pub gamma: f64,
}

impl PrecisionWeighting {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    pub fn weighted_efe(&self, efe: &[f64]) -> Vec<f64> {
        efe.iter().map(|&g| self.gamma * g).collect()
    }

    /// Update γ = 1 / E_π[|G(π)|] based on expected free energy under current policy.
    pub fn update_precision(&mut self, efe: &[f64], action_probs: &[f64]) {
        let expected_g: f64 = efe
            .iter()
            .zip(action_probs.iter())
            .map(|(&g, &p)| g.abs() * p)
            .sum();
        if expected_g > 1e-10 {
            self.gamma = 1.0 / expected_g;
        }
    }
}

/// Iterates perception (VB update) and action (policy selection).
pub struct PerceptionActionLoop {
    pub model: GenerativeModel,
    pub vb: VariationalBayes,
    pub efe: ExpectedFreeEnergy,
    pub policy: PolicySelection,
    pub precision: PrecisionWeighting,
    pub beliefs: Vec<f64>,
}

impl PerceptionActionLoop {
    pub fn new(model: GenerativeModel, precision_gamma: f64) -> Self {
        let n_states = model.n_states;
        let beliefs = model.prior.clone();
        Self {
            model,
            vb: VariationalBayes,
            efe: ExpectedFreeEnergy,
            policy: PolicySelection::new(precision_gamma),
            precision: PrecisionWeighting::new(precision_gamma),
            beliefs: if beliefs.is_empty() {
                vec![1.0 / n_states as f64; n_states]
            } else {
                beliefs
            },
        }
    }

    pub fn perceive(&mut self, obs: &[TernaryVal]) {
        self.beliefs = self.vb.update_posterior(&self.model, &self.beliefs, obs);
    }

    pub fn act(&mut self) -> TernaryVal {
        let efe_vals = self.efe.all_actions(&self.model, &self.beliefs);
        let weighted = self.precision.weighted_efe(&efe_vals);
        let action_probs = self.policy.action_distribution(&weighted);
        self.precision.update_precision(&efe_vals, &action_probs);
        self.policy.select_ternary(&weighted)
    }

    pub fn step(&mut self, obs: &[TernaryVal]) -> TernaryVal {
        self.perceive(obs);
        self.act()
    }

    pub fn belief_entropy(&self) -> f64 {
        self.beliefs
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.ln())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_model() -> GenerativeModel {
        let n_states = 3;
        let n_obs_dims = 1;
        // Each state prefers the corresponding ternary observation
        let likelihood = vec![
            vec![[0.8, 0.1, 0.1]],
            vec![[0.1, 0.8, 0.1]],
            vec![[0.1, 0.1, 0.8]],
        ];
        // Action 0 (=-1): tends toward state 0; action 1 (=0): stays; action 2 (=+1): state 2
        let t0 = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.5, 0.3, 0.2],
            vec![0.3, 0.4, 0.3],
        ];
        let t1 = vec![
            vec![0.33, 0.34, 0.33],
            vec![0.33, 0.34, 0.33],
            vec![0.33, 0.34, 0.33],
        ];
        let t2 = vec![
            vec![0.1, 0.2, 0.7],
            vec![0.2, 0.3, 0.5],
            vec![0.1, 0.2, 0.7],
        ];
        let prior = vec![1.0 / 3.0; 3];
        GenerativeModel::new(n_states, n_obs_dims, likelihood, vec![t0, t1, t2], prior)
    }

    #[test]
    fn test_generative_model_construction() {
        let m = simple_model();
        assert_eq!(m.n_states, 3);
        assert_eq!(m.n_actions, 3);
    }

    #[test]
    fn test_vb_posterior_normalizes() {
        let m = simple_model();
        let vb = VariationalBayes;
        let prior = vec![1.0 / 3.0; 3];
        let obs = vec![0i8]; // zero observation
        let post = vb.update_posterior(&m, &prior, &obs);
        let sum: f64 = post.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vb_posterior_updates_correctly() {
        let m = simple_model();
        let vb = VariationalBayes;
        let prior = vec![1.0 / 3.0; 3];
        // Observe -1 → state 0 should have highest posterior
        let post = vb.update_posterior(&m, &prior, &[-1i8]);
        assert!(post[0] > post[1]);
        assert!(post[0] > post[2]);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let p = vec![0.3, 0.4, 0.3];
        let kl = VariationalBayes::kl_divergence(&p, &p);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_nonneg() {
        let q = vec![0.5, 0.3, 0.2];
        let p = vec![0.2, 0.5, 0.3];
        let kl = VariationalBayes::kl_divergence(&q, &p);
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_efe_returns_scalar() {
        let m = simple_model();
        let efe_calc = ExpectedFreeEnergy;
        let posterior = vec![1.0 / 3.0; 3];
        let g = efe_calc.compute(&m, &posterior, 0);
        assert!(g.is_finite());
    }

    #[test]
    fn test_efe_all_actions_length() {
        let m = simple_model();
        let efe_calc = ExpectedFreeEnergy;
        let posterior = vec![1.0 / 3.0; 3];
        let gs = efe_calc.all_actions(&m, &posterior);
        assert_eq!(gs.len(), 3);
    }

    #[test]
    fn test_policy_distribution_sums_to_one() {
        let pol = PolicySelection::new(1.0);
        let efe = vec![0.5, 0.2, 0.8];
        let dist = pol.action_distribution(&efe);
        let sum: f64 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_policy_selects_lowest_efe() {
        let pol = PolicySelection::new(10.0); // high precision → greedy
        let efe = vec![1.0, 0.1, 0.5]; // action 1 has lowest EFE
        assert_eq!(pol.select_action(&efe), 1);
    }

    #[test]
    fn test_policy_select_ternary_in_range() {
        let pol = PolicySelection::new(1.0);
        let efe = vec![0.5, 0.2, 0.8];
        let a = pol.select_ternary(&efe);
        assert!(a == -1 || a == 0 || a == 1);
    }

    #[test]
    fn test_precision_weighting_scales() {
        let pw = PrecisionWeighting::new(2.0);
        let efe = vec![1.0, 2.0, 3.0];
        let w = pw.weighted_efe(&efe);
        assert!((w[0] - 2.0).abs() < 1e-10);
        assert!((w[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_update() {
        let mut pw = PrecisionWeighting::new(1.0);
        let efe = vec![1.0, 2.0, 3.0];
        let probs = vec![0.5, 0.3, 0.2];
        pw.update_precision(&efe, &probs);
        assert!(pw.gamma.is_finite() && pw.gamma > 0.0);
    }

    #[test]
    fn test_perception_action_loop_returns_ternary() {
        let m = simple_model();
        let mut pal = PerceptionActionLoop::new(m, 1.0);
        let action = pal.step(&[-1i8]);
        assert!(action == -1 || action == 0 || action == 1);
    }

    #[test]
    fn test_belief_entropy_decreases_with_strong_evidence() {
        let m = simple_model();
        let mut pal = PerceptionActionLoop::new(m, 1.0);
        let h0 = pal.belief_entropy();
        // Repeatedly observe state-0 signal
        for _ in 0..5 {
            pal.perceive(&[-1i8]);
        }
        let h1 = pal.belief_entropy();
        assert!(h1 < h0 + 1e-6); // entropy should decrease or stay
    }
}
