//! Active Inference over ternary (−1, 0, +1) action and observation spaces.
//!
//! Implements the free-energy minimisation framework where agents select actions
//! by minimising Expected Free Energy (EFE) across a three-valued signal space.

/// Map ternary value {-1, 0, +1} to array index {0, 1, 2}.
fn obs_to_idx(v: i8) -> usize {
    match v {
        -1 => 0,
        1  => 2,
        _  => 1,
    }
}

/// Map array index {0, 1, 2} to ternary value {-1, 0, +1}.
fn idx_to_ternary(i: usize) -> i8 {
    match i {
        0 => -1,
        2 => 1,
        _ => 0,
    }
}

fn vec_normalize(v: &mut Vec<f64>) {
    let s: f64 = v.iter().sum();
    if s > 1e-12 {
        for x in v.iter_mut() { *x /= s; }
    } else {
        let n = v.len() as f64;
        for x in v.iter_mut() { *x = 1.0 / n; }
    }
}

fn vec_entropy(dist: &[f64]) -> f64 {
    dist.iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.ln())
        .sum()
}

// ─── GenerativeModel ────────────────────────────────────────────────────────

/// Encodes beliefs about how states generate observations and how actions drive transitions.
///
/// - `prior`: distribution over n_states
/// - `likelihood`: n_states rows, each `[p(obs=-1|s), p(obs=0|s), p(obs=+1|s)]`
/// - `transitions`: 3 matrices (one per ternary action), each n_states × n_states
pub struct GenerativeModel {
    pub prior: Vec<f64>,
    pub likelihood: Vec<[f64; 3]>,
    pub transitions: Vec<Vec<Vec<f64>>>,
}

impl GenerativeModel {
    /// Uniform initialisation over `n_states`.
    pub fn new(n_states: usize) -> Self {
        let n = n_states.max(1);
        let p = 1.0 / n as f64;
        let uniform_row = vec![p; n];
        let mat = vec![uniform_row; n];
        Self {
            prior: vec![p; n],
            likelihood: vec![[1.0 / 3.0; 3]; n],
            transitions: vec![mat.clone(), mat.clone(), mat],
        }
    }

    /// Predicted observation distribution: marginalise likelihood over prior.
    pub fn predict_observation(&self) -> [f64; 3] {
        let mut pred = [0.0f64; 3];
        for (s, &ps) in self.prior.iter().enumerate() {
            if s < self.likelihood.len() {
                for o in 0..3 {
                    pred[o] += ps * self.likelihood[s][o];
                }
            }
        }
        let s: f64 = pred.iter().sum();
        if s > 1e-12 { for p in &mut pred { *p /= s; } }
        pred
    }

    /// Return a new model whose prior is the result of taking `action`.
    pub fn transition(&self, action: i8) -> GenerativeModel {
        let aidx = obs_to_idx(action);
        let n = self.prior.len();
        let mut new_prior = vec![0.0f64; n];
        let mat = &self.transitions[aidx];
        for (from, &ps) in self.prior.iter().enumerate() {
            if from < mat.len() {
                for (to, &t) in mat[from].iter().enumerate() {
                    if to < n { new_prior[to] += ps * t; }
                }
            }
        }
        vec_normalize(&mut new_prior);
        GenerativeModel {
            prior: new_prior,
            likelihood: self.likelihood.clone(),
            transitions: self.transitions.clone(),
        }
    }
}

// ─── VariationalBayes ───────────────────────────────────────────────────────

/// Posterior update via Bayes rule: q(s) ∝ p(obs | s) × prior(s).
pub struct VariationalBayes;

impl VariationalBayes {
    /// Return the per-state likelihood column for a given observation.
    pub fn likelihood_col(model: &GenerativeModel, obs: i8) -> Vec<f64> {
        let oidx = obs_to_idx(obs);
        model.likelihood.iter().map(|row| row[oidx]).collect()
    }

    /// Bayesian update: posterior ∝ likelihood_col × prior.
    pub fn update(prior: &[f64], likelihood_col: &[f64]) -> Vec<f64> {
        let mut posterior: Vec<f64> = prior.iter()
            .zip(likelihood_col.iter())
            .map(|(&p, &l)| p * l)
            .collect();
        vec_normalize(&mut posterior);
        posterior
    }
}

// ─── ExpectedFreeEnergy ─────────────────────────────────────────────────────

/// EFE(a) = ambiguity + risk.
///
/// ambiguity = H\[p(o|a)\] — entropy of predicted observations after action a
/// risk      = KL(p(o|a) ‖ C) — divergence from preferred observations C
pub struct ExpectedFreeEnergy;

impl ExpectedFreeEnergy {
    pub fn compute(model: &GenerativeModel, action: i8, preferences: &[f64]) -> f64 {
        let future = model.transition(action);
        let pred = future.predict_observation();

        let ambiguity = vec_entropy(&pred);

        // Normalise preferences to a distribution over {-1,0,+1}
        let psum: f64 = preferences.iter().take(3).sum();
        let mut risk = 0.0f64;
        for i in 0..3_usize.min(preferences.len()) {
            let po = pred[i];
            let c = if psum > 1e-12 { preferences[i] / psum } else { 1.0 / 3.0 };
            if po > 1e-12 && c > 1e-12 {
                risk += po * (po / c).ln();
            }
        }
        ambiguity + risk
    }
}

// ─── PolicySelection ────────────────────────────────────────────────────────

/// Select the ternary action {-1, 0, +1} that minimises EFE.
pub struct PolicySelection;

impl PolicySelection {
    pub fn select(model: &GenerativeModel, preferences: &[f64]) -> i8 {
        let mut best_action = 0i8;
        let mut best_efe = f64::INFINITY;
        for &a in &[-1i8, 0, 1] {
            let efe = ExpectedFreeEnergy::compute(model, a, preferences);
            if efe < best_efe {
                best_efe = efe;
                best_action = a;
            }
        }
        best_action
    }
}

// ─── PerceptionActionLoop ───────────────────────────────────────────────────

/// Closed-loop cycle: observe → infer → act → update.
pub struct PerceptionActionLoop {
    pub model: GenerativeModel,
    pub preferences: Vec<f64>,
    pub beliefs: Vec<f64>,
}

impl PerceptionActionLoop {
    pub fn new(model: GenerativeModel, preferences: Vec<f64>) -> Self {
        let beliefs = model.prior.clone();
        Self { model, preferences, beliefs }
    }

    /// Run one cycle given `observation`. Returns the chosen action.
    pub fn step(&mut self, observation: i8) -> i8 {
        // Infer
        let lik = VariationalBayes::likelihood_col(&self.model, observation);
        self.beliefs = VariationalBayes::update(&self.beliefs, &lik);
        self.model.prior = self.beliefs.clone();

        // Act
        let action = PolicySelection::select(&self.model, &self.preferences);

        // Update
        self.model = self.model.transition(action);
        self.beliefs = self.model.prior.clone();

        action
    }
}

// ─── PrecisionWeighting ─────────────────────────────────────────────────────

/// Attention-like precision scaling over ternary modalities.
///
/// Higher precision → beliefs from that modality weighted more heavily.
/// Precision decays toward zero as prediction error grows.
pub struct PrecisionWeighting {
    pub weights: Vec<f64>,
}

impl PrecisionWeighting {
    pub fn new(n_modalities: usize) -> Self {
        Self { weights: vec![1.0; n_modalities] }
    }

    /// Scale beliefs by precision weights and renormalise.
    pub fn weight_beliefs(&self, beliefs: &[f64]) -> Vec<f64> {
        let n = beliefs.len().min(self.weights.len());
        let mut out: Vec<f64> = (0..n).map(|i| beliefs[i] * self.weights[i]).collect();
        vec_normalize(&mut out);
        out
    }

    /// Exponential moving-average update: large error → precision shrinks.
    pub fn update_precision(&mut self, prediction_error: f64, modality: usize) {
        if modality < self.weights.len() {
            let new_val = 1.0 / (1.0 + prediction_error.abs());
            self.weights[modality] = 0.9 * self.weights[modality] + 0.1 * new_val;
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }
    fn sums_to_one(v: &[f64]) -> bool { close(v.iter().sum::<f64>(), 1.0) }

    #[test]
    fn test_generative_model_uniform_prior() {
        let m = GenerativeModel::new(4);
        assert_eq!(m.prior.len(), 4);
        assert!(sums_to_one(&m.prior));
        assert!(close(m.prior[0], 0.25));
    }

    #[test]
    fn test_generative_model_single_state() {
        let m = GenerativeModel::new(1);
        assert!(sums_to_one(&m.prior));
        let pred = m.predict_observation();
        assert!(close(pred.iter().sum::<f64>(), 1.0));
    }

    #[test]
    fn test_predict_observation_uniform_likelihood() {
        let m = GenerativeModel::new(3);
        let pred = m.predict_observation();
        assert!(close(pred[0], 1.0 / 3.0));
        assert!(close(pred[1], 1.0 / 3.0));
        assert!(close(pred[2], 1.0 / 3.0));
    }

    #[test]
    fn test_generative_model_transition_preserves_normalisation() {
        let m = GenerativeModel::new(3);
        let m2 = m.transition(-1);
        assert!(sums_to_one(&m2.prior));
        let m3 = m2.transition(1);
        assert!(sums_to_one(&m3.prior));
        let m4 = m3.transition(0);
        assert!(sums_to_one(&m4.prior));
    }

    #[test]
    fn test_variational_bayes_update_basic() {
        let prior = vec![0.5, 0.5];
        let lik = vec![0.9, 0.1];
        let post = VariationalBayes::update(&prior, &lik);
        assert!(sums_to_one(&post));
        assert!(post[0] > post[1]);
    }

    #[test]
    fn test_variational_bayes_update_zero_likelihood() {
        let prior = vec![0.5, 0.5];
        let lik = vec![0.0, 0.0];
        let post = VariationalBayes::update(&prior, &lik);
        assert!(sums_to_one(&post));
        assert!(!post[0].is_nan());
    }

    #[test]
    fn test_variational_bayes_likelihood_col() {
        let mut m = GenerativeModel::new(2);
        m.likelihood[0] = [0.8, 0.1, 0.1];
        m.likelihood[1] = [0.2, 0.7, 0.1];
        let col = VariationalBayes::likelihood_col(&m, -1);
        assert!(close(col[0], 0.8));
        assert!(close(col[1], 0.2));
    }

    #[test]
    fn test_expected_free_energy_finite_for_all_actions() {
        let m = GenerativeModel::new(2);
        let prefs = vec![0.7, 0.2, 0.1];
        for &a in &[-1i8, 0, 1] {
            let efe = ExpectedFreeEnergy::compute(&m, a, &prefs);
            assert!(efe.is_finite(), "EFE not finite for action {}", a);
            assert!(efe >= 0.0);
        }
    }

    #[test]
    fn test_expected_free_energy_uniform_prefs_equals_ambiguity() {
        // With uniform preferences risk → 0, EFE ≈ entropy of predicted obs
        let m = GenerativeModel::new(3);
        let prefs = vec![1.0 / 3.0; 3];
        let efe = ExpectedFreeEnergy::compute(&m, 0, &prefs);
        let expected_ambiguity = (3.0f64).ln(); // uniform obs dist → max entropy
        assert!(close(efe, expected_ambiguity));
    }

    #[test]
    fn test_policy_selection_returns_valid_action() {
        let m = GenerativeModel::new(3);
        let prefs = vec![0.6, 0.3, 0.1];
        let a = PolicySelection::select(&m, &prefs);
        assert!(a == -1 || a == 0 || a == 1);
    }

    #[test]
    fn test_perception_action_loop_valid_actions_over_time() {
        let m = GenerativeModel::new(2);
        let prefs = vec![0.5, 0.3, 0.2];
        let mut pal = PerceptionActionLoop::new(m, prefs);
        for obs in [-1i8, 0, 1, -1, 0, 1] {
            let a = pal.step(obs);
            assert!(a == -1 || a == 0 || a == 1, "invalid action {}", a);
        }
    }

    #[test]
    fn test_perception_action_loop_beliefs_stay_normalised() {
        let m = GenerativeModel::new(3);
        let prefs = vec![1.0 / 3.0; 3];
        let mut pal = PerceptionActionLoop::new(m, prefs);
        for obs in [-1i8, 1, 0, -1, 1, 0] {
            pal.step(obs);
            assert!(sums_to_one(&pal.beliefs), "beliefs not normalised");
        }
    }

    #[test]
    fn test_precision_weighting_scales_beliefs() {
        let mut pw = PrecisionWeighting::new(3);
        pw.weights = vec![2.0, 1.0, 0.5];
        let beliefs = vec![0.33, 0.33, 0.34];
        let w = pw.weight_beliefs(&beliefs);
        assert!(sums_to_one(&w));
        assert!(w[0] > w[2]);
    }

    #[test]
    fn test_precision_update_shrinks_on_large_error() {
        let mut pw = PrecisionWeighting::new(2);
        let initial = pw.weights[0];
        pw.update_precision(100.0, 0);
        assert!(pw.weights[0] < initial);
    }

    #[test]
    fn test_precision_weighting_empty_beliefs() {
        let pw = PrecisionWeighting::new(3);
        let w = pw.weight_beliefs(&[]);
        assert_eq!(w.len(), 0);
    }
}
