use nalgebra::DVector;
use crate::soul::geometry::{compute_nf, project_to_ball, SOUL_DIM};
use crate::soul::hyperbolic::{geodesic_distance, exp_map, log_map};
use crate::soul::manifold::StrobePhase;

pub const VFE_EQUILIBRIUM:     f64   = 0.05;
pub const MAX_CYCLES:          usize = 25;
pub const PLASTICITY_MOMENTUM: f64   = 0.99;

/// Prior uncertainty in Poincaré ball units.
/// High σ = loose prior — belief can move freely toward observations.
pub const SIGMA_PRIOR: f64 = 1.5;
/// Observation uncertainty — how much each query shifts belief.
/// Lower σ = high-precision observation = stronger update.
pub const SIGMA_OBS:   f64 = 1.0;

const DEFAULT_CURVATURE: f64 = crate::soul::geometry::INITIAL_CURVATURE;

// ── Belief state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BeliefState {
    pub position:        DVector<f64>,
    pub confidence:      f64,
    pub vfe:             f64,
    /// KL[Q(s|o) || P(s)] — complexity: distance from prior, precision-weighted
    pub complexity:      f64,
    /// E_Q[-log P(o|s)] — accuracy: prediction error, precision-weighted
    pub accuracy:        f64,
    /// Expected Free Energy — drives active inference policy selection
    pub efe:             f64,
    /// Information gain: how much this observation reduces uncertainty (exploration)
    pub epistemic_value: f64,
    /// Preference satisfaction: how well outcome matches prior (exploitation)
    pub pragmatic_value: f64,
    /// Current posterior uncertainty — tightens as belief converges
    pub sigma:           f64,
    pub cycle:           usize,
}

impl BeliefState {
    pub fn new(soul: &DVector<f64>) -> Self {
        let pos = project_to_ball(soul);
        Self {
            position:        pos,
            confidence:      0.0,
            vfe:             f64::MAX,
            complexity:      0.0,
            accuracy:        0.0,
            efe:             f64::MAX,
            epistemic_value: 0.0,
            pragmatic_value: 0.0,
            sigma:           SIGMA_PRIOR,
            cycle:           0,
        }
    }
}

// ── Core VFE ─────────────────────────────────────────────────────────────────

/// Convergence VFE — geodesic sum used by the minimisation loop.
/// Approximates VFE in the large-σ limit.
/// Kept unchanged so existing calibration and tests are preserved.
pub fn compute_vfe(
    belief:      &DVector<f64>,
    prior:       &DVector<f64>,
    observation: &DVector<f64>,
    curvature:   f64,
) -> f64 {
    geodesic_distance(belief, prior,       curvature)
  + geodesic_distance(belief, observation, curvature)
}

/// Proper VFE decomposed into complexity + accuracy with precision weighting.
///
///   VFE = KL[Q(s|o) || P(s)]           +  E_Q[-log P(o|s)]
///       = d(belief, prior)² / (2σP²)   +  d(belief, obs)² / (2σO²)
///       = complexity                   +  accuracy
///
/// Precision weighting (1/σ²) means:
///   - High-certainty priors (low σP) resist being overridden.
///   - High-certainty observations (low σO) pull belief strongly.
///
/// Returns (vfe, complexity, accuracy).
pub fn compute_vfe_components(
    belief:      &DVector<f64>,
    prior:       &DVector<f64>,
    observation: &DVector<f64>,
    sigma_prior: f64,
    sigma_obs:   f64,
    curvature:   f64,
) -> (f64, f64, f64) {
    let d_prior = geodesic_distance(belief, prior,       curvature);
    let d_obs   = geodesic_distance(belief, observation, curvature);

    let complexity = d_prior.powi(2) / (2.0 * sigma_prior.powi(2));
    let accuracy   = d_obs.powi(2)   / (2.0 * sigma_obs.powi(2));

    (complexity + accuracy, complexity, accuracy)
}

// ── Expected Free Energy ─────────────────────────────────────────────────────

/// Expected Free Energy — the quantity minimised in active inference.
///
///   EFE(π) = − epistemic_value − pragmatic_value
///
///   epistemic_value = log(σ_prior / σ_posterior)
///     Information gain: entropy reduction from acting under policy π.
///     High epistemic value → action reveals information (exploration).
///
///   pragmatic_value = − d(belief, preferred_obs)² / (2σP²)
///     Preference satisfaction: how well expected outcome matches goal.
///     High pragmatic value → action achieves preferred state (exploitation).
///
/// Minimising EFE balances exploration and exploitation — the same
/// objective that drives perception, action, and learning in Friston's
/// active inference framework.
pub fn compute_efe(
    belief:          &DVector<f64>,
    prior:           &DVector<f64>,
    preferred_obs:   &DVector<f64>,
    sigma_prior:     f64,
    sigma_posterior: f64,
    curvature:       f64,
) -> (f64, f64, f64) {
    // Epistemic: entropy reduction (always ≥ 0 when posterior is tighter)
    let epistemic = (sigma_prior / sigma_posterior.max(1e-10)).ln().max(0.0);

    // Pragmatic: negative squared distance to preferred outcome
    let d_pref    = geodesic_distance(belief, preferred_obs, curvature);
    let pragmatic = -(d_pref.powi(2) / (2.0 * sigma_prior.powi(2)));

    let efe = -epistemic - pragmatic;
    (efe, epistemic, pragmatic)
}

// ── Bayesian update on the Poincaré ball ─────────────────────────────────────

/// Kalman-like Bayesian belief update in hyperbolic space.
///
/// Euclidean Kalman filter:
///   K       = σP² / (σP² + σO²)            — Kalman gain
///   μ_post  = μ_prior + K(obs − μ_prior)   — weighted mean
///   σ_post² = σP² · σO² / (σP² + σO²)     — harmonic precision sum
///
/// Hyperbolic adaptation (replaces Euclidean arithmetic with Riemannian ops):
///   v       = log_{prior}(observation)     — obs in tangent space at prior
///   μ_post  = exp_{prior}(K · v)           — geodesic move by Kalman gain
///   σ_post  = sqrt(σP² · σO² / (σP² + σO²)) — exact for wrapped Gaussians
///
/// The posterior is the minimum-surprise belief: it is closer to the
/// observation when σO < σP (trust observations more), and closer to the
/// prior when σP < σO (trust prior more).
pub fn bayesian_update(
    prior:       &DVector<f64>,
    observation: &DVector<f64>,
    sigma_prior: f64,
    sigma_obs:   f64,
    curvature:   f64,
) -> (DVector<f64>, f64) {
    let k = sigma_prior.powi(2) / (sigma_prior.powi(2) + sigma_obs.powi(2));

    let v         = log_map(prior, observation, curvature);
    let posterior = exp_map(prior, &(v * k), curvature);

    let sigma_post = ((sigma_prior.powi(2) * sigma_obs.powi(2))
        / (sigma_prior.powi(2) + sigma_obs.powi(2))).sqrt();

    (posterior, sigma_post)
}

// ── Prediction error ─────────────────────────────────────────────────────────

/// Prediction error in the tangent space at `belief`.
/// This is the hyperbolic equivalent of (observation − belief projection).
pub fn prediction_error(
    belief:      &DVector<f64>,
    observation: &DVector<f64>,
    curvature:   f64,
) -> DVector<f64> {
    log_map(belief, observation, curvature)
}

// ── Precision-weighted gradient step ─────────────────────────────────────────

/// Riemannian gradient descent on VFE with precision weighting.
///
/// Gradient of VFE w.r.t. belief position in tangent space:
///   ∇VFE = −(1/σP²) · v_prior − (1/σO²) · v_obs
///
/// Gradient descent: μ_new = exp_μ(−η · ∇VFE)
///                         = exp_μ(η/σO² · v_obs + η/σP² · v_prior)
///
/// With SIGMA_OBS < SIGMA_PRIOR, observations pull more strongly than priors.
/// This is precision-weighted prediction error minimisation.
pub fn vfe_step(
    belief:      &DVector<f64>,
    prior:       &DVector<f64>,
    observation: &DVector<f64>,
    lr:          f64,
    curvature:   f64,
) -> DVector<f64> {
    let v_obs   = log_map(belief, observation, curvature);
    let v_prior = log_map(belief, prior,       curvature);

    let precision_obs   = 1.0 / SIGMA_OBS.powi(2);
    let precision_prior = 1.0 / SIGMA_PRIOR.powi(2);

    let tangent = v_obs * precision_obs + v_prior * precision_prior;
    exp_map(belief, &(tangent * lr), curvature)
}

// ── Minimisation loop ────────────────────────────────────────────────────────

pub fn minimise_vfe(
    soul:        &DVector<f64>,
    attractor:   &DVector<f64>,
    observation: &DVector<f64>,
    lr:          f64,
) -> (BeliefState, Vec<VFERecord>) {
    minimise_vfe_with_curvature(soul, attractor, observation, lr, DEFAULT_CURVATURE)
}

pub fn minimise_vfe_with_curvature(
    soul:        &DVector<f64>,
    attractor:   &DVector<f64>,
    observation: &DVector<f64>,
    lr:          f64,
    curvature:   f64,
) -> (BeliefState, Vec<VFERecord>) {
    let mut belief  = BeliefState::new(soul);
    let mut history = Vec::new();

    let attractor   = project_to_ball(attractor);
    let observation = project_to_ball(observation);

    // Posterior sigma — starts at SIGMA_PRIOR, tightens as belief converges
    let mut sigma = SIGMA_PRIOR;

    println!("\n[UnifiedOmniAGI] VFE minimisation starting...");

    for cycle in 0..MAX_CYCLES {
        belief.cycle = cycle;

        // ── Convergence criterion (unchanged) ────────────────────────────
        belief.vfe = compute_vfe(&belief.position, &attractor, &observation, curvature);

        // ── Proper decomposition: complexity + accuracy ──────────────────
        let (_, complexity, accuracy) = compute_vfe_components(
            &belief.position, &attractor, &observation,
            sigma, SIGMA_OBS, curvature,
        );
        belief.complexity = complexity;
        belief.accuracy   = accuracy;

        // ── Expected Free Energy ─────────────────────────────────────────
        // Compute expected posterior sigma from Bayesian update formula
        let sigma_post_expected = ((sigma.powi(2) * SIGMA_OBS.powi(2))
            / (sigma.powi(2) + SIGMA_OBS.powi(2))).sqrt();

        let (efe, epistemic, pragmatic) = compute_efe(
            &belief.position, &attractor, &observation,
            sigma, sigma_post_expected, curvature,
        );
        belief.efe             = efe;
        belief.epistemic_value = epistemic;
        belief.pragmatic_value = pragmatic;
        belief.sigma           = sigma;

        // ── Confidence and diagnostics ───────────────────────────────────
        let nf      = compute_nf(&belief.position);
        let pe      = prediction_error(&belief.position, &observation, curvature);
        let pe_norm = pe.norm();

        belief.confidence = (-belief.vfe / (VFE_EQUILIBRIUM * 20.0)).exp();

        let record = VFERecord {
            cycle,
            vfe:             belief.vfe,
            complexity,
            accuracy,
            efe,
            epistemic_value: epistemic,
            pragmatic_value: pragmatic,
            confidence:      belief.confidence,
            nf,
            pe_norm,
            sigma,
        };

        println!(
            "  [Cycle {:2}] VFE={:.6} C={:.4} A={:.4} EFE={:.4} conf={:.4} NF={:.4} PE={:.4}",
            cycle, belief.vfe, complexity, accuracy, efe,
            belief.confidence, nf, pe_norm
        );
        history.push(record);

        if belief.vfe <= VFE_EQUILIBRIUM {
            println!(
                "  [UnifiedOmniAGI] Equilibrium reached at cycle {}. VFE={:.6}",
                cycle, belief.vfe
            );
            break;
        }

        // ── Precision-weighted gradient step ─────────────────────────────
        let lr_cycle = lr * (1.0 - cycle as f64 / MAX_CYCLES as f64).max(0.1);
        belief.position = vfe_step(
            &belief.position, &attractor, &observation, lr_cycle, curvature,
        );

        // Sigma tightens as belief converges toward observation
        let vfe_factor = (belief.vfe / (VFE_EQUILIBRIUM * 10.0)).min(1.0);
        sigma = (sigma * (0.85 + 0.10 * vfe_factor)).max(0.01);
    }

    if belief.vfe > VFE_EQUILIBRIUM {
        println!(
            "  [UnifiedOmniAGI] Max cycles reached. Final VFE={:.6}",
            belief.vfe
        );
    }

    (belief, history)
}

// ── Supporting types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VFERecord {
    pub cycle:           usize,
    pub vfe:             f64,
    pub complexity:      f64,
    pub accuracy:        f64,
    pub efe:             f64,
    pub epistemic_value: f64,
    pub pragmatic_value: f64,
    pub confidence:      f64,
    pub nf:              f64,
    pub pe_norm:         f64,
    pub sigma:           f64,
}

pub fn phase_from_vfe(vfe: f64) -> StrobePhase {
    if vfe <= 0.03       { StrobePhase::Transcendent }
    else if vfe <= 0.1   { StrobePhase::Understanding }
    else if vfe <= 0.25  { StrobePhase::Engaged }
    else if vfe <= 0.5   { StrobePhase::Aware }
    else                 { StrobePhase::Dark }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ball_vec(seed: f64) -> DVector<f64> {
        let v: Vec<f64> = (0..SOUL_DIM)
            .map(|i| ((i as f64 + seed) * 1.7).sin() * 0.3)
            .collect();
        project_to_ball(&DVector::from_vec(v))
    }

    #[test]
    fn test_belief_state_inside_ball() {
        let belief = BeliefState::new(&ball_vec(1.0));
        assert!(belief.position.norm() < 1.0);
        assert!(belief.position.norm() > 0.0);
    }

    #[test]
    fn test_unit_sphere_input_clamped() {
        let v: Vec<f64> = (0..SOUL_DIM).map(|i| (i as f64 * 0.03).sin()).collect();
        let raw  = DVector::from_vec(v);
        let unit = &raw / raw.norm();
        assert!(BeliefState::new(&unit).position.norm() < 1.0);
    }

    #[test]
    fn test_vfe_non_negative() {
        assert!(compute_vfe(&ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0), DEFAULT_CURVATURE) >= 0.0);
    }

    #[test]
    fn test_vfe_zero_when_aligned() {
        let v = ball_vec(1.0);
        assert!(compute_vfe(&v, &v, &v, DEFAULT_CURVATURE).abs() < 1e-8);
    }

    #[test]
    fn test_vfe_components_non_negative() {
        let (vfe, c, a) = compute_vfe_components(
            &ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0),
            SIGMA_PRIOR, SIGMA_OBS, DEFAULT_CURVATURE,
        );
        assert!(vfe >= 0.0 && c >= 0.0 && a >= 0.0);
    }

    #[test]
    fn test_vfe_decreases() {
        let (_, history) = minimise_vfe(&ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0), 0.1);
        assert!(!history.is_empty());
        assert!(history.last().unwrap().vfe <= history[0].vfe + 1e-8);
    }

    #[test]
    fn test_confidence_not_zero() {
        let (belief, _) = minimise_vfe(&ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0), 0.1);
        assert!(belief.confidence > 0.01);
    }

    #[test]
    fn test_vfe_step_stays_in_ball() {
        let next = vfe_step(&ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0), 0.1, DEFAULT_CURVATURE);
        assert!(next.norm() < 1.0);
    }

    #[test]
    fn test_bayesian_update_reduces_sigma() {
        let (_, sigma_post) = bayesian_update(
            &ball_vec(1.0), &ball_vec(2.0), SIGMA_PRIOR, SIGMA_OBS, DEFAULT_CURVATURE,
        );
        assert!(sigma_post < SIGMA_PRIOR, "Posterior sigma must be tighter than prior");
    }

    #[test]
    fn test_efe_epistemic_non_negative() {
        let (_, epistemic, _) = compute_efe(
            &ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0),
            SIGMA_PRIOR, SIGMA_PRIOR * 0.85, DEFAULT_CURVATURE,
        );
        assert!(epistemic >= 0.0);
    }

    #[test]
    fn test_phase_from_vfe() {
        assert_eq!(phase_from_vfe(0.02), StrobePhase::Transcendent);
        assert_eq!(phase_from_vfe(0.07), StrobePhase::Understanding);
        assert_eq!(phase_from_vfe(0.20), StrobePhase::Engaged);
        assert_eq!(phase_from_vfe(0.40), StrobePhase::Aware);
        assert_eq!(phase_from_vfe(0.90), StrobePhase::Dark);
    }
}

