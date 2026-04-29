use nalgebra::DVector;
use crate::soul::geometry::{compute_nf, project_to_ball, SOUL_DIM};
use crate::soul::hyperbolic::{geodesic_distance, exp_map, log_map};
use crate::soul::manifold::StrobePhase;

pub const VFE_EQUILIBRIUM: f64 = 0.05;
pub const MAX_CYCLES: usize = 25;
pub const PLASTICITY_MOMENTUM: f64 = 0.99;

/// Curvature used throughout VFE — matches manifold default.
/// Pass the live manifold curvature when available.
const DEFAULT_CURVATURE: f64 = crate::soul::geometry::INITIAL_CURVATURE;

// ── Belief state ────────────────────────────────────────────────────────────

/// The belief state lives INSIDE the Poincaré ball (norm strictly < 1).
/// It is never normalised to the unit sphere.
#[derive(Debug, Clone)]
pub struct BeliefState {
    pub position:   DVector<f64>,
    pub confidence: f64,
    pub vfe:        f64,
    pub cycle:      usize,
}

impl BeliefState {
    pub fn new(soul: &DVector<f64>) -> Self {
        // Keep the soul inside the Poincaré ball — do NOT normalise to unit sphere.
        // project_to_ball handles NaN, inf, and boundary clamping.
        let pos = project_to_ball(soul);
        Self {
            position:   pos,
            confidence: 0.0,
            vfe:        f64::MAX,
            cycle:      0,
        }
    }
}

// ── VFE on the Poincaré ball ─────────────────────────────────────────────────

/// Variational Free Energy in hyperbolic space.
///
/// F = KL[Q || P] – log p(y | x)
///
/// Both terms are approximated using hyperbolic geodesic distance:
///   KL[Q || P]       ≈  d_hyp(belief, prior)        — complexity
///   –log p(y | x)    ≈  d_hyp(belief, observation)  — accuracy
///
/// This is valid because in the Poincaré ball, geodesic distance is the
/// natural divergence measure. Cosine similarity and (1 – cos_sim) are
/// Euclidean/spherical concepts and must not be used here.
pub fn compute_vfe(
    belief:      &DVector<f64>,
    prior:       &DVector<f64>,
    observation: &DVector<f64>,
    curvature:   f64,
) -> f64 {
    let kl       = geodesic_distance(belief, prior,       curvature);
    let accuracy = geodesic_distance(belief, observation, curvature);
    kl + accuracy
}

/// Prediction error in the tangent space at `belief` on the Poincaré ball.
///
/// Uses log_map to map `observation` into the tangent space at `belief`.
/// The result is the hyperbolic equivalent of (obs – projection of obs onto belief).
pub fn prediction_error(
    belief:      &DVector<f64>,
    observation: &DVector<f64>,
    curvature:   f64,
) -> DVector<f64> {
    log_map(belief, observation, curvature)
}

/// Single Riemannian gradient descent step on the Poincaré ball.
///
/// The gradient lives in the tangent space at `belief`. We:
///   1. Map prior and observation into the tangent space via log_map.
///   2. Combine them (prior term downweighted to avoid overshooting).
///   3. Map back to the ball via exp_map.
///
/// This replaces the old `belief + lr * gradient` + `normalise` pattern,
/// which was operating on the wrong manifold.
pub fn vfe_step(
    belief:      &DVector<f64>,
    prior:       &DVector<f64>,
    observation: &DVector<f64>,
    lr:          f64,
    curvature:   f64,
) -> DVector<f64> {
    let v_obs   = log_map(belief, observation, curvature);
    let v_prior = log_map(belief, prior,       curvature);

    // Prior term given 0.3 weight — keeps belief anchored without overshooting.
    let tangent = v_obs + v_prior * 0.3;

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

/// Full minimisation with explicit curvature — call this from quorum once the
/// live manifold curvature is threaded through.
pub fn minimise_vfe_with_curvature(
    soul:        &DVector<f64>,
    attractor:   &DVector<f64>,
    observation: &DVector<f64>,
    lr:          f64,
    curvature:   f64,
) -> (BeliefState, Vec<VFERecord>) {
    let mut belief = BeliefState::new(soul);
    let mut history = Vec::new();

    // Ensure attractor and observation are inside the ball before we start.
    let attractor   = project_to_ball(attractor);
    let observation = project_to_ball(observation);

    println!("\n[UnifiedOmniAGI] VFE minimisation starting...");

    for cycle in 0..MAX_CYCLES {
        belief.cycle = cycle;

        belief.vfe = compute_vfe(&belief.position, &attractor, &observation, curvature);
        let nf     = compute_nf(&belief.position);
        let pe     = prediction_error(&belief.position, &observation, curvature);
        let pe_norm = pe.norm();

        // Confidence: rescale VFE so equilibrium threshold maps to ~0.95 confidence.
        // VFE_EQUILIBRIUM=0.05 → exp(-0.05/0.1) = exp(-0.5) ≈ 0.61 at threshold,
        // which gives useful gradient. Using raw exp(-vfe) collapses to 0 when VFE>5.
        belief.confidence = (-belief.vfe / (VFE_EQUILIBRIUM * 20.0)).exp();

        let record = VFERecord {
            cycle,
            vfe:        belief.vfe,
            confidence: belief.confidence,
            nf,
            pe_norm,
        };

        println!(
            "  [Cycle {:2}] VFE={:.6} conf={:.4} NF={:.4} PE={:.4}",
            cycle, belief.vfe, belief.confidence, nf, pe_norm
        );
        history.push(record);

        if belief.vfe <= VFE_EQUILIBRIUM {
            println!(
                "  [UnifiedOmniAGI] Equilibrium reached at cycle {}. VFE={:.6}",
                cycle, belief.vfe
            );
            break;
        }

        let lr_cycle = lr * (1.0 - cycle as f64 / MAX_CYCLES as f64).max(0.1);
        belief.position = vfe_step(
            &belief.position, &attractor, &observation, lr_cycle, curvature,
        );
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
    pub cycle:      usize,
    pub vfe:        f64,
    pub confidence: f64,
    pub nf:         f64,
    pub pe_norm:    f64,
}

pub fn phase_from_vfe(vfe: f64) -> StrobePhase {
    if vfe <= 0.03 {
        StrobePhase::Transcendent
    } else if vfe <= 0.1 {
        StrobePhase::Understanding
    } else if vfe <= 0.25 {
        StrobePhase::Engaged
    } else if vfe <= 0.5 {
        StrobePhase::Aware
    } else {
        StrobePhase::Dark
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::geometry::project_to_ball;

    /// Mid-ball test vector — inside Poincaré ball, not on boundary.
    fn ball_vec(seed: f64) -> DVector<f64> {
        let v: Vec<f64> = (0..SOUL_DIM)
            .map(|i| ((i as f64 + seed) * 1.7).sin() * 0.3)
            .collect();
        project_to_ball(&DVector::from_vec(v))
    }

    #[test]
    fn test_belief_state_inside_ball() {
        // BeliefState must not normalise to unit sphere.
        let soul = ball_vec(1.0);
        let belief = BeliefState::new(&soul);
        assert!(
            belief.position.norm() < 1.0,
            "BeliefState must be inside ball, norm={}",
            belief.position.norm()
        );
        assert!(belief.position.norm() > 0.0);
    }

    #[test]
    fn test_unit_sphere_input_clamped() {
        // If a unit-sphere vector slips through, BeliefState must clamp it.
        let v: Vec<f64> = (0..SOUL_DIM).map(|i| (i as f64 * 0.03).sin()).collect();
        let raw = DVector::from_vec(v);
        let unit = &raw / raw.norm(); // norm == 1.0
        let belief = BeliefState::new(&unit);
        assert!(
            belief.position.norm() < 1.0,
            "Unit-sphere input must be clamped inside ball, norm={}",
            belief.position.norm()
        );
    }

    #[test]
    fn test_vfe_non_negative() {
        let c = DEFAULT_CURVATURE;
        let vfe = compute_vfe(&ball_vec(1.0), &ball_vec(2.0), &ball_vec(3.0), c);
        assert!(vfe >= 0.0, "VFE must be non-negative, got {}", vfe);
    }

    #[test]
    fn test_vfe_zero_when_aligned() {
        let v = ball_vec(1.0);
        let vfe = compute_vfe(&v, &v, &v, DEFAULT_CURVATURE);
        assert!(vfe.abs() < 1e-8, "VFE should be ~0 when aligned, got {}", vfe);
    }

    #[test]
    fn test_vfe_decreases() {
        let c = DEFAULT_CURVATURE;
        let soul = ball_vec(1.0);
        let prior = ball_vec(2.0);
        let obs = ball_vec(3.0);
        let (_, history) = minimise_vfe(&soul, &prior, &obs, 0.1);
        assert!(!history.is_empty());
        let first = history[0].vfe;
        let last  = history.last().unwrap().vfe;
        assert!(last <= first + 1e-8, "VFE should decrease: {} -> {}", first, last);
    }

    #[test]
    fn test_confidence_not_zero() {
        let soul  = ball_vec(1.0);
        let prior = ball_vec(2.0);
        let obs   = ball_vec(3.0);
        let (belief, _) = minimise_vfe(&soul, &prior, &obs, 0.1);
        assert!(
            belief.confidence > 0.01,
            "Confidence must be visible, got {}",
            belief.confidence
        );
    }

    #[test]
    fn test_vfe_step_stays_in_ball() {
        let c     = DEFAULT_CURVATURE;
        let b     = ball_vec(1.0);
        let prior = ball_vec(2.0);
        let obs   = ball_vec(3.0);
        let next  = vfe_step(&b, &prior, &obs, 0.1, c);
        assert!(
            next.norm() < 1.0,
            "vfe_step result must stay inside ball, norm={}",
            next.norm()
        );
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
