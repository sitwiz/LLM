//! Soul geometry — Poincaré ball model.
//! NF is entropy-based coherence from the paper, not just norm.
//! ball_depth() gives ‖x‖ for zone classification.

use nalgebra::DVector;
use crate::soul::hyperbolic::{
    clamp_to_ball, sphere_to_ball, geodesic_distance,
    exp_map, log_map, SAFE_MAX_NORM,
};

pub const SOUL_DIM:          usize = 256;
pub const PHI:               f64   = 1.6180339887498948482;
pub const GOLDEN_RATIO:      f64   = PHI;
pub const INITIAL_CURVATURE: f64   = 1.0;

/// Project a raw embedding vector into the Poincaré ball at fixed depth 0.4.
/// Used for soul initialisation — personalities start at mid-ball depth.
pub fn project_to_ball(x: &DVector<f64>) -> DVector<f64> {
    let norm = x.norm();
    if norm < 1e-10 {
        return DVector::zeros(x.len());
    }
    let target_norm = 0.4_f64;
    clamp_to_ball(&(x * (target_norm / norm)), SAFE_MAX_NORM)
}

/// Project an embedding into the Poincaré ball preserving natural depth.
/// Used for concept insertion — preserves depth variation so concepts
/// spread across zones based on their content.
pub fn project_to_ball_natural(x: &DVector<f64>) -> DVector<f64> {
    let norm = x.norm();
    if norm < 1e-10 {
        let mut v = DVector::zeros(x.len());
        v[0] = 0.15;
        return v;
    }
    let natural_depth = (0.35 * (1.0 + norm).ln()).clamp(0.05, 0.85);
    clamp_to_ball(&(x * (natural_depth / norm)), SAFE_MAX_NORM)
}

/// Convert an existing unit-sphere soul to the Poincaré ball.
pub fn normalise_to_ball(x: &DVector<f64>) -> DVector<f64> {
    sphere_to_ball(x, SAFE_MAX_NORM)
}

/// Legacy normalise — deprecated.
#[deprecated(note = "Hyperbolic souls must not be forced to unit norm")]
pub fn normalise(x: &DVector<f64>) -> DVector<f64> {
    let norm = x.norm();
    if norm < 1e-10 { return x.clone(); }
    x / norm
}

/// Entropy-based coherence from the paper.
/// NF = 1 - H/log(n)  where H = -Σ pi·log(pi) and pi = xi²/Σxj²
/// NF near 1.0 = coherent, energy concentrated in few dimensions.
/// NF near 0.0 = incoherent, energy spread uniformly — forbidden zone.
/// This is the Psi metric from Pereira's stroboscopic model.
pub fn compute_nf(x: &DVector<f64>) -> f64 {
    let sq_sum: f64 = x.iter().map(|v| v * v).sum::<f64>().max(1e-10);
    let entropy: f64 = x.iter()
        .map(|v| (v * v) / sq_sum)
        .filter(|&p| p > 1e-10)
        .map(|p| -p * p.ln())
        .sum();
    1.0 - entropy / (x.len() as f64).ln().max(1e-10)
}

/// Depth in the Poincaré ball — ‖x‖.
/// Separate from NF coherence.
/// Used for zone classification, attractor depth, and manifold position.
pub fn ball_depth(x: &DVector<f64>) -> f64 {
    x.norm()
}

/// Update soul toward target using hyperbolic geodesic interpolation.
/// Momentum = 1/PHI ≈ 0.618.
pub fn update_soul(current: &DVector<f64>, target: &DVector<f64>) -> DVector<f64> {
    let lr      = 1.0 - (1.0 / PHI);
    let v       = log_map(current, target, INITIAL_CURVATURE);
    let updated = exp_map(current, &(v * lr), INITIAL_CURVATURE);
    clamp_to_ball(&updated, SAFE_MAX_NORM)
}

/// Initialise a soul from a domain text embedding.
pub fn soul_from_embedding(embedding: &[f64]) -> DVector<f64> {
    assert!(embedding.len() >= SOUL_DIM,
        "Embedding too short: {} < {}", embedding.len(), SOUL_DIM);
    project_to_ball(&DVector::from_vec(embedding[..SOUL_DIM].to_vec()))
}

/// Curvature at epoch — decays from INITIAL toward MIN_CURVATURE.
/// Maps to Pereira's expanding hypersphere — as universe ages, curvature flattens.
pub fn curvature_at_epoch(epoch: u32) -> f64 {
    let c = INITIAL_CURVATURE / (1.0 + epoch as f64 * 0.01);
    c.max(crate::soul::manifold::MIN_CURVATURE)
}

/// True if soul is in the forbidden zone.
/// Uses entropy-based NF — forbidden when NF < 0.05 OR near boundary.
pub fn in_forbidden_zone(x: &DVector<f64>) -> bool {
    let nf   = compute_nf(x);
    let norm = ball_depth(x);
    nf < 0.05 || norm < 0.05 || norm > 0.98
}

/// Geodesic distance between two soul positions at default curvature.
pub fn soul_distance(x: &DVector<f64>, y: &DVector<f64>) -> f64 {
    geodesic_distance(x, y, INITIAL_CURVATURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(seed: f64) -> DVector<f64> {
       let v: Vec<f64> = (0..SOUL_DIM)
           .map(|i| ((i as f64 + seed) * 1.7).sin() * 0.03)
           .collect();
       project_to_ball(&DVector::from_vec(v))
    }

    #[test]
    fn test_project_to_ball_in_range() {
        let b = project_to_ball(&rand_vec(1.0));
        assert!(b.norm() < 1.0);
        assert!(b.norm() > 0.0);
    }

    #[test]
    fn test_project_to_ball_natural_in_range() {
        let v: Vec<f64> = (0..SOUL_DIM)
            .map(|i| ((i as f64) * 1.7).sin() * 0.5)
            .collect();
        let b = project_to_ball_natural(&DVector::from_vec(v));
        assert!(b.norm() >= 0.05);
        assert!(b.norm() <= 0.85);
    }

    #[test]
    fn test_compute_nf_coherent() {
        // Concentrated vector — high NF
        let mut x = DVector::zeros(SOUL_DIM);
        x[0] = 1.0;
        let nf = compute_nf(&x);
        assert!(nf > 0.9, "Concentrated vector should have high NF, got {}", nf);
    }

    #[test]
    fn test_compute_nf_incoherent() {
        // Uniform vector — low NF
        let x = DVector::from_element(SOUL_DIM, 1.0 / (SOUL_DIM as f64).sqrt());
        let nf = compute_nf(&x);
        assert!(nf < 0.1, "Uniform vector should have low NF, got {}", nf);
    }

    #[test]
    fn test_ball_depth_is_norm() {
        let x = rand_vec(1.0);
        assert!((ball_depth(&x) - x.norm()).abs() < 1e-12);
    }

    #[test]
    fn test_update_soul_moves_toward_target() {
        let current = rand_vec(1.0);
        let target  = rand_vec(2.0);
        let updated = update_soul(&current, &target);
        assert!(soul_distance(&updated, &target) < soul_distance(&current, &target));
    }

    #[test]
    fn test_update_soul_stays_in_ball() {
        let current = rand_vec(1.0);
        let target  = rand_vec(5.0);
        assert!(update_soul(&current, &target).norm() < 1.0);
    }

    #[test]
    fn test_forbidden_zone_origin() {
       // Vector with norm well below 0.05 — forbidden by norm check
       let mut x = DVector::zeros(SOUL_DIM);
       x[0] = 0.01;  // norm = 0.01 < 0.05 threshold
       assert!(in_forbidden_zone(&x));
    }    
    
    #[test]
    fn test_not_forbidden_mid() {
        // Mid-ball vector with concentrated energy — should not be forbidden
        let mut x = DVector::zeros(SOUL_DIM);
        x[0] = 0.5;
        assert!(!in_forbidden_zone(&x));
    }

    #[test]
    fn test_curvature_decays() {
        let c0   = curvature_at_epoch(0);
        let c10  = curvature_at_epoch(10);
        let c100 = curvature_at_epoch(100);
        assert!(c0 >= c10 && c10 >= c100);
        assert!(c100 >= crate::soul::manifold::MIN_CURVATURE);
    }

    #[test]
    fn test_soul_from_embedding() {
        let emb: Vec<f64> = (0..SOUL_DIM).map(|i| (i as f64).sin()).collect();
        assert!(soul_from_embedding(&emb).norm() < 1.0);
    }
}
