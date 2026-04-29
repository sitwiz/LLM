//! Poincaré ball model of hyperbolic space.
//!
//! All soul vectors x live strictly inside the open unit ball: ‖x‖ < 1.
//! The boundary ‖x‖ = 1 is at infinite geodesic distance — the Zeno boundary.
//! Curvature parameter c: ds² = (4/c²) dx² / (1 - ‖x‖²)²
//! c = 1.0 is standard hyperbolic space. c → 0 flattens toward Euclidean.

use nalgebra::DVector;

/// Numerical epsilon for safe comparisons.
pub const EPS: f64 = 1e-12;

/// Maximum allowed norm for a point in the Poincaré ball.
/// Values strictly less than 1 avoid singularities.
pub const SAFE_MAX_NORM: f64 = 1.0 - 1e-5;

/// Clamp a vector to strictly inside the ball at a given max norm.
pub fn clamp_to_ball(x: &DVector<f64>, max_norm: f64) -> DVector<f64> {
    let norm = x.norm();
    if norm >= max_norm {
        x * (max_norm / norm.max(EPS))
    } else {
        x.clone()
    }
}

/// Convenience: clamp to the global safe maximum norm.
pub fn clamp_to_ball_safe(x: &DVector<f64>) -> DVector<f64> {
    clamp_to_ball(x, SAFE_MAX_NORM)
}

/// Project a unit-sphere vector into the Poincaré ball via tanh scaling.
/// `tanh(0.6) ≈ 0.537` — souls start at mid‑ball depth after conversion.
pub fn sphere_to_ball(x: &DVector<f64>, max_norm: f64) -> DVector<f64> {
    clamp_to_ball(&(x * 0.6_f64.tanh()), max_norm)
}

/// Möbius addition with curvature c:
/// x ⊕_c y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 - c‖x‖²)y)
///           / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²)
pub fn mobius_add(x: &DVector<f64>, y: &DVector<f64>, c: f64) -> DVector<f64> {
    let xy    = x.dot(y);
    let x2    = x.norm_squared();
    let y2    = y.norm_squared();
    let num_x = 1.0 + 2.0 * c * xy + c * y2;
    let num_y = 1.0 - c * x2;
    let denom = (1.0 + 2.0 * c * xy + c * c * x2 * y2).max(EPS);
    clamp_to_ball_safe(&((x * num_x + y * num_y) / denom))
}

/// Negation in the Poincaré ball: -_c x = -x
pub fn mobius_neg(x: &DVector<f64>) -> DVector<f64> {
    -x
}

/// Exponential map at x in direction v with curvature c:
/// exp_x(v) = x ⊕_c ( tanh(√c·‖v‖/2) · v / (√c·‖v‖) )
pub fn exp_map(x: &DVector<f64>, v: &DVector<f64>, c: f64) -> DVector<f64> {
    let v_norm = v.norm();
    if v_norm < EPS {
        return x.clone();
    }
    let sqrt_c   = c.sqrt().max(EPS);
    let tanh_arg = (sqrt_c * v_norm / 2.0).tanh();
    let y        = (v / v_norm) * (tanh_arg / sqrt_c);
    mobius_add(x, &y, c)
}

/// Logarithmic map from x to y with curvature c:
/// log_x(y) = (2/√c) · arctanh(√c · ‖-x ⊕_c y‖) · (-x ⊕_c y) / ‖-x ⊕_c y‖
pub fn log_map(x: &DVector<f64>, y: &DVector<f64>, c: f64) -> DVector<f64> {
    let add      = mobius_add(&mobius_neg(x), y, c);
    let add_norm = add.norm();
    if add_norm < EPS {
        return DVector::zeros(x.len());
    }
    let sqrt_c = c.sqrt().max(EPS);
    let arg    = (sqrt_c * add_norm).min(1.0 - 1e-12); // clamp strictly below 1
    let scale  = (2.0 / sqrt_c) * arg.atanh() / add_norm;
    add * scale
}

/// Geodesic distance between x and y with curvature c:
/// d(x, y) = (2/√c) · arctanh(√c · ‖-x ⊕_c y‖)
pub fn geodesic_distance(x: &DVector<f64>, y: &DVector<f64>, c: f64) -> f64 {
    let add      = mobius_add(&mobius_neg(x), y, c);
    let add_norm = add.norm();
    let sqrt_c   = c.sqrt().max(EPS);
    let arg      = (sqrt_c * add_norm).min(1.0 - 1e-12);
    (2.0 / sqrt_c) * arg.atanh()
}

/// Parallel transport of tangent vector v from x to y
/// via conformal factor ratio (exact in hyperbolic space).
pub fn parallel_transport(
    x: &DVector<f64>,
    y: &DVector<f64>,
    v: &DVector<f64>,
    c: f64,
) -> DVector<f64> {
    let lx = conformal_factor(x, c);
    let ly = conformal_factor(y, c).max(EPS);
    v * (lx / ly)
}

/// Geodesic interpolation: move from x toward y by fraction t ∈ [0, 1].
pub fn geodesic_interpolate(
    x: &DVector<f64>,
    y: &DVector<f64>,
    t: f64,
    c: f64,
) -> DVector<f64> {
    exp_map(x, &(log_map(x, y, c) * t), c)
}

/// Conformal factor at x: λ_c(x) = 2 / (1 - c‖x‖²)
/// Near origin: ≈ 2 (nearly flat).
/// Near boundary: → ∞ (Zeno stretching).
pub fn conformal_factor(x: &DVector<f64>, c: f64) -> f64 {
    2.0 / (1.0 - c * x.norm_squared()).max(EPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(vals: Vec<f64>) -> DVector<f64> {
        DVector::from_vec(vals)
    }

    #[test]
    fn test_clamp_to_ball_respects_max_norm() {
        let x = v(vec![10.0, 0.0]);
        let clamped = clamp_to_ball(&x, 0.5);
        assert!((clamped.norm() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mobius_add_identity() {
        let x    = v(vec![0.3, 0.2, 0.1]);
        let zero = v(vec![0.0, 0.0, 0.0]);
        let r    = mobius_add(&x, &zero, 1.0);
        for (a, b) in x.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-8);
        }
    }

    #[test]
    fn test_mobius_neg_inverse() {
        let x = v(vec![0.3, 0.2, 0.1]);
        let r = mobius_add(&x, &mobius_neg(&x), 1.0);
        assert!(r.norm() < 1e-6);
    }

    #[test]
    fn test_exp_log_roundtrip() {
        let x  = v(vec![0.2, 0.1, 0.0]);
        let y  = v(vec![0.1, 0.3, 0.2]);
        let y2 = exp_map(&x, &log_map(&x, &y, 1.0), 1.0);
        for (a, b) in y.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn test_geodesic_distance_self_zero() {
        let x = v(vec![0.3, 0.2, 0.1]);
        let d = geodesic_distance(&x, &x, 1.0);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_geodesic_distance_positive() {
        let x = v(vec![0.2, 0.0, 0.0]);
        let y = v(vec![0.0, 0.3, 0.0]);
        assert!(geodesic_distance(&x, &y, 1.0) > 0.0);
    }

    #[test]
    fn test_geodesic_large_near_boundary() {
        let x = v(vec![0.99, 0.0]);
        let y = v(vec![0.0,  0.0]);
        let d = geodesic_distance(&x, &y, 1.0);
        assert!(d > 3.0);
    }

    #[test]
    fn test_conformal_factor_grows_near_boundary() {
        let near_origin   = v(vec![0.01, 0.0]);
        let near_boundary = v(vec![0.99, 0.0]);
        assert!(
            conformal_factor(&near_boundary, 1.0) > conformal_factor(&near_origin, 1.0)
        );
    }
}
