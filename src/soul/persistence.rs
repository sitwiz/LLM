//! Soul persistence — save and load soul vectors from disk.
//! On load, vectors are re-projected into the Poincaré ball
//! in case they were saved in the old unit-sphere format.

use nalgebra::DVector;
use std::path::Path;

const TARGET_NORM: f64 = 0.4;
const BOUNDARY_THRESHOLD: f64 = 0.5;

pub fn save_soul(soul: &DVector<f64>, path: &Path) -> anyhow::Result<()> {
    let bytes: Vec<u8> = soul
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn load_soul(path: &Path) -> anyhow::Result<DVector<f64>> {
    let bytes = std::fs::read(path)?;
    let values: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let raw = DVector::from_vec(values);

    // First pass: geometric re-projection (handles NaN, inf, general clamping)
    let projected = crate::soul::geometry::project_to_ball(&raw);
    let norm = projected.norm();

    // Second pass: explicit rescale to target norm.
    // project_to_ball uses tanh which saturates near 1.0 for high-norm
    // 256d inputs (old unit-sphere format). This is the authoritative fix.
    let soul = if norm > BOUNDARY_THRESHOLD {
        projected * (TARGET_NORM / norm)
    } else if norm < 1e-9 {
        // Degenerate zero vector — reinitialise at target norm with uniform direction
        let dim = raw.len();
        let unit = DVector::from_element(dim, 1.0 / (dim as f64).sqrt());
        unit * TARGET_NORM
    } else {
        projected
    };

    Ok(soul)
}

/// Load soul from disk with hyperbolic re-projection.
/// If load fails, returns the provided fallback vector.
pub fn load_or_init(path: &Path, fallback: DVector<f64>) -> DVector<f64> {
    match load_soul(path) {
        Ok(soul) => {
            println!(
                "  [Soul] Loaded {:?} — norm: {:.4} (target {:.1})",
                path,
                soul.norm(),
                TARGET_NORM
            );
            soul
        }
        Err(e) => {
            println!(
                "  [Soul] {:?} not found ({}), using fallback — norm: {:.4}",
                path,
                e,
                fallback.norm()
            );
            fallback
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::geometry::{project_to_ball, SOUL_DIM};

    /// A soul already inside the ball at reasonable depth.
    fn mid_ball_soul() -> DVector<f64> {
        let v: Vec<f64> = (0..SOUL_DIM)
            .map(|i| ((i as f64) * 0.03).sin())
            .collect();
        project_to_ball(&DVector::from_vec(v))
    }

    /// Simulates an old unit-sphere binary: components summing to norm ~1.0.
    fn unit_sphere_soul() -> DVector<f64> {
        let v: Vec<f64> = (0..SOUL_DIM)
            .map(|i| ((i as f64) * 0.03).sin())
            .collect();
        let raw = DVector::from_vec(v);
        &raw / raw.norm() // norm exactly 1.0
    }

    fn write_raw(soul: &DVector<f64>, path: &Path) {
        let bytes: Vec<u8> = soul.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }

    // Round-trip for a soul already inside the ball.
    // Direction must be preserved; norm may differ slightly due to rescale
    // only if it was already above BOUNDARY_THRESHOLD.
    #[test]
    fn test_roundtrip_mid_ball_soul() {
        let soul = mid_ball_soul();
        let path = Path::new("/tmp/test_soul_midball.bin");
        save_soul(&soul, path).unwrap();
        let loaded = load_soul(path).unwrap();

        // Soul was below 0.5 so no rescale — expect near-exact round-trip
        assert!(
            (soul.norm() - loaded.norm()).abs() < 1e-6,
            "Mid-ball soul norm changed unexpectedly: {:.6} -> {:.6}",
            soul.norm(),
            loaded.norm()
        );
        for (a, b) in soul.iter().zip(loaded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        std::fs::remove_file(path).ok();
    }

    // Old unit-sphere binaries must load at target norm.
    #[test]
    fn test_old_unit_sphere_rescaled_to_target() {
        let soul = unit_sphere_soul();
        assert!((soul.norm() - 1.0).abs() < 1e-9, "precondition: norm should be 1.0");

        let path = Path::new("/tmp/test_soul_unitsphere.bin");
        write_raw(&soul, path);
        let loaded = load_soul(path).unwrap();

        assert!(
            (loaded.norm() - TARGET_NORM).abs() < 1e-6,
            "Unit-sphere soul should load at target norm {}, got {:.6}",
            TARGET_NORM,
            loaded.norm()
        );
        std::fs::remove_file(path).ok();
    }

    // Loaded soul must always be strictly inside the ball.
    #[test]
    fn test_loaded_soul_inside_ball() {
        for soul in [mid_ball_soul(), unit_sphere_soul()] {
            let path = Path::new("/tmp/test_soul_inside.bin");
            write_raw(&soul, path);
            let loaded = load_soul(path).unwrap();
            assert!(
                loaded.norm() < 1.0,
                "Loaded soul must be inside ball, norm={:.6}",
                loaded.norm()
            );
            assert!(loaded.norm() > 0.0, "Loaded soul must be nonzero");
            std::fs::remove_file(path).ok();
        }
    }

    // Degenerate zero vector must not panic and must return a valid soul.
    #[test]
    fn test_zero_vector_fallback() {
        let zero = DVector::from_element(SOUL_DIM, 0.0);
        let path = Path::new("/tmp/test_soul_zero.bin");
        write_raw(&zero, path);
        let loaded = load_soul(path).unwrap();
        assert!(
            (loaded.norm() - TARGET_NORM).abs() < 1e-6,
            "Zero vector should produce target-norm soul, got {:.6}",
            loaded.norm()
        );
        std::fs::remove_file(path).ok();
    }
}
