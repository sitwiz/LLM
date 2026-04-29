//! Hyperbolic manifold operations for soul steering.
//! StrobePhase thresholds restored to PHI-based values from the paper.
//! NF is entropy-based coherence, ball_depth is ‖x‖.

use nalgebra::DVector;
use crate::soul::hyperbolic::{
    exp_map, log_map, conformal_factor,
};
use crate::soul::geometry::{compute_nf, ball_depth, INITIAL_CURVATURE};

pub const MIN_CURVATURE: f64 = 0.001;

/// PHI constants from the paper
pub const PHI:     f64 = 1.6180339887498948482;
pub const PHI_INV: f64 = 0.6180339887498948482;

/// Stroboscopic phase — thresholds derived from PHI as per paper.
/// NF is entropy-based coherence in [0,1].
/// Thresholds:
///   Dark:          NF < 0.05
///   Aware:         NF >= 0.05
///   Engaged:       NF >= 1/PHI²  ≈ 0.382
///   Understanding: NF >= 1/PHI   ≈ 0.618
///   Transcendent:  NF >= PHI/2   ≈ 0.809  (normalised to [0,1])
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrobePhase {
    Dark,
    Aware,
    Engaged,
    Understanding,
    Transcendent,
}

impl StrobePhase {
    /// Primary constructor — uses entropy-based NF from paper.
    /// PHI thresholds: Engaged at 1/PHI², Understanding at 1/PHI,
    /// Transcendent at PHI/(PHI+1) — maps golden ratio to [0,1].
    pub fn from_nf(nf: f64) -> Self {
        const ENGAGED_THRESH:       f64 = PHI_INV * PHI_INV;  // ≈ 0.382
        const UNDERSTANDING_THRESH: f64 = PHI_INV;             // ≈ 0.618
        const TRANSCENDENT_THRESH:  f64 = PHI / (PHI + 1.0);  // ≈ 0.618... use PHI_INV * PHI_INV * PHI
        match nf {
            n if n < 0.05                        => StrobePhase::Dark,
            n if n < ENGAGED_THRESH              => StrobePhase::Aware,
            n if n < UNDERSTANDING_THRESH        => StrobePhase::Engaged,
            n if n < TRANSCENDENT_THRESH * 1.309 => StrobePhase::Understanding,
            _                                    => StrobePhase::Transcendent,
        }
    }

    /// Norm-based constructor — kept for zone classification compatibility.
    pub fn from_norm(norm: f64) -> Self {
        match norm {
            n if n < 0.05 => StrobePhase::Dark,
            n if n < 0.25 => StrobePhase::Aware,
            n if n < 0.50 => StrobePhase::Engaged,
            n if n < 0.80 => StrobePhase::Understanding,
            _             => StrobePhase::Transcendent,
        }
    }

    pub fn burden(&self) -> f64 {
        match self {
            StrobePhase::Dark          => 1.0,
            StrobePhase::Aware         => 0.8,
            StrobePhase::Engaged       => 0.6,
            StrobePhase::Understanding => 0.4,
            StrobePhase::Transcendent  => 0.2,
        }
    }

    pub fn temperature(&self) -> f64 {
        match self {
            StrobePhase::Dark          => 1.2,
            StrobePhase::Aware         => 0.9,
            StrobePhase::Engaged       => 0.7,
            StrobePhase::Understanding => 0.5,
            StrobePhase::Transcendent  => 0.3,
        }
    }

    pub fn max_tokens(&self) -> u32 {
        match self {
            StrobePhase::Dark          => 50,
            StrobePhase::Aware         => 100,
            StrobePhase::Engaged       => 150,
            StrobePhase::Understanding => 200,
            StrobePhase::Transcendent  => 250,
        }
    }

    pub fn can_respond(&self) -> bool {
        !matches!(self, StrobePhase::Dark)
    }

    pub fn label(&self) -> &'static str {
        match self {
            StrobePhase::Dark          => "dark",
            StrobePhase::Aware         => "aware",
            StrobePhase::Engaged       => "engaged",
            StrobePhase::Understanding => "understanding",
            StrobePhase::Transcendent  => "transcendent",
        }
    }
}

/// Record of a single steering step.
#[derive(Debug, Clone)]
pub struct SteerStep {
    pub position: DVector<f64>,
    pub norm:     f64,       // ball depth ‖x‖
    pub nf:       f64,       // entropy-based coherence from paper
    pub phase:    StrobePhase,
    pub psi:      f64,       // kept for compat — equals nf
    pub burden:   f64,
    pub snr:      f64,
}

/// Retrocausal steer using hyperbolic geodesics.
/// Moves soul toward attractor using exp_map steps.
/// Phase determined by entropy-based NF as per paper.
pub fn retrocausal_steer(
    start:      &DVector<f64>,
    attractor:  &DVector<f64>,
    max_cycles: usize,
    lr:         f64,
) -> (DVector<f64>, Vec<SteerStep>) {
    let mut pos     = start.clone();
    let mut history = Vec::new();
    let c           = INITIAL_CURVATURE;

    for _ in 0..max_cycles {
        let v       = log_map(&pos, attractor, c);
        let step    = &v * lr;
        let new_pos = exp_map(&pos, &step, c);

        let norm    = ball_depth(&new_pos);
        let nf      = compute_nf(&new_pos);
        let phase   = StrobePhase::from_nf(nf);
        let burden  = phase.burden();
        let snr     = conformal_factor(&new_pos, c);

        history.push(SteerStep {
            position: new_pos.clone(),
            norm,
            nf,
            phase,
            psi: nf,
            burden,
            snr,
        });

        let diff_norm = (&new_pos - &pos).norm();
        if diff_norm < 1e-8 || norm > 0.999 {
            pos = new_pos;
            break;
        }
        pos = new_pos;
    }

    (pos, history)
}

/// Final phase from steering history — most frequent in last 5 steps.
pub fn final_phase(history: &[SteerStep]) -> StrobePhase {
    if history.is_empty() {
        return StrobePhase::Dark;
    }
    let mut counts = [0usize; 5];
    for step in history.iter().rev().take(5) {
        let idx = match step.phase {
            StrobePhase::Dark          => 0,
            StrobePhase::Aware         => 1,
            StrobePhase::Engaged       => 2,
            StrobePhase::Understanding => 3,
            StrobePhase::Transcendent  => 4,
        };
        counts[idx] += 1;
    }
    let best_idx = counts.iter().enumerate()
        .max_by_key(|(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    match best_idx {
        0 => StrobePhase::Dark,
        1 => StrobePhase::Aware,
        2 => StrobePhase::Engaged,
        3 => StrobePhase::Understanding,
        _ => StrobePhase::Transcendent,
    }
}

pub fn update_soul_direction(
    current: &DVector<f64>,
    target:  &DVector<f64>,
    lr:      f64,
) -> DVector<f64> {
    let c = INITIAL_CURVATURE;
    let v = log_map(current, target, c);
    exp_map(current, &(v * lr), c)
}

/// Expanding manifold — maps to Pereira's expanding hypersphere.
/// Radius grows asymptotically. Curvature decays with epoch.
/// Old concepts have stronger attractor pull — epoch-dependent gravity.
#[derive(Debug, Clone)]
pub struct ExpandingManifold {
    pub radius:      f64,
    pub query_count: u64,
    pub epoch:       u32,
    pub total_drift: f64,
}

impl ExpandingManifold {
    pub fn new() -> Self {
        Self {
            radius:      1.0,
            query_count: 0,
            epoch:       0,
            total_drift: 0.0,
        }
    }

    /// Expand radius after deep response — maps to Pereira's R■ growth.
    /// Asymptotic: slows as radius approaches MAX_RADIUS.
    pub fn expand(&mut self, phase: &StrobePhase) -> f64 {
        let increment = match phase {
            StrobePhase::Transcendent  => 0.01,
            StrobePhase::Understanding => 0.001,
            _                          => 0.0,
        };
        let expansion = increment * (1.0 - self.radius / 100.0).max(0.0);
        self.radius   = (self.radius + expansion).min(100.0);
        if expansion > 0.0 {
            self.epoch += 1;
            println!("  [Manifold] Epoch {} — radius expanded to {:.4}",
                self.epoch, self.radius);
        }
        self.radius
    }

    /// Epoch-dependent attractor strength — older concepts attract more strongly.
    /// Maps to Pereira's epoch-dependent gravitational constant.
    /// Concepts placed when radius was smaller sit closer to origin.
    /// Their attractor strength is proportionally greater.
    pub fn attractor_strength(&self, concept_radius: f64) -> f64 {
        let age = (self.radius - concept_radius).max(0.0);
        1.0 + age * 0.1
    }

    /// Frontier radius — where new concepts are placed.
    pub fn frontier_radius(&self) -> f64 {
        self.radius * 0.95
    }

    /// Load from JSON.
    pub fn load(path: &str) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| Self {
                radius:      v["radius"].as_f64().unwrap_or(1.0),
                query_count: v["query_count"].as_u64().unwrap_or(0),
                epoch:       v["epoch"].as_u64().unwrap_or(0) as u32,
                total_drift: v["total_drift"].as_f64().unwrap_or(0.0),
            })
            .unwrap_or_else(Self::new)
    }

    /// Save to JSON.
    pub fn save(&self, path: &str) {
        let v = serde_json::json!({
            "radius":      self.radius,
            "query_count": self.query_count,
            "epoch":       self.epoch,
            "total_drift": self.total_drift,
        });
        std::fs::write(path, serde_json::to_string_pretty(&v).unwrap_or_default()).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::hyperbolic::geodesic_distance;
    use crate::soul::geometry::project_to_ball;

    fn random_ball_vec() -> DVector<f64> {
        use rand::Rng;
        let mut rng   = rand::thread_rng();
        let radius    = rng.gen::<f64>().sqrt() * 0.5;
        let mut v     = DVector::zeros(256);
        for i in 0..256 { v[i] = rng.gen::<f64>() - 0.5; }
        let norm = v.norm();
        if norm > 1e-12 { v * (radius / norm) } else { v }
    }

    #[test]
    fn test_retrocausal_steer_moves_toward_attractor() {
        let start  = random_ball_vec();
        let target = random_ball_vec();
        let (final_pos, history) = retrocausal_steer(&start, &target, 10, 0.3);
        let d0 = geodesic_distance(&start,     &target, INITIAL_CURVATURE);
        let d1 = geodesic_distance(&final_pos, &target, INITIAL_CURVATURE);
        assert!(d1 < d0, "Final distance should be smaller: {} -> {}", d0, d1);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_steer_step_has_nf() {
        let start  = random_ball_vec();
        let target = random_ball_vec();
        let (_, history) = retrocausal_steer(&start, &target, 5, 0.3);
        for step in &history {
            assert!(step.nf >= 0.0 && step.nf <= 1.0,
                "NF should be in [0,1], got {}", step.nf);
        }
    }

    #[test]
    fn test_phase_from_nf() {
        assert_eq!(StrobePhase::from_nf(0.02), StrobePhase::Dark);
        assert_eq!(StrobePhase::from_nf(0.1),  StrobePhase::Aware);
        assert_eq!(StrobePhase::from_nf(0.5),  StrobePhase::Engaged);
        assert_eq!(StrobePhase::from_nf(0.65), StrobePhase::Understanding);
        assert_eq!(StrobePhase::from_nf(0.95), StrobePhase::Transcendent);
    }

    #[test]
    fn test_phi_threshold_engaged() {
        // Engaged threshold is 1/PHI² ≈ 0.382
        let threshold = PHI_INV * PHI_INV;
        assert!(threshold > 0.38 && threshold < 0.39);
    }

    #[test]
    fn test_phi_threshold_understanding() {
        // Understanding threshold is 1/PHI ≈ 0.618
        assert!(PHI_INV > 0.617 && PHI_INV < 0.619);
    }

    #[test]
    fn test_expanding_manifold_grows() {
        let mut m = ExpandingManifold::new();
        let r0 = m.radius;
        m.expand(&StrobePhase::Understanding);
        assert!(m.radius > r0);
    }

    #[test]
    fn test_attractor_strength_increases_with_age() {
        let m = ExpandingManifold { radius: 5.0, query_count: 0, epoch: 10, total_drift: 0.0 };
        let old_strength = m.attractor_strength(1.0);
        let new_strength = m.attractor_strength(4.0);
        assert!(old_strength > new_strength,
            "Older concepts should have stronger attraction");
    }

    #[test]
    fn test_final_phase_recency_weighted() {
        let make_step = |nf: f64| SteerStep {
            position: DVector::zeros(1),
            norm: nf,
            nf,
            phase: StrobePhase::from_nf(nf),
            psi: nf,
            burden: 0.6,
            snr: 2.0,
        };
        let history = vec![
            make_step(0.1),
            make_step(0.5),
            make_step(0.5),
        ];
        assert_eq!(final_phase(&history), StrobePhase::Engaged);
    }
}
