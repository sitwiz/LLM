use serde::{Serialize, Deserialize};
use crate::soul::manifold::StrobePhase;

const INITIAL_RADIUS: f64 = 1.0;
const UNDERSTANDING_INCREMENT: f64 = 0.001;
const TRANSCENDENT_INCREMENT: f64  = 0.01;
pub const MAX_RADIUS: f64 = 100.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandingManifold {
    pub radius:      f64,
    pub query_count: u64,
    pub epoch:       u32,
    pub total_drift: f64,
}

impl ExpandingManifold {
    pub fn new() -> Self {
        Self {
            radius:      INITIAL_RADIUS,
            query_count: 0,
            epoch:       0,
            total_drift: 0.0,
        }
    }

    pub fn expand(&mut self, phase: &StrobePhase) -> f64 {
        self.query_count += 1;
        let increment = match phase {
            StrobePhase::Transcendent  => TRANSCENDENT_INCREMENT,
            StrobePhase::Understanding => UNDERSTANDING_INCREMENT,
            _                          => 0.0,
        };

        if increment > 0.0 {
            let old_radius = self.radius;
            let expansion = increment * (1.0 - self.radius / MAX_RADIUS).max(0.0);
            self.radius = (self.radius + expansion).min(MAX_RADIUS);
            self.total_drift += self.radius - old_radius;

            if (self.radius * 100.0) as u32 > (old_radius * 100.0) as u32 {
                self.epoch += 1;
                println!("  [Manifold] Epoch {} — radius expanded to {:.4}",
                    self.epoch, self.radius);
            }
        }

        self.radius
    }

    pub fn frontier_radius(&self) -> f64 {
        self.radius * 0.95
    }

    pub fn expansion_factor(&self) -> f64 {
        self.radius / INITIAL_RADIUS
    }

    pub fn attractor_strength(&self, concept_radius: f64) -> f64 {
        let age = (self.radius - concept_radius).max(0.0);
        1.0 + age * 0.1
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expansion_on_transcendent() {
        let mut m = ExpandingManifold::new();
        let r0 = m.radius;
        m.expand(&StrobePhase::Transcendent);
        assert!(m.radius > r0);
    }

    #[test]
    fn test_no_expansion_on_dark() {
        let mut m = ExpandingManifold::new();
        let r0 = m.radius;
        m.expand(&StrobePhase::Dark);
        assert!((m.radius - r0).abs() < 1e-10);
    }

    #[test]
    fn test_asymptotic_limit() {
        let mut m = ExpandingManifold::new();
        for _ in 0..100000 {
            m.expand(&StrobePhase::Transcendent);
        }
        assert!(m.radius < MAX_RADIUS);
    }

    #[test]
    fn test_older_concepts_stronger() {
        let m = ExpandingManifold {
            radius: 2.0, query_count: 100, epoch: 5, total_drift: 1.0
        };
        let old_strength = m.attractor_strength(0.5);
        let new_strength = m.attractor_strength(1.8);
        assert!(old_strength > new_strength);
    }
}
