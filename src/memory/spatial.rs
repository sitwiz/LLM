//! Spatial memory index — hyperbolic version with complex wave packet interference.
//!
//! Each concept is a wave packet in the Poincaré ball with:
//!   - position μ — mean geodesic location
//!   - width σ — uncertainty, consolidates toward 0 via Zeno property
//!   - phase φ — from personality domain, enables destructive interference
//!
//! Retrieval score uses full interference term:
//!   score = exp(-d²/2σ²) × cos(φ_C + k×d - φ_query) × strength
//!
//! Domain separation is achieved by consolidating each concept toward its
//! personality's domain embedding direction rather than toward the shared origin.
//! The anchor is passed in from the caller (Quorum) which holds the real
//! semantic embeddings — no hardcoding needed.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::soul::hyperbolic::geodesic_distance;
use crate::soul::INITIAL_CURVATURE;
use std::f64::consts::PI;

pub const WAVE_K:     f64 = 0.8;
pub const SIGMA_INIT: f64 = 0.5;
pub const SIGMA_MIN:  f64 = 0.05;
pub const SIGMA_MAX:  f64 = 0.8;

/// Phase angle for each personality domain.
pub fn personality_phase(personality: &str) -> f64 {
    match personality {
        "Khaos"          => 0.0,
        "Tartaros"       => PI / 4.0,
        "Gaia"           => PI / 2.0,
        "Eros"           => 3.0 * PI / 4.0,
        "UnifiedOmniAGI" => PI,
        _                => PI / 2.0,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryZone {
    Forbidden,
    Core,
    Working,
    Frontier,
}

impl MemoryZone {
    pub fn from_norm(norm: f64) -> Self {
        if      norm < 0.05 { MemoryZone::Forbidden }
        else if norm < 0.25 { MemoryZone::Core }
        else if norm < 0.60 { MemoryZone::Working }
        else                { MemoryZone::Frontier }
    }

    pub fn label(&self) -> &str {
        match self {
            MemoryZone::Forbidden => "forbidden",
            MemoryZone::Core      => "core",
            MemoryZone::Working   => "working",
            MemoryZone::Frontier  => "frontier",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptPoint {
    pub name:         String,
    pub position:     Vec<f64>,
    pub personality:  String,
    pub zone:         MemoryZone,
    pub visit_count:  u32,
    pub strength:     f64,
    pub norm:         f64,
    pub epoch:        u32,
    pub target_depth: f64,
    pub sigma:        f64,
    pub phase:        f64,
}

impl ConceptPoint {
    pub fn new(
        name:        &str,
        position:    &DVector<f64>,
        _frontier:   f64,
        personality: &str,
        strength:    f64,
        _radius:     f64,
        epoch:       u32,
    ) -> Self {
        let norm = position.norm();
        let mut hash: u64 = 5381;
        for b in name.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(b as u64);
        }
        let depth_variation = (hash % 1000) as f64 / 10000.0;

        Self {
            name:         name.to_string(),
            position:     position.iter().cloned().collect(),
            personality:  personality.to_string(),
            zone:         MemoryZone::from_norm(norm),
            visit_count:  1,
            strength,
            norm,
            epoch,
            target_depth: 0.60 + depth_variation,
            sigma:        SIGMA_INIT,
            phase:        personality_phase(personality),
        }
    }

    pub fn position_vec(&self) -> DVector<f64> {
        DVector::from_vec(self.position.clone())
    }

    pub fn reinforce(&mut self, amount: f64) {
        self.visit_count += 1;
        self.strength = 1.2 + (self.visit_count as f64).ln() * amount;
    }

    /// Update depth and sigma from governance signal — Zeno consolidation.
    ///
    /// anchor is the personality's domain embedding projected to unit length.
    /// Concepts drift toward their personality's semantic region of the ball,
    /// not toward the shared origin — keeping domains separated after consolidation.
    pub fn update_depth(&mut self, approved: bool, anchor: &DVector<f64>) {
        // Precision weighting — high-certainty memories resist updating.
        // precision = 1/σ² — normalised to 0..1 range using SIGMA_MAX.
        // Low sigma (consolidated) → small update_rate → barely moves.
        // High sigma (frontier)    → large update_rate → updates freely.
        let update_rate = (self.sigma / SIGMA_MAX).min(1.0);

        if approved {
            self.target_depth = (self.target_depth * (1.0 - 0.03 * update_rate)).max(0.05);
            self.sigma        = (self.sigma * (1.0 - 0.05 * update_rate)).max(SIGMA_MIN);
        } else {
            self.target_depth = (self.target_depth * (1.0 + 0.02 * update_rate)).min(0.85);
            self.sigma        = (self.sigma * (1.0 + 0.02 * update_rate)).min(SIGMA_MAX);
        }
        let new_pos = anchor * self.target_depth;
        self.position = new_pos.iter().cloned().collect();
        self.norm     = self.target_depth;
        self.zone     = MemoryZone::from_norm(self.norm);
   }
}
pub struct SpatialIndex {
    pub concepts:    Vec<ConceptPoint>,
    pub soul_radius: f64,
    pub curvature:   f64,
}

impl SpatialIndex {
    pub fn new(soul_radius: f64) -> Self {
        Self {
            concepts:  Vec::new(),
            soul_radius,
            curvature: INITIAL_CURVATURE,
        }
    }

    pub fn load(path: &str, soul_radius: f64) -> Self {
        let concepts = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<ConceptPoint>>(&s).ok())
            .unwrap_or_default();
        println!("  [Memory] Loaded {} concepts from disk.", concepts.len());
        Self {
            concepts,
            soul_radius,
            curvature: INITIAL_CURVATURE,
        }
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        std::fs::write(path, serde_json::to_string(&self.concepts)?)?;
        Ok(())
    }

    pub fn insert(&mut self, concept: ConceptPoint) {
        if let Some(existing) = self.concepts.iter_mut()
            .find(|c| c.name == concept.name && c.personality == concept.personality)
        {
            existing.reinforce(0.2);
            println!("  [Memory] Reinforced: {:?} (visits={} strength={:.3})",
                existing.name, existing.visit_count, existing.strength);
        } else {
            println!("  [Memory] New concept: {:?} zone={} norm={:.3} σ={:.3} φ={:.3}",
                concept.name, concept.zone.label(), concept.norm,
                concept.sigma, concept.phase);
            self.concepts.push(concept);
        }
    }

    /// k nearest neighbours using complex wave packet interference score.
    ///
    /// score(C,q) = exp(-d²/2σ²) × cos(φ_C + k×d - φ_query) × strength
    pub fn nearest_with_phase(
        &self,
        query:       &DVector<f64>,
        query_phase: f64,
        k:           usize,
    ) -> Vec<&ConceptPoint> {
        if self.concepts.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(f64, &ConceptPoint)> = self.concepts.iter()
            .filter(|c| c.zone != MemoryZone::Forbidden)
            .filter_map(|c| {
                let d            = geodesic_distance(query, &c.position_vec(), self.curvature);
                let gaussian     = (-d * d / (2.0 * c.sigma * c.sigma)).exp();
                let phase_arg    = c.phase + WAVE_K * d - query_phase;
                let interference = phase_arg.cos();

                let raw_score = gaussian * interference.max(0.0);
                let score     = raw_score * c.strength.max(1.0);

                if score > 1e-10 {
                    Some((-score, c))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, c)| c).collect()
    }

    /// Standard nearest — geodesic distance only, no phase.
    pub fn nearest(&self, query: &DVector<f64>, k: usize) -> Vec<&ConceptPoint> {
        if self.concepts.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(f64, &ConceptPoint)> = self.concepts.iter()
            .filter(|c| c.zone != MemoryZone::Forbidden)
            .map(|c| {
                let dist  = geodesic_distance(query, &c.position_vec(), self.curvature);
                let score = dist / c.strength.max(1.0);
                (score, c)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, c)| c).collect()
    }

    /// Update depth and sigma of concept by name after governance vote.
    /// anchor must be the personality's domain embedding unit vector.
    pub fn consolidate_depth(&mut self, name: &str, approved: bool, anchor: &DVector<f64>) {
        if let Some(concept) = self.concepts.iter_mut().find(|c| c.name == name) {
            let old_depth = concept.target_depth;
            let old_sigma = concept.sigma;
            concept.update_depth(approved, anchor);
            println!("  [Memory] Consolidate {:?}: depth {:.4}->{:.4} σ {:.4}->{:.4} zone={}",
                &name[..name.len().min(40)],
                old_depth, concept.target_depth,
                old_sigma, concept.sigma,
                concept.zone.label());
        }
    }

    pub fn update_zones(&mut self) {
        for c in &mut self.concepts {
            c.zone = MemoryZone::from_norm(c.norm);
        }
    }

    pub fn len(&self)      -> usize { self.concepts.len() }
    pub fn is_empty(&self) -> bool  { self.concepts.is_empty() }

    pub fn set_curvature(&mut self, c: f64) {
        self.curvature = c.max(crate::soul::manifold::MIN_CURVATURE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::geometry::{project_to_ball, SOUL_DIM};
    use nalgebra::DVector;

    fn ball_vec(seed: f64, dim: usize) -> DVector<f64> {
        let v: Vec<f64> = (0..dim)
            .map(|i| ((i as f64 + seed) * 1.7).sin())
            .collect();
        project_to_ball(&DVector::from_vec(v))
    }

    /// Simple orthogonal test anchor — idx selects which block of SOUL_DIM is positive.
    fn test_anchor(idx: usize) -> DVector<f64> {
        let dim   = SOUL_DIM;
        let block = dim / 5;
        let start = idx * block;
        let mut v = vec![-0.1f64; dim];
        for i in start..(start + block).min(dim) {
            v[i] = 1.0;
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        DVector::from_vec(v.iter().map(|x| x / norm).collect())
    }

    #[test]
    fn test_zone_from_norm() {
        assert_eq!(MemoryZone::from_norm(0.01), MemoryZone::Forbidden);
        assert_eq!(MemoryZone::from_norm(0.1),  MemoryZone::Core);
        assert_eq!(MemoryZone::from_norm(0.4),  MemoryZone::Working);
        assert_eq!(MemoryZone::from_norm(0.7),  MemoryZone::Frontier);
    }

    #[test]
    fn test_personality_phases_distinct() {
        let phases = [
            personality_phase("Khaos"),
            personality_phase("Tartaros"),
            personality_phase("Gaia"),
            personality_phase("Eros"),
            personality_phase("UnifiedOmniAGI"),
        ];
        for i in 0..phases.len() {
            for j in (i+1)..phases.len() {
                assert!((phases[i] - phases[j]).abs() > 0.1,
                    "Personalities {} and {} have same phase", i, j);
            }
        }
    }

    #[test]
    fn test_anchor_consolidation_keeps_domains_separate() {
        let mut index = SpatialIndex::new(1.0);
        let pos_rust  = ball_vec(1.0, SOUL_DIM);
        let pos_philo = ball_vec(9.0, SOUL_DIM);

        let rust_concept = ConceptPoint::new(
            "fix memory leak", &pos_rust, 1.0, "Gaia", 1.2, 1.0, 0
        );
        let philo_concept = ConceptPoint::new(
            "what is consciousness", &pos_philo, 1.0, "Khaos", 1.2, 1.0, 0
        );

        index.concepts.push(rust_concept);
        index.concepts.push(philo_concept);

        let anchor_gaia  = test_anchor(0);
        let anchor_khaos = test_anchor(2);

        for _ in 0..30 {
            index.consolidate_depth("fix memory leak",       true, &anchor_gaia);
            index.consolidate_depth("what is consciousness", true, &anchor_khaos);
        }

        let rust_pos  = index.concepts[0].position_vec();
        let philo_pos = index.concepts[1].position_vec();
        let dot = rust_pos.dot(&philo_pos) / (rust_pos.norm() * philo_pos.norm()).max(1e-10);

        assert!(dot < 0.5,
            "Domains collapsed after consolidation: dot={:.3}", dot);
    }

    #[test]
    fn test_same_phase_higher_score_than_orthogonal() {
        let mut index = SpatialIndex::new(1.0);
        let pos_same = ball_vec(1.0, SOUL_DIM);
        let pos_orth = ball_vec(1.0, SOUL_DIM);

        let mut c_same = ConceptPoint::new("same", &pos_same, 1.0, "Khaos", 1.2, 1.0, 0);
        c_same.phase = 0.0;

        let mut c_orth = ConceptPoint::new("orth", &pos_orth, 1.0, "Gaia", 1.2, 1.0, 0);
        c_orth.phase = PI / 2.0;

        index.concepts.push(c_same);
        index.concepts.push(c_orth);

        let results = index.nearest_with_phase(&pos_same, 0.0, 2);
        if results.len() >= 1 {
            assert_eq!(results[0].name, "same",
                "Same-phase concept should rank first");
        }
    }

    #[test]
    fn test_opposite_phase_suppressed() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, SOUL_DIM);

        let mut c_opposite = ConceptPoint::new("opp", &pos, 1.0, "Khaos", 1.2, 1.0, 0);
        c_opposite.phase = PI;

        index.concepts.push(c_opposite);

        let results = index.nearest_with_phase(&pos, 0.0, 5);
        assert!(results.is_empty() || results[0].name != "opp",
            "Opposite phase concept should be suppressed");
    }

    #[test]
    fn test_insert_and_nearest() {
        let mut index = SpatialIndex::new(1.0);
        let pos_a = ball_vec(1.0, 32);
        let pos_b = ball_vec(9.0, 32);
        index.insert(ConceptPoint::new("alpha", &pos_a, 1.0, "Khaos", 1.2, 1.0, 0));
        index.insert(ConceptPoint::new("beta",  &pos_b, 1.0, "Gaia",  1.2, 1.0, 0));
        let nearest = index.nearest(&ball_vec(1.1, 32), 1);
        assert_eq!(nearest.len(), 1);
        assert_eq!(nearest[0].name, "alpha");
    }

    #[test]
    fn test_reinforce_increases_visits() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, 32);
        index.insert(ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 0));
        index.insert(ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 0));
        assert_eq!(index.concepts[0].visit_count, 2);
    }

    #[test]
    fn test_nearest_excludes_forbidden() {
        let mut index = SpatialIndex::new(1.0);
        let mut forbidden = ConceptPoint::new(
            "void", &DVector::zeros(32), 1.0, "Khaos", 1.2, 1.0, 0
        );
        forbidden.zone = MemoryZone::Forbidden;
        index.concepts.push(forbidden);
        let nearest = index.nearest(&ball_vec(1.0, 32), 5);
        assert!(nearest.iter().all(|c| c.zone != MemoryZone::Forbidden));
    }

    #[test]
    fn test_update_zones() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, 32);
        let mut c = ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 0);
        c.zone = MemoryZone::Core;
        c.norm = 0.7;
        index.concepts.push(c);
        index.update_zones();
        assert_eq!(index.concepts[0].zone, MemoryZone::Frontier);
    }

    #[test]
    fn test_epoch_stamped_on_insert() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, 32);
        index.insert(ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 5));
        assert_eq!(index.concepts[0].epoch, 5);
    }

    #[test]
    fn test_depth_consolidates_on_approval() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, 32);
        index.insert(ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 0));
        let initial_depth = index.concepts[0].target_depth;
        let initial_sigma = index.concepts[0].sigma;
        let anchor = test_anchor(0);
        index.consolidate_depth("x", true, &anchor);
        assert!(index.concepts[0].target_depth < initial_depth);
        assert!(index.concepts[0].sigma < initial_sigma);
    }

    #[test]
    fn test_depth_drifts_on_block() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, 32);
        let mut c = ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 0);
        c.target_depth = 0.5;
        index.concepts.push(c);
        let anchor = test_anchor(0);
        index.consolidate_depth("x", false, &anchor);
        assert!(index.concepts[0].target_depth > 0.5);
    }

    #[test]
    fn test_zeno_never_reaches_zero() {
        let mut index = SpatialIndex::new(1.0);
        let pos = ball_vec(1.0, 32);
        index.insert(ConceptPoint::new("x", &pos, 1.0, "Khaos", 1.2, 1.0, 0));
        let anchor = test_anchor(0);
        for _ in 0..1000 {
            index.consolidate_depth("x", true, &anchor);
        }
        assert!(index.concepts[0].target_depth >= 0.05);
        assert!(index.concepts[0].sigma >= SIGMA_MIN);
    }
}
