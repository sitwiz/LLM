//! Layer — Attractor formation mechanics.
//!
//! An attractor is a concept that has been located in the Poincaré ball
//! by one or more personalities. Its position is the weighted geodesic
//! centroid of all contributing soul positions.
//!
//! Formation:   new concept creates a new attractor
//! Reinforcement: existing concept pulls attractor toward new soul position
//! Merge:       two attractors closer than AUTO_MERGE_DIST become one
//! Split:       one attractor with high contributor variance becomes two
//! Conflict:    two personalities pulling in opposite directions — flagged

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use crate::soul::hyperbolic::{geodesic_distance, geodesic_interpolate};
use crate::soul::geometry::INITIAL_CURVATURE;

/// Auto-merge threshold — concepts closer than this are the same attractor.
/// Set from empirical data: minimum observed separation between distinct
/// concepts is 0.8435. This is well below that.
pub const AUTO_MERGE_DIST:   f64 = 0.08;

/// Merge candidate threshold — flag for review but do not auto-merge.
pub const MERGE_CANDIDATE_DIST: f64 = 0.25;

/// Split trigger — contributor position variance above this means the
/// attractor is being pulled in genuinely different directions.
pub const SPLIT_VARIANCE_THRESHOLD: f64 = 0.40;

/// A single contributor to an attractor — records which personality
/// engaged with this concept and where their soul was at the time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttractorContributor {
    pub personality: String,
    pub position:    Vec<f64>,   // soul position at time of contribution
    pub weight:      f64,        // contribution weight — increases with reinforcement
    pub visits:      u32,
}

impl AttractorContributor {
    pub fn new(personality: &str, position: &DVector<f64>) -> Self {
        Self {
            personality: personality.to_string(),
            position:    position.iter().cloned().collect(),
            weight:      1.0,
            visits:      1,
        }
    }

    pub fn position_vec(&self) -> DVector<f64> {
        DVector::from_vec(self.position.clone())
    }

    pub fn reinforce(&mut self, position: &DVector<f64>) {
        self.visits += 1;
        self.weight  = 1.0 + (self.visits as f64).ln();
        // Pull stored position toward new soul position
        let t       = 0.1 * (1.0 - 1.0 / self.visits as f64);
        let updated = geodesic_interpolate(
            &self.position_vec(), position, t, INITIAL_CURVATURE,
        );
        self.position = updated.iter().cloned().collect();
    }
}

/// An attractor — a concept located in the Poincaré ball by consensus
/// of all personalities that have engaged with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attractor {
    pub name:         String,
    pub position:     Vec<f64>,          // weighted geodesic centroid
    pub contributors: Vec<AttractorContributor>,
    pub total_visits: u32,
    pub strength:     f64,
    pub epoch:        u32,
    pub split_flag:   bool,              // true if contributor variance is high
    pub norm:         f64,
}

impl Attractor {
    /// Create a new attractor from a first contribution.
    pub fn new(
        name:        &str,
        position:    &DVector<f64>,
        personality: &str,
        epoch:       u32,
    ) -> Self {
        Self {
            name:         name.to_string(),
            position:     position.iter().cloned().collect(),
            contributors: vec![AttractorContributor::new(personality, position)],
            total_visits: 1,
            strength:     1.2,
            epoch,
            split_flag:   false,
            norm:         position.norm(),
        }
    }

    pub fn position_vec(&self) -> DVector<f64> {
        DVector::from_vec(self.position.clone())
    }

    /// Absorb a new contribution from a personality.
    /// Updates the attractor position toward the weighted centroid.
    pub fn contribute(
        &mut self,
        personality: &str,
        soul_position: &DVector<f64>,
    ) {
        self.total_visits += 1;
        self.strength = 1.2 + (self.total_visits as f64).ln() * 0.2;

        // Update or add contributor
        if let Some(existing) = self.contributors.iter_mut()
            .find(|c| c.personality == personality)
        {
            existing.reinforce(soul_position);
        } else {
            self.contributors.push(
                AttractorContributor::new(personality, soul_position)
            );
        }

        // Recompute weighted centroid across all contributors
        self.position = self.compute_centroid().iter().cloned().collect();
        self.norm     = self.position_vec().norm();

        // Check split condition
        self.split_flag = self.contributor_variance() > SPLIT_VARIANCE_THRESHOLD;
        if self.split_flag {
            println!("  [Attractor] Split flag set on {:?} — contributor variance={:.4}",
                self.name, self.contributor_variance());
        }
    }

    /// Weighted geodesic centroid of all contributor positions.
    /// Uses iterative Fréchet mean approximation on the Poincaré ball.
    fn compute_centroid(&self) -> DVector<f64> {
        if self.contributors.is_empty() {
            return self.position_vec();
        }

        let total_weight: f64 = self.contributors.iter().map(|c| c.weight).sum();
        if total_weight < 1e-10 {
            return self.position_vec();
        }

        // Iterative weighted Fréchet mean — 5 iterations is sufficient
        let mut centroid = self.position_vec();
        for _ in 0..5 {
            let mut step = DVector::zeros(centroid.len());
            for c in &self.contributors {
                let w   = c.weight / total_weight;
                let log = crate::soul::hyperbolic::log_map(
                    &centroid, &c.position_vec(), INITIAL_CURVATURE,
                );
                step += log * w;
            }
            centroid = crate::soul::hyperbolic::exp_map(
                &centroid, &step, INITIAL_CURVATURE,
            );
        }
        centroid
    }

    /// Mean pairwise geodesic distance between contributor positions.
    /// High variance means the attractor should split.
    pub fn contributor_variance(&self) -> f64 {
        if self.contributors.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..self.contributors.len() {
            for j in (i+1)..self.contributors.len() {
                let a = self.contributors[i].position_vec();
                let b = self.contributors[j].position_vec();
                total += geodesic_distance(&a, &b, INITIAL_CURVATURE);
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total / count as f64 }
    }

    /// Merge another attractor into this one.
    /// All contributors from other are absorbed.
    pub fn merge(&mut self, other: &Attractor) {
        println!("  [Attractor] Merging {:?} into {:?}", other.name, self.name);
        for c in &other.contributors {
            self.contribute(&c.personality, &c.position_vec());
        }
        self.total_visits += other.total_visits;
        self.strength = 1.2 + (self.total_visits as f64).ln() * 0.2;
    }

    /// Split this attractor into two along the axis of maximum contributor spread.
    /// Returns two new attractors. The original should be removed.
    pub fn split(&self, epoch: u32) -> (Attractor, Attractor) {
        println!("  [Attractor] Splitting {:?} — variance={:.4}",
            self.name, self.contributor_variance());

        // Sort contributors by their projection onto the first principal axis
        // Approximated by sorting by norm distance from centroid
        let centroid = self.position_vec();
        let mut sorted = self.contributors.clone();
        sorted.sort_by(|a, b| {
            let da = geodesic_distance(&a.position_vec(), &centroid, INITIAL_CURVATURE);
            let db = geodesic_distance(&b.position_vec(), &centroid, INITIAL_CURVATURE);
            da.partial_cmp(&db).unwrap()
        });

        let mid        = sorted.len() / 2;
        let group_a    = &sorted[..mid.max(1)];
        let group_b    = &sorted[mid.max(1)..];

        let name_a = format!("{} [A]", self.name);
        let name_b = format!("{} [B]", self.name);

        let pos_a = if group_a.is_empty() {
            self.position_vec()
        } else {
            group_a[0].position_vec()
        };
        let pos_b = if group_b.is_empty() {
            self.position_vec()
        } else {
            group_b[0].position_vec()
        };

        let mut att_a = Attractor::new(&name_a, &pos_a, &group_a[0].personality, epoch);
        let mut att_b = Attractor::new(&name_b, &pos_b, &group_b[0].personality, epoch);

        for c in group_a.iter().skip(1) {
            att_a.contribute(&c.personality, &c.position_vec());
        }
        for c in group_b.iter().skip(1) {
            att_b.contribute(&c.personality, &c.position_vec());
        }

        (att_a, att_b)
    }
}

/// The attractor registry — owns all attractors and handles
/// formation, merge detection, and split triggering.
pub struct AttractorRegistry {
    pub attractors: Vec<Attractor>,
}

impl AttractorRegistry {
    pub fn new() -> Self {
        Self { attractors: Vec::new() }
    }

    /// Insert or update an attractor for a concept.
    /// Returns merge candidates if any are found.
    pub fn insert(
        &mut self,
        name:          &str,
        soul_position: &DVector<f64>,
        personality:   &str,
        epoch:         u32,
    ) -> Vec<String> {
        // Find existing attractor by name
        if let Some(existing) = self.attractors.iter_mut()
            .find(|a| a.name == name)
        {
            existing.contribute(personality, soul_position);
            println!("  [Attractor] Reinforced {:?} contributors={} strength={:.3}",
                name, existing.contributors.len(), existing.strength);
        } else {
            let attractor = Attractor::new(name, soul_position, personality, epoch);
            println!("  [Attractor] Formed {:?} pos_norm={:.4}",
                name, attractor.norm);
            self.attractors.push(attractor);
        }

        // Check for merge candidates
        self.find_merge_candidates(name)
    }

    /// Find attractors that are close enough to the named attractor
    /// to be merge candidates.
    fn find_merge_candidates(&self, name: &str) -> Vec<String> {
        let source = match self.attractors.iter().find(|a| a.name == name) {
            Some(a) => a,
            None    => return Vec::new(),
        };

        let mut candidates = Vec::new();

        for other in &self.attractors {
            if other.name == name { continue; }
            let dist = geodesic_distance(
                &source.position_vec(),
                &other.position_vec(),
                INITIAL_CURVATURE,
            );

            if dist < AUTO_MERGE_DIST {
                println!("  [Attractor] AUTO-MERGE: {:?} and {:?} dist={:.4}",
                    name, other.name, dist);
                candidates.push(other.name.clone());
            } else if dist < MERGE_CANDIDATE_DIST {
                println!("  [Attractor] Merge candidate: {:?} and {:?} dist={:.4}",
                    name, other.name, dist);
                candidates.push(other.name.clone());
            }
        }

        candidates
    }

    /// Execute pending auto-merges — call after insert.
    pub fn execute_merges(&mut self) {
        let mut to_merge: Vec<(String, String)> = Vec::new();

        for i in 0..self.attractors.len() {
            for j in (i+1)..self.attractors.len() {
                let dist = geodesic_distance(
                    &self.attractors[i].position_vec(),
                    &self.attractors[j].position_vec(),
                    INITIAL_CURVATURE,
                );
                if dist < AUTO_MERGE_DIST {
                    to_merge.push((
                        self.attractors[i].name.clone(),
                        self.attractors[j].name.clone(),
                    ));
                }
            }
        }

        for (keep, absorb) in to_merge {
            let absorb_data = match self.attractors.iter()
                .find(|a| a.name == absorb)
                .cloned()
            {
                Some(d) => d,
                None    => continue,
            };
            if let Some(keeper) = self.attractors.iter_mut()
                .find(|a| a.name == keep)
            {
                keeper.merge(&absorb_data);
            }
            self.attractors.retain(|a| a.name != absorb);
        }
    }

    /// Execute pending splits — call after dream cycle.
    pub fn execute_splits(&mut self, epoch: u32) {
        let split_names: Vec<String> = self.attractors.iter()
            .filter(|a| a.split_flag && a.contributors.len() >= 2)
            .map(|a| a.name.clone())
            .collect();

        for name in split_names {
            if let Some(pos) = self.attractors.iter().position(|a| a.name == name) {
                let (att_a, att_b) = self.attractors[pos].split(epoch);
                self.attractors.remove(pos);
                self.attractors.push(att_a);
                self.attractors.push(att_b);
            }
        }
    }

    pub fn len(&self)      -> usize { self.attractors.len() }
    pub fn is_empty(&self) -> bool  { self.attractors.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::geometry::project_to_ball;

    fn ball_vec(seed: f64) -> DVector<f64> {
        let v: Vec<f64> = (0..256)
            .map(|i| ((i as f64 + seed) * 1.7).sin() * 0.3)
            .collect();
        project_to_ball(&DVector::from_vec(v))
    }

    #[test]
    fn test_attractor_formation() {
        let pos = ball_vec(1.0);
        let att = Attractor::new("consciousness", &pos, "Khaos", 0);
        assert_eq!(att.name, "consciousness");
        assert_eq!(att.contributors.len(), 1);
        assert!(att.norm < 1.0);
    }

    #[test]
    fn test_contribution_adds_personality() {
        let pos_a = ball_vec(1.0);
        let pos_b = ball_vec(2.0);
        let mut att = Attractor::new("consciousness", &pos_a, "Khaos", 0);
        att.contribute("UnifiedOmniAGI", &pos_b);
        assert_eq!(att.contributors.len(), 2);
        assert_eq!(att.total_visits, 2);
    }

    #[test]
    fn test_strength_increases_with_visits() {
        let pos = ball_vec(1.0);
        let mut att = Attractor::new("x", &pos, "Khaos", 0);
        let s0 = att.strength;
        att.contribute("Gaia", &ball_vec(2.0));
        att.contribute("Eros", &ball_vec(3.0));
        assert!(att.strength > s0);
    }

    #[test]
    fn test_centroid_inside_ball() {
        let pos_a = ball_vec(1.0);
        let pos_b = ball_vec(5.0);
        let mut att = Attractor::new("x", &pos_a, "Khaos", 0);
        att.contribute("Gaia", &pos_b);
        assert!(att.position_vec().norm() < 1.0);
    }

    #[test]
    fn test_registry_insert_and_reinforce() {
        let mut reg = AttractorRegistry::new();
        let pos = ball_vec(1.0);
        reg.insert("consciousness", &pos, "Khaos", 0);
        reg.insert("consciousness", &ball_vec(2.0), "UnifiedOmniAGI", 0);
        assert_eq!(reg.attractors.len(), 1);
        assert_eq!(reg.attractors[0].total_visits, 2);
    }

    #[test]
    fn test_registry_merge_close_concepts() {
        let mut reg = AttractorRegistry::new();
        let pos_a = ball_vec(1.0);
        // Create two slightly different positions that are very close
        let mut pos_b = pos_a.clone();
        pos_b[0] += 0.001;
        let pos_b = project_to_ball(&pos_b);
        reg.insert("concept_a", &pos_a, "Khaos", 0);
        reg.insert("concept_b", &pos_b, "Gaia",  0);
        reg.execute_merges();
        assert_eq!(reg.attractors.len(), 1);
    }

    #[test]
    fn test_no_merge_for_distant_concepts() {
        let mut reg = AttractorRegistry::new();
        reg.insert("consciousness", &ball_vec(1.0), "Khaos", 0);
        reg.insert("rust_memory",   &ball_vec(9.0), "Gaia",  0);
        reg.execute_merges();
        assert_eq!(reg.attractors.len(), 2);
    }

    #[test]
    fn test_contributor_variance_single() {
        let pos = ball_vec(1.0);
        let att = Attractor::new("x", &pos, "Khaos", 0);
        assert_eq!(att.contributor_variance(), 0.0);
    }

    #[test]
    fn test_split_produces_two_attractors() {
        let mut reg = AttractorRegistry::new();
        let pos = ball_vec(1.0);
        reg.insert("x", &pos,        "Khaos",    0);
        reg.insert("x", &ball_vec(9.0), "Gaia",  0);
        // Force split flag
        reg.attractors[0].split_flag = true;
        reg.execute_splits(1);
        assert_eq!(reg.attractors.len(), 2);
    }
}
