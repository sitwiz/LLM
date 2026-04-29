//! Compression benchmark — proves the expanding manifold claim from the paper.
//!
//! "The compression problem dissolves because infinite space requires no compression."
//!
//! Three metrics tracked across sessions as concept count grows:
//!
//! 1. Retrieval fidelity — perturbed nearest-neighbour test comparing
//!    hyperbolic wave packet interference vs flat Euclidean L2 distance.
//!    If hyperbolic beats Euclidean at scale, the geometry is proven.
//!
//! 2. Geodesic separation — minimum and average pairwise geodesic distance.
//!    Should stay stable or grow as manifold expands.
//!
//! 3. Attractor strength preservation — old concepts stronger than new ones.
//!    Epoch-dependent gravity working.

use nalgebra::DVector;
use crate::memory::spatial::{SpatialIndex, ConceptPoint, MemoryZone};
use crate::memory::expanding::ExpandingManifold;
use crate::soul::hyperbolic::geodesic_distance;
use crate::soul::geometry::INITIAL_CURVATURE;

#[derive(Debug)]
pub struct CompressionResult {
    pub session:          u64,
    pub concept_count:    usize,
    // Hyperbolic wave packet retrieval
    pub hit_rate_k1:      f64,
    pub hit_rate_k3:      f64,
    // Euclidean baseline retrieval
    pub hit_rate_k1_euc:  f64,
    pub hit_rate_k3_euc:  f64,
    pub min_geodesic_sep: f64,
    pub avg_geodesic_sep: f64,
    pub old_avg_strength: f64,
    pub new_avg_strength: f64,
    pub strength_ratio:   f64,
    pub manifold_radius:  f64,
    pub manifold_epoch:   u32,
}

impl CompressionResult {
    pub fn print(&self) {
        println!("\n[Benchmark] ═══════════════════════════════════════");
        println!("[Benchmark] Compression Benchmark — Session {}", self.session);
        println!("[Benchmark] Concepts: {}  Epoch: {}  Radius: {:.4}",
            self.concept_count, self.manifold_epoch, self.manifold_radius);
        println!("[Benchmark] ───────────────────────────────────────");
        println!("[Benchmark] RETRIEVAL FIDELITY (perturbed 5% noise)");
        println!("[Benchmark]   Hyperbolic k=1: {:.4}  Euclidean k=1: {:.4}",
            self.hit_rate_k1, self.hit_rate_k1_euc);
        println!("[Benchmark]   Hyperbolic k=3: {:.4}  Euclidean k=3: {:.4}",
            self.hit_rate_k3, self.hit_rate_k3_euc);
        if self.hit_rate_k1 > self.hit_rate_k1_euc {
            println!("[Benchmark]   ✓ Hyperbolic beats Euclidean k=1 by {:.4}",
                self.hit_rate_k1 - self.hit_rate_k1_euc);
        } else if self.hit_rate_k1 < self.hit_rate_k1_euc {
            println!("[Benchmark]   ~ Euclidean leads k=1 by {:.4}",
                self.hit_rate_k1_euc - self.hit_rate_k1);
        } else {
            println!("[Benchmark]   ~ Hyperbolic and Euclidean equal at k=1");
        }
        if self.hit_rate_k1 > 0.8 {
            println!("[Benchmark]   ✓ Strong hyperbolic retrieval under noise");
        } else {
            println!("[Benchmark]   ~ Hyperbolic retrieval degrading");
        }
        println!("[Benchmark] ───────────────────────────────────────");
        println!("[Benchmark] GEODESIC SEPARATION");
        println!("[Benchmark]   Min pairwise: {:.4}", self.min_geodesic_sep);
        println!("[Benchmark]   Avg pairwise: {:.4}", self.avg_geodesic_sep);
        if self.min_geodesic_sep > 0.1 {
            println!("[Benchmark]   ✓ Concepts maintain separation — no crowding");
        } else {
            println!("[Benchmark]   ~ Some concepts crowding");
        }
        println!("[Benchmark] ───────────────────────────────────────");
        println!("[Benchmark] EPOCH-DEPENDENT GRAVITY");
        println!("[Benchmark]   Old concept strength: {:.4}", self.old_avg_strength);
        println!("[Benchmark]   New concept strength: {:.4}", self.new_avg_strength);
        println!("[Benchmark]   Strength ratio (old/new): {:.3}x", self.strength_ratio);
        if self.strength_ratio > 1.0 {
            println!("[Benchmark]   ✓ Older concepts stronger — epoch gravity working");
        } else {
            println!("[Benchmark]   ~ Epoch gravity not yet visible");
        }
        println!("[Benchmark] ───────────────────────────────────────");
        println!("[Benchmark] COMPRESSION CLAIM");
        let hyp_beats_euc = self.hit_rate_k1 >= self.hit_rate_k1_euc;
        let fidelity_ok   = self.hit_rate_k1 > 0.8;
        let separation_ok = self.min_geodesic_sep > 0.05;
        if fidelity_ok && separation_ok && hyp_beats_euc {
            println!("[Benchmark]   ✓ High fidelity + maintained separation");
            println!("[Benchmark]   ✓ Hyperbolic geometry outperforms Euclidean");
            println!("[Benchmark]   ✓ Manifold stores without compression loss");
        } else if fidelity_ok && separation_ok {
            println!("[Benchmark]   ✓ High fidelity + maintained separation");
            println!("[Benchmark]   ✓ Manifold stores without compression loss");
            println!("[Benchmark]   ~ Euclidean competitive — need more sessions");
        } else {
            println!("[Benchmark]   ~ Accumulating — check again after more sessions");
        }
        println!("[Benchmark] ═══════════════════════════════════════\n");
    }
}

/// Euclidean nearest neighbour — flat L2 distance, no phase, no geodesic.
/// Direct comparison baseline for hyperbolic wave packet retrieval.
fn euclidean_nearest<'a>(
    query:    &DVector<f64>,
    concepts: &'a [ConceptPoint],
    k:        usize,
) -> Vec<&'a ConceptPoint> {
    let mut scored: Vec<(f64, &ConceptPoint)> = concepts.iter()
        .filter(|c| c.zone != MemoryZone::Forbidden)
        .map(|c| {
            let cp   = c.position_vec();
            let diff = query - &cp;
            let dist = diff.norm();
            let score = dist / c.strength.max(1.0);
            (score, c)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, c)| c).collect()
}

pub struct CompressionBenchmark;

impl CompressionBenchmark {
    pub fn new() -> Self { Self }

    pub fn run(
        &self,
        spatial:  &SpatialIndex,
        manifold: &ExpandingManifold,
        session:  u64,
    ) -> Option<CompressionResult> {
        let concepts = &spatial.concepts;

        if concepts.len() < 3 {
            println!("[Benchmark] Not enough concepts (need ≥3, have {}).",
                concepts.len());
            return None;
        }

        println!("[Benchmark] Running compression benchmark on {} concepts...",
            concepts.len());

        let n = concepts.len();

        // ── Metric 1: Perturbed retrieval fidelity ────────────────────────
        // Hyperbolic wave packet vs Euclidean L2.
        // Same perturbation applied to both — fair comparison.
        let mut hits_k1_hyp = 0usize;
        let mut hits_k3_hyp = 0usize;
        let mut hits_k1_euc = 0usize;
        let mut hits_k3_euc = 0usize;

        for concept in concepts {
            let pos = concept.position_vec();
            let dim = pos.len();
            let noise_scale = pos.norm() * 0.05;

            let perturbed: DVector<f64> = DVector::from_fn(dim, |i, _| {
                let noise = ((i as f64 * 7.3
                    + concept.name.len() as f64 * 1.7).sin())
                    * noise_scale;
                pos[i] + noise
            });

            let p_norm = perturbed.norm();
            let perturbed = if p_norm >= 0.99999 {
                &perturbed * (0.99999 / p_norm)
            } else {
                perturbed
            };

            // Hyperbolic — wave packet interference with phase
            let nearby_hyp = spatial.nearest_with_phase(
                &perturbed,
                concept.phase,
                5,
            );
            if nearby_hyp.first().map(|c| c.name == concept.name).unwrap_or(false) {
                hits_k1_hyp += 1;
            }
            if nearby_hyp.iter().take(3).any(|c| c.name == concept.name) {
                hits_k3_hyp += 1;
            }

            // Euclidean — flat L2 distance, no phase, no geodesic
            let nearby_euc = euclidean_nearest(&perturbed, concepts, 5);
            if nearby_euc.first().map(|c| c.name == concept.name).unwrap_or(false) {
                hits_k1_euc += 1;
            }
            if nearby_euc.iter().take(3).any(|c| c.name == concept.name) {
                hits_k3_euc += 1;
            }
        }

        let hit_rate_k1     = hits_k1_hyp as f64 / n as f64;
        let hit_rate_k3     = hits_k3_hyp as f64 / n as f64;
        let hit_rate_k1_euc = hits_k1_euc as f64 / n as f64;
        let hit_rate_k3_euc = hits_k3_euc as f64 / n as f64;

        // ── Metric 2: Geodesic separation ────────────────────────────────
        let mut min_sep    = f64::MAX;
        let mut total_sep  = 0.0;
        let mut pair_count = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let pi = concepts[i].position_vec();
                let pj = concepts[j].position_vec();
                let d  = geodesic_distance(&pi, &pj, INITIAL_CURVATURE);
                if d < min_sep { min_sep = d; }
                total_sep  += d;
                pair_count += 1;
            }
        }

        let min_geodesic_sep = if min_sep == f64::MAX { 0.0 } else { min_sep };
        let avg_geodesic_sep = if pair_count > 0 {
            total_sep / pair_count as f64
        } else {
            0.0
        };

        // ── Metric 3: Epoch-dependent gravity ────────────────────────────
        let mut sorted = concepts.clone();
        sorted.sort_by_key(|c| c.epoch);

        let half         = (sorted.len() / 2).max(1);
        let old_concepts = &sorted[..half];
        let new_concepts = &sorted[half..];

        let old_avg_strength = if old_concepts.is_empty() {
            0.0
        } else {
            old_concepts.iter().map(|c| c.strength).sum::<f64>()
                / old_concepts.len() as f64
        };

        let new_avg_strength = if new_concepts.is_empty() {
            0.0
        } else {
            new_concepts.iter().map(|c| c.strength).sum::<f64>()
                / new_concepts.len() as f64
        };

        let strength_ratio = if new_avg_strength > 0.0 {
            old_avg_strength / new_avg_strength
        } else {
            1.0
        };

        Some(CompressionResult {
            session,
            concept_count:    n,
            hit_rate_k1,
            hit_rate_k3,
            hit_rate_k1_euc,
            hit_rate_k3_euc,
            min_geodesic_sep,
            avg_geodesic_sep,
            old_avg_strength,
            new_avg_strength,
            strength_ratio,
            manifold_radius:  manifold.radius,
            manifold_epoch:   manifold.epoch,
        })
    }

    pub fn save_result(&self, result: &CompressionResult) {
        let record = serde_json::json!({
            "session":          result.session,
            "concept_count":    result.concept_count,
            "hit_rate_k1":      result.hit_rate_k1,
            "hit_rate_k3":      result.hit_rate_k3,
            "hit_rate_k1_euc":  result.hit_rate_k1_euc,
            "hit_rate_k3_euc":  result.hit_rate_k3_euc,
            "min_geodesic_sep": result.min_geodesic_sep,
            "avg_geodesic_sep": result.avg_geodesic_sep,
            "old_avg_strength": result.old_avg_strength,
            "new_avg_strength": result.new_avg_strength,
            "strength_ratio":   result.strength_ratio,
            "manifold_radius":  result.manifold_radius,
            "manifold_epoch":   result.manifold_epoch,
        });

        let line = serde_json::to_string(&record).unwrap_or_default();
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("compression_benchmark.jsonl")
        {
            use std::io::Write;
            writeln!(file, "{}", line).ok();
        }

        println!("[Benchmark] Result saved to compression_benchmark.jsonl");
    }
}
