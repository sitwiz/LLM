//! BrainSystem — owns all 14 regions, processes queries,
//! learns from NeoCorticalMesh approval signal.
//! Provides emergent depth assignment based on region activation.

use nalgebra::{DVector, DMatrix};
use crate::brain::region::{BrainRegion, create_all_regions};
use crate::brain::health::HealthMonitor;
use serde::{Serialize, Deserialize};
use std::path::Path;

pub const BRAIN_CONTEXT_DIM: usize = 256;
const HIDDEN_DIM:             usize = 64;
const NUM_REGIONS:            usize = 14;
const PROJECTION_DIM:         usize = NUM_REGIONS * HIDDEN_DIM; // 896

/// Depth bias per region — maps cognitive function to manifold depth.
/// Abstract/philosophical regions → deep (low norm).
/// Concrete/novelty regions → frontier (high norm).
pub const REGION_DEPTH_BIAS: [f64; 14] = [
    0.20,  // frontal_lobe      — planning, reasoning — deepest
    0.35,  // temporal_lobe     — language, memory
    0.45,  // parietal_lobe     — spatial, math
    0.62,  // occipital_lobe    — pattern, visual
    0.25,  // insular_lobe      — awareness, emotion — deep
    0.22,  // limbic            — reward, fear — deep
    0.48,  // thalamus          — attention, filter
    0.50,  // hypothalamus      — resource, balance
    0.60,  // cerebellum        — precision, skill
    0.65,  // midbrain          — novelty, alert — frontier
    0.55,  // pons              — relay, bridge
    0.70,  // medulla_oblongata — safety baseline
    0.75,  // pituitary_gland   — global regulation
    0.72,  // meninges          — boundary, context
];

/// Per-session metrics for the intelligence growth curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMetrics {
    pub session:           u64,
    pub queries_processed: u64,
    pub approvals:         u64,
    pub blocks:            u64,
    pub approval_rate:     f64,
    pub avg_system_snr:    f64,
    pub avg_depth:         f64,
}

impl BrainMetrics {
    pub fn new(session: u64) -> Self {
        Self {
            session,
            queries_processed: 0,
            approvals:         0,
            blocks:            0,
            approval_rate:     0.0,
            avg_system_snr:    0.0,
            avg_depth:         0.0,
        }
    }

    pub fn record(&mut self, approved: bool, system_snr: f64, depth: f64) {
        self.queries_processed += 1;
        if approved { self.approvals += 1; } else { self.blocks += 1; }
        self.approval_rate  = self.approvals as f64 / self.queries_processed as f64;
        self.avg_system_snr = 0.9 * self.avg_system_snr + 0.1 * system_snr;
        self.avg_depth      = 0.9 * self.avg_depth      + 0.1 * depth;
    }
}

pub struct BrainSystem {
    pub regions:    Vec<BrainRegion>,
    pub monitor:    HealthMonitor,
    pub metrics:    BrainMetrics,
    projection:     DMatrix<f64>,
    session_id:     u64,
}

impl BrainSystem {
    pub fn new() -> Self {
        let regions    = create_all_regions();
        let session_id = Self::load_session_id();
        let metrics    = Self::load_metrics(session_id);
        let projection = Self::load_or_init_projection();

        println!("  [Brain] {} regions online. Session {}. Approval rate: {:.3}",
            regions.len(), session_id, metrics.approval_rate);

        Self {
            regions,
            monitor:   HealthMonitor::new(),
            metrics,
            projection,
            session_id,
        }
    }

    /// Process query through all 14 regions.
    /// Returns (context_vector, emergent_depth).
    /// Depth emerges from weighted region activation — not hardcoded.
    pub fn process_query(&mut self, query: &str) -> (DVector<f64>, f64) {
        let token_ids = Self::tokenise(query);

        for region in &mut self.regions {
            region.reset_hidden();
        }

        for &token_id in &token_ids {
            for region in &mut self.regions {
                region.process_token(token_id);
            }
        }

        let context = self.aggregate_hidden_states();
        let depth   = self.compute_emergent_depth();

        println!("  [Brain] Context norm={:.4} Depth={:.4} SNR={:.3}",
            context.norm(), depth, self.system_snr());

        (context, depth)
    }

    /// Compute emergent depth from region activation patterns.
    /// Regions that fired strongly pull the depth toward their bias.
    /// Abstract regions (frontal, limbic) pull deep.
    /// Concrete regions (cerebellum, midbrain) pull to frontier.
    pub fn compute_emergent_depth(&self) -> f64 {
        let total_snr: f64 = self.regions.iter().map(|r| r.health.snr).sum();
        let total_snr = total_snr.max(1e-10);

        let mut weighted_depth = 0.0f64;
        let mut total_weight   = 0.0f64;

        for (i, region) in self.regions.iter().enumerate() {
            let activation = region.hidden.norm();
            let snr_weight = region.health.snr / total_snr;
            let weight     = activation * snr_weight;
            let bias       = REGION_DEPTH_BIAS.get(i).copied().unwrap_or(0.48);

            weighted_depth += weight * bias;
            total_weight   += weight;
        }

        if total_weight < 1e-10 {
            0.45  // default working depth
        } else {
            (weighted_depth / total_weight).clamp(0.10, 0.85)
        }
    }

    /// Reinforce regions from governance approval signal.
    pub fn reinforce(&mut self, query: &str, approved: bool) {
        let words: Vec<String> = query.split_whitespace()
            .map(|w| w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let lr = if approved { 0.001 } else { 0.0005 };

        for region in &mut self.regions {
            for word in &words {
                region.experience_word(word, &words, approved);
            }
            if !words.is_empty() {
                let token_id  = Self::word_to_id(&words[words.len() - 1]);
                let target_id = if approved {
                    token_id
                } else {
                    (token_id + 1) % crate::brain::gru::VOCAB_SIZE
                };
                region.learn_from_token(token_id, target_id, lr);
            }
            let signal = if approved { 4.0 } else { 2.0 };
            let noise  = if approved { 0.5 } else { 2.0 };
            region.health.update(signal, noise);
        }

        let snr   = self.system_snr();
        let depth = self.compute_emergent_depth();
        self.metrics.record(approved, snr, depth);

        println!("  [Brain] Reinforced {}. Approval rate: {:.3} Avg depth: {:.4}",
            if approved { "+" } else { "-" },
            self.metrics.approval_rate,
            self.metrics.avg_depth);

        self.monitor.apply_interventions(&mut self.regions);
    }

    /// Consolidate word weights after dream cycle.
    pub fn consolidate(&mut self) {
        println!("  [Brain] Consolidating {} regions...", self.regions.len());
        for region in &mut self.regions {
            region.consolidate();
        }
    }

    /// Save all weights and metrics to disk.
    pub fn save(&self) {
        for region in &self.regions {
            region.save();
        }
        self.save_metrics();
        self.save_projection();
        self.save_session_id(self.session_id + 1);
        println!("  [Brain] Saved. Session {} complete.", self.session_id);
    }

    pub fn system_snr(&self) -> f64 {
        let total: f64 = self.regions.iter().map(|r| r.health.snr).sum();
        total / self.regions.len() as f64
    }

    pub fn region_snrs(&self) -> Vec<(String, f64)> {
        self.regions.iter()
            .map(|r| (r.name.clone(), r.health.snr))
            .collect()
    }
    pub fn metrics(&self) -> &BrainMetrics { &self.metrics }

    // ── Private helpers ──────────────────────────────────────────────────

    fn aggregate_hidden_states(&self) -> DVector<f64> {
        let mut concatenated = DVector::zeros(PROJECTION_DIM);
        let total_snr: f64   = self.regions.iter().map(|r| r.health.snr).sum();
        let total_snr        = total_snr.max(1e-10);

        for (i, region) in self.regions.iter().enumerate() {
            let weight = region.health.snr / total_snr;
            let offset = i * HIDDEN_DIM;
            for j in 0..HIDDEN_DIM {
                concatenated[offset + j] = region.hidden[j] * weight;
            }
        }

        let projected = &self.projection * &concatenated;
        let norm      = projected.norm().max(1e-10);
        projected / norm * 0.2
    }

    fn tokenise(text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| Self::word_to_id(w))
            .collect()
    }

    fn word_to_id(word: &str) -> usize {
        let mut hash: usize = 5381;
        for b in word.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(b as usize);
        }
        hash % crate::brain::gru::VOCAB_SIZE
    }

    fn load_or_init_projection() -> DMatrix<f64> {
        let path = Path::new("nn_weights/projection.bin");
        if path.exists() {
            if let Ok(bytes) = std::fs::read(path) {
                let floats: Vec<f64> = bytes.chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                if floats.len() == BRAIN_CONTEXT_DIM * PROJECTION_DIM {
                    return DMatrix::from_vec(
                        BRAIN_CONTEXT_DIM, PROJECTION_DIM, floats
                    );
                }
            }
        }
        let scale = 1.0 / (PROJECTION_DIM as f64).sqrt();
        use rand::Rng;
        let mut rng = rand::thread_rng();
        DMatrix::from_fn(BRAIN_CONTEXT_DIM, PROJECTION_DIM, |_, _| {
            rng.gen::<f64>() * scale * 2.0 - scale
        })
    }

    fn save_projection(&self) {
        let bytes: Vec<u8> = self.projection.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        std::fs::create_dir_all("nn_weights").ok();
        std::fs::write("nn_weights/projection.bin", bytes).ok();
    }

    fn load_session_id() -> u64 {
        std::fs::read_to_string("nn_weights/session_id.txt")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    fn save_session_id(&self, id: u64) {
        std::fs::write("nn_weights/session_id.txt", id.to_string()).ok();
    }

    fn load_metrics(session_id: u64) -> BrainMetrics {
        std::fs::read_to_string("nn_weights/brain_metrics.json")
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| BrainMetrics::new(session_id))
    }

    fn save_metrics(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.metrics) {
            std::fs::write("nn_weights/brain_metrics.json", json).ok();
        }
    }
}
