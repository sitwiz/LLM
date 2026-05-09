//! BrainSystem — owns all 14 regions, runs hierarchical predictive coding,
//! learns from NeoCorticalMesh approval signal.
//!
//! Query processing runs three passes:
//!   1. Forward  — tokens through all 14 regions, update hidden states
//!   2. Top-down — cortical → intermediate → subcortical → brainstem (predictions)
//!   3. Bottom-up — brainstem → subcortical → intermediate → cortical (PE)
//!
//! Depth and context emerge from precision-weighted region activations.

use nalgebra::{DVector, DMatrix};
use crate::brain::region::{BrainRegion, HierarchyLevel, create_all_regions};
use crate::brain::health::HealthMonitor;
use serde::{Serialize, Deserialize};
use std::path::Path;

pub const BRAIN_CONTEXT_DIM: usize = 256;
const HIDDEN_DIM:             usize = 64;
const NUM_REGIONS:            usize = 14;
const PROJECTION_DIM:         usize = NUM_REGIONS * HIDDEN_DIM;

/// Depth bias per region — maps cognitive function to manifold depth.
/// Abstract/philosophical regions → deep (low norm).
/// Concrete/novelty regions → frontier (high norm).
pub const REGION_DEPTH_BIAS: [f64; 14] = [
    0.20,  // frontal_lobe      — planning, reasoning — deepest
    0.35,  // temporal_lobe     — language, memory
    0.45,  // parietal_lobe     — spatial, math
    0.62,  // occipital_lobe    — pattern, visual
    0.25,  // insular_lobe      — awareness, emotion — deep
    0.40,  // pituitary_gland   — global regulation
    0.38,  // meninges          — boundary, context
    0.22,  // limbic            — reward, fear — deep
    0.48,  // thalamus          — attention, filter
    0.50,  // hypothalamus      — resource, balance
    0.65,  // midbrain          — novelty, alert — frontier
    0.60,  // cerebellum        — precision, skill
    0.55,  // pons              — relay, bridge
    0.70,  // medulla_oblongata — safety baseline
];

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

        Self { regions, monitor: HealthMonitor::new(), metrics, projection, session_id }
    }

    /// Process query through all 14 regions with hierarchical PE.
    ///
    /// Pass 1 — Forward: all regions process query tokens.
    /// Pass 2 — Top-down: cortical regions predict intermediate,
    ///          intermediate predict subcortical, subcortical predict brainstem.
    /// Pass 3 — Bottom-up: brainstem computes PE, propagates upward.
    ///          Each level updates precision and applies local correction.
    pub fn process_query(&mut self, query: &str) -> (DVector<f64>, f64) {
        let token_ids = Self::tokenise(query);

        // ── Pass 1: forward ──────────────────────────────────────────────
        for region in &mut self.regions {
            region.reset_hidden();
        }
        for &token_id in &token_ids {
            for region in &mut self.regions {
                region.process_token(token_id);
            }
        }

        // ── Pass 2 & 3: hierarchical PE ──────────────────────────────────
        self.run_hierarchical_pe_pass();

        let context = self.aggregate_hidden_states();
        let depth   = self.compute_emergent_depth();

        println!("  [Brain] Context norm={:.4} Depth={:.4} SNR={:.3}",
            context.norm(), depth, self.system_snr());

        (context, depth)
    }

    /// Hierarchical predictive coding pass.
    ///
    /// Top-down: higher levels send predictions to lower levels.
    ///   The average hidden state of level N is scaled by that level's
    ///   mean precision and sent as the expected state for level N-1.
    ///
    /// Bottom-up: lower levels compute PE = hidden − received_prediction.
    ///   The precision-weighted average PE of level N is sent upward to
    ///   nudge the hidden states of level N+1 (surprise propagation).
    ///   Each region updates its precision and applies local correction.
    ///
    /// This implements the core message-passing loop of predictive coding.
    fn run_hierarchical_pe_pass(&mut self) {
        let dim = self.regions[0].hidden.len();

        // ── Collect level statistics (immutable) ─────────────────────────
        struct LevelStats {
            level:         u8,
            hidden_avg:    DVector<f64>,
            precision_avg: f64,
        }

        let level_stats: Vec<LevelStats> = (1u8..=4).filter_map(|lv| {
            let at_level: Vec<_> = self.regions.iter()
                .filter(|r| r.level.value() == lv)
                .collect();
            if at_level.is_empty() { return None; }
            let n = at_level.len() as f64;
            let hidden_sum = at_level.iter()
                .fold(DVector::zeros(dim), |acc, r| acc + &r.hidden);
            let prec_sum = at_level.iter().map(|r| r.precision).sum::<f64>();
            Some(LevelStats {
                level:         lv,
                hidden_avg:    hidden_sum / n,
                precision_avg: prec_sum / n,
            })
        }).collect();

        // ── Top-down: send predictions downward ─────────────────────────
        // Level N predicts what level N-1 should have computed.
        // Prediction is precision-gated: confident levels send stronger signals.
        for region in &mut self.regions {
            let lv = region.level.value();
            if lv == 4 {
                // Cortical — top of hierarchy, receives no prediction
                region.receive_prediction(DVector::zeros(dim));
                continue;
            }
            // Find the level above this region
            if let Some(above) = level_stats.iter().find(|s| s.level == lv + 1) {
                let confidence = 0.7 * above.precision_avg / (above.precision_avg + 1.0);
                region.receive_prediction(&above.hidden_avg * confidence);
            }
        }

        // ── Bottom-up: compute PE, store results ─────────────────────────
        // We need to collect PE first (requires mut borrow), then propagate.
        struct PEResult {
            level:        u8,
            pe_weighted:  DVector<f64>,  // precision-weighted PE vector
            pe_magnitude: f64,
        }

        let mut pe_results: Vec<PEResult> = Vec::new();
        for region in &mut self.regions {
            let pe_weighted  = region.compute_prediction_error();
            let pe_magnitude = region.pe_magnitude;
            pe_results.push(PEResult {
                level:       region.level.value(),
                pe_weighted,
                pe_magnitude,
            });
            region.update_precision();
        }

        // Average PE per level for propagation upward
        let level_pe_avg: Vec<(u8, DVector<f64>, f64)> = (1u8..=4).filter_map(|lv| {
            let at_level: Vec<_> = pe_results.iter()
                .filter(|r| r.level == lv)
                .collect();
            if at_level.is_empty() { return None; }
            let n   = at_level.len() as f64;
            let avg_pe = at_level.iter()
                .fold(DVector::zeros(dim), |acc, r| acc + &r.pe_weighted) / n;
            let avg_mag = at_level.iter().map(|r| r.pe_magnitude).sum::<f64>() / n;
            Some((lv, avg_pe, avg_mag))
        }).collect();

        // ── Propagate PE upward and apply local corrections ───────────────
        // Regions at level N receive the PE from level N-1 (one below).
        // They nudge their hidden state in the direction of the PE signal
        // (surprise from below means the higher level's prediction was off).
        for region in &mut self.regions {
            let lv = region.level.value();

            // Receive PE from the level below (bottom-up surprise signal)
            if lv > 1 {
                if let Some((_, pe_below, _)) = level_pe_avg.iter().find(|(l, _, _)| *l == lv - 1) {
                    // Nudge hidden state: higher level updates its model
                    // in response to surprise from below
                    let nudge = pe_below * 0.05;
                    region.hidden = (&region.hidden + &nudge).map(|v| v.clamp(-5.0, 5.0));
                }
            }

            // Local PE correction — region adjusts to reduce its own surprise
            if region.pe_magnitude > 0.01 {
                region.apply_pe_correction(0.1);
            }
        }

        // ── Log hierarchical PE summary ───────────────────────────────────
        let level_names = ["brainstem", "subcortical", "intermediate", "cortical"];
        for (lv, _, avg_mag) in &level_pe_avg {
            let name = level_names.get((*lv as usize).saturating_sub(1))
                .copied().unwrap_or("unknown");
            let avg_prec = self.regions.iter()
                .filter(|r| r.level.value() == *lv)
                .map(|r| r.precision)
                .sum::<f64>()
                / self.regions.iter().filter(|r| r.level.value() == *lv).count().max(1) as f64;
            println!("  [PE] L{} ({}) PE={:.4} precision={:.3}",
                lv, name, avg_mag, avg_prec);
        }
    }

    /// Compute emergent depth from precision-weighted region activation.
    ///
    /// Regions that fired strongly AND have high precision pull depth
    /// toward their bias. This means well-calibrated, active regions
    /// dominate depth assignment — unreliable or quiet regions contribute less.
    pub fn compute_emergent_depth(&self) -> f64 {
        let total_weight: f64 = self.regions.iter()
            .map(|r| r.hidden.norm() * r.health.snr * r.precision)
            .sum::<f64>()
            .max(1e-10);

        let mut weighted_depth = 0.0f64;
        let mut sum_w          = 0.0f64;

        for (i, region) in self.regions.iter().enumerate() {
            let activation = region.hidden.norm();
            let weight     = activation * region.health.snr * region.precision / total_weight;
            let bias       = REGION_DEPTH_BIAS.get(i).copied().unwrap_or(0.48);
            weighted_depth += weight * bias;
            sum_w          += weight;
        }

        if sum_w < 1e-10 { 0.45 }
        else { (weighted_depth / sum_w).clamp(0.10, 0.85) }
    }

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
            self.metrics.approval_rate, self.metrics.avg_depth);

        self.monitor.apply_interventions(&mut self.regions);
    }

    pub fn consolidate(&mut self) {
        println!("  [Brain] Consolidating {} regions...", self.regions.len());
        for region in &mut self.regions {
            region.consolidate();
        }
    }

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
        self.regions.iter().map(|r| (r.name.clone(), r.health.snr)).collect()
    }

    pub fn metrics(&self) -> &BrainMetrics { &self.metrics }

    // ── Private helpers ──────────────────────────────────────────────────

    /// Aggregate hidden states into context vector.
    /// Weighted by SNR × precision — reliable, active regions dominate.
    fn aggregate_hidden_states(&self) -> DVector<f64> {
        let mut concatenated = DVector::zeros(PROJECTION_DIM);
        let total_weight: f64 = self.regions.iter()
            .map(|r| r.health.snr * r.precision)
            .sum::<f64>()
            .max(1e-10);

        for (i, region) in self.regions.iter().enumerate() {
            let weight = region.health.snr * region.precision / total_weight;
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
        text.split_whitespace().map(|w| Self::word_to_id(w)).collect()
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
                    return DMatrix::from_vec(BRAIN_CONTEXT_DIM, PROJECTION_DIM, floats);
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

