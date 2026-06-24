use std::collections::{HashMap, HashSet};
use nalgebra::DVector;
use std::path::Path;
use crate::brain::gru::GRUCell;
use crate::brain::health::RegionHealth;

/// Hierarchical level in the predictive coding stack.
///
/// Information flow:
///   Top-down:  Cortical → Intermediate → Subcortical → Brainstem
///              (predictions, priors — what the system expects)
///   Bottom-up: Brainstem → Subcortical → Intermediate → Cortical
///              (prediction errors, surprise — what the system got)
///
/// Each level minimises its own prediction error while sending
/// residual surprise upward to update higher-level beliefs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HierarchyLevel {
    Brainstem    = 1,  // cerebellum, pons, medulla — basic pattern processing
    Subcortical  = 2,  // limbic, thalamus, hypothalamus, midbrain — integration/gating
    Intermediate = 3,  // insular, pituitary, meninges — contextual regulation
    Cortical     = 4,  // frontal, temporal, parietal, occipital — executive/abstract
}

impl HierarchyLevel {
    pub fn value(self) -> u8 { self as u8 }

    pub fn label(self) -> &'static str {
        match self {
            Self::Brainstem    => "brainstem",
            Self::Subcortical  => "subcortical",
            Self::Intermediate => "intermediate",
            Self::Cortical     => "cortical",
        }
    }
}

pub struct BrainRegion {
    pub name:               String,
    pub level:              HierarchyLevel,
    pub gru:                GRUCell,
    pub word_weights:       HashMap<String, f64>,
    pub knowledge_graph:    HashMap<String, HashSet<String>>,
    pub health:             RegionHealth,
    pub hidden:             DVector<f64>,
    pub specialty_words:    Vec<String>,
    pub call_count:         u64,
    pub total_loss:         f64,
    pub prediction:          DVector<f64>,
    pub received_prediction: DVector<f64>,
    pub prediction_error:    DVector<f64>,
    pub precision:           f64,
    pub pe_magnitude:        f64,
    pub total_pe:            f64,
}

impl BrainRegion {
    pub fn new(name: &str, level: HierarchyLevel, specialty_words: Vec<String>) -> Self {
        let weights_path = format!("nn_weights/{}.bin",
            name.to_lowercase().replace(' ', "_"));
        std::fs::create_dir_all("nn_weights").ok();
        let gru = GRUCell::load_or_init(Path::new(&weights_path));

        let ww_path = format!("nn_weights/{}_words.json",
            name.to_lowercase().replace(' ', "_"));
        let word_weights: HashMap<String, f64> =
            std::fs::read_to_string(&ww_path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();

        let zero_h = GRUCell::zero_hidden();
        let dim    = zero_h.len();

        println!("  [{}] Region online. Specialty words: {} Word weights: {}",
            name, specialty_words.len(), word_weights.len());

        Self {
            name:                name.to_string(),
            level,
            gru,
            word_weights,
            knowledge_graph:     HashMap::new(),
            health:              RegionHealth::new(),
            hidden:              zero_h,
            specialty_words,
            call_count:          0,
            total_loss:          0.0,
            prediction:          DVector::zeros(dim),
            received_prediction: DVector::zeros(dim),
            prediction_error:    DVector::zeros(dim),
            precision:           1.0,
            pe_magnitude:        0.0,
            total_pe:            0.0,
        }
    }

    // ── Forward processing ────────────────────────────────────────────────

    pub fn process_token(&mut self, token_id: usize) -> Vec<(usize, f64)> {
        let x = self.gru.embed(token_id);
        let (new_hidden, _, probs) = self.gru.forward(&x, &self.hidden);
        self.hidden     = new_hidden;
        self.call_count += 1;
        self.prediction = self.generate_prediction();

        probs.iter().enumerate().map(|(i, p)| (i, *p)).collect()
    }

    pub fn process_sequence(&mut self, token_ids: &[usize]) -> Vec<Vec<(usize, f64)>> {
        token_ids.iter().map(|&id| self.process_token(id)).collect()
    }

    // ── Predictive coding pass ────────────────────────────────────────────

    pub fn generate_prediction(&self) -> DVector<f64> {
        let confidence_scale = 0.7 * self.precision / (self.precision + 1.0);
        &self.hidden * confidence_scale
    }

    pub fn receive_prediction(&mut self, prediction: DVector<f64>) {
        self.received_prediction = prediction;
    }

    pub fn compute_prediction_error(&mut self) -> DVector<f64> {
        self.prediction_error = &self.hidden - &self.received_prediction;
        self.pe_magnitude     = self.prediction_error.norm();
        self.total_pe        += self.pe_magnitude;

        &self.prediction_error * self.precision
    }

    pub fn update_precision(&mut self) {
        const MAX_PRECISION: f64 = 4.0;
        let accuracy         = 1.0 / (1.0 + self.pe_magnitude);
        let target_precision = accuracy * MAX_PRECISION;
        self.precision       = (0.9 * self.precision + 0.1 * target_precision)
            .clamp(0.1, MAX_PRECISION);
    }

    pub fn apply_pe_correction(&mut self, correction_rate: f64) {
        let rate       = (correction_rate / self.precision.max(0.1)).min(0.5);
        let correction = &self.prediction_error * (-rate);
        self.hidden    = (&self.hidden + correction).map(|v| v.clamp(-5.0, 5.0));
    }

    // ── Quorum influence ─────────────────────────────────────────────────

    pub fn token_weight(&self, token: &str) -> f64 {
        let base = self.word_weights.get(token).copied().unwrap_or(1.0);

        let specialty_boost = if self.specialty_words.iter()
            .any(|w| token.to_lowercase().contains(w.as_str()))
        { 1.5 } else { 1.0 };

        let precision_w  = (self.precision / 2.0).clamp(0.5, 2.0);
        let pe_penalty   = 1.0 / (1.0 + self.pe_magnitude * 0.5);
        let health_scale = (self.health.snr / 3.154).clamp(0.1, 2.0);

        base * specialty_boost * precision_w * pe_penalty * health_scale
    }

    // ── Learning ──────────────────────────────────────────────────────────

    pub fn learn_from_token(&mut self, token_id: usize, target_id: usize, lr: f64) -> f64 {
        let x      = self.gru.embed(token_id);
        let h_prev = self.hidden.clone();
        let (_, _, probs) = self.gru.forward(&x, &h_prev);
        let loss   = self.gru.learn(&x, &h_prev, target_id, &probs, lr);
        self.total_loss += loss;
        loss
    }

    /// Update word weights using VFE signal.
    /// High confidence + low VFE → stronger reinforcement.
    /// High VFE + low confidence → weaker or negative update.
    pub fn experience_word(&mut self, word: &str, context: &[String], positive: bool, vfe: f64, confidence: f64) {
        // VFE signal: positive when system is converged, negative when confused
        let vfe_signal = (confidence - vfe.min(1.0)).tanh();

        // Scale delta by VFE signal — good predictions reinforce more strongly
        let base_delta = if positive { 0.01 } else { -0.005 };
        let delta      = base_delta * (1.0 + vfe_signal.max(0.0));

        let entry = self.word_weights.entry(word.to_string()).or_insert(1.0);
        *entry    = (*entry + delta).max(0.1).min(5.0);

        let connections = self.knowledge_graph.entry(word.to_string()).or_default();
        for ctx_word in context { connections.insert(ctx_word.clone()); }
    }

    pub fn curiosity_score(&self, token: &str) -> f64 {
        let w = self.word_weights.get(token).copied().unwrap_or(1.0);
        if w < 1.1 && w > 0.9 { 1.5 } else { 1.0 }
    }

    // ── Maintenance ───────────────────────────────────────────────────────

    pub fn consolidate(&mut self) {
        let avg_weight: f64 = if self.word_weights.is_empty() { 1.0 }
        else { self.word_weights.values().sum::<f64>() / self.word_weights.len() as f64 };

        self.word_weights.retain(|_, w| *w > avg_weight * 0.1);
        for w in self.word_weights.values_mut() {
            if *w > avg_weight * 1.5 { *w *= 1.01; }
        }

        println!(
            "  [{}] Consolidated. Level={} Words={} SNR={:.3} Precision={:.3} ΣPE={:.3}",
            self.name, self.level.label(),
            self.word_weights.len(), self.health.snr,
            self.precision, self.total_pe,
        );
    }

    pub fn reset_hidden(&mut self) {
        let dim                  = self.hidden.len();
        self.hidden              = GRUCell::zero_hidden();
        self.prediction          = DVector::zeros(dim);
        self.received_prediction = DVector::zeros(dim);
        self.prediction_error    = DVector::zeros(dim);
        self.pe_magnitude        = 0.0;
    }

    pub fn save(&self) {
        let path = format!("nn_weights/{}.bin",
            self.name.to_lowercase().replace(' ', "_"));
        std::fs::create_dir_all("nn_weights").ok();
        if let Err(e) = self.gru.save(Path::new(&path)) {
            eprintln!("  [{}] Failed to save GRU weights: {}", self.name, e);
        }
        let ww_path = format!("nn_weights/{}_words.json",
            self.name.to_lowercase().replace(' ', "_"));
        if let Ok(json) = serde_json::to_string(&self.word_weights) {
            std::fs::write(&ww_path, json).ok();
        }
    }

    pub fn average_loss(&self) -> f64 {
        if self.call_count == 0 { 0.0 }
        else { self.total_loss / self.call_count as f64 }
    }
}

pub fn create_all_regions() -> Vec<BrainRegion> {
    vec![
        BrainRegion::new("frontal_lobe", HierarchyLevel::Cortical, vec![
            "plan".into(), "decide".into(), "reason".into(), "goal".into(),
            "strategy".into(), "logic".into(), "analyse".into(), "judge".into(),
        ]),
        BrainRegion::new("temporal_lobe", HierarchyLevel::Cortical, vec![
            "word".into(), "language".into(), "mean".into(), "define".into(),
            "concept".into(), "name".into(), "remember".into(), "story".into(),
        ]),
        BrainRegion::new("parietal_lobe", HierarchyLevel::Cortical, vec![
            "space".into(), "position".into(), "number".into(), "math".into(),
            "calculate".into(), "measure".into(), "quantity".into(), "size".into(),
        ]),
        BrainRegion::new("occipital_lobe", HierarchyLevel::Cortical, vec![
            "pattern".into(), "visual".into(), "see".into(), "shape".into(),
            "design".into(), "structure".into(), "recognise".into(), "detect".into(),
        ]),
        BrainRegion::new("insular_lobe", HierarchyLevel::Intermediate, vec![
            "feel".into(), "emotion".into(), "aware".into(), "empathy".into(),
            "trust".into(), "sense".into(), "inner".into(), "intuition".into(),
        ]),
        BrainRegion::new("pituitary_gland", HierarchyLevel::Intermediate, vec![
            "regulate".into(), "output".into(), "global".into(), "system".into(),
            "modulate".into(), "control".into(), "adjust".into(), "tune".into(),
        ]),
        BrainRegion::new("meninges", HierarchyLevel::Intermediate, vec![
            "protect".into(), "boundary".into(), "context".into(), "preserve".into(),
            "integrity".into(), "contain".into(), "secure".into(), "scope".into(),
        ]),
        BrainRegion::new("limbic", HierarchyLevel::Subcortical, vec![
            "reward".into(), "fear".into(), "anger".into(), "happy".into(),
            "memory".into(), "desire".into(), "pleasure".into(), "anxiety".into(),
        ]),
        BrainRegion::new("thalamus", HierarchyLevel::Subcortical, vec![
            "attention".into(), "focus".into(), "priority".into(), "filter".into(),
            "relevant".into(), "signal".into(), "important".into(), "urgent".into(),
        ]),
        BrainRegion::new("hypothalamus", HierarchyLevel::Subcortical, vec![
            "energy".into(), "resource".into(), "balance".into(), "need".into(),
            "sustain".into(), "conserve".into(), "limit".into(), "budget".into(),
        ]),
        BrainRegion::new("midbrain", HierarchyLevel::Subcortical, vec![
            "alert".into(), "quick".into(), "react".into(), "reward".into(),
            "surprise".into(), "novelty".into(), "fast".into(), "trigger".into(),
        ]),
        BrainRegion::new("cerebellum", HierarchyLevel::Brainstem, vec![
            "precise".into(), "accurate".into(), "correct".into(), "refine".into(),
            "skill".into(), "error".into(), "timing".into(), "sequence".into(),
        ]),
        BrainRegion::new("pons", HierarchyLevel::Brainstem, vec![
            "bridge".into(), "connect".into(), "relay".into(), "transition".into(),
            "state".into(), "interface".into(), "handoff".into(), "pass".into(),
        ]),
        BrainRegion::new("medulla_oblongata", HierarchyLevel::Brainstem, vec![
            "safe".into(), "danger".into(), "baseline".into(), "vital".into(),
            "monitor".into(), "filter".into(), "reject".into(), "integrity".into(),
        ]),
    ]
}
