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

/// A single brain region with GRU processing and hierarchical PE machinery.
///
/// Each region participates in two passes per query:
///
///   1. Forward pass — process tokens, update hidden state (bottom-up).
///   2. Top-down pass — receive prediction from level above, compute PE,
///      update precision, apply correction, relay PE upward.
///
/// Precision tracks long-run prediction accuracy — regions that consistently
/// predict well gain authority (higher precision → stronger predictions,
/// more weight in quorum voting).
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

    // ── Predictive coding state ───────────────────────────────────────────

    /// Top-down prediction sent to regions one level below.
    /// Derived from this region's hidden state — what it expects lower
    /// levels to compute. Precision-weighted: confident regions send
    /// stronger predictions.
    pub prediction:          DVector<f64>,

    /// Prediction received FROM the level above.
    /// Cortical regions (top of hierarchy) receive zeros.
    pub received_prediction: DVector<f64>,

    /// Prediction error: how much this region's actual hidden state
    /// deviates from what the level above expected.
    ///
    ///   PE = hidden − received_prediction
    ///
    /// Large PE = surprise = useful bottom-up signal for higher levels.
    /// Small PE = prediction was accurate = little to learn.
    pub prediction_error:    DVector<f64>,

    /// Precision = 1 / σ² — reliability of this region's predictions.
    ///
    /// Increases when predictions were accurate (observed PE small),
    /// decreases when surprised (observed PE large).
    ///
    /// High-precision regions:
    ///   • Send stronger top-down predictions (suppress lower levels)
    ///   • Have more weight in quorum voting
    ///   • Resist correction from bottom-up PE (trust their model)
    pub precision:           f64,

    /// Scalar PE magnitude for logging and SNR coupling.
    pub pe_magnitude:        f64,

    /// Cumulative PE across all queries in this session.
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

    /// Process a single token, update hidden state, regenerate prediction.
    pub fn process_token(&mut self, token_id: usize) -> Vec<(usize, f64)> {
        let x = self.gru.embed(token_id);
        let (new_hidden, _, probs) = self.gru.forward(&x, &self.hidden);
        self.hidden    = new_hidden;
        self.call_count += 1;
        self.prediction = self.generate_prediction();

        let signal = probs.iter().cloned().fold(0.0f64, f64::max);
        let noise  = probs.iter().sum::<f64>() / probs.len() as f64;
        self.health.update(signal, noise);

        probs.iter().enumerate().map(|(i, p)| (i, *p)).collect()
    }

    pub fn process_sequence(&mut self, token_ids: &[usize]) -> Vec<Vec<(usize, f64)>> {
        token_ids.iter().map(|&id| self.process_token(id)).collect()
    }

    // ── Predictive coding pass ────────────────────────────────────────────

    /// Generate top-down prediction for the level below.
    ///
    /// The prediction is the hidden state attenuated by a precision factor.
    /// Confident regions (high precision) send stronger predictions,
    /// exerting more top-down influence on lower-level processing.
    ///
    /// Attenuation at 0.7 prevents over-suppression of lower levels,
    /// preserving their capacity to signal genuine surprise.
    pub fn generate_prediction(&self) -> DVector<f64> {
        let confidence_scale = 0.7 * self.precision / (self.precision + 1.0);
        &self.hidden * confidence_scale
    }

    /// Accept a top-down prediction from the level above.
    pub fn receive_prediction(&mut self, prediction: DVector<f64>) {
        self.received_prediction = prediction;
    }

    /// Compute prediction error and return precision-weighted version.
    ///
    ///   PE_raw = hidden − received_prediction
    ///   PE_weighted = precision × PE_raw
    ///
    /// The weighted PE is what propagates upward. High-precision regions
    /// send louder error signals — their surprises matter more.
    ///
    /// Internally, pe_magnitude (the scalar norm) is updated here and
    /// used for precision updating and logging.
    pub fn compute_prediction_error(&mut self) -> DVector<f64> {
        self.prediction_error = &self.hidden - &self.received_prediction;
        self.pe_magnitude     = self.prediction_error.norm();
        self.total_pe        += self.pe_magnitude;

        &self.prediction_error * self.precision
    }

    /// Update precision from recent prediction accuracy.
    ///
    /// Precision converges to a value that reflects long-run accuracy:
    ///   accuracy = 1 / (1 + pe_magnitude)   — in [0, 1]
    ///   precision_target = accuracy × MAX_PRECISION
    ///
    /// Exponential moving average with α=0.1 keeps precision stable
    /// while adapting over time. Clamped to [0.1, 4.0] to prevent
    /// degenerate collapse or runaway amplification.
    pub fn update_precision(&mut self) {
        const MAX_PRECISION: f64 = 4.0;
        let accuracy        = 1.0 / (1.0 + self.pe_magnitude);
        let target_precision = accuracy * MAX_PRECISION;
        self.precision       = (0.9 * self.precision + 0.1 * target_precision)
            .clamp(0.1, MAX_PRECISION);
    }

    /// Apply precision-gated PE correction to hidden state.
    ///
    /// When PE is large, this region partially adjusts its hidden state
    /// toward the received prediction — implementing the recognition model
    /// update in active inference:
    ///
    ///   correction = −correction_rate / precision × PE
    ///
    /// The 1/precision gating means high-confidence regions resist being
    /// corrected (they trust their hidden state more than the incoming PE),
    /// while uncertain regions update more freely.
    pub fn apply_pe_correction(&mut self, correction_rate: f64) {
        let rate       = (correction_rate / self.precision.max(0.1)).min(0.5);
        let correction = &self.prediction_error * (-rate);
        self.hidden    = (&self.hidden + correction).map(|v| v.clamp(-5.0, 5.0));
    }

    // ── Quorum influence ─────────────────────────────────────────────────

    /// Token weight incorporating hierarchical precision and PE.
    ///
    /// A region's influence on quorum voting is modulated by:
    ///   • base weight from learned word_weights
    ///   • specialty boost for domain-matched tokens
    ///   • precision weight — accurate predictors have more authority
    ///   • PE penalty — surprised regions lose confidence temporarily
    ///   • health scale — depleted SNR reduces influence
    pub fn token_weight(&self, token: &str) -> f64 {
        let base = self.word_weights.get(token).copied().unwrap_or(1.0);

        let specialty_boost = if self.specialty_words.iter()
            .any(|w| token.to_lowercase().contains(w.as_str()))
        { 1.5 } else { 1.0 };

        // Precision boost: well-calibrated regions have more say
        let precision_w  = (self.precision / 2.0).clamp(0.5, 2.0);
        // PE penalty: regions currently surprised have less authority
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

    pub fn experience_word(&mut self, word: &str, context: &[String], positive: bool) {
        let delta = if positive { 0.01 } else { -0.005 };
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
        let dim              = self.hidden.len();
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

/// All 14 brain regions with explicit hierarchy levels.
///
/// Cortical (4):     frontal, temporal, parietal, occipital
/// Intermediate (3): insular, pituitary, meninges
/// Subcortical (2):  limbic, thalamus, hypothalamus, midbrain
/// Brainstem (1):    cerebellum, pons, medulla_oblongata
pub fn create_all_regions() -> Vec<BrainRegion> {
    vec![
        // ── Cortical — executive and abstract ────────────────────────────
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
        // ── Intermediate — contextual regulation ─────────────────────────
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
        // ── Subcortical — integration and gating ─────────────────────────
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
        // ── Brainstem — basic processing ─────────────────────────────────
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

