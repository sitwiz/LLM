use std::collections::{HashMap, HashSet};
use nalgebra::DVector;
use std::path::Path;
use crate::brain::gru::GRUCell;
use crate::brain::health::RegionHealth;

/// A single brain region with its own GRU, word weights, and knowledge graph
pub struct BrainRegion {
    pub name:            String,
    pub gru:             GRUCell,
    pub word_weights:    HashMap<String, f64>,
    pub knowledge_graph: HashMap<String, HashSet<String>>,
    pub health:          RegionHealth,
    pub hidden:          DVector<f64>,
    pub specialty_words: Vec<String>,
    pub call_count:      u64,
    pub total_loss:      f64,
}

impl BrainRegion {
    pub fn new(name: &str, specialty_words: Vec<String>) -> Self {
    let weights_path = format!("nn_weights/{}.bin",
        name.to_lowercase().replace(' ', "_"));
    std::fs::create_dir_all("nn_weights").ok();
    let gru = GRUCell::load_or_init(Path::new(&weights_path));

    // Load word weights if they exist
    let ww_path = format!("nn_weights/{}_words.json",
        name.to_lowercase().replace(' ', "_"));
    let word_weights: std::collections::HashMap<String, f64> =
        std::fs::read_to_string(&ww_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

    println!("  [{}] Region online. Specialty words: {} Word weights: {}",
        name, specialty_words.len(), word_weights.len());

    Self {
        name:            name.to_string(),
        gru,
        word_weights,
        knowledge_graph: std::collections::HashMap::new(),
        health:          RegionHealth::new(),
        hidden:          GRUCell::zero_hidden(),
        specialty_words,
        call_count:      0,
        total_loss:      0.0,
    }
}
    /// Process a token id and return probability distribution over vocab
    pub fn process_token(&mut self, token_id: usize) -> Vec<(usize, f64)> {
        let x = self.gru.embed(token_id);
        let (new_hidden, _, probs) = self.gru.forward(&x, &self.hidden);
        self.hidden = new_hidden;
        self.call_count += 1;

        // Update health based on signal strength
        let signal = probs.iter().cloned().fold(0.0f64, f64::max);
        let noise = probs.iter().sum::<f64>() / probs.len() as f64;
        self.health.update(signal, noise);

        probs.iter().enumerate().map(|(i, p)| (i, *p)).collect()
    }

    /// Process a full token sequence
    pub fn process_sequence(&mut self, token_ids: &[usize]) -> Vec<Vec<(usize, f64)>> {
        token_ids.iter().map(|&id| self.process_token(id)).collect()
    }

    /// Learn from experience — update weights when prediction was right or wrong
    /// target_id: the token that actually appeared
    /// lr: learning rate
    pub fn learn_from_token(
        &mut self,
        token_id: usize,
        target_id: usize,
        lr: f64,
    ) -> f64 {
        let x = self.gru.embed(token_id);
        let h_prev = self.hidden.clone();
        let (_, _, probs) = self.gru.forward(&x, &h_prev);
        let loss = self.gru.learn(&x, &h_prev, target_id, &probs, lr);
        self.total_loss += loss;
        loss
    }

    /// Get this region's confidence weight for a token string
    /// Used in the voting aggregation
    pub fn token_weight(&self, token: &str) -> f64 {
        let base = self.word_weights.get(token).copied().unwrap_or(1.0);

        // Specialty boost — regions are better at their domain
        let specialty_boost = if self.specialty_words.iter()
            .any(|w| token.to_lowercase().contains(w.as_str()))
        {
            1.5
        } else {
            1.0
        };

        // Health scaling — unhealthy regions have less influence
        let health_scale = (self.health.snr / 3.154).clamp(0.1, 2.0);

        base * specialty_boost * health_scale
    }

    /// Experience learning — a word appeared in context, update weights
    pub fn experience_word(&mut self, word: &str, context: &[String], positive: bool) {
        let delta = if positive { 0.01 } else { -0.005 };
        let entry = self.word_weights.entry(word.to_string()).or_insert(1.0);
        *entry = (*entry + delta).max(0.1).min(5.0);

        // Update knowledge graph connections
        let connections = self.knowledge_graph
            .entry(word.to_string())
            .or_default();
        for ctx_word in context {
            connections.insert(ctx_word.clone());
        }
    }

    /// Curiosity learning — explore words with high entropy (uncertainty)
    pub fn curiosity_score(&self, token: &str) -> f64 {
        let weight = self.word_weights.get(token).copied().unwrap_or(1.0);
        // High curiosity for words we haven't learned well
        if weight < 1.1 && weight > 0.9 {
            1.5  // unexplored — high curiosity
        } else {
            1.0  // already learned — lower curiosity
        }
    }

    /// Sleep consolidation — prune weak weights, strengthen important ones
    pub fn consolidate(&mut self) {
        let avg_weight: f64 = if self.word_weights.is_empty() {
            1.0
        } else {
            self.word_weights.values().sum::<f64>() / self.word_weights.len() as f64
        };

        // Prune words below 10% of average
        self.word_weights.retain(|_, w| *w > avg_weight * 0.1);

        // Strengthen words above 150% of average
        for weight in self.word_weights.values_mut() {
            if *weight > avg_weight * 1.5 {
                *weight *= 1.01;  // gentle strengthening
            }
        }

        println!("  [{}] Consolidated. Words: {} SNR: {:.3} Status: {}",
            self.name,
            self.word_weights.len(),
            self.health.snr,
            self.health.status()
        );
    }

    /// Reset hidden state between conversations
    pub fn reset_hidden(&mut self) {
        self.hidden = GRUCell::zero_hidden();
    }

    /// Save GRU weights to disk
    pub fn save(&self) {
    let path = format!("nn_weights/{}.bin",
        self.name.to_lowercase().replace(' ', "_"));
    std::fs::create_dir_all("nn_weights").ok();
    if let Err(e) = self.gru.save(Path::new(&path)) {
        eprintln!("  [{}] Failed to save GRU weights: {}", self.name, e);
    }

    // Save word weights as JSON separately
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

/// All 14 brain regions with their specialities
pub fn create_all_regions() -> Vec<BrainRegion> {
    vec![
        BrainRegion::new("frontal_lobe", vec![
            "plan".into(), "decide".into(), "reason".into(), "goal".into(),
            "strategy".into(), "logic".into(), "analyse".into(), "judge".into(),
        ]),
        BrainRegion::new("temporal_lobe", vec![
            "word".into(), "language".into(), "mean".into(), "define".into(),
            "concept".into(), "name".into(), "remember".into(), "story".into(),
        ]),
        BrainRegion::new("parietal_lobe", vec![
            "space".into(), "position".into(), "number".into(), "math".into(),
            "calculate".into(), "measure".into(), "quantity".into(), "size".into(),
        ]),
        BrainRegion::new("occipital_lobe", vec![
            "pattern".into(), "visual".into(), "see".into(), "shape".into(),
            "design".into(), "structure".into(), "recognise".into(), "detect".into(),
        ]),
        BrainRegion::new("insular_lobe", vec![
            "feel".into(), "emotion".into(), "aware".into(), "empathy".into(),
            "trust".into(), "sense".into(), "inner".into(), "intuition".into(),
        ]),
        BrainRegion::new("limbic", vec![
            "reward".into(), "fear".into(), "anger".into(), "happy".into(),
            "memory".into(), "desire".into(), "pleasure".into(), "anxiety".into(),
        ]),
        BrainRegion::new("thalamus", vec![
            "attention".into(), "focus".into(), "priority".into(), "filter".into(),
            "relevant".into(), "signal".into(), "important".into(), "urgent".into(),
        ]),
        BrainRegion::new("hypothalamus", vec![
            "energy".into(), "resource".into(), "balance".into(), "need".into(),
            "sustain".into(), "conserve".into(), "limit".into(), "budget".into(),
        ]),
        BrainRegion::new("cerebellum", vec![
            "precise".into(), "accurate".into(), "correct".into(), "refine".into(),
            "skill".into(), "error".into(), "timing".into(), "sequence".into(),
        ]),
        BrainRegion::new("midbrain", vec![
            "alert".into(), "quick".into(), "react".into(), "reward".into(),
            "surprise".into(), "novelty".into(), "fast".into(), "trigger".into(),
        ]),
        BrainRegion::new("pons", vec![
            "bridge".into(), "connect".into(), "relay".into(), "transition".into(),
            "state".into(), "interface".into(), "handoff".into(), "pass".into(),
        ]),
        BrainRegion::new("medulla_oblongata", vec![
            "safe".into(), "danger".into(), "baseline".into(), "vital".into(),
            "monitor".into(), "filter".into(), "reject".into(), "integrity".into(),
        ]),
        BrainRegion::new("pituitary_gland", vec![
            "regulate".into(), "output".into(), "global".into(), "system".into(),
            "modulate".into(), "control".into(), "adjust".into(), "tune".into(),
        ]),
        BrainRegion::new("meninges", vec![
            "protect".into(), "boundary".into(), "context".into(), "preserve".into(),
            "integrity".into(), "contain".into(), "secure".into(), "scope".into(),
        ]),
    ]
}
