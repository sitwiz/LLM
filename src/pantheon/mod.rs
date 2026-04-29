pub mod khaos;
pub mod gaia;
pub mod tartaros;
pub mod eros;

use nalgebra::DVector;
use crate::soul::geometry::update_soul;
use crate::soul::manifold::{retrocausal_steer, final_phase};
use crate::ollama::OllamaClient;

#[derive(Debug, Clone)]
pub struct SpeakResult {
    pub activated:  bool,
    pub domain:     Option<String>,
    pub response:   Option<String>,
    pub soul_drift: f64,
    pub psi_final:  f64,
    pub phase:      String,
}

pub trait God {
    fn name(&self) -> &str;
    fn system_prompt(&self) -> &str;
    fn triggers(&self) -> &[(&str, &[&str])];
    fn soul(&self) -> &DVector<f64>;
    fn soul_mut(&mut self) -> &mut DVector<f64>;
    fn client(&self) -> &OllamaClient;

    fn should_activate(&self, query: &str) -> Option<String> {
        let q = query.to_lowercase();
        for (domain, keywords) in self.triggers() {
            for kw in *keywords {
                if q.contains(kw) {
                    return Some(domain.to_string());
                }
            }
        }
        None
    }

    /// Speak using a pre-computed semantic embedding as the attractor
    /// The quorum passes the nomic embedding so the soul navigates
    /// toward semantically meaningful positions on the manifold
    fn speak(&mut self, query: &str, attractor: &DVector<f64>) -> SpeakResult {
        let domain = match self.should_activate(query) {
            Some(d) => d,
            None => return SpeakResult {
                activated:  false,
                domain:     None,
                response:   None,
                soul_drift: 0.0,
                psi_final:  0.0,
                phase:      "dark".to_string(),
            },
        };

        println!("\n[{}] Activated on domain: {}", self.name(), domain);

        // Use the semantic attractor — same embedding used for routing
        let (new_pos, history) = retrocausal_steer(self.soul(), attractor, 20, 0.3);

        let psi_final = history.last().map(|h| h.psi).unwrap_or(0.0);
        let burden    = history.last().map(|h| h.burden).unwrap_or(0.0);
        let snr       = history.last().map(|h| h.snr).unwrap_or(0.0);

        let phase = final_phase(&history);

        println!("[{}] Final phase: {} Psi={:.4} Burden={:.4} SNR={:.2}",
            self.name(), phase.label(), psi_final, burden, snr);

        if !phase.can_respond() {
            let old_soul = self.soul().clone();
            let new_soul = update_soul(&old_soul, &new_pos);
            *self.soul_mut() = new_soul;
            return SpeakResult {
                activated:  true,
                domain:     Some(domain),
                response:   None,
                soul_drift: 0.0,
                psi_final,
                phase:      phase.label().to_string(),
            };
        }

        let soul_context = format!(
            "\nYour current geometric state: Phase={} Psi={:.3} Burden={:.3} SNR={:.2}",
            phase.label(), psi_final, burden, snr
        );
        let conditioned_prompt = format!("{}{}", query, soul_context);

        let response = self.client()
            .generate(
                &conditioned_prompt,
                self.system_prompt(),
                phase.temperature(),
                phase.max_tokens(),
            )
            .unwrap_or_else(|e| format!("[{} unreachable: {}]", self.name(), e));

        let old_soul = self.soul().clone();
        let new_soul = update_soul(&old_soul, &new_pos);
        let drift    = (&new_soul - &old_soul).norm();
        *self.soul_mut() = new_soul;

        println!("[{}] Soul drift: {:.6}", self.name(), drift);

        SpeakResult {
            activated:  true,
            domain:     Some(domain),
            response:   Some(response),
            soul_drift: drift,
            psi_final,
            phase:      phase.label().to_string(),
        }
    }
}
