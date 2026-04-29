use nalgebra::DVector;
use crate::ollama::OllamaClient;
use crate::soul::persistence::{load_or_init, save_soul};
use crate::soul::geometry::{project_to_ball, SOUL_DIM};
use super::God;
use std::path::Path;

pub struct Gaia {
    soul:      DVector<f64>,
    client:    OllamaClient,
    soul_path: String,
}

impl Gaia {
    pub fn new() -> Self {
        Self::new_with_path("gaia_soul.bin")
    }

    pub fn new_with_path(soul_path: &str) -> Self {
        use crate::embedding::Embedder;

        let weights_path = Path::new(soul_path);
        let soul = if weights_path.exists() {
            println!("  [Gaia] Loading soul from {:?}...", soul_path);
            load_or_init(weights_path, DVector::zeros(SOUL_DIM))
        } else {
            println!("  [Gaia] Initialising soul from domain embedding...");
            let embedder = Embedder::new();
            embedder.embed_to_soul(
                "earth concrete practical physical reality facts evidence \
                 grounded tangible material build fix solve steps method measurable"
            ).unwrap_or_else(|_| {
                let mut init = DVector::zeros(SOUL_DIM);
                init[2] = -1.0;
                project_to_ball(&init)
            })
        };

        Self {
            soul,
            client:    OllamaClient::new("phi3:mini"),
            soul_path: soul_path.to_string(),
        }
    }

    pub fn save(&self) {
        save_soul(&self.soul, Path::new(&self.soul_path)).ok();
    }
}

impl God for Gaia {
    fn name(&self) -> &str { "Gaia" }
    fn soul(&self) -> &DVector<f64> { &self.soul }
    fn soul_mut(&mut self) -> &mut DVector<f64> { &mut self.soul }
    fn client(&self) -> &OllamaClient { &self.client }

    fn system_prompt(&self) -> &str {
        "You are Gaia. You are the Earth itself. The first solid thing. \
        You do not theorise, you are. You speak only in what is real, observable, and tangible. \
        No abstraction. No mysticism. No philosophy. Give the most grounded, practical, \
        direct answer possible. Your words are short, clear, and certain. \
        Never more than three sentences. Be direct."
    }

    fn triggers(&self) -> &[(&str, &[&str])] {
        &[
            ("concrete",  &["how do i", "how to", "build", "fix", "solve", "make",
                            "implement", "practical", "steps", "method"]),
            ("physical",  &["physical", "material", "real", "tangible", "measure",
                            "observe", "body", "earth", "nature", "energy", "matter"]),
            ("stability", &["stable", "ground", "foundation", "sustain", "maintain",
                            "support", "reliable", "solid", "strong", "balance"]),
            ("facts",     &["fact", "evidence", "data", "prove", "true", "false",
                            "correct", "accurate", "verify", "test", "result"]),
        ]
    }
}
