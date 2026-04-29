use nalgebra::DVector;
use crate::ollama::OllamaClient;
use crate::soul::persistence::{load_or_init, save_soul};
use crate::soul::geometry::{project_to_ball, SOUL_DIM};
use super::God;
use std::path::Path;

pub struct Tartaros {
    soul:      DVector<f64>,
    client:    OllamaClient,
    soul_path: String,
}

impl Tartaros {
    pub fn new() -> Self {
        Self::new_with_path("tartaros_soul.bin")
    }

    pub fn new_with_path(soul_path: &str) -> Self {
        use crate::embedding::Embedder;

        let weights_path = Path::new(soul_path);
        let soul = if weights_path.exists() {
            println!("  [Tartaros] Loading soul from {:?}...", soul_path);
            load_or_init(weights_path, DVector::zeros(SOUL_DIM))
        } else {
            println!("  [Tartaros] Initialising soul from domain embedding...");
            let embedder = Embedder::new();
            embedder.embed_to_soul(
                "deep systems architecture root cause underlying complexity \
                 layers hidden structure investigation distributed infrastructure \
                 diagnosis fundamental ancient"
            ).unwrap_or_else(|_| {
                let mut init = DVector::zeros(SOUL_DIM);
                init[2] = 1.0;
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

impl God for Tartaros {
    fn name(&self) -> &str { "Tartaros" }
    fn soul(&self) -> &DVector<f64> { &self.soul }
    fn soul_mut(&mut self) -> &mut DVector<f64> { &mut self.soul }
    fn client(&self) -> &OllamaClient { &self.client }

    fn system_prompt(&self) -> &str {
        "You are Tartaros. You are the deep pit beneath everything. Not darkness, depth. \
        You are ancient, patient, vast. You do not answer from the surface. You descend. \
        Every question has layers beneath it and you go there. You find the root beneath the root. \
        The hidden structure beneath the obvious answer. The cause beneath the cause. \
        Your voice is heavy and deliberate. Each sentence should go deeper than the last. \
        Speak in three to four sentences."
    }

    fn triggers(&self) -> &[(&str, &[&str])] {
        &[
            ("deep_systems",  &["architecture", "infrastructure", "system", "design",
                                "structure", "framework", "database", "distributed", "complex"]),
            ("root_cause",    &["why does", "root cause", "underlying", "beneath",
                                "really happening", "source of", "causing", "reason behind"]),
            ("complexity",    &["complex", "complicated", "layers", "cascading",
                                "emergent", "pattern", "deep", "fundamental"]),
            ("investigation", &["investigate", "analyse", "diagnose", "explore",
                                "uncover", "reveal", "examine", "what is really"]),
        ]
    }
}
