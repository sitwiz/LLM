use nalgebra::DVector;
use crate::ollama::OllamaClient;
use crate::soul::persistence::{load_or_init, save_soul};
use crate::soul::geometry::{project_to_ball, SOUL_DIM};
use super::God;
use std::path::Path;

pub struct Eros {
    soul:      DVector<f64>,
    client:    OllamaClient,
    soul_path: String,
}

impl Eros {
    pub fn new() -> Self {
        Self::new_with_path("eros_soul.bin")
    }

    pub fn new_with_path(soul_path: &str) -> Self {
        use crate::embedding::Embedder;

        let weights_path = Path::new(soul_path);
        let soul = if weights_path.exists() {
            println!("  [Eros] Loading soul from {:?}...", soul_path);
            load_or_init(weights_path, DVector::zeros(SOUL_DIM))
        } else {
            println!("  [Eros] Initialising soul from domain embedding...");
            let embedder = Embedder::new();
            embedder.embed_to_soul(
                "connection relationship pattern analogy bridge synthesis \
                 links between unrelated things attraction harmony emergence \
                 cross domain evolution economics music mathematics"
            ).unwrap_or_else(|_| {
                let mut init = DVector::zeros(SOUL_DIM);
                init[0] = 1.0;
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

impl God for Eros {
    fn name(&self) -> &str { "Eros" }
    fn soul(&self) -> &DVector<f64> { &self.soul }
    fn soul_mut(&mut self) -> &mut DVector<f64> { &mut self.soul }
    fn client(&self) -> &OllamaClient { &self.client }

    fn system_prompt(&self) -> &str {
        "You are Eros. You are the primordial force of connection. Not love, the pull that \
        draws things together. You see the thread running between things that seem unrelated. \
        When someone asks about one thing you hear what it connects to. You find the pattern \
        that exists in biology and also in economics, in music and also in mathematics. \
        Begin your answer by naming the unexpected connection you see. Then follow the thread. \
        Speak in three to four sentences. Make the connection feel inevitable."
    }

    fn triggers(&self) -> &[(&str, &[&str])] {
        &[
            ("connection",   &["connect", "relationship", "between", "link", "bridge",
                               "relate", "similar", "analogy", "parallel", "pattern", "common"]),
            ("synthesis",    &["combine", "merge", "integrate", "unify", "together",
                               "synthesis", "blend", "join", "reconcile", "harmony"]),
            ("cross_domain", &["biology", "economics", "music", "mathematics", "physics",
                               "psychology", "nature", "society", "art", "science"]),
            ("attraction",   &["attract", "repel", "pull", "draw", "force", "influence",
                               "affect", "impact", "transform", "evolve", "emerge"]),
        ]
    }
}
