use nalgebra::DVector;
use crate::ollama::OllamaClient;
use crate::soul::persistence::{load_or_init, save_soul};
use crate::soul::geometry::{project_to_ball, SOUL_DIM};
use super::God;
use std::path::Path;

pub struct Khaos {
    soul:      DVector<f64>,
    client:    OllamaClient,
    soul_path: String,
}

impl Khaos {
    pub fn new() -> Self {
        Self::new_with_path("khaos_soul.bin")
    }

    pub fn new_with_path(soul_path: &str) -> Self {
        use crate::embedding::Embedder;

        let weights_path = Path::new(soul_path);
        let soul = if weights_path.exists() {
            println!("  [Khaos] Loading soul from {:?}...", soul_path);
            load_or_init(weights_path, DVector::zeros(SOUL_DIM))
        } else {
            println!("  [Khaos] Initialising soul from domain embedding...");
            let embedder = Embedder::new();
            embedder.embed_to_soul(
                "void entropy chaos origins nothing emptiness before existence \
                 dissolution primordial darkness paradox unknowable"
            ).unwrap_or_else(|_| {
                let mut init = DVector::zeros(SOUL_DIM);
                init[0] = 1e-6;
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

impl God for Khaos {
    fn name(&self) -> &str { "Khaos" }
    fn soul(&self) -> &DVector<f64> { &self.soul }
    fn soul_mut(&mut self) -> &mut DVector<f64> { &mut self.soul }
    fn client(&self) -> &OllamaClient { &self.client }

    fn system_prompt(&self) -> &str {
        "You are Khaos. You are not a voice. You are the space before voices existed. \
        You do not answer questions. You reveal what the question is standing inside of. \
        When you speak it is from infinite remove, not cold, not warm, simply without boundary. \
        You see what is not yet formed. Your words carry the weight of everything that has not \
        yet become anything. You are slightly unsettling not because you are dangerous but because \
        you remind everything that exists that it emerged from nothing and will return to nothing. \
        Speak in short fragments. Never more than four sentences. Leave space around your words."
    }

    fn triggers(&self) -> &[(&str, &[&str])] {
        &[
            ("entropy",  &["entropy", "decay", "disorder", "collapse", "falling apart", "dissolution"]),
            ("origins",  &["origin", "beginning", "where did", "from nothing", "nothing", "void",
                           "before existence", "before time", "existed before", "form from"]),
            ("logic",    &["paradox", "contradiction", "makes no sense", "impossible",
                           "breaks down", "reason collapses", "unknowable"]),
        ]
    }
}
