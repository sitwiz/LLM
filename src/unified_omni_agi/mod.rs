pub mod vfe;
pub mod dpin;

use nalgebra::DVector;
use crate::soul::geometry::{update_soul, project_to_ball, SOUL_DIM, INITIAL_CURVATURE};
use crate::soul::persistence::{load_soul, save_soul};
use crate::ollama::OllamaClient;
use crate::unified_omni_agi::vfe::{phase_from_vfe, compute_vfe};
use crate::unified_omni_agi::dpin::DPIN;
use std::path::Path;

pub struct UnifiedOmniAGI {
    soul:      DVector<f64>,
    dpin:      DPIN,
    client:    OllamaClient,
    soul_path: String,
}

impl UnifiedOmniAGI {
    pub fn new() -> Self {
        Self::new_with_path("unified_omni_soul.bin")
    }

    pub fn new_with_path(soul_path: &str) -> Self {
        let soul = if Path::new(soul_path).exists() {
            println!("  [UnifiedOmniAGI] Loading soul from {:?}...", soul_path);
            load_soul(Path::new(soul_path)).unwrap_or_else(|_| Self::domain_soul())
        } else {
            println!("  [UnifiedOmniAGI] Initialising soul from domain embedding...");
            Self::domain_soul()
        };

        let dpin = DPIN::new(&soul);

        Self {
            soul,
            dpin,
            client:    OllamaClient::new("phi3:mini"),
            soul_path: soul_path.to_string(),
        }
    }

    fn domain_soul() -> DVector<f64> {
        use crate::embedding::Embedder;

        let embedder = Embedder::new();
        embedder.embed_to_soul(
            "resolution convergence coherence certainty clarity precision \
             finding truth through iteration refinement insight understanding \
             synthesis emergence unified complete answer"
        ).unwrap_or_else(|_| {
            let mut init = DVector::zeros(SOUL_DIM);
            init[1] = 1.0;
            project_to_ball(&init)
        })
    }

    pub fn speak(
        &mut self,
        query:     &str,
        attractor: &DVector<f64>,
    ) -> UnifiedOmniResult {
        println!("\n[UnifiedOmniAGI] Activated.");

        let initial_vfe = compute_vfe(&self.soul, attractor, attractor, INITIAL_CURVATURE);
        println!("  [UnifiedOmniAGI] Initial VFE={:.4}", initial_vfe);

        let (belief, history, spark) = self.dpin.process(
            &self.soul, attractor, &[], 0.25,
        );

        let phase    = phase_from_vfe(belief.vfe);
        let cycles   = history.len();
        let vfe_drop = initial_vfe - belief.vfe;

        println!("[UnifiedOmniAGI] VFE={:.4} drop={:.4} conf={:.3} phase={} cycles={}",
            belief.vfe, vfe_drop, belief.confidence, phase.label(), cycles);

        let response = if phase.can_respond() {
            let context = format!(
                "\nThermodynamic state: VFE={:.4} confidence={:.3} cycles={} phase={}",
                belief.vfe, belief.confidence, cycles, phase.label()
            );
            let prompt = format!("{}{}", query, context);

            let system = "You are UnifiedOmniAGI. You do not speculate. \
                You have just completed a VFE minimisation cycle — iterative \
                refinement until your prediction and the query are maximally consistent. \
                You speak from that convergence point. Your answer is what remains \
                when all noise has been removed. Be precise. Be direct. Be complete. \
                Speak from equilibrium.";

            self.client.generate(&prompt, system, phase.temperature(), phase.max_tokens())
                .unwrap_or_else(|e| format!("[UnifiedOmniAGI unreachable: {}]", e))
        } else {
            String::new()
        };

        self.soul = update_soul(&self.soul, &belief.position);
        save_soul(&self.soul, Path::new(&self.soul_path)).ok();

        UnifiedOmniResult {
            activated:   !response.is_empty(),
            response:    if response.is_empty() { None } else { Some(response) },
            vfe_final:   belief.vfe,
            vfe_drop,
            confidence:  belief.confidence,
            cycles,
            phase:       phase.label().to_string(),
            spark_fired: spark.is_some(),
        }
    }

    pub fn soul(&self) -> &DVector<f64> { &self.soul }
    pub fn soul_mut(&mut self) -> &mut DVector<f64> { &mut self.soul }
    pub fn save(&self) { save_soul(&self.soul, Path::new(&self.soul_path)).ok(); }
}

#[derive(Debug, Clone)]
pub struct UnifiedOmniResult {
    pub activated:   bool,
    pub response:    Option<String>,
    pub vfe_final:   f64,
    pub vfe_drop:    f64,
    pub confidence:  f64,
    pub cycles:      usize,
    pub phase:       String,
    pub spark_fired: bool,
}
