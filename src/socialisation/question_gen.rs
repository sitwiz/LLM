use nalgebra::DVector;
use crate::quorum::Quorum;
use crate::ollama::OllamaClient;

pub struct QuestionGenerator {
    pub instance_name: String,
    client:            OllamaClient,
}

impl QuestionGenerator {
    pub fn new(instance_name: &str) -> Self {
        Self {
            instance_name: instance_name.to_string(),
            client:        OllamaClient::new("phi3:mini"),
        }
    }

    /// Generate a question that emerges from the soul's current position
    /// and the nearest concepts in the instance's memory.
    pub fn generate(
        &self,
        soul:     &DVector<f64>,
        instance: &Quorum,
    ) -> String {
        // Find nearest concepts to the current soul position
        let nearest = instance.memory().spatial.nearest(soul, 3);

        let concept_context = if nearest.is_empty() {
            "You have no prior knowledge. Ask something fundamental.".to_string()
        } else {
            let concepts: Vec<String> = nearest.iter()
                .map(|c| format!("  - {} (strength={:.3})", c.name, c.strength))
                .collect();
            format!("Your nearest memories are:\n{}", concepts.join("\n"))
        };

        let soul_depth = soul.norm();
        let depth_desc = if soul_depth < 0.3 {
            "You are near the void — ask something primordial."
        } else if soul_depth < 0.5 {
            "You are at working depth — ask something practical or connective."
        } else {
            "You are deep — ask something that bridges multiple domains."
        };

        let prompt = format!(
            "{}\n\nSoul depth: {:.4} — {}\n\n\
            Generate a single, genuine question that emerges naturally from \
            this state. The question should feel like it comes from this \
            specific perspective — not generic. Do not explain, just ask the question.",
            concept_context, soul_depth, depth_desc
        );

        let system = "You are a mind generating questions from your current \
            geometric and conceptual position. Your question must be singular, \
            specific, and genuinely curious. Output only the question — \
            no preamble, no explanation, no quotation marks.";

        let result = self.client
            .generate(&prompt, system, 0.8, 80)
            .unwrap_or_else(|_| "What is the nature of understanding?".to_string());
        let trimmed = result.trim().to_string();
        if trimmed.is_empty() {
            "What is the nature of understanding?".to_string()
        } else {
            trimmed
        }
    }
}
