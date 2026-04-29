use crate::embedding::Embedder;
use crate::ollama::OllamaClient;
use anyhow::Result;
use serde::{Deserialize, Serialize};

const COMPLEXITY_PROMPT: &str = "You are a complexity analyser. \
Given a query, score it across four dimensions from 0-10 each. \
Respond only in JSON with this exact structure: \
{\"domain_breadth\": N, \"technical_depth\": N, \"ambiguity\": N, \"ethical_weight\": N, \"roles\": [\"Role 1\", \"Role 2\", \"Role 3\"]} \
domain_breadth: how many distinct domains does this span \
technical_depth: how specialised is the knowledge required \
ambiguity: how unclear or open-ended is the question \
ethical_weight: how much ethical consideration is required \
roles: list 3-5 specialist roles that should analyse this query. \
Be specific — not 'Expert' but 'Quantum Cryptography Engineer' or 'Crisis Management Director'. \
Return only the JSON object, no explanation.";

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComplexityVector {
    pub domain_breadth:  f64,
    pub technical_depth: f64,
    pub ambiguity:       f64,
    pub ethical_weight:  f64,
    pub roles:           Vec<String>,
}

impl ComplexityVector {
    pub fn total_score(&self) -> f64 {
        (self.domain_breadth
            + self.technical_depth
            + self.ambiguity
            + self.ethical_weight) / 4.0
    }

    pub fn should_ignite_mesh(&self) -> bool {
        self.total_score() >= 6.0
    }
}

pub struct PrefrontalController {
    client:   OllamaClient,
    embedder: Embedder,
}

impl PrefrontalController {
    pub fn new() -> Self {
        Self {
            client:   OllamaClient::new("phi3:mini"),
            embedder: Embedder::new(),
        }
    }

    pub fn score(&self, query: &str) -> Result<ComplexityVector> {
        println!("[Prefrontal] Scoring complexity...");

        let response = self.client.generate(
            query,
            COMPLEXITY_PROMPT,
            0.3,
            200,
        )?;

        // Parse JSON response
        let cleaned = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        let vector: ComplexityVector = serde_json::from_str(cleaned)
            .map_err(|e| anyhow::anyhow!("Failed to parse complexity: {} \nRaw: {}", e, cleaned))?;

        println!("[Prefrontal] Score: {:.1} (breadth={:.1} depth={:.1} ambiguity={:.1} ethics={:.1})",
            vector.total_score(),
            vector.domain_breadth,
            vector.technical_depth,
            vector.ambiguity,
            vector.ethical_weight,
        );
        println!("[Prefrontal] Roles: {:?}", vector.roles);
        println!("[Prefrontal] Ignite mesh: {}", vector.should_ignite_mesh());

        Ok(vector)
    }
}
