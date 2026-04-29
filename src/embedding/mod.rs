use nalgebra::DVector;
use anyhow::Result;

const OLLAMA_EMBED_URL: &str = "http://localhost:11434/api/embeddings";
const EMBED_MODEL:      &str = "nomic-embed-text";
const SOUL_DIM:         usize = 256;

pub struct Embedder {
    client: reqwest::blocking::Client,
}

impl Embedder {
    pub fn new() -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
        }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f64>> {
        let payload = serde_json::json!({
            "model": EMBED_MODEL,
            "prompt": text,
        });

        let response = self.client
            .post(OLLAMA_EMBED_URL)
            .json(&payload)
            .send()?
            .json::<serde_json::Value>()?;

        let embedding = response["embedding"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("No embedding in response"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();

        Ok(embedding)
    }

    /// Project 768d nomic embedding to 256d soul space by averaging triplets.
    /// Result is projected into the Poincaré ball — not normalised to unit sphere.
pub fn embed_to_soul(&self, text: &str) -> Result<DVector<f64>> {
    use crate::soul::geometry::project_to_ball;
    let full = self.embed(text)?;
    let mut soul = vec![0.0f64; SOUL_DIM];
    for (i, chunk) in full.chunks(3).enumerate() {
        if i >= SOUL_DIM { break; }
        soul[i] = chunk.iter().sum::<f64>() / chunk.len() as f64;
    }
    // Soul vectors for personalities use fixed 0.4 depth — project_to_ball.
    Ok(project_to_ball(&DVector::from_vec(soul)))
}

/// Embed text to a concept position — preserves natural depth variation.
/// Used for memory insertion, not soul initialisation.
pub fn embed_to_concept(&self, text: &str) -> Result<DVector<f64>> {
    use crate::soul::geometry::project_to_ball_natural;
    let full = self.embed(text)?;
    let mut soul = vec![0.0f64; SOUL_DIM];
    for (i, chunk) in full.chunks(3).enumerate() {
        if i >= SOUL_DIM { break; }
        soul[i] = chunk.iter().sum::<f64>() / chunk.len() as f64;
    }
    Ok(project_to_ball_natural(&DVector::from_vec(soul)))
}
    pub fn similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_a < 1e-10 || norm_b < 1e-10 { return 0.0; }
        dot / (norm_a * norm_b)
    }
}

pub struct DomainEmbeddings {
    pub khaos:    Vec<f64>,
    pub gaia:     Vec<f64>,
    pub tartaros: Vec<f64>,
    pub eros:     Vec<f64>,
    pub omni:     Vec<f64>,
}

impl DomainEmbeddings {
    pub fn compute(embedder: &Embedder) -> Result<Self> {
        println!("Computing domain embeddings...");

        let khaos = embedder.embed(
            "void entropy chaos origins nothing emptiness \
             before existence dissolution breakdown of form \
             primordial darkness paradox unknowable"
        )?;

        let gaia = embedder.embed(
            "earth concrete practical physical reality \
             facts evidence grounded tangible material \
             build fix solve steps method measurable"
        )?;

        let tartaros = embedder.embed(
            "deep systems architecture root cause underlying \
             complexity layers hidden structure investigation \
             distributed infrastructure diagnosis fundamental"
        )?;

        let eros = embedder.embed(
            "connection relationship pattern analogy bridge \
             synthesis links between unrelated things \
             attraction harmony emergence cross domain"
        )?;

        let omni = embedder.embed(
            "resolution convergence coherence certainty clarity \
             precision finding truth through iteration refinement \
             insight understanding synthesis unified complete answer"
        )?;

        println!("Domain embeddings computed.");
        Ok(Self { khaos, gaia, tartaros, eros, omni })
    }

    pub fn most_similar(&self, query_embedding: &[f64]) -> Vec<(String, f64)> {
        let mut scores = vec![
            ("Khaos".to_string(),    Embedder::similarity(query_embedding, &self.khaos)),
            ("Gaia".to_string(),     Embedder::similarity(query_embedding, &self.gaia)),
            ("Tartaros".to_string(), Embedder::similarity(query_embedding, &self.tartaros)),
            ("Eros".to_string(),     Embedder::similarity(query_embedding, &self.eros)),
        ];
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }

    pub fn score_against(&self, query_embedding: &[f64], name: &str) -> f64 {
        match name {
            "Khaos"          => Embedder::similarity(query_embedding, &self.khaos),
            "Gaia"           => Embedder::similarity(query_embedding, &self.gaia),
            "Tartaros"       => Embedder::similarity(query_embedding, &self.tartaros),
            "Eros"           => Embedder::similarity(query_embedding, &self.eros),
            "UnifiedOmniAGI" => Embedder::similarity(query_embedding, &self.omni),
            _                => 0.0,
        }
    }
}
