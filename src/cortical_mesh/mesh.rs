use crate::cortical_mesh::prefrontal::{PrefrontalController, ComplexityVector};
use crate::cortical_mesh::corpus_callosum::CorpusCallosum;
use crate::cortical_mesh::node::{MeshNode, NodeOutput};
use crate::ollama::OllamaClient;
use anyhow::Result;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use std::fs;

const SYNTHESIS_PROMPT: &str = "You are the PrefrontalController synthesis layer. \
You receive analyses from multiple specialist nodes who have debated across two rounds. \
Synthesise their final positions into one authoritative response. \
Address the Devil's Advocate concerns explicitly. \
Be comprehensive but structured. Lead with the key finding, \
then supporting analysis, then caveats. Maximum 400 words.";

const CACHE_PATH: &str = "complex_insights.json";
const SIMILARITY_THRESHOLD: f64 = 0.85;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CachedInsight {
    pub query:       String,
    pub response:    String,
    pub embedding:   Vec<f64>,
    pub soul_psi:    f64,
    pub soul_burden: f64,
    pub complexity:  f64,
}

#[derive(Debug, Clone)]
pub struct MeshResult {
    pub query:      String,
    pub response:   String,
    pub complexity: ComplexityVector,
    pub nodes_used: Vec<String>,
    pub rounds:     usize,
    pub from_cache: bool,
}

pub struct CorticalMesh {
    prefrontal: PrefrontalController,
    client:     OllamaClient,
    cache:      Vec<CachedInsight>,
}

impl CorticalMesh {
    pub fn new() -> Self {
        let cache = Self::load_cache();
        println!("[CorticalMesh] Initialised. Cache: {} insights.", cache.len());
        Self {
            prefrontal: PrefrontalController::new(),
            client:     OllamaClient::new("phi3:mini"),
            cache,
        }
    }

    pub fn process(
        &mut self,
        query: &str,
        god_context: &str,
        soul_psi: f64,
        soul_burden: f64,
        query_embedding: &[f64],
    ) -> Result<MeshResult> {

        // Check cache
        if let Some(cached) = self.check_cache(query_embedding, soul_psi) {
            println!("[CorticalMesh] Cache hit.");
            return Ok(MeshResult {
                query:      query.to_string(),
                response:   cached.response.clone(),
                complexity: ComplexityVector {
                    domain_breadth:  cached.complexity,
                    technical_depth: cached.complexity,
                    ambiguity:       cached.complexity,
                    ethical_weight:  cached.complexity,
                    roles:           vec![],
                },
                nodes_used: vec!["cache".to_string()],
                rounds:     0,
                from_cache: true,
            });
        }

        let complexity = self.prefrontal.score(query)?;

        println!("\n[CorticalMesh] ═══════════════════════════════════");
        println!("[CorticalMesh] Igniting mesh. Score: {:.1}", complexity.total_score());
        println!("[CorticalMesh] Spawning {} nodes in parallel...", complexity.roles.len() + 1);
        println!("[CorticalMesh] ═══════════════════════════════════\n");

        let mut roles = complexity.roles.clone();
        roles.push("Devil's Advocate".to_string());
        let node_roles: Vec<String> = roles.clone();

        // Shared CorpusCallosum
        let cc = Arc::new(CorpusCallosum::new());

        // Round 1 — all nodes fire simultaneously using threads
        let node_ids: Vec<String> = roles.iter()
            .map(|r| format!("{}-{:x}", r.to_lowercase().replace(' ', "-"), rand::random::<u32>()))
            .collect();

        let round1_outputs: Arc<Mutex<Vec<NodeOutput>>> = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::new();

        for (i, role) in roles.iter().enumerate() {
            let role = role.clone();
            let node_id = node_ids[i].clone();
            let all_ids = node_ids.clone();
            let cc_ref = Arc::clone(&cc);
            let outputs_ref = Arc::clone(&round1_outputs);
            let query = query.to_string();
            let god_context = god_context.to_string();
            let soul_psi = soul_psi;
            let soul_burden = soul_burden;

            let handle = std::thread::spawn(move || {
                println!("[Node {}] Round 1 starting...", role);

                let node = MeshNode::new(&role, &god_context, soul_psi, soul_burden);
                let client = OllamaClient::new("phi3:mini");

                let system = node.system_prompt_str();
                let prompt = format!("Query: {}\n\nGod context: {}", query, god_context);

                match client.generate(&prompt, &system, 0.7, 250) {
                    Ok(analysis) => {
                        println!("[Node {}] Round 1 complete.", role);

                        // Broadcast to all other nodes
                        cc_ref.broadcast(&node_id, &analysis, 1, &all_ids);

                        let mut outputs = outputs_ref.lock().unwrap();
                        outputs.push(NodeOutput {
                            role:     role.clone(),
                            round:    1,
                            analysis: analysis.clone(),
                            concerns: vec![],
                        });
                    }
                    Err(e) => println!("[Node {}] Round 1 failed: {}", role, e),
                }
            });

            handles.push(handle);
        }

        // Wait for all round 1 threads
        for handle in handles {
            handle.join().ok();
        }

        println!("[CorticalMesh] Round 1 complete. Starting round 2...");

        // Round 2 — all nodes read round 1 outputs and refine
        let round1_done = round1_outputs.lock().unwrap().clone();
        let round2_outputs: Arc<Mutex<Vec<NodeOutput>>> = Arc::new(Mutex::new(Vec::new()));

        let mut handles2 = Vec::new();

        for (i, role) in roles.iter().enumerate() {
            let role = role.clone();
            let node_id = node_ids[i].clone();
            let cc_ref = Arc::clone(&cc);
            let outputs_ref = Arc::clone(&round2_outputs);
            let query = query.to_string();
            let god_context = god_context.to_string();
            let round1 = round1_done.clone();

            // Find this node's round 1 output
            let my_round1 = round1.iter()
                .find(|o| o.role == role)
                .map(|o| o.analysis.clone())
                .unwrap_or_default();

            // Read messages from other nodes
            let messages = cc_ref.flush(&node_id);
            let others: String = messages.iter()
                .map(|m| format!("[{}]: {}", m.from, m.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            let handle = std::thread::spawn(move || {
                if my_round1.is_empty() { return; }

                println!("[Node {}] Round 2 starting ({} peer messages)...",
                    role, messages.len());

                let node = MeshNode::new(&role, &god_context, 0.0, 0.0);
                let client = OllamaClient::new("phi3:mini");
                let system = node.system_prompt_str();

                let prompt = format!(
                    "Query: {}\n\nYour round 1 analysis: {}\n\n\
                    Other specialists said:\n{}\n\n\
                    Refine your analysis. Address their concerns. \
                    What do you now think differently?",
                    query, my_round1, others
                );

                match client.generate(&prompt, &system, 0.6, 250) {
                    Ok(refined) => {
                        println!("[Node {}] Round 2 complete.", role);
                        let mut outputs = outputs_ref.lock().unwrap();
                        outputs.push(NodeOutput {
                            role:     role.clone(),
                            round:    2,
                            analysis: refined,
                            concerns: vec![],
                        });
                    }
                    Err(e) => println!("[Node {}] Round 2 failed: {}", role, e),
                }
            });

            handles2.push(handle);
        }

        for handle in handles2 {
            handle.join().ok();
        }

        println!("[CorticalMesh] Round 2 complete. Synthesising...");

        // Synthesise — prefer round 2, fall back to round 1
        let round2_done = round2_outputs.lock().unwrap().clone();
        let mut synthesis_input = format!("Query: {}\n\nGod context: {}\n\n", query, god_context);

        for role in &roles {
            let output = round2_done.iter().find(|o| &o.role == role)
                .or_else(|| round1_done.iter().find(|o| &o.role == role));
            if let Some(o) = output {
                synthesis_input.push_str(&format!(
                    "[{}] (Round {}):\n{}\n\n", o.role, o.round, o.analysis
                ));
            }
        }
        synthesis_input.push_str("Synthesise into a final authoritative response.");

        let response = self.client.generate(
            &synthesis_input, SYNTHESIS_PROMPT, 0.6, 500
        )?;

        let insight = CachedInsight {
            query:       query.to_string(),
            response:    response.clone(),
            embedding:   query_embedding.to_vec(),
            soul_psi,
            soul_burden,
            complexity:  complexity.total_score(),
        };
        self.cache.push(insight);
        self.save_cache();

        Ok(MeshResult {
            query:      query.to_string(),
            response,
            complexity,
            nodes_used: node_roles,
            rounds:     2,
            from_cache: false,
        })
    }

    fn check_cache(&self, query_embedding: &[f64], current_psi: f64) -> Option<&CachedInsight> {
        use crate::embedding::Embedder;
        for cached in &self.cache {
            let sim = Embedder::similarity(query_embedding, &cached.embedding);
            let soul_drift = (current_psi - cached.soul_psi).abs();
            if sim >= SIMILARITY_THRESHOLD && soul_drift < 2.0 {
                return Some(cached);
            }
        }
        None
    }

    fn load_cache() -> Vec<CachedInsight> {
        fs::read_to_string(CACHE_PATH)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    fn save_cache(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.cache) {
            fs::write(CACHE_PATH, json).ok();
        }
    }
}
