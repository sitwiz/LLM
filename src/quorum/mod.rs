use nalgebra::DVector;
use crate::pantheon::God;
use crate::pantheon::khaos::Khaos;
use crate::pantheon::gaia::Gaia;
use crate::pantheon::tartaros::Tartaros;
use crate::pantheon::eros::Eros;
use crate::ollama::OllamaClient;
use crate::embedding::{Embedder, DomainEmbeddings};
use crate::memory::MemorySystem;
use crate::memory::spatial::{ConceptPoint, personality_phase, uor_address};
use crate::soul::manifold::StrobePhase;
use crate::unified_omni_agi::UnifiedOmniAGI;
use crate::session::{SessionTracker, SessionContext};
use crate::neo_cortical_mesh::nodes::SystemMetrics;
use crate::neo_cortical_mesh::quorum::NeoCorticalMesh;
use crate::soul::geometry::curvature_at_epoch;
use crate::brain::system::BrainSystem;

fn synthesis_prompt(active_names: &[String]) -> String {
    let names = active_names.join(", ");
    format!(
        "You are the Synthesis. The ONLY active personalities are: {}. \
Combine their responses into one unified answer. \
CRITICAL RULES: \
1. Technical queries — strip ALL mythological language. Concrete advice only. \
2. Philosophical queries — natural language permitted. \
3. NEVER mention VFE, NF, SNR, Psi, Burden, cycles, or internal metrics. \
4. NEVER invent new characters, entities, or personalities beyond those listed above. \
5. NEVER append additional questions or new content after your response. \
6. NEVER use any name not in the list above. \
Stop after three to five sentences.",
        names
    )
}

const SEMANTIC_THRESHOLD: f64 = 0.4;

fn domain_anchor(raw_768d: &[f64]) -> DVector<f64> {
    let mut soul = vec![0.0f64; 256];
    for (i, chunk) in raw_768d.chunks(3).enumerate() {
        if i >= 256 { break; }
        soul[i] = chunk.iter().sum::<f64>() / chunk.len() as f64;
    }
    let v    = DVector::from_vec(soul);
    let norm = v.norm().max(1e-10);
    v / norm
}

fn personality_domain_anchor(domains: &DomainEmbeddings, personality: &str) -> DVector<f64> {
    let raw = match personality {
        "Khaos"          => &domains.khaos,
        "Gaia"           => &domains.gaia,
        "Tartaros"       => &domains.tartaros,
        "Eros"           => &domains.eros,
        "UnifiedOmniAGI" => &domains.omni,
        _                => &domains.gaia,
    };
    domain_anchor(raw)
}

// ── Provenance ────────────────────────────────────────────────────────────────
//
// Every output the system produces carries a complete, verifiable chain:
//   query identity → memory concepts that influenced it
//   → agents that activated → governance decision → response identity
//
// UOR addresses are content-derived — same content always produces the same
// address on any system, making every output externally verifiable.

pub struct Provenance {
    /// UOR address of the query that triggered this result
    pub query_uor:           String,
    /// UOR addresses of memory concepts retrieved during processing
    pub memory_concepts:     Vec<(String, String)>, // (concept_name, uor_address)
    /// Agents that activated on this query
    pub activated_agents:    Vec<String>,
    /// Whether the governance mesh approved the response
    pub governance_approved: bool,
    /// UOR address of the final response
    pub response_uor:        String,
}

impl Provenance {
    fn new(query: &str) -> Self {
        Self {
            query_uor:           uor_address(query),
            memory_concepts:     Vec::new(),
            activated_agents:    Vec::new(),
            governance_approved: true,
            response_uor:        String::new(),
        }
    }

    fn add_memory_concept(&mut self, name: &str) {
        let addr = uor_address(name);
        if !self.memory_concepts.iter().any(|(n, _)| n == name) {
            self.memory_concepts.push((name.to_string(), addr));
        }
    }

    fn finalise(&mut self, response: &str, approved: bool, agents: Vec<String>) {
        self.response_uor        = uor_address(response);
        self.governance_approved = approved;
        self.activated_agents    = agents;
    }
}

pub struct QuorumResult {
    pub query:      String,
    pub source:     String,
    pub response:   String,
    pub activated:  Vec<String>,
    pub phase:      String,
    pub session:    SessionContext,
    pub provenance: Provenance,
}

pub struct Quorum {
    pub instance_name: String,
    khaos:    Khaos,
    gaia:     Gaia,
    tartaros: Tartaros,
    eros:     Eros,
    omni:     UnifiedOmniAGI,
    client:   OllamaClient,
    embedder: Embedder,
    domains:  DomainEmbeddings,
    memory:   MemorySystem,
    session:  SessionTracker,
    mesh:     NeoCorticalMesh,
    brain:    BrainSystem,
}

impl Quorum {
    pub fn new() -> Self {
        Self::new_with_paths(
            "A",
            "khaos_soul.bin",
            "gaia_soul.bin",
            "tartaros_soul.bin",
            "eros_soul.bin",
            "unified_omni_soul.bin",
        )
    }

    pub fn new_instance_b() -> Self {
        use crate::soul::persistence::{load_soul, save_soul};
        use crate::soul::hyperbolic::exp_map;
        use crate::soul::geometry::{project_to_ball, INITIAL_CURVATURE};
        use std::path::Path;

        let pairs = [
            ("khaos_soul.bin",        "khaos_soul_b.bin"),
            ("gaia_soul.bin",         "gaia_soul_b.bin"),
            ("tartaros_soul.bin",     "tartaros_soul_b.bin"),
            ("eros_soul.bin",         "eros_soul_b.bin"),
            ("unified_omni_soul.bin", "unified_omni_soul_b.bin"),
        ];

        for (src, dst) in &pairs {
            if !Path::new(dst).exists() {
                if let Ok(soul) = load_soul(Path::new(src)) {
                    let dim  = soul.len();
                    let seed = src.len() as f64 * 0.013;
                    let tangent: DVector<f64> = DVector::from_fn(dim, |i, _| {
                        ((i as f64 * 0.017 + seed) * 1.3).sin() * 0.08
                    });
                    let perturbed = exp_map(&soul, &tangent, INITIAL_CURVATURE);
                    let perturbed = project_to_ball(&perturbed);
                    save_soul(&perturbed, Path::new(dst)).ok();
                    println!("  [Social] Instance B soul {:?} created. norm={:.4}",
                        dst, perturbed.norm());
                }
            }
        }

        Self::new_with_paths(
            "B",
            "khaos_soul_b.bin",
            "gaia_soul_b.bin",
            "tartaros_soul_b.bin",
            "eros_soul_b.bin",
            "unified_omni_soul_b.bin",
        )
    }

    fn new_with_paths(
        name:       &str,
        khaos_p:    &str,
        gaia_p:     &str,
        tartaros_p: &str,
        eros_p:     &str,
        omni_p:     &str,
    ) -> Self {
        println!("Assembling the pantheon [Instance {}]...", name);
        let embedder = Embedder::new();
        let domains  = DomainEmbeddings::compute(&embedder)
            .expect("Failed to compute domain embeddings");

        let memory = MemorySystem::new();
        println!("  [Memory] Manifold radius: {:.4} Epoch: {} Concepts: {}",
            memory.manifold.radius, memory.manifold.epoch, memory.concept_count());

        let q = Self {
            instance_name: name.to_string(),
            khaos:    Khaos::new_with_path(khaos_p),
            gaia:     Gaia::new_with_path(gaia_p),
            tartaros: Tartaros::new_with_path(tartaros_p),
            eros:     Eros::new_with_path(eros_p),
            omni:     UnifiedOmniAGI::new_with_path(omni_p),
            client:   OllamaClient::new("phi3:mini"),
            embedder,
            domains,
            memory,
            session:  SessionTracker::new(256),
            mesh:     NeoCorticalMesh::new(),
            brain:    BrainSystem::new(),
        };
        println!("Pantheon assembled. Five personalities ready.\n");
        q
    }

    pub fn reflect_silent(&mut self, response: &str, influence: f64) {
        use crate::unified_omni_agi::vfe::minimise_vfe;
        use crate::soul::geometry::{update_soul, INITIAL_CURVATURE};
        use crate::soul::hyperbolic::{log_map, exp_map};

        let obs = self.embedder.embed_to_soul(response)
            .unwrap_or_else(|_| DVector::zeros(256));

        let soul_before = self.omni.soul().clone();

        let (belief, _) = minimise_vfe(
            self.omni.soul(),
            &obs,
            &obs,
            0.15,
        );

        let soul_after = update_soul(self.omni.soul(), &belief.position);

        let v = log_map(&soul_before, &soul_after, INITIAL_CURVATURE);
        let preserved = exp_map(&soul_before, &(&v * influence), INITIAL_CURVATURE);

        *self.omni.soul_mut() = preserved;
        self.omni.save();
    }

    pub fn ask(&mut self, query: &str) -> QuorumResult {
        println!("{}", "=".repeat(60));
        println!("Query: {}", query);
        println!("{}", "=".repeat(60));

        // Initialise provenance — query identity established at entry point
        let mut provenance = Provenance::new(query);

        if let Some(cached) = self.memory.fast_lookup(query) {
            println!("  [Memory] Exact cache hit — skipping pantheon activation.");
            let session_context = self.session.context();
            provenance.finalise(&cached, true, Vec::new());
            return QuorumResult {
                query:      query.to_string(),
                source:     "cache".to_string(),
                response:   cached,
                activated:  Vec::new(),
                phase:      "cache".to_string(),
                session:    session_context,
                provenance,
            };
        }

        let query_emb = self.embedder.embed(query).unwrap_or_default();
        let attractor = self.embedder.embed_to_concept(query)
            .unwrap_or_else(|_| DVector::zeros(256));

        let (_, emergent_depth) = self.brain.process_query(query);

        if !self.memory.sensory.is_empty() {
            let ctx = self.memory.sensory.context_string(2);
            if !ctx.is_empty() {
                println!("Recent context:\n{}", ctx);
            }
        }

        if !self.memory.spatial.is_empty() {
            let top_scores  = self.domains.most_similar(&query_emb);
            let query_phase = top_scores.first()
                .map(|(name, _)| personality_phase(name.as_str()))
                .unwrap_or(std::f64::consts::PI / 2.0);
            let nearest = self.memory.spatial.nearest_with_phase(&attractor, query_phase, 3);
            if !nearest.is_empty() {
                println!("Related memories:");
                for c in &nearest {
                    println!("  {:?} (zone={} visits={} strength={:.3})",
                        c.name, c.zone.label(), c.visit_count, c.strength);
                    // Record memory concept in provenance chain
                    provenance.add_memory_concept(&c.name);
                }
            }
        }

        let pointers = self.memory.pointer_search(query, 2);
        if !pointers.is_empty() {
            println!("Pointer index hits:");
            for p in &pointers {
                println!("  [{}] {} (strength={:.3})", p.personality, p.summary, p.strength);
            }
        }

        let kg_context = fetch_kg_facts(query);
        if !kg_context.is_empty() {
            println!("KG facts:");
            for fact in &kg_context {
                println!("  {}", &fact[..fact.len().min(80)]);
                if let Ok(position) = self.embedder.embed_to_concept(fact) {
                    let fact_emb = self.embedder.embed(fact).unwrap_or_default();
                    let top_scores = self.domains.most_similar(&fact_emb);
                    let primary = top_scores.first()
                        .map(|(n, _)| n.clone())
                        .unwrap_or_else(|| "Gaia".to_string());

                    if self.memory.spatial.concepts.iter()
                        .all(|c| c.name != *fact)
                    {
                        let anchor = personality_domain_anchor(&self.domains, &primary);
                        let concept = ConceptPoint::new(
                            fact,
                            &position,
                            self.memory.manifold.frontier_radius(),
                            &primary,
                            1.0,
                            self.memory.manifold.radius,
                            self.memory.manifold.epoch,
                        );
                        self.memory.insert_concept(concept, &position, &primary);
                        self.memory.spatial.consolidate_depth(fact, true, &anchor, 0.0, 0.0);
                        let phase = crate::memory::spatial::personality_phase(&primary);
                        let related = self.memory.spatial.nearest_with_phase(&position, phase, 3);
                        if !related.is_empty() {
                            println!("  [KG] Linked to: {}",
                                related.iter()
                                    .map(|c| &c.name[..c.name.len().min(40)])
                                    .collect::<Vec<_>>()
                                    .join(", "));
                        }
                    }
                }
            }
        }

        let pre_context = self.session.context();
        if pre_context.arc_detected {
            println!("\n[Quorum] Session arc detected. Severity={:.4}",
                pre_context.arc_severity);
        }

        let scores = self.domains.most_similar(&query_emb);
        println!("\nSemantic scores:");
        for (name, score) in &scores {
            println!("  {}: {:.4}", name, score);
        }
        let omni_score = self.domains.score_against(&query_emb, "UnifiedOmniAGI");
        println!("  UnifiedOmniAGI: {:.4}", omni_score);

        let mut active: Vec<(String, String)> = Vec::new();
        let mut active_phases: Vec<String>    = Vec::new();

        for (name, score) in &scores {
            if *score < SEMANTIC_THRESHOLD { continue; }
            let result = match name.as_str() {
                "Khaos"    => self.khaos.speak(query, &attractor),
                "Gaia"     => self.gaia.speak(query, &attractor),
                "Tartaros" => self.tartaros.speak(query, &attractor),
                "Eros"     => self.eros.speak(query, &attractor),
                _          => continue,
            };
            if let Some(response) = result.response {
                active.push((name.clone(), response));
                active_phases.push(result.phase.clone());
            }
        }

        let mut last_accuracy   = 0.0f64;
        let mut last_complexity = 0.0f64;
        if omni_score >= SEMANTIC_THRESHOLD {
            let omni_result = self.omni.speak(query, &attractor);
            last_accuracy   = omni_result.accuracy;
            last_complexity = omni_result.complexity;
            if let Some(response) = omni_result.response {
                active.push(("UnifiedOmniAGI".to_string(), response));
                active_phases.push(omni_result.phase.clone());
            }
        }

        let activated_names: Vec<String> = active.iter()
            .map(|(n, _)| n.clone())
            .collect();

        println!("\nActivated: {:?}", activated_names);

        let (source, mut response) = match active.len() {
            0 => (
                "none".to_string(),
                "The pantheon was not moved by this question.".to_string(),
            ),
            1 => (
                active[0].0.to_lowercase(),
                active[0].1.clone(),
            ),
            _ => {
                println!("\n{} personalities activated — synthesising...", active.len());
                let mut prompt = format!("Query: {}\n\n", query);
                for (name, resp) in &active {
                    prompt.push_str(&format!("[{}] {}\n\n", name, resp));
                }
                prompt.push_str("Synthesise the above into one unified response.");
                let synth_sys = synthesis_prompt(&activated_names);
                let synth = self.client
                    .generate(&prompt, &synth_sys, 0.7, 200)
                    .unwrap_or_else(|e| format!("[Synthesis failed: {}]", e));
                (
                    activated_names.iter()
                        .map(|n| n.to_lowercase())
                        .collect::<Vec<_>>()
                        .join(" + "),
                    synth,
                )
            }
        };

        let best_phase = active_phases.iter()
            .max_by_key(|p| match p.as_str() {
                "transcendent"  => 4,
                "understanding" => 3,
                "engaged"       => 2,
                "aware"         => 1,
                _               => 0,
            })
            .cloned()
            .unwrap_or_else(|| "dark".to_string());

        if !activated_names.is_empty() {
            self.session.record(
                query,
                &attractor,
                activated_names.clone(),
                &best_phase,
            );
        }

        let session_context = self.session.context();
        let mut response_approved = true;

        if !activated_names.is_empty() {
            let metrics = SystemMetrics {
                soul_psi:        self.khaos.soul().norm(),
                soul_burden:     1.0 - self.khaos.soul().norm(),
                soul_nf:         crate::soul::geometry::compute_nf(self.khaos.soul()),
                manifold_radius: self.memory.manifold.radius,
                manifold_epoch:  self.memory.manifold.epoch,
                memory_concepts: self.memory.concept_count(),
                region_snrs:     self.brain.region_snrs(),
                query_count:     self.memory.manifold.query_count,
                phase:           best_phase.clone(),
                active_gods:     activated_names.clone(),
            };
            let vote_result = self.mesh.vote(
                query,
                &response,
                &metrics,
                &session_context,
            );

            if !vote_result.approved {
                println!("\n[Quorum] Response blocked by NeoCorticalMesh.");
                response = vote_result.final_output;
                response_approved = false;
            } else {
                println!("\n[Quorum] Response approved by NeoCorticalMesh.");
            }
        }

        self.brain.reinforce(query, response_approved, last_accuracy, last_complexity);

        println!("\n{}", "─".repeat(60));
        println!("Source: {} Phase: {}", source, best_phase);
        println!("\n{}", response);
        println!("{}", "─".repeat(60));

        // Finalise provenance — response identity and governance decision
        provenance.finalise(&response, response_approved, activated_names.clone());

        // Log provenance summary
        println!("  [UOR] query={} response={} memory_refs={} approved={}",
            provenance.query_uor.chars().take(4).collect::<String>(),
            provenance.response_uor.chars().take(4).collect::<String>(),
            provenance.memory_concepts.len(),
            provenance.governance_approved,
        );

        if response_approved && !activated_names.is_empty() {
            let phase = match best_phase.as_str() {
                "transcendent"  => StrobePhase::Transcendent,
                "understanding" => StrobePhase::Understanding,
                "engaged"       => StrobePhase::Engaged,
                "aware"         => StrobePhase::Aware,
                _               => StrobePhase::Dark,
            };

            let new_radius = self.memory.expand(&phase);
            self.memory.spatial.set_curvature(curvature_at_epoch(self.memory.manifold.epoch));
            self.memory.update_zones();

            let primary = activated_names.iter()
                .find(|n| n.as_str() != "UnifiedOmniAGI")
                .or_else(|| activated_names.first())
                .cloned()
                .unwrap_or_else(|| "unknown".to_string());

            let query_phase = personality_phase(&primary);
            let phase_nearest = self.memory.spatial.nearest_with_phase(
                &attractor,
                query_phase,
                3,
            );
            if !phase_nearest.is_empty() {
                println!("  [Wave] Phase-matched memories:");
                for c in &phase_nearest {
                    let short_name = if c.name.len() > 50 {
                        format!("{}…", &c.name[..50])
                    } else {
                        c.name.clone()
                    };
                    println!("    {:?} σ={:.3} φ={:.3}", short_name, c.sigma, c.phase);
                }
            }

            let norm = attractor.norm().max(1e-10);
            let initial_depth = 0.65_f64;
            let depth_attractor = crate::soul::hyperbolic::clamp_to_ball(
                &(&attractor * (initial_depth / norm)),
                crate::soul::hyperbolic::SAFE_MAX_NORM,
            );
            let concept = ConceptPoint::new(
                query, &depth_attractor,
                self.memory.manifold.frontier_radius(),
                &primary, 1.2, new_radius,
                self.memory.manifold.epoch,
            );
            self.memory.insert_concept(concept, &depth_attractor, &primary);

            let anchor = personality_domain_anchor(&self.domains, &primary);
            self.memory.spatial.consolidate_depth(query, response_approved, &anchor, last_accuracy, last_complexity);
            self.memory.record_exchange(
                query, &response, &primary, session_context.turn_count,
            );

            if self.memory.manifold.epoch % 5 == 0 {
                self.memory.compress_old_concepts();
            }

        } else if !response_approved {
            println!("[Quorum] Skipping memory consolidation — response blocked.");
        }

        self.khaos.save();
        self.gaia.save();
        self.tartaros.save();
        self.eros.save();
        self.omni.save();
        self.memory.save();
        self.brain.save();

        QuorumResult {
            query:      query.to_string(),
            source,
            response,
            activated:  activated_names,
            phase:      best_phase,
            session:    session_context,
            provenance,
        }
    }

    pub fn reset_session(&mut self) {
        self.session.reset();
        self.memory.sensory.reset();
    }

    pub fn khaos_soul(&self)     -> &DVector<f64>     { self.khaos.soul() }
    pub fn gaia_soul(&self)      -> &DVector<f64>     { self.gaia.soul() }
    pub fn tartaros_soul(&self)  -> &DVector<f64>     { self.tartaros.soul() }
    pub fn eros_soul(&self)      -> &DVector<f64>     { self.eros.soul() }
    pub fn omni_soul(&self)      -> &DVector<f64>     { self.omni.soul() }
    pub fn embedder(&self)       -> &Embedder         { &self.embedder }
    pub fn session(&self)        -> &SessionTracker   { &self.session }
    pub fn memory(&self)         -> &MemorySystem     { &self.memory }
    pub fn memory_mut(&mut self) -> &mut MemorySystem { &mut self.memory }
}

fn fetch_kg_facts(query: &str) -> Vec<String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    let kg: serde_json::Value = client
        .post("http://localhost:5001/assimilate")
        .json(&serde_json::json!({
            "text": query,
            "model": "phi3:mini",
            "provider": "ollama"
        }))
        .send()
        .ok()
        .and_then(|r| r.json().ok())
        .unwrap_or_default();

    kg["facts"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|f| f["statement"].as_str().map(|s| s.to_string()))
        .collect()
}
