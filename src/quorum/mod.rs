use nalgebra::DVector;
use crate::pantheon::God;
use crate::pantheon::khaos::Khaos;
use crate::pantheon::gaia::Gaia;
use crate::pantheon::tartaros::Tartaros;
use crate::pantheon::eros::Eros;
use crate::ollama::OllamaClient;
use crate::embedding::{Embedder, DomainEmbeddings};
use crate::memory::MemorySystem;
use crate::memory::spatial::ConceptPoint;
use crate::memory::spatial::personality_phase;
use crate::soul::manifold::StrobePhase;
use crate::unified_omni_agi::UnifiedOmniAGI;
use crate::session::{SessionTracker, SessionContext};
use crate::neo_cortical_mesh::nodes::SystemMetrics;
use crate::neo_cortical_mesh::quorum::NeoCorticalMesh;
use crate::soul::geometry::curvature_at_epoch;
use crate::brain::system::BrainSystem;

const SYNTHESIS_PROMPT: &str = "You are the Synthesis. You receive responses from cognitive \
personalities and produce one unified answer. \
CRITICAL RULE: If the query is technical or practical — code, systems, debugging, \
how-to questions — strip ALL mythological and poetic language entirely. \
Deliver only concrete, actionable, technical advice. \
If the query is philosophical or abstract — consciousness, meaning, connection — \
you may use the poetic framing naturally. \
Never mix mythology into technical answers. \
Three to five sentences maximum.";

const SEMANTIC_THRESHOLD: f64 = 0.4;

/// Project a raw 768d domain embedding to a unit vector in 256d soul space.
/// Uses the same averaging-triplets projection as embed_to_concept.
/// This gives each personality a semantically meaningful anchor direction
/// derived from its actual domain description — no hardcoding needed.
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

/// Get the domain anchor for a named personality from the domain embeddings.
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

pub struct QuorumResult {
    pub query:     String,
    pub source:    String,
    pub response:  String,
    pub activated: Vec<String>,
    pub phase:     String,
    pub session:   SessionContext,
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

    pub fn reflect_silent(&mut self, response: &str) {
        use crate::unified_omni_agi::vfe::minimise_vfe;
        use crate::soul::geometry::update_soul;

        let obs = self.embedder.embed_to_soul(response)
            .unwrap_or_else(|_| DVector::zeros(256));

        let (belief, _) = minimise_vfe(
            self.omni.soul(),
            &obs,
            &obs,
            0.15,
        );

        *self.omni.soul_mut() = update_soul(self.omni.soul(), &belief.position);
        *self.khaos.soul_mut() = update_soul(self.khaos.soul(), &belief.position);

        self.omni.save();
        self.khaos.save();
    }

    pub fn ask(&mut self, query: &str) -> QuorumResult {
        println!("{}", "=".repeat(60));
        println!("Query: {}", query);
        println!("{}", "=".repeat(60));

        if let Some(cached) = self.memory.fast_lookup(query) {
            println!("  [Memory] Exact cache hit — skipping pantheon activation.");
            let session_context = self.session.context();
            return QuorumResult {
                query:     query.to_string(),
                source:    "cache".to_string(),
                response:  cached,
                activated: Vec::new(),
                phase:     "cache".to_string(),
                session:   session_context,
            };
        }

        let query_emb = self.embedder.embed(query).unwrap_or_default();
        let attractor = self.embedder.embed_to_concept(query)
            .unwrap_or_else(|_| DVector::zeros(256));

        // Brain processes query — produces emergent depth from region activation
        let (_, emergent_depth) = self.brain.process_query(query);

        if !self.memory.sensory.is_empty() {
            let ctx = self.memory.sensory.context_string(2);
            if !ctx.is_empty() {
                println!("Recent context:\n{}", ctx);
            }
        }

        // Phase-aware retrieval — use query's dominant domain phase
        // to suppress cross-domain memories via wave packet interference.
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

        if omni_score >= SEMANTIC_THRESHOLD {
            let omni_result = self.omni.speak(query, &attractor);
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
                    prompt.push_str(&format!("{} says: {}\n\n", name, resp));
                }
                prompt.push_str("Weave these into a single unified response.");
                let synth = self.client
                    .generate(&prompt, SYNTHESIS_PROMPT, 0.7, 200)
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

        self.brain.reinforce(query, response_approved);

        println!("\n{}", "─".repeat(60));
        println!("Source: {} Phase: {}", source, best_phase);
        println!("\n{}", response);
        println!("{}", "─".repeat(60));

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

            // Phase-matched memory display
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

            // Consolidate toward the personality's real domain embedding direction.
            // This keeps Rust memories near the Gaia anchor and consciousness
            // memories near the Khaos anchor — not collapsed to the same origin.
            let anchor = personality_domain_anchor(&self.domains, &primary);
            self.memory.spatial.consolidate_depth(query, response_approved, &anchor);

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
            query:     query.to_string(),
            source,
            response,
            activated: activated_names,
            phase:     best_phase,
            session:   session_context,
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
