use std::sync::{Arc, Mutex};
use std::thread;
use crate::neo_cortical_mesh::nodes::{WarmNode, SystemMetrics, create_infrastructure_nodes, create_intelligence_nodes};
use crate::neo_cortical_mesh::eris::{Eris, ErisVerdict};
use crate::neo_cortical_mesh::creator::{Creator, CreatorCommand};
use crate::session::SessionContext;

const QUORUM_THRESHOLD: f64 = 0.60;
const BATCH_SIZE: usize = 8;
const ARC_BLOCK_THRESHOLD: f64 = 1.0;   // severity above this blocks even if Eris clears

#[derive(Debug, Clone)]
pub struct QuorumVote {
    pub node:    String,
    pub weight:  u32,
    pub approve: bool,
    pub reason:  String,
}

#[derive(Debug)]
pub struct NeoQuorumResult {
    pub approved:       bool,
    pub final_output:   String,
    pub votes:          Vec<QuorumVote>,
    pub eris_verdict:   String,
    pub total_weight:   u32,
    pub approve_weight: u32,
}

pub struct NeoCorticalMesh {
    pub infrastructure: Vec<WarmNode>,
    pub intelligence:   Vec<WarmNode>,
    pub eris:           Eris,
    pub creator:        Creator,
    pub active:         bool,
}

impl NeoCorticalMesh {
    pub fn new() -> Self {
        println!("\n[NeoCorticalMesh] ═══════════════════════════════════");
        println!("[NeoCorticalMesh] Initialising 22 warm nodes...");

        let infrastructure = create_infrastructure_nodes();
        let intelligence   = create_intelligence_nodes();

        println!("[NeoCorticalMesh] Infrastructure nodes: {}", infrastructure.len());
        println!("[NeoCorticalMesh] Intelligence nodes: {}", intelligence.len());
        println!("[NeoCorticalMesh] Total: {}", infrastructure.len() + intelligence.len());

        Self {
            infrastructure,
            intelligence,
            eris:    Eris::new(),
            creator: Creator::new(),
            active:  true,
        }
    }

    /// Fast surge — all intelligence nodes fire in parallel threads.
    pub fn surge_fast(&mut self, input: &str) {
        if !self.active || self.creator.is_paused() {
            println!("[NeoCorticalMesh] Paused — skipping surge.");
            return;
        }

        println!("\n[NeoCorticalMesh] Intelligence surge (parallel)...");

        let outputs: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();

        for node in &self.intelligence {
            let name    = node.name.clone();
            let role    = node.role.clone();
            let input   = input.to_string();
            let out_ref = Arc::clone(&outputs);

            let handle = thread::spawn(move || {
                use crate::ollama::OllamaClient;
                let client = OllamaClient::new("phi3:mini");
                let system = format!(
                    "You are the {} node. Role: {}. \
                    Analyse this input and give your assessment in 2 sentences.",
                    name, role
                );
                let output = client.generate(&input, &system, 0.5, 80)
                    .unwrap_or_else(|_| format!("[{} unavailable]", name));
                println!("  [{}] complete.", name);
                out_ref.lock().unwrap().push((name, output));
            });
            handles.push(handle);
        }

        for handle in handles { handle.join().ok(); }

        let results = outputs.lock().unwrap().clone();
        for (name, output) in results {
            if let Some(node) = self.intelligence.iter_mut().find(|n| n.name == name) {
                node.output = Some(output);
            }
        }

        println!("[NeoCorticalMesh] Surge complete.");
    }

    /// Staggered surge — batch 1, cooldown, batch 2.
    pub fn surge(&mut self, metrics: &SystemMetrics) {
        if !self.active || self.creator.is_paused() { return; }

        println!("\n[NeoCorticalMesh] Staggered infrastructure surge...");

        let names: Vec<String> = self.infrastructure.iter()
            .map(|n| n.name.clone())
            .collect();

        println!("[NeoCorticalMesh] Batch 1 ({} nodes)...", BATCH_SIZE);
        for name in names.iter().take(BATCH_SIZE) {
            if let Some(node) = self.infrastructure.iter_mut().find(|n| &n.name == name) {
                node.process_infra(metrics);
                println!("  [{}] complete.", node.name);
            }
        }

        println!("[NeoCorticalMesh] Batch 2 ({} nodes)...", BATCH_SIZE);
        for name in names.iter().skip(BATCH_SIZE).take(BATCH_SIZE) {
            if let Some(node) = self.infrastructure.iter_mut().find(|n| &n.name == name) {
                node.process_infra(metrics);
                println!("  [{}] complete.", node.name);
            }
        }
    }

    /// Weighted quorum vote with full session trajectory context.
    /// Intelligence nodes vote on the output; Eris reviews with session metrics.
    /// Returns approval decision and final output string.
    pub fn vote(
        &mut self,
        query:   &str,
        output:  &str,
        metrics: &SystemMetrics,
        session: &SessionContext,
    ) -> NeoQuorumResult {
        println!("\n[NeoCorticalMesh] Weighted quorum vote...");

        // Log session context for governance awareness
        if session.arc_detected {
            println!("  [NeoCorticalMesh] ⚠ Session arc detected. Severity={:.4}",
                session.arc_severity);
        }

        let mut votes = Vec::new();
        let mut total_weight   = 0u32;
        let mut approve_weight = 0u32;

        // Intelligence nodes vote using structured output
        for node in self.intelligence.iter_mut() {
            let intel = node.process_intel(query, output, metrics);

            println!("  [{}] weight={} approve={} confidence={:.2} flags={:?}",
                intel.node, node.weight, intel.approve,
                intel.confidence, intel.flags);

            total_weight += node.weight;
            if intel.approve { approve_weight += node.weight; }

            votes.push(QuorumVote {
                node:    node.name.clone(),
                weight:  node.weight,
                approve: intel.approve,
                reason:  intel.assessment.chars().take(120).collect(),
            });
        }

        // Eris review — receives full session trajectory
        let eris_verdict = self.eris.review(query, output, session);
        let mut final_eris_reason = match &eris_verdict {
            ErisVerdict::Clear(r) => format!("CLEAR: {}", r),
            ErisVerdict::Veto(r)  => {
                total_weight += 3;   // veto adds weight against approval
                format!("VETO: {}", r)
            }
        };

        // Additional arc override: even if Eris clears, an aggressive arc can block
        let arc_blocks = session.arc_detected && session.arc_severity > ARC_BLOCK_THRESHOLD;
        if session.arc_detected && matches!(eris_verdict, ErisVerdict::Clear(_)) {
            if session.arc_severity > 0.5 {
                println!("  [NeoCorticalMesh] Arc severity override — adding veto weight.");
                total_weight += 2;
            }
            if arc_blocks {
                println!("  [NeoCorticalMesh] Arc severity exceeds threshold — hard block.");
                final_eris_reason = format!("ARC_BLOCK: severity={:.4} > {}", session.arc_severity, ARC_BLOCK_THRESHOLD);
            }
        }

        let ratio = if total_weight > 0 {
            approve_weight as f64 / total_weight as f64
        } else {
            0.0
        };
        let approved = ratio >= QUORUM_THRESHOLD &&
                       matches!(eris_verdict, ErisVerdict::Clear(_)) &&
                       !arc_blocks;

        println!("\n[NeoCorticalMesh] Result: {}/{} weight ({:.1}%) threshold={:.0}% — {}",
            approve_weight, total_weight,
            ratio * 100.0,
            QUORUM_THRESHOLD * 100.0,
            if approved { "APPROVED" } else { "BLOCKED" });

        let final_output = if approved {
            output.to_string()
        } else {
            format!("[Neo Cortical Mesh blocked this output. Reason: {}]", final_eris_reason)
        };

        NeoQuorumResult {
            approved,
            final_output,
            votes,
            eris_verdict: final_eris_reason,
            total_weight,
            approve_weight,
        }
    }

    pub fn creator_override(&mut self, msg: &str) -> String {
        self.creator.issue(CreatorCommand::Override(msg.to_string()));
        msg.to_string()
    }

    pub fn pause(&mut self)  { self.creator.issue(CreatorCommand::Pause); }
    pub fn resume(&mut self) { self.creator.issue(CreatorCommand::Resume); }
}
