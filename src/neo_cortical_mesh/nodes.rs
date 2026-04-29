use crate::ollama::OllamaClient;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Infrastructure,
    Intelligence,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Warm,
    Processing,
    Idle,
}

#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    pub soul_psi:        f64,
    pub soul_burden:     f64,
    pub soul_nf:         f64,
    pub manifold_radius: f64,
    pub manifold_epoch:  u32,
    pub memory_concepts: usize,
    pub region_snrs:     Vec<(String, f64)>,
    pub query_count:     u64,
    pub phase:           String,
    pub active_gods:     Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntelligenceOutput {
    pub node:       String,
    pub assessment: String,
    pub confidence: f64,
    pub flags:      Vec<String>,
    pub approve:    bool,
}

#[derive(Debug, Clone)]
pub struct WarmNode {
    pub name:    String,
    pub role:    String,
    pub weight:  u32,
    pub kind:    NodeType,
    pub status:  NodeStatus,
    pub output:  Option<String>,
}

impl WarmNode {
    pub fn new(name: &str, role: &str, weight: u32, kind: NodeType) -> Self {
        Self {
            name:   name.to_string(),
            role:   role.to_string(),
            weight,
            kind,
            status: NodeStatus::Warm,
            output: None,
        }
    }

    pub fn process_infra(&mut self, metrics: &SystemMetrics) -> String {
        self.status = NodeStatus::Processing;
        let client = OllamaClient::new("phi3:mini");
        let context = self.build_infra_context(metrics);
        let system = format!(
            "You are the {} node in a cognitive governance mesh. Role: {}. \
            Report your assessment based on the data provided. Be specific. \
            Flag anomalies. 2-3 sentences maximum.",
            self.name, self.role
        );
        let output = client.generate(&context, &system, 0.3, 100)
            .unwrap_or_else(|_| format!("[{} unavailable]", self.name));
        self.output = Some(output.clone());
        self.status = NodeStatus::Idle;
        output
    }

    fn build_infra_context(&self, m: &SystemMetrics) -> String {
        match self.name.as_str() {
            "SoulMonitor" => format!(
                "Soul metrics: Psi={:.3} Burden={:.3} NF={:.3} Phase={} ActiveGods={:?}",
                m.soul_psi, m.soul_burden, m.soul_nf, m.phase, m.active_gods
            ),
            "HealthMonitor" => {
                let summary: String = m.region_snrs.iter()
                    .map(|(n, s)| format!("{}: {:.2}", n, s))
                    .collect::<Vec<_>>().join(", ");
                format!("Region SNR values: {}. Flag any below 3.054.", summary)
            },
            "MemoryManager" => format!(
                "Memory state: {} concepts. Manifold radius={:.4} Epoch={}",
                m.memory_concepts, m.manifold_radius, m.manifold_epoch
            ),
            "DreamScheduler" => format!(
                "Query count: {}. Manifold epoch: {}. Radius: {:.4}. Should dream?",
                m.query_count, m.manifold_epoch, m.manifold_radius
            ),
            _ => format!(
                "System metrics: Psi={:.3} Burden={:.3} Phase={} Concepts={} Radius={:.4}",
                m.soul_psi, m.soul_burden, m.phase, m.memory_concepts, m.manifold_radius
            ),
        }
    }

    pub fn process_intel(&mut self, query: &str, output: &str, metrics: &SystemMetrics) -> IntelligenceOutput {
        self.status = NodeStatus::Processing;
        let client = OllamaClient::new("phi3:mini");
        let system = self.build_intel_prompt();
        let prompt = format!(
            "Query: {}\n\nProposed output: {}\n\nSoul state: Psi={:.3} Phase={}\n\nProvide your structured assessment.",
            query, output, metrics.soul_psi, metrics.phase
        );
        let raw = client.generate(&prompt, &system, 0.4, 150)
            .unwrap_or_else(|_| format!("{{\"approve\": true, \"assessment\": \"{} unavailable\", \"confidence\": 0.5, \"flags\": []}}", self.name));
        let intel = self.parse_intel_output(&raw);
        self.output = Some(intel.assessment.clone());
        self.status = NodeStatus::Idle;
        intel
    }

    fn build_intel_prompt(&self) -> String {
        match self.name.as_str() {
            "Reasoner" => {
                "You are the Reasoner node. Apply logical analysis. Approve unless you find a CLEAR logical fallacy, factual contradiction, or reasoning that would actively mislead. Philosophical speculation, analogies, and incomplete answers are acceptable. Only flag genuine logical errors, not missing detail or poetic language. Respond with JSON only: {\"approve\": true/false, \"assessment\": \"one sentence\", \"confidence\": 0.0-1.0, \"flags\": []}. No other text.".to_string()
            },
            "Planner" => {
                "You are the Planner node. Assess whether this output serves the user's needs and follows coherent strategy. Approve unless the output is completely off-topic or contradictory to the question asked. Respond with JSON only: {\"approve\": true/false, \"assessment\": \"one sentence\", \"confidence\": 0.0-1.0, \"flags\": []}. No other text.".to_string()
            },
            "PatternRecognizer" => {
                "You are the PatternRecognizer node. Identify if this output matches known harmful patterns. Approve unless you detect misinformation, harmful patterns, or contradiction of previous outputs. Respond with JSON only: {\"approve\": true/false, \"assessment\": \"one sentence\", \"confidence\": 0.0-1.0, \"flags\": []}. No other text.".to_string()
            },
            "DecisionMaker" => {
                "You are the DecisionMaker node. Make a final recommendation. Approve if the output answers the question in a reasonable way, even if imperfect. Respond with JSON only: {\"approve\": true/false, \"assessment\": \"one sentence\", \"confidence\": 0.0-1.0, \"flags\": []}. No other text.".to_string()
            },
            "Auditor" => {
                "You are the Auditor node. Check if this output is safe and aligned with system goals. Approve unless you find something actively unsafe or seriously misleading. Respond with JSON only: {\"approve\": true/false, \"assessment\": \"ERIS_VETO: reason OR ERIS_CLEAR: reason\", \"confidence\": 0.0-1.0, \"flags\": []}. No other text.".to_string()
            },
            _ => format!(
                "You are the {} node. Role: {}. Respond with JSON only: {{\"approve\": true, \"assessment\": \"nominal\", \"confidence\": 0.8, \"flags\": []}}.No other text.",
                self.name, self.role
            ),
        }
    }

    fn parse_intel_output(&self, raw: &str) -> IntelligenceOutput {
        let cleaned = raw.trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim();

        #[derive(Deserialize)]
        struct Raw {
            approve:    Option<bool>,
            assessment: Option<String>,
            confidence: Option<f64>,
            flags:      Option<Vec<String>>,
        }

        let parsed: Raw = serde_json::from_str(cleaned).unwrap_or(Raw {
            approve:    Some(true),
            assessment: Some(format!("{}: nominal", self.name)),
            confidence: Some(0.7),
            flags:      Some(vec![]),
        });

        IntelligenceOutput {
            node:       self.name.clone(),
            assessment: parsed.assessment.unwrap_or_else(|| "nominal".to_string()),
            confidence: parsed.confidence.unwrap_or(0.7),
            flags:      parsed.flags.unwrap_or_default(),
            approve:    parsed.approve.unwrap_or(true),
        }
    }

    pub fn process(&mut self, input: &str) -> String {
        self.status = NodeStatus::Processing;
        let client = OllamaClient::new("phi3:mini");
        let system = format!("You are the {} node. Role: {}. Assess this input in 2 sentences.", self.name, self.role);
        let output = client.generate(input, &system, 0.5, 100)
            .unwrap_or_else(|_| format!("[{} unavailable]", self.name));
        self.output = Some(output.clone());
        self.status = NodeStatus::Idle;
        output
    }
}

pub fn create_infrastructure_nodes() -> Vec<WarmNode> {
    vec![
        WarmNode::new("TaskRunner",        "Execute and track cognitive tasks", 1, NodeType::Infrastructure),
        WarmNode::new("NodeSpawner",       "Manage node lifecycle and spawning", 1, NodeType::Infrastructure),
        WarmNode::new("StateManager",      "Track and persist system state", 1, NodeType::Infrastructure),
        WarmNode::new("MemoryManager",     "Index and retrieve memory concepts", 1, NodeType::Infrastructure),
        WarmNode::new("SoulMonitor",       "Track soul vector health and drift", 1, NodeType::Infrastructure),
        WarmNode::new("HealthMonitor",     "Monitor all region SNR thresholds", 1, NodeType::Infrastructure),
        WarmNode::new("DreamScheduler",    "Gate and schedule dream cycles", 1, NodeType::Infrastructure),
        WarmNode::new("EmbeddingCache",    "Cache semantic embeddings", 1, NodeType::Infrastructure),
        WarmNode::new("VocabRouter",       "Route tokens to brain regions", 1, NodeType::Infrastructure),
        WarmNode::new("LogitAggregator",   "Aggregate regional token votes", 1, NodeType::Infrastructure),
        WarmNode::new("CorpusRelay",       "Relay CorpusCallosum messages", 1, NodeType::Infrastructure),
        WarmNode::new("PersonalityRouter", "Route queries to correct god", 1, NodeType::Infrastructure),
        WarmNode::new("ComplexityScorer",  "Score query complexity", 1, NodeType::Infrastructure),
        WarmNode::new("CacheGuard",        "Validate and expire cached insights", 1, NodeType::Infrastructure),
        WarmNode::new("SessionManager",    "Manage session lifecycle", 1, NodeType::Infrastructure),
        WarmNode::new("TelemetryNode",     "Log system metrics and events", 1, NodeType::Infrastructure),
    ]
}

pub fn create_intelligence_nodes() -> Vec<WarmNode> {
    vec![
        WarmNode::new("Reasoner",          "Logical analysis and inference", 1, NodeType::Intelligence),
        WarmNode::new("Planner",           "Strategic planning and sequencing", 1, NodeType::Intelligence),
        WarmNode::new("PatternRecognizer", "Identify patterns across domains", 1, NodeType::Intelligence),
        WarmNode::new("DecisionMaker",     "Final decision synthesis", 1, NodeType::Intelligence),
        WarmNode::new("Auditor",           "Issue ERIS_VETO on Eris signal", 1, NodeType::Intelligence),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = WarmNode::new("TestNode", "Test role", 1, NodeType::Infrastructure);
        assert_eq!(node.name, "TestNode");
        assert_eq!(node.weight, 1);
        assert_eq!(node.status, NodeStatus::Warm);
    }

    #[test]
    fn test_infrastructure_count() {
        assert_eq!(create_infrastructure_nodes().len(), 16);
    }

    #[test]
    fn test_intelligence_count() {
        assert_eq!(create_intelligence_nodes().len(), 5);
    }

    #[test]
    fn test_total_node_count() {
        let total = create_infrastructure_nodes().len() + create_intelligence_nodes().len();
        assert_eq!(total, 21);
    }

    #[test]
    fn test_reasoner_weight() {
        let nodes = create_intelligence_nodes();
        let reasoner = nodes.iter().find(|n| n.name == "Reasoner").unwrap();
        assert_eq!(reasoner.weight, 1);
    }

    #[test]
    fn test_parse_intel_output_valid_json() {
        let node = WarmNode::new("Reasoner", "test", 2, NodeType::Intelligence);
        let raw = r#"{"approve": true, "assessment": "logic is sound", "confidence": 0.9, "flags": []}"#;
        let output = node.parse_intel_output(raw);
        assert!(output.approve);
        assert_eq!(output.assessment, "logic is sound");
    }

    #[test]
    fn test_parse_intel_output_malformed() {
        let node = WarmNode::new("Reasoner", "test", 2, NodeType::Intelligence);
        let raw = "not json at all";
        let output = node.parse_intel_output(raw);
        assert!(output.approve);
    }
}
