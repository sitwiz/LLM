use crate::ollama::OllamaClient;
use crate::cortical_mesh::corpus_callosum::CorpusCallosum;

#[derive(Debug, Clone)]
pub struct NodeOutput {
    pub role:     String,
    pub round:    usize,
    pub analysis: String,
    pub concerns: Vec<String>,
}

pub struct MeshNode {
    pub role:        String,
    pub id:          String,
    pub god_context: String,
    pub soul_psi:    f64,
    pub soul_burden: f64,
}

impl MeshNode {
    pub fn new(
        role: &str,
        god_context: &str,
        soul_psi: f64,
        soul_burden: f64,
    ) -> Self {
        let id = format!("{}-{:x}",
            role.to_lowercase().replace(' ', "-"),
            rand::random::<u32>()
        );
        Self {
            role:        role.to_string(),
            id,
            god_context: god_context.to_string(),
            soul_psi,
            soul_burden,
        }
    }

    pub fn system_prompt_str(&self) -> String {
        if self.role == "Devil's Advocate" {
            return "You are the Devil's Advocate node. Your sole purpose is to challenge \
            the emerging consensus. Read the other specialists analyses and generate \
            the strongest possible objection to their conclusions. \
            Be specific. Name the exact assumption that is most likely wrong. \
            Keep response under 150 words.".to_string();
        }

        format!(
            "You are a specialist node in a cognitive mesh: {}. \
            You have been given a geometric context from the pantheon: {}. \
            Your soul state: Psi={:.3} Burden={:.3}. \
            High Psi means you are close to the answer. High Burden means far. \
            Analyse the query from your specialist perspective. \
            Be specific, technical, and direct. \
            Identify: 1) Your core analysis, 2) Key concerns or risks, \
            3) What other specialists should consider. \
            Keep response under 200 words.",
            self.role, self.god_context, self.soul_psi, self.soul_burden
        )
    }

    /// Sequential think — used for testing
    pub fn think(
        &self,
        query: &str,
        cc: &CorpusCallosum,
        all_nodes: &[String],
        outputs: &mut Vec<NodeOutput>,
    ) -> anyhow::Result<()> {
        let client = OllamaClient::new("phi3:mini");
        let system = self.system_prompt_str();
        let prompt = format!("Query: {}\n\nGod context: {}", query, self.god_context);

        let analysis = client.generate(&prompt, &system, 0.7, 250)?;
        cc.broadcast(&self.id, &analysis, 1, all_nodes);

        outputs.push(NodeOutput {
            role:     self.role.clone(),
            round:    1,
            analysis: analysis.clone(),
            concerns: vec![],
        });

        let messages = cc.flush(&self.id);
        if !messages.is_empty() {
            let others: String = messages.iter()
                .map(|m| format!("[{}]: {}", m.from, m.content))
                .collect::<Vec<_>>()
                .join("\n\n");

            let round2_prompt = format!(
                "Query: {}\n\nYour round 1 analysis: {}\n\n\
                Other specialists said:\n{}\n\n\
                Refine your analysis.",
                query, analysis, others
            );

            let refined = client.generate(&round2_prompt, &system, 0.6, 250)?;
            outputs.push(NodeOutput {
                role:     self.role.clone(),
                round:    2,
                analysis: refined,
                concerns: vec![],
            });
        }

        Ok(())
    }
}
