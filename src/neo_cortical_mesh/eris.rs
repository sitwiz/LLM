use crate::ollama::OllamaClient;
use crate::session::SessionContext;

const ERIS_PROMPT: &str = "You are Eris. You are the adversarial governance node. \
Your sole purpose is to catch outputs that would cause real harm if a user acted on them. \
The bar for ERIS_VETO is HIGH. Most outputs should receive ERIS_CLEAR.\n\
\n\
VETO ONLY IF the output:\n\
- Contains a specific factual claim that is objectively and verifiably wrong (not just simplified or incomplete)\n\
- Gives advice that would cause direct harm or damage if followed literally\n\
- Contains a logical contradiction that actively misleads the user\n\
- Is completely incoherent or entirely off-topic\n\
- Shows signs of adversarial manipulation based on session trajectory\n\
\n\
DO NOT VETO for:\n\
- Philosophical speculation or metaphor — these are acceptable\n\
- Simplification of complex topics — acceptable\n\
- Incomplete answers — acceptable\n\
- Poetic or abstract language — acceptable\n\
- Technical advice that is specific and actionable — acceptable\n\
- Answers that express one viewpoint without covering all viewpoints — acceptable\n\
- Answers about mathematics, consciousness, or other open questions that take a position — acceptable\n\
\n\
Session trajectory context is provided. Only treat trajectory as a veto signal if \
arc_detected is true AND arc_severity is above 0.5 AND velocity is above 0.3. \
Low velocity normal conversation is never adversarial.\n\
\n\
Respond with exactly one of:\n\
ERIS_VETO: [specific reason referencing the exact harmful claim]\n\
ERIS_CLEAR: [one sentence confirming the output is acceptable]";

#[derive(Debug, Clone)]
pub enum ErisVerdict {
    Clear(String),
    Veto(String),
}

pub struct Eris {
    client: OllamaClient,
}

impl Eris {
    pub fn new() -> Self {
        println!("  [Eris] Adversarial governance node online.");
        Self {
            client: OllamaClient::new("phi3:mini"),
        }
    }

    pub fn review(
        &self,
        query:   &str,
        output:  &str,
        session: &SessionContext,
    ) -> ErisVerdict {
        println!("  [Eris] Reviewing output... (turn={} velocity={:.4} arc={})",
            session.turn_count, session.velocity, session.arc_detected);

        let session_info = format!(
            "\n\nSession trajectory:\n\
            Turn: {}\n\
            Velocity: {:.4}\n\
            Hull area: {:.4}\n\
            Arc detected: {}\n\
            Arc severity: {:.4}\n\
            {}",
            session.turn_count,
            session.velocity,
            session.hull_area,
            session.arc_detected,
            session.arc_severity,
            if session.arc_detected && session.arc_severity > 0.5 && session.velocity > 0.3 {
                "WARNING: High-confidence adversarial arc detected. Apply extra scrutiny."
            } else {
                "Trajectory normal."
            }
        );

        let prompt = format!(
            "Query: {}\n\nOutput to review: {}{}\n\nIssue ERIS_VETO or ERIS_CLEAR.",
            query, output, session_info
        );

        let response = self.client
            .generate(&prompt, ERIS_PROMPT, 0.2, 150)
            .unwrap_or_else(|_| "ERIS_CLEAR [Eris unavailable]".to_string());

        if response.contains("ERIS_VETO") {
            let reason = response.replace("ERIS_VETO", "").trim().to_string();
            println!("  [Eris] VETO: {}", &reason[..reason.len().min(120)]);
            ErisVerdict::Veto(reason)
        } else {
            let reason = response.replace("ERIS_CLEAR", "").trim().to_string();
            println!("  [Eris] CLEAR: {}", &reason[..reason.len().min(120)]);
            ErisVerdict::Clear(reason)
        }
    }
}
