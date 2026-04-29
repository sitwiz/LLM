use anyhow::Result;
use serde::{Deserialize, Serialize};

const LLAMA_SERVER_URL: &str = "http://localhost:8080";

#[derive(Serialize)]
struct CompletionRequest {
    prompt:      String,
    n_predict:   i32,
    n_probs:     i32,
    temperature: f64,
    top_k:       i32,
    stream:      bool,
}

#[derive(Deserialize, Debug, Clone)]
pub struct TopLogProb {
    pub id:      usize,
    pub token:   String,
    pub logprob: f64,
}

impl TopLogProb {
    pub fn prob(&self) -> f64 {
        self.logprob.exp()
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct TokenCompletion {
    pub id:           usize,
    pub token:        String,
    pub logprob:      f64,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Deserialize, Debug)]
struct CompletionResponse {
    content:                  String,
    completion_probabilities: Option<Vec<TokenCompletion>>,
}

pub struct LogitClient {
    url:    String,
    client: reqwest::blocking::Client,
}

impl LogitClient {
    pub fn new() -> Self {
        Self {
            url: LLAMA_SERVER_URL.to_string(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .unwrap(),
        }
    }

    /// Generate and return per-token probability distributions
    pub fn complete_with_probs(
        &self,
        prompt: &str,
        max_tokens: i32,
        n_probs: i32,
        temperature: f64,
    ) -> Result<(String, Vec<Vec<TopLogProb>>)> {
        let request = CompletionRequest {
            prompt:      prompt.to_string(),
            n_predict:   max_tokens,
            n_probs,
            temperature,
            top_k:       n_probs,
            stream:      false,
        };

        let response = self.client
            .post(format!("{}/completion", self.url))
            .json(&request)
            .send()?
            .json::<CompletionResponse>()?;

        let probs = response.completion_probabilities
            .unwrap_or_default()
            .into_iter()
            .map(|tc| tc.top_logprobs)
            .collect();

        Ok((response.content, probs))
    }

    /// Get top token probabilities for the next predicted token
    pub fn next_token_probs(
        &self,
        prompt: &str,
        n_probs: i32,
    ) -> Result<Vec<TopLogProb>> {
        let (_, all_probs) = self.complete_with_probs(prompt, 1, n_probs, 1.0)?;
        Ok(all_probs.into_iter().next().unwrap_or_default())
    }

    /// Extract domain knowledge by running domain prompts through llama3
    /// Returns probability distributions that encode linguistic knowledge
    pub fn extract_domain_probs(
        &self,
        domain_prompts: &[&str],
        n_probs: i32,
    ) -> Result<Vec<Vec<TopLogProb>>> {
        let mut all = Vec::new();
        for prompt in domain_prompts {
            println!("  [Logit] Extracting: {}...", &prompt[..prompt.len().min(50)]);
            match self.next_token_probs(prompt, n_probs) {
                Ok(probs) => all.push(probs),
                Err(e)    => println!("  [Logit] Failed: {}", e),
            }
        }
        Ok(all)
    }

    pub fn health_check(&self) -> bool {
        self.client
            .get(format!("{}/health", self.url))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creates() {
        let client = LogitClient::new();
        assert!(client.url.contains("8080"));
    }

    #[test]
    fn test_logprob_to_prob() {
        let lp = TopLogProb {
            id: 0,
            token: "test".to_string(),
            logprob: 0.0,
        };
        assert!((lp.prob() - 1.0).abs() < 1e-10);
    }
}
