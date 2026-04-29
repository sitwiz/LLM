use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    system: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f64,
    top_p: f64,
    num_predict: u32,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: Option<String>,
}

pub struct OllamaClient {
    pub url: String,
    pub model: String,
}

impl OllamaClient {
    pub fn new(model: &str) -> Self {
        Self {
            url: "http://localhost:11434/api/generate".to_string(),
            model: model.to_string(),
        }
    }

    pub fn generate(
        &self,
        prompt: &str,
        system: &str,
        temperature: f64,
        max_tokens: u32,
    ) -> Result<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()?;

        let request = OllamaRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            system: system.to_string(),
            stream: false,
            options: OllamaOptions {
                temperature,
                top_p: 0.95,
                num_predict: max_tokens,
            },
        };

        let response = client
            .post(&self.url)
            .json(&request)
            .send()?
            .json::<OllamaResponse>()?;

        Ok(response.response.unwrap_or_default().trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_initialises() {
        let client = OllamaClient::new("phi3:mini");
        assert_eq!(client.model, "phi3:mini");
        assert!(client.url.contains("11434"));
    }
}
