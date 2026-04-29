use std::collections::HashMap;
use crate::brain::region::BrainRegion;

/// Aggregate token probability distributions from all 14 regions
/// Each region votes weighted by its SNR health and word weights
pub fn weighted_vote(
    regions: &mut Vec<BrainRegion>,
    token_ids: &[usize],
    vocab: &[String],
) -> Vec<(String, f64)> {
    let mut aggregated: HashMap<String, f64> = HashMap::new();

    // Initialise all vocab tokens with zero
    for token in vocab {
        aggregated.insert(token.clone(), 0.0);
    }

    let total_weight: f64 = regions.iter()
        .map(|r| r.health.snr / 3.154)
        .sum::<f64>()
        .max(1e-10);

    for region in regions.iter_mut() {
        let last_id = token_ids.last().copied().unwrap_or(0);
        let probs = region.process_token(last_id);
        let weight = (region.health.snr / 3.154) / total_weight;

        // Map region probability over vocab tokens using word weights
        for (i, token) in vocab.iter().enumerate() {
            let region_prob = probs.get(i).map(|(_, p)| *p).unwrap_or(0.0);
            let word_bias = region.token_weight(token);
            *aggregated.entry(token.clone()).or_insert(0.0) +=
                region_prob * weight * word_bias;
        }
    }

    let mut result: Vec<(String, f64)> = aggregated.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    result
}
/// Anti-bias correction
/// Prevents any single token from dominating the vote across all regions
/// This is what the blueprint calls "bias voting" in ethics/bias_voting
pub fn bias_correction(
    votes: Vec<(String, f64)>,
    dominance_threshold: f64,
) -> Vec<(String, f64)> {
    let total: f64 = votes.iter().map(|(_, p)| p).sum::<f64>().max(1e-10);

    votes.into_iter().map(|(token, prob)| {
        let normalised = prob / total;
        let corrected = if normalised > dominance_threshold {
            // Reduce dominant tokens — square root dampening
            prob * (dominance_threshold / normalised).sqrt()
        } else {
            prob
        };
        (token, corrected)
    }).collect()
}

/// Compute SNR for each region and return health summary
pub fn compute_snr_health(regions: &[BrainRegion]) -> Vec<(String, f64, &str)> {
    regions.iter().map(|r| {
        (r.name.clone(), r.health.snr, r.health.status())
    }).collect()
}

/// Select the winning token from the voted distribution
/// Uses weighted sampling rather than argmax for diversity
pub fn sample_token(votes: &[(String, f64)], temperature: f64) -> String {
    if votes.is_empty() {
        return "<unk>".to_string();
    }

    // Apply temperature
    let scaled: Vec<f64> = votes.iter()
        .map(|(_, s)| s / temperature.max(0.01))
        .collect();

    let total: f64 = scaled.iter().sum::<f64>().max(1e-10);
    let normalised: Vec<f64> = scaled.iter().map(|s| s / total).collect();

    // Sample from distribution
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;

    for (i, p) in normalised.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return votes[i].0.clone();
        }
    }

    votes[0].0.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bias_correction_reduces_dominance() {
        let votes = vec![
            ("the".to_string(), 0.8),
            ("a".to_string(), 0.1),
            ("an".to_string(), 0.1),
        ];
        let corrected = bias_correction(votes, 0.5);
        let total: f64 = corrected.iter().map(|(_, p)| p).sum();
        // The dominant token should be reduced
        let dominant = corrected.iter().find(|(t, _)| t == "the").unwrap();
        assert!(dominant.1 / total < 0.8);
    }

    #[test]
    fn test_sample_token_returns_valid() {
        let votes = vec![
            ("hello".to_string(), 0.6),
            ("world".to_string(), 0.4),
        ];
        let token = sample_token(&votes, 1.0);
        assert!(token == "hello" || token == "world");
    }
}
