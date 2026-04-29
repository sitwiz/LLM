use crate::brain::logit_server::LogitClient;
use crate::brain::region::BrainRegion;
use crate::brain::gru::VOCAB_SIZE;
use anyhow::Result;
use std::collections::HashMap;

fn domain_prompts(region_name: &str) -> Vec<&'static str> {
    match region_name {
        "frontal_lobe" => vec![
            "To solve this problem logically, I should first",
            "The most rational decision here would be to",
            "Analysing the situation carefully, the best strategy is",
            "Step by step reasoning leads me to conclude",
            "The goal is clear and the plan to achieve it is",
        ],
        "temporal_lobe" => vec![
            "The meaning of this word in context is",
            "This story begins with a narrative about",
            "The definition of this concept can be stated as",
            "In language and communication, the key idea is",
            "Recalling from memory, the sequence of events was",
        ],
        "parietal_lobe" => vec![
            "Calculating the numerical result gives us",
            "The spatial relationship between these objects is",
            "Measuring the quantity precisely reveals",
            "The mathematical formula for this is",
            "Arranging these elements in order yields",
        ],
        "occipital_lobe" => vec![
            "The pattern I recognise in this data is",
            "Visually analysing the structure reveals",
            "The repeating motif in this sequence is",
            "Detecting anomalies in the pattern shows",
            "The structural similarity between these is",
        ],
        "insular_lobe" => vec![
            "The emotional dimension of this situation is",
            "My sense of empathy tells me that",
            "This feeling of awareness suggests",
            "The intuitive response to this is",
            "The inner experience of this moment is",
        ],
        "limbic" => vec![
            "The emotional response to this situation is",
            "Fear and anxiety arise when",
            "The rewarding feeling comes from",
            "Memory of this emotional experience shows",
            "The motivational drive behind this is",
        ],
        "thalamus" => vec![
            "The most important signal to attend to is",
            "Filtering out the noise, the key message is",
            "Prioritising what matters most here means",
            "The urgent information that requires attention is",
            "Routing this signal to the right destination involves",
        ],
        "hypothalamus" => vec![
            "The energy requirements for this task are",
            "Maintaining balance and homeostasis requires",
            "Resource conservation in this context means",
            "The survival imperative here is to",
            "Regulating the system to maintain equilibrium involves",
        ],
        "cerebellum" => vec![
            "The precise sequence of steps required is",
            "Error correction in this procedure requires",
            "Fine-tuning the accuracy of this process means",
            "The coordinated timing of these actions is",
            "Skill refinement through practice leads to",
        ],
        "midbrain" => vec![
            "Alert to this sudden change, the response is",
            "The novelty of this stimulus triggers",
            "Quick reaction to this urgent signal means",
            "The dopamine response to this reward is",
            "Switching attention to this new stimulus involves",
        ],
        "pons" => vec![
            "Bridging between these two systems requires",
            "The transition from one state to another involves",
            "Relaying this signal upward means",
            "Connecting the lower and higher functions involves",
            "The interface between these processes is",
        ],
        "medulla_oblongata" => vec![
            "The safety check for this action confirms",
            "Baseline monitoring of vital functions shows",
            "Filtering this input for potential danger reveals",
            "The integrity check of this data shows",
            "Critical system monitoring indicates",
        ],
        "pituitary_gland" => vec![
            "System-wide regulation requires adjusting",
            "The global output modulation here means",
            "Calibrating the entire system involves",
            "Broadcasting this signal to all subsystems",
            "The master regulatory response is to",
        ],
        "meninges" => vec![
            "Protecting the boundary of this context means",
            "Context preservation requires maintaining",
            "Scope validation of this request shows",
            "The protective boundary here ensures",
            "Integrity of the conversation context requires",
        ],
        _ => vec!["The response to this query is"],
    }
}

pub fn initialise_region_from_llama(
    region: &mut BrainRegion,
    client: &LogitClient,
    n_probs: usize,
) -> Result<()> {
    println!("\n[Initialiser] Extracting domain knowledge for {}...", region.name);

    let prompts = domain_prompts(&region.name);
    let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_ref()).collect();

    let domain_probs = client.extract_domain_probs(&prompt_refs, n_probs as i32)?;

    if domain_probs.is_empty() {
        println!("  [Initialiser] No probs received for {}. Using random init.", region.name);
        return Ok(());
    }

    // Aggregate probability scores across all domain prompts
    let mut vocab_scores: HashMap<String, f64> = HashMap::new();
    let mut token_ids: HashMap<String, usize> = HashMap::new();

    for token_probs in &domain_probs {
        for tp in token_probs {
            let entry = vocab_scores.entry(tp.token.clone()).or_insert(0.0);
            *entry += tp.prob();
            token_ids.insert(tp.token.clone(), tp.id);
        }
    }

    // Normalise scores
    let total: f64 = vocab_scores.values().sum::<f64>().max(1e-10);
    for score in vocab_scores.values_mut() {
        *score /= total;
    }

    // Update region word weights with domain knowledge
    for (token, score) in &vocab_scores {
        region.word_weights.insert(token.clone(), 1.0 + score * 10.0);
    }

    // Sort by score for reporting and weight initialisation
    let mut sorted_tokens: Vec<(String, f64, usize)> = vocab_scores.iter()
        .map(|(t, s)| (t.clone(), *s, *token_ids.get(t).unwrap_or(&0)))
        .collect();
    sorted_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  [Initialiser] {} — top domain tokens:", region.name);
    for (token, score, id) in sorted_tokens.iter().take(5) {
        println!("    id={} {:?} score={:.4}", id, token, score);
    }

    // Update GRU output bias for top tokens using real token ids
    // This biases the region toward its specialty domain from query one
    for (_, score, token_id) in sorted_tokens.iter().take(n_probs) {
        let safe_id = token_id % VOCAB_SIZE;
        region.gru.by[safe_id] += score * 0.5;
    }

    println!("  [Initialiser] {} initialised with {} domain tokens.",
        region.name, sorted_tokens.len());

    Ok(())
}

pub fn initialise_all_regions(
    regions: &mut Vec<BrainRegion>,
    client: &LogitClient,
) -> Result<()> {
    println!("\n[Initialiser] ═══════════════════════════════════════");
    println!("[Initialiser] Initialising 14 brain regions from llama3...");
    println!("[Initialiser] ═══════════════════════════════════════\n");

    for region in regions.iter_mut() {
        match initialise_region_from_llama(region, client, 50) {
            Ok(_) => {
                region.save();
                println!("  [Initialiser] {} saved.", region.name);
            }
            Err(e) => {
                println!("  [Initialiser] {} failed: {}. Using random weights.", region.name, e);
            }
        }
    }

    println!("\n[Initialiser] All regions initialised and saved.");
    println!("[Initialiser] ═══════════════════════════════════════\n");

    Ok(())
}

pub fn needs_initialisation(regions: &[BrainRegion]) -> bool {
    let path = format!("nn_weights/{}.bin",
        regions[0].name.to_lowercase().replace(' ', "_"));
    !std::path::Path::new(&path).exists()
}
