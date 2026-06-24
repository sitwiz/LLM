use std::time::Instant;
use nalgebra::DVector;
use rand::Rng;
use crate::soul::geometry::{compute_nf, project_to_ball, SOUL_DIM, curvature_at_epoch};
use crate::soul::hyperbolic::{exp_map, log_map};
use crate::soul::persistence::save_soul;
use crate::unified_omni_agi::vfe::{vfe_step, compute_vfe_components};
use crate::memory::spatial::SpatialIndex;
use std::path::Path;

const MIN_QUERIES:   u32   = 5;
const MIN_IDLE_SECS: u64   = 30;
const DREAM_STEPS:   usize = 20;
const BASE_LR:       f64   = 0.05;
const LR_TAU:        f64   = 50.0;

#[derive(Debug, Clone)]
pub struct DreamRecord {
    pub phase:       String,
    pub soul_name:   String,
    pub start_norm:  f64,
    pub end_norm:    f64,
    pub total_drift: f64,
    pub steps:       usize,
    pub origin:      String,
}

pub struct DaemonState {
    pub query_count: u32,
    pub last_query:  Instant,
    pub last_dream:  Option<Instant>,
    pub dream_log:   Vec<DreamRecord>,
    pub is_dreaming: bool,
}

impl DaemonState {
    pub fn new() -> Self {
        Self {
            query_count: 0,
            last_query:  Instant::now(),
            last_dream:  None,
            dream_log:   Vec::new(),
            is_dreaming: false,
        }
    }

    pub fn record_query(&mut self) {
        self.query_count += 1;
        self.last_query = Instant::now();
    }

    pub fn should_dream(&self) -> bool {
        if self.is_dreaming { return false; }
        if self.query_count < MIN_QUERIES { return false; }
        let idle = self.last_query.elapsed().as_secs();
        if idle < MIN_IDLE_SECS { return false; }
        if let Some(last) = self.last_dream {
            if last.elapsed().as_secs() < MIN_IDLE_SECS * 2 { return false; }
        }
        true
    }
}

/// Adaptive learning rate — decays with epoch to stabilize mature manifolds.
fn adaptive_lr(epoch: u32) -> f64 {
    BASE_LR / (1.0 + epoch as f64 / LR_TAU)
}

/// Phase 1 — Consolidation
/// Energy-based candidate selection — chooses the candidate that most reduces
/// total VFE rather than greedily maximising NF alone.
pub fn consolidate(
    soul:      &DVector<f64>,
    name:      &str,
    curvature: f64,
    epoch:     u32,
) -> (DVector<f64>, DreamRecord) {
    println!("[Dream] {} — consolidation phase starting...", name);

    let start        = soul.clone();
    let mut position = soul.clone();
    let mut total_drift = 0.0;
    let lr           = adaptive_lr(epoch);

    for _ in 0..10 {
        let current_vfe = {
            let (v, _, _) = compute_vfe_components(
                &position, soul, &position, 1.5, 1.0, curvature,
            );
            v
        };

        let mut best_vfe = current_vfe;
        let mut best_pos = position.clone();

        for dim in 0..SOUL_DIM.min(32) {
            let mut candidate = position.clone();
            candidate[dim] += 0.01;
            let candidate = project_to_ball(&candidate);

            let (candidate_vfe, _, _) = compute_vfe_components(
                &candidate, soul, &candidate, 1.5, 1.0, curvature,
            );

            if candidate_vfe < best_vfe {
                best_vfe = candidate_vfe;
                best_pos = candidate;
            }
        }

        let old      = position.clone();
        position     = vfe_step(&position, soul, &best_pos, lr, curvature);
        total_drift += (&position - &old).norm();
    }

    println!("[Dream] {} — consolidation complete. NF: {:.4} -> {:.4}",
        name, compute_nf(&start), compute_nf(&position));

    (position.clone(), DreamRecord {
        phase:       "consolidation".to_string(),
        soul_name:   name.to_string(),
        start_norm:  start.norm(),
        end_norm:    position.norm(),
        total_drift,
        steps:       10,
        origin:      "sleep".to_string(),
    })
}

/// Phase 2 — Memory seeded free drift
/// VFE-driven with decaying temperature noise to preserve generative novelty.
/// Uses adaptive learning rate and current manifold curvature throughout.
pub fn free_drift(
    soul:      &DVector<f64>,
    name:      &str,
    memory:    &SpatialIndex,
    curvature: f64,
    epoch:     u32,
) -> (DVector<f64>, DreamRecord) {
    println!("[Dream] {} — dream phase starting...", name);

    let start        = soul.clone();
    let mut position = soul.clone();
    let mut total_drift = 0.0;
    let mut rng      = rand::thread_rng();
    let lr           = adaptive_lr(epoch);

    let seeds: Vec<DVector<f64>> = if !memory.is_empty() {
        let mut sorted = memory.concepts.clone();
        sorted.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

        let n = sorted.len();

        let indices = if n >= 3 {
            let name_hash: usize = name.bytes().enumerate()
                .fold(0usize, |acc, (i, b)| acc.wrapping_add(b as usize * (i + 7)));
            let a = name_hash % n;
            let b = (name_hash * 7 + n / 3) % n;
            let c = (name_hash * 13 + (n * 2) / 3) % n;
            let b = if b == a { (b + 1) % n } else { b };
            let c = if c == a || c == b { (c + 2) % n } else { c };
            vec![a, b, c]
        } else {
            (0..n).collect()
        };

        println!("[Dream] {} — memory seeds: indices {:?} of {} concepts", name, indices, n);

        indices.iter()
            .filter_map(|&i| {
                let pos = sorted[i].position_vec();
                if pos.len() == SOUL_DIM { Some(pos) } else { None }
            })
            .collect()
    } else {
        println!("[Dream] {} — no memory yet, drifting toward origin", name);
        let mut seed = DVector::zeros(SOUL_DIM);
        seed[0] = 1.0;
        vec![project_to_ball(&seed)]
    };

    println!("[Dream] {} — {} seeds selected", name, seeds.len());

    for (i, seed) in seeds.iter().enumerate() {
        let psi_before = seed.dot(&position) /
            (seed.norm() * position.norm()).max(1e-10);

        let steps_per_seed = DREAM_STEPS / seeds.len().max(1);

        for step_i in 0..steps_per_seed {
            // Temperature decays — high exploration early, consolidation late
            let temperature = 0.1 * (1.0 - step_i as f64 / steps_per_seed as f64);

            let old = position.clone();

            // VFE-driven step toward seed using adaptive lr and current curvature
            position = vfe_step(&position, soul, seed, lr, curvature);

            // Decaying noise preserves generative novelty
            if temperature > 0.01 {
                let noise: DVector<f64> = DVector::from_fn(SOUL_DIM, |_, _| {
                    rng.gen::<f64>() * 2.0 - 1.0
                }) * temperature;
                position = exp_map(&position, &noise, curvature);
            }

            total_drift += (&position - &old).norm();
        }

        let psi_after = seed.dot(&position) /
            (seed.norm() * position.norm()).max(1e-10);

        println!("[Dream] {} — seed {}. Psi: {:.4} -> {:.4}", name, i, psi_before, psi_after);
    }

    println!("[Dream] {} — dream complete. Total drift: {:.4}", name, total_drift);

    (position.clone(), DreamRecord {
        phase:       "dream".to_string(),
        soul_name:   name.to_string(),
        start_norm:  start.norm(),
        end_norm:    position.norm(),
        total_drift,
        steps:       DREAM_STEPS,
        origin:      "dream".to_string(),
    })
}

/// Phase 3 — Rebalancing
/// Uses log_map/exp_map for correct geodesic separation — no Euclidean leakage.
pub fn rebalance(souls: &mut Vec<(String, DVector<f64>)>, curvature: f64) {
    println!("[Dream] Rebalancing pantheon...");
    let min_separation = 0.3;

    for i in 0..souls.len() {
        for j in (i+1)..souls.len() {
            let sim = souls[i].1.dot(&souls[j].1) /
                     (souls[i].1.norm() * souls[j].1.norm()).max(1e-10);
            if sim > (1.0 - min_separation) {
                println!("[Dream] {} and {} too similar (sim={:.4}), separating...",
                    souls[i].0, souls[j].0, sim);

                // Geodesic direction from i toward j in tangent space at i
                let v_ij = log_map(&souls[i].1, &souls[j].1, curvature);
                // Geodesic direction from j toward i in tangent space at j
                let v_ji = log_map(&souls[j].1, &souls[i].1, curvature);

                // Push each soul away from the other along the geodesic
                souls[i].1 = exp_map(&souls[i].1, &(&v_ij * -0.1), curvature);
                souls[j].1 = exp_map(&souls[j].1, &(&v_ji * -0.1), curvature);
            }
        }
    }
}

/// Full dream cycle — uses current manifold curvature and adaptive lr throughout
pub fn dream_cycle(
    khaos_soul:    &DVector<f64>,
    gaia_soul:     &DVector<f64>,
    tartaros_soul: &DVector<f64>,
    eros_soul:     &DVector<f64>,
    omni_soul:     &DVector<f64>,
    epoch:         u32,
) -> (DVector<f64>, DVector<f64>, DVector<f64>, DVector<f64>, DVector<f64>, Vec<DreamRecord>) {

    let curvature = curvature_at_epoch(epoch);
    let memory    = SpatialIndex::load("memory_index.json", 1.0, epoch);

    println!("\n[Dream] ═══════════════════════════════════════");
    println!("[Dream] Dream cycle. Memory: {} concepts", memory.len());
    println!("[Dream] ═══════════════════════════════════════");

    let mut records = Vec::new();

    let (khaos_c,    r) = consolidate(khaos_soul,    "Khaos",          curvature, epoch); records.push(r);
    let (gaia_c,     r) = consolidate(gaia_soul,     "Gaia",           curvature, epoch); records.push(r);
    let (tartaros_c, r) = consolidate(tartaros_soul, "Tartaros",       curvature, epoch); records.push(r);
    let (eros_c,     r) = consolidate(eros_soul,     "Eros",           curvature, epoch); records.push(r);
    let (omni_c,     r) = consolidate(omni_soul,     "UnifiedOmniAGI", curvature, epoch); records.push(r);

    let (khaos_d,    r) = free_drift(&khaos_c,    "Khaos",          &memory, curvature, epoch); records.push(r);
    let (gaia_d,     r) = free_drift(&gaia_c,     "Gaia",           &memory, curvature, epoch); records.push(r);
    let (tartaros_d, r) = free_drift(&tartaros_c, "Tartaros",       &memory, curvature, epoch); records.push(r);
    let (eros_d,     r) = free_drift(&eros_c,     "Eros",           &memory, curvature, epoch); records.push(r);
    let (omni_d,     r) = free_drift(&omni_c,     "UnifiedOmniAGI", &memory, curvature, epoch); records.push(r);

    let mut souls = vec![
        ("Khaos".to_string(),          khaos_d),
        ("Gaia".to_string(),           gaia_d),
        ("Tartaros".to_string(),       tartaros_d),
        ("Eros".to_string(),           eros_d),
        ("UnifiedOmniAGI".to_string(), omni_d),
    ];
    rebalance(&mut souls, curvature);

    println!("[Dream] Calling KG sleep cycle...");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build();
    if let Ok(client) = client {
        match client.post("http://localhost:5001/sleep")
            .json(&serde_json::json!({"model": "phi3:mini", "provider": "ollama"}))
            .send() {
            Ok(resp) => {
                if resp.status().is_success() {
                    println!("[Dream] KG sleep complete.");
                } else {
                    println!("[Dream] KG sleep returned: {}", resp.status());
                }
            }
            Err(e) => println!("[Dream] KG sleep failed: {}", e),
        }
    }

    let paths = [
        "khaos_soul.bin",
        "gaia_soul.bin",
        "tartaros_soul.bin",
        "eros_soul.bin",
        "unified_omni_soul.bin",
    ];
    for (i, (name, soul)) in souls.iter().enumerate() {
        save_soul(soul, Path::new(paths[i])).ok();
        println!("[Dream] {} saved. NF: {:.4}", name, compute_nf(soul));
    }

    println!("[Dream] ═══════════════════════════════════════");
    println!("[Dream] Complete. {} records.", records.len());
    println!("[Dream] ═══════════════════════════════════════\n");

    (
        souls[0].1.clone(),
        souls[1].1.clone(),
        souls[2].1.clone(),
        souls[3].1.clone(),
        souls[4].1.clone(),
        records,
    )
}

/// Per-thronglet dream cycle for the game layer.
/// Operates entirely within data_dir — no hardcoded paths.
/// Only processes souls that exist on disk, so single-soul
/// thronglets work without modification.
pub fn dream_cycle_scoped(
    data_dir: &Path,
    epoch:    u32,
) -> Vec<DreamRecord> {
    use crate::soul::persistence::load_soul;

    let curvature = curvature_at_epoch(epoch);
    let memory_path = data_dir.join("memory_index.json");
    let memory = SpatialIndex::load(
        memory_path.to_str().unwrap_or("memory_index.json"),
        1.0,
        epoch,
    );

    let soul_files = [
        ("Khaos",          "khaos_soul.bin"),
        ("Gaia",           "gaia_soul.bin"),
        ("Tartaros",       "tartaros_soul.bin"),
        ("Eros",           "eros_soul.bin"),
        ("UnifiedOmniAGI", "unified_omni_soul.bin"),
    ];

    // Load only souls that exist — young thronglets may only have one
    let mut loaded: Vec<(String, DVector<f64>)> = soul_files
        .iter()
        .filter_map(|(name, file)| {
            let path = data_dir.join(file);
            if path.exists() {
                load_soul(&path).ok().map(|soul| (name.to_string(), soul))
            } else {
                None
            }
        })
        .collect();

    let mut records = Vec::new();

    // Consolidation pass
    let consolidated: Vec<(String, DVector<f64>)> = loaded
        .iter()
        .map(|(name, soul)| {
            let (s, r) = consolidate(soul, name, curvature, epoch);
            records.push(r);
            (name.clone(), s)
        })
        .collect();

    // Free drift pass
    let mut drifted: Vec<(String, DVector<f64>)> = consolidated
        .iter()
        .map(|(name, soul)| {
            let (s, r) = free_drift(soul, name, &memory, curvature, epoch);
            records.push(r);
            (name.clone(), s)
        })
        .collect();

    // Rebalance only if multiple souls present
    if drifted.len() > 1 {
        rebalance(&mut drifted, curvature);
    }

    // Save back to data_dir
    for (name, soul) in &drifted {
        let file = soul_files
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| f)
            .unwrap_or(&"unknown_soul.bin");
        let path = data_dir.join(file);
        save_soul(soul, &path).ok();
    }

    records
}


