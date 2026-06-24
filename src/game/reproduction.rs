use crate::game::thronglet::{Lineage, SoulArchetype, Thronglet, WorldPosition};
use crate::soul::hyperbolic::{exp_map, log_map};
use nalgebra::DVector;
use rand::Rng;
use std::path::PathBuf;
use uuid::Uuid;

const BIRTH_NOISE_SIGMA: f64 = 0.02;   // small perturbation from midpoint
const ATTRACTOR_INHERIT_N: usize = 3;  // top N attractors from each parent

pub struct ReproductionResult {
    pub child: Thronglet,
    // Soul vector for the child — caller writes this to child's data_dir
    pub initial_soul_vector: DVector<f64>,
    // Concept seeds inherited from parents — caller inserts into child's spatial index
    pub inherited_concept_seeds: Vec<InheritedConcept>,
}

pub struct InheritedConcept {
    pub query: String,
    pub position: DVector<f64>,   // inherits parent position, sigma widens at birth
    pub sigma: f64,
    pub strength: f64,            // starts weaker than parents' — not fully consolidated
}

pub fn reproduce(
    parent_a: &Thronglet,
    parent_b: &Thronglet,
    soul_a: &DVector<f64>,
    soul_b: &DVector<f64>,
    attractors_a: Vec<(String, DVector<f64>, f64)>, // (query, position, strength)
    attractors_b: Vec<(String, DVector<f64>, f64)>,
    curvature: f64,
    data_root: &PathBuf,
    position: WorldPosition,
    name: String,
    now: u64,
) -> Result<ReproductionResult, ReproductionError> {
    if !parent_a.can_reproduce {
        return Err(ReproductionError::NotMature(parent_a.id));
    }
    if !parent_b.can_reproduce {
        return Err(ReproductionError::NotMature(parent_b.id));
    }
    if parent_a.id == parent_b.id {
        return Err(ReproductionError::SameParent);
    }

    // Soul archetype: 60% dominant parent (higher maturity), 40% other
    let dominant_soul = {
        let mut rng = rand::thread_rng();
        let use_a = rng.gen_bool(if parent_a.maturity >= parent_b.maturity { 0.6 } else { 0.4 });
        if use_a { parent_a.dominant_soul.clone() } else { parent_b.dominant_soul.clone() }
    };

    // Soul vector: geodesic midpoint + noise tangent
    // The child starts between its parents on the manifold, not at the origin
    // This means it inherits the "neighbourhood" of both parents' knowledge
    let midpoint = geodesic_midpoint(soul_a, soul_b, curvature);
    let noise = random_tangent(midpoint.len(), BIRTH_NOISE_SIGMA);
    let initial_soul_vector = exp_map(&midpoint, &noise, curvature);

    // Inherited concepts: top N attractors from each parent
    // Sigma is widened — the child has the concept but it's uncertain
    // It will consolidate (sigma tightening) through its own experience
    let top_a = top_attractors(attractors_a, ATTRACTOR_INHERIT_N);
    let top_b = top_attractors(attractors_b, ATTRACTOR_INHERIT_N);

    let inherited_concept_seeds: Vec<InheritedConcept> = top_a.into_iter().chain(top_b).map(|(query, pos, strength)| {
        InheritedConcept {
            query,
            position: pos,
            sigma: 0.35,              // wider than parent's consolidated sigma
            strength: strength * 0.5, // starts at half parent strength
        }
    }).collect();

    let lineage = Lineage::from_parents(
        parent_a,
        parent_b,
        attractors_a_queries(&inherited_concept_seeds, ATTRACTOR_INHERIT_N),
        attractors_b_queries(&inherited_concept_seeds, ATTRACTOR_INHERIT_N),
    );

    let child_data_dir = data_root.join("thronglets").join(Uuid::new_v4().to_string());

    let mut child = Thronglet::new(
        name,
        dominant_soul,
        child_data_dir,
        position,
        now,
    );
    child.lineage = lineage;

    // Child starts not at Dark phase — it inherits a head start
    // from its parents' knowledge seeds, so it enters Aware immediately
    // This is the mechanical expression of inheritance
    child.stats.concept_count = inherited_concept_seeds.len();
    child.stats.soul_norm = initial_soul_vector.norm();
    child.update_phase();

    Ok(ReproductionResult {
        child,
        initial_soul_vector,
        inherited_concept_seeds,
    })
}

fn geodesic_midpoint(a: &DVector<f64>, b: &DVector<f64>, curvature: f64) -> DVector<f64> {
    // log_map gives the tangent direction from a toward b
    // Half that tangent + exp_map gets the midpoint
    let tangent = log_map(a, b, curvature);
    let half_tangent = &tangent * 0.5;
    exp_map(a, &half_tangent, curvature)
}

fn random_tangent(dim: usize, sigma: f64) -> DVector<f64> {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (0..dim).map(|_| rng.gen::<f64>() * sigma * 2.0 - sigma).collect();
    DVector::from_vec(v)
}

fn top_attractors(
    mut attractors: Vec<(String, DVector<f64>, f64)>,
    n: usize,
) -> Vec<(String, DVector<f64>, f64)> {
    attractors.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    attractors.truncate(n);
    attractors
}

fn attractors_a_queries(seeds: &[InheritedConcept], n: usize) -> Vec<String> {
    seeds.iter().take(n).map(|s| s.query.clone()).collect()
}

fn attractors_b_queries(seeds: &[InheritedConcept], n: usize) -> Vec<String> {
    seeds.iter().skip(n).map(|s| s.query.clone()).collect()
}

#[derive(Debug)]
pub enum ReproductionError {
    NotMature(uuid::Uuid),
    SameParent,
    IncompatibleZones,
}

impl std::fmt::Display for ReproductionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::NotMature(id) => write!(f, "thronglet {id} has not reached maturity"),
            Self::SameParent => write!(f, "cannot reproduce with self"),
            Self::IncompatibleZones => write!(f, "both parents must be in Working or Frontier zone"),
        }
    }
}
