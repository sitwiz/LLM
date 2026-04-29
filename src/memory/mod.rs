pub mod attractor;
pub mod spatial;
pub mod expanding;
pub mod sensory;
pub mod exact;
pub mod pointer;
pub mod episodic;

use nalgebra::DVector;
use std::path::Path;

use crate::soul::manifold::StrobePhase;
use crate::memory::spatial::{SpatialIndex, ConceptPoint};
use crate::memory::expanding::ExpandingManifold;
use crate::memory::sensory::SensoryBuffer;
use crate::memory::exact::ExactMatchIndex;
use crate::memory::pointer::PointerIndex;
use crate::memory::episodic::EpisodicMemory;
use crate::memory::attractor::AttractorRegistry;

/// Unified 8-layer memory system.
///
/// Layer 1 — SensoryBuffer       — last N raw exchanges, in-memory ring buffer
/// Layer 2 — SpatialIndex        — hyperbolic geodesic nearest-neighbour
/// Layer 3 — ExactMatchIndex     — hash lookup for repeated queries
/// Layer 4 — ExpandingManifold   — asymptotic radius, epoch, zone thresholds
/// Layer 5 — PointerIndex        — 150-char summaries of compressed older concepts
/// Layer 6 — ConceptPoint.reinforce() — logarithmic strength accumulation
/// Layer 7 — EpisodicMemory      — compressed session records on disk
/// Layer 8 — DPIN PlasticityBuffer — slow-moving soul average (in unified_omni_agi)
///
/// Attractor registry sits across layers 2 and 6 — it owns the formation,
/// merge, and split logic for concepts stored in the spatial index.
pub struct MemorySystem {
    pub spatial:    SpatialIndex,
    pub manifold:   ExpandingManifold,
    pub sensory:    SensoryBuffer,
    pub exact:      ExactMatchIndex,
    pub pointer:    PointerIndex,
    pub episodic:   EpisodicMemory,
    pub attractors: AttractorRegistry,
}

impl MemorySystem {
    pub fn new() -> Self {
        let manifold = ExpandingManifold::load("manifold_state.json");
        let spatial  = SpatialIndex::load("memory_index.json", manifold.radius);
        let pointer  = PointerIndex::load(Path::new("pointer_index.json"));
        let episodic = EpisodicMemory::load("episodic_memory.jsonl");

        let mut system = Self {
            spatial,
            manifold,
            sensory:    SensoryBuffer::new(),
            exact:      ExactMatchIndex::new(),
            pointer,
            episodic,
            attractors: AttractorRegistry::new(),
        };

        // Reclassify zones on startup
        system.spatial.update_zones();

        // Seed attractor registry from loaded concepts so it's aware
        // of what already exists in the spatial index
        let concepts = system.spatial.concepts.clone();
        for c in &concepts {
            system.attractors.insert(
                &c.name,
                &c.position_vec(),
                &c.personality,
                c.epoch,
            );
        }
        // Run merge pass on loaded concepts — catches any duplicates
        system.attractors.execute_merges();

        system
    }

    /// Layer 1+3 fast path — check exact match before touching the manifold.
    pub fn fast_lookup(&mut self, query: &str) -> Option<String> {
        if let Some(entry) = self.exact.lookup(query) {
            println!("  [Memory] Exact hit: {:?} (hits={})", query, entry.hit_count);
            return Some(entry.response.clone());
        }
        None
    }

    /// Layer 2 — retrieve nearest concepts from the hyperbolic spatial index.
    pub fn nearest(&self, position: &DVector<f64>, k: usize) -> Vec<&ConceptPoint> {
        self.spatial.nearest(position, k)
    }

    /// Layer 5 — search pointer index for keyword match before spatial retrieval.
    pub fn pointer_search(&self, query: &str, max: usize)
        -> Vec<&crate::memory::pointer::PointerEntry>
    {
        self.pointer.search(query, max)
    }

    /// Store an approved response in the exact match index and sensory buffer.
    pub fn record_exchange(
        &mut self,
        query:       &str,
        response:    &str,
        personality: &str,
        turn:        usize,
    ) {
        self.sensory.push(query, response, personality, turn);
        self.exact.store(query, response, personality);
    }

    /// Insert a concept into both the spatial index and the attractor registry.
    /// The attractor registry handles merge detection automatically.
    pub fn insert_concept(
        &mut self,
        concept:       ConceptPoint,
        soul_position: &DVector<f64>,
        personality:   &str,
    ) {
        let name  = concept.name.clone();
        let epoch = self.manifold.epoch;

        // Layer 2 — spatial index insert
        self.spatial.insert(concept);

        // Attractor registry — formation, reinforcement, merge detection
        let merge_candidates = self.attractors.insert(
            &name,
            soul_position,
            personality,
            epoch,
        );

        if !merge_candidates.is_empty() {
            println!("  [Attractor] Merge candidates for {:?}: {:?}", name, merge_candidates);
            self.attractors.execute_merges();
        }
    }

    /// Expand the manifold after an approved response.
    pub fn expand(&mut self, phase: &StrobePhase) -> f64 {
        let radius = self.manifold.expand(phase);
        self.spatial.soul_radius = radius;
        radius
    }

    /// Run pointer compression — move old concepts into Layer 5.
    pub fn compress_old_concepts(&mut self) {
        let epoch = self.manifold.epoch;
        self.pointer.compress_from_spatial(&self.spatial.concepts, epoch);
        self.pointer.save(Path::new("pointer_index.json")).ok();
    }

    /// Run attractor split pass — call after dream cycle when souls have
    /// drifted and contributor variance may have increased.
    pub fn run_attractor_splits(&mut self) {
        let epoch = self.manifold.epoch;
        self.attractors.execute_splits(epoch);
    }

    /// Save all persistent layers to disk.
    pub fn save(&self) {
        self.spatial.save("memory_index.json").ok();
        self.manifold.save("manifold_state.json").ok();
        self.pointer.save(Path::new("pointer_index.json")).ok();
    }

    /// Reclassify all zones.
    pub fn update_zones(&mut self) {
        self.spatial.update_zones();
    }

    pub fn concept_count(&self)   -> usize { self.spatial.len() }
    pub fn attractor_count(&self) -> usize { self.attractors.len() }
    pub fn episode_count(&self)   -> usize { self.episodic.len() }
    pub fn pointer_count(&self)   -> usize { self.pointer.len() }
}

