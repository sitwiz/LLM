use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::path::PathBuf;

// Maps directly to your existing VFE phase progression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DevelopmentPhase {
    Dark,           // newborn — no LLM, behavioural only
    Aware,          // fragments forming, minimal language
    Engaged,        // active learning, socialisation meaningful
    Understanding,  // full voice, tool acquisition
    Transcendent,   // deep attractors, strange and unpredictable
}

// Maps to your existing manifold zones
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ManifoldZone {
    Forbidden,  // norm < 0.05 — newborn state
    Core,       // norm < 0.25 — early childhood
    Working,    // norm < 0.60 — active development
    Frontier,   // norm > 0.60 — mature, still expanding
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SoulArchetype {
    Khaos,       // void, entropy, origins — speaks in fragments
    Gaia,        // concrete, practical, systems
    Tartaros,    // deep infrastructure, failure, the abyss
    Eros,        // connection, cross-domain synthesis
    UnifiedOmni, // meta-reasoner, equilibrium
}

impl SoulArchetype {
    // How many additional souls are active at each development phase
    // Infant = 1 (dominant only), adult = all 5
    pub fn active_souls(&self, phase: &DevelopmentPhase) -> Vec<SoulArchetype> {
        match phase {
            DevelopmentPhase::Dark => vec![self.clone()],
            DevelopmentPhase::Aware => vec![self.clone(), SoulArchetype::UnifiedOmni],
            DevelopmentPhase::Engaged => vec![
                self.clone(),
                SoulArchetype::UnifiedOmni,
                self.complementary(),
            ],
            DevelopmentPhase::Understanding | DevelopmentPhase::Transcendent => {
                SoulArchetype::all()
            }
        }
    }

    // Eros connects opposites — Khaos/Gaia, Tartaros/Eros
    pub fn complementary(&self) -> SoulArchetype {
        match self {
            SoulArchetype::Khaos => SoulArchetype::Gaia,
            SoulArchetype::Gaia => SoulArchetype::Khaos,
            SoulArchetype::Tartaros => SoulArchetype::Eros,
            SoulArchetype::Eros => SoulArchetype::Tartaros,
            SoulArchetype::UnifiedOmni => SoulArchetype::Khaos,
        }
    }

    pub fn all() -> Vec<SoulArchetype> {
        vec![
            SoulArchetype::Khaos,
            SoulArchetype::Gaia,
            SoulArchetype::Tartaros,
            SoulArchetype::Eros,
            SoulArchetype::UnifiedOmni,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldPosition {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepSchedule {
    pub last_slept_at: u64,    // unix timestamp
    pub next_sleep_at: u64,    // every 3600 seconds
    pub sleep_duration_secs: u64,  // 60 seconds
    pub is_sleeping: bool,
    pub dream_count: u32,      // total dreams completed — proxy for age
}

impl SleepSchedule {
    pub fn new(now: u64) -> Self {
        Self {
            last_slept_at: now,
            next_sleep_at: now + 3600,
            sleep_duration_secs: 60,
            is_sleeping: false,
            dream_count: 0,
        }
    }

    pub fn should_sleep(&self, now: u64) -> bool {
        !self.is_sleeping && now >= self.next_sleep_at
    }

    pub fn wake_at(&self) -> u64 {
        self.last_slept_at + self.sleep_duration_secs
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lineage {
    pub generation: u32,
    pub parent_ids: Vec<Uuid>,       // empty for gen 0
    pub inherited_attractors: Vec<String>, // top attractor queries from each parent
}

impl Lineage {
    pub fn genesis() -> Self {
        Self {
            generation: 0,
            parent_ids: vec![],
            inherited_attractors: vec![],
        }
    }

    pub fn from_parents(
        parent_a: &Thronglet,
        parent_b: &Thronglet,
        attractors_a: Vec<String>,
        attractors_b: Vec<String>,
    ) -> Self {
        Self {
            generation: parent_a.lineage.generation.max(parent_b.lineage.generation) + 1,
            parent_ids: vec![parent_a.id, parent_b.id],
            inherited_attractors: attractors_a.into_iter().chain(attractors_b).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrongletStats {
    pub concept_count: usize,
    pub epoch: u32,
    pub soul_norm: f64,          // distance from origin — growth indicator
    pub avg_vfe: f64,            // rolling average VFE over last N queries
    pub approval_rate: f64,      // quorum approval rate
    pub total_queries: u64,
    pub last_query_phase: DevelopmentPhase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thronglet {
    pub id: Uuid,
    pub name: String,

    // Identity
    pub dominant_soul: SoulArchetype,
    pub lineage: Lineage,

    // Filesystem paths — per-thronglet isolated instance
    // each thronglet gets its own directory: data/thronglets/{id}/
    pub data_dir: PathBuf,

    // Development state — derived from the underlying soul/memory state
    pub phase: DevelopmentPhase,
    pub zone: ManifoldZone,
    pub stats: ThrongletStats,

    // Lifecycle
    pub sleep: SleepSchedule,
    pub born_at: u64,

    // Reproduction
    pub maturity: f64,       // 0.0 → 1.0
    pub offspring_count: u32,
    pub can_reproduce: bool, // maturity >= threshold AND in Working/Frontier zone

    // World
    pub position: WorldPosition,
}

impl Thronglet {
    pub fn new(
        name: String,
        dominant_soul: SoulArchetype,
        data_dir: PathBuf,
        position: WorldPosition,
        now: u64,
    ) -> Self {
        let id = Uuid::new_v4();
        Self {
            id,
            name,
            dominant_soul,
            lineage: Lineage::genesis(),
            data_dir,
            phase: DevelopmentPhase::Dark,
            zone: ManifoldZone::Forbidden,
            stats: ThrongletStats {
                concept_count: 0,
                epoch: 0,
                soul_norm: 0.0,
                avg_vfe: 3.5,   // high VFE at birth — maximum uncertainty
                approval_rate: 0.0,
                total_queries: 0,
                last_query_phase: DevelopmentPhase::Dark,
            },
            sleep: SleepSchedule::new(now),
            born_at: now,
            maturity: 0.0,
            offspring_count: 0,
            can_reproduce: false,
            position,
        }
    }

    // Derive phase from soul_norm and concept_count
    // Mirrors the VFE phase gating in your existing system
    pub fn update_phase(&mut self) {
        self.phase = match (self.stats.soul_norm, self.stats.concept_count) {
            (n, c) if n < 0.05 || c < 5   => DevelopmentPhase::Dark,
            (n, c) if n < 0.25 || c < 20  => DevelopmentPhase::Aware,
            (n, c) if n < 0.45 || c < 50  => DevelopmentPhase::Engaged,
            (n, c) if n < 0.60 || c < 90  => DevelopmentPhase::Understanding,
            _                              => DevelopmentPhase::Transcendent,
        };

        self.zone = match self.stats.soul_norm {
            n if n < 0.05 => ManifoldZone::Forbidden,
            n if n < 0.25 => ManifoldZone::Core,
            n if n < 0.60 => ManifoldZone::Working,
            _             => ManifoldZone::Frontier,
        };

        // Maturity gates reproduction
        // Requires Working zone minimum + enough concepts + enough dreams
        self.maturity = {
            let norm_score    = (self.stats.soul_norm / 0.60).min(1.0);
            let concept_score = (self.stats.concept_count as f64 / 80.0).min(1.0);
            let dream_score   = (self.sleep.dream_count as f64 / 10.0).min(1.0);
            (norm_score * 0.4 + concept_score * 0.4 + dream_score * 0.2)
        };

        self.can_reproduce = self.maturity >= 1.0
            && matches!(self.zone, ManifoldZone::Working | ManifoldZone::Frontier)
            && self.phase != DevelopmentPhase::Dark;
    }
}
