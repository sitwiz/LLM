use crate::game::thronglet::{
    DevelopmentPhase, Lineage, ManifoldZone, SleepSchedule,
    SoulArchetype, Thronglet, ThrongletStats, WorldPosition,
};
use crate::soul::geometry::{INITIAL_CURVATURE, SOUL_DIM};
use crate::soul::persistence::save_soul;
use nalgebra::DVector;
use redis::{AsyncCommands, Client};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

const NEWBORN_NORM: f64 = 0.03;

pub struct PopulationManager {
    redis:     Client,
    data_root: PathBuf,
}

impl PopulationManager {
    pub fn new(redis: Client, data_root: PathBuf) -> Self {
        Self { redis, data_root }
    }

    pub async fn spawn(
        &self,
        name:      String,
        archetype: SoulArchetype,
        position:  WorldPosition,
    ) -> Result<Thronglet, SpawnError> {
        let now = unix_now();
        let id  = Uuid::new_v4();

        let data_dir = self.data_root
            .join("thronglets")
            .join(id.to_string());

        std::fs::create_dir_all(&data_dir)
            .map_err(SpawnError::Io)?;

        let soul = newborn_soul(&archetype);

        let soul_path = data_dir.join(archetype_filename(&archetype));
        save_soul(&soul, &soul_path)
            .map_err(|e| SpawnError::SoulSave(e.to_string()))?;

        let mut thronglet = Thronglet::new(
            name,
            archetype,
            data_dir,
            position,
            now,
        );
        thronglet.id              = id;
        thronglet.stats.soul_norm = soul.norm();
        thronglet.update_phase();

        self.register(&thronglet)
            .await
            .map_err(|e| SpawnError::Redis(e.to_string()))?;

        println!(
            "[Population] Spawned {:?} ({}) id={} norm={:.4}",
            thronglet.dominant_soul,
            thronglet.name,
            thronglet.id,
            thronglet.stats.soul_norm,
        );

        Ok(thronglet)
    }

    pub async fn spawn_with_soul(
        &self,
        name:       String,
        archetype:  SoulArchetype,
        position:   WorldPosition,
        soul:       DVector<f64>,
        attractors: Vec<(String, DVector<f64>, f64)>,
    ) -> Result<Thronglet, SpawnError> {
        use crate::memory::spatial::{SpatialIndex, ConceptPoint};

        let now      = unix_now();
        let id       = Uuid::new_v4();
        let data_dir = self.data_root.join("thronglets").join(id.to_string());

        std::fs::create_dir_all(&data_dir).map_err(SpawnError::Io)?;

        // Save inherited soul vector
        let soul_path = data_dir.join(archetype_filename(&archetype));
        save_soul(&soul, &soul_path)
            .map_err(|e| SpawnError::SoulSave(e.to_string()))?;

        // Pre-populate spatial index with inherited attractors
        let index_path = data_dir.join("memory_index.json");
        let mut index  = SpatialIndex::new(1.0);
        for (query, pos, strength) in &attractors {
            let concept = ConceptPoint::new(
                query, pos, 0.6,
                &format!("{:?}", archetype),
                *strength * 0.5,
                1.5, 0,
            );
            index.insert(concept);
        }
        index.save(index_path.to_str().unwrap_or(""))
            .map_err(|e| SpawnError::SoulSave(e.to_string()))?;

        let mut thronglet = Thronglet::new(name, archetype, data_dir, position, now);
        thronglet.id                  = id;
        thronglet.stats.soul_norm     = soul.norm();
        thronglet.stats.concept_count = attractors.len();
        thronglet.update_phase();

        self.register(&thronglet).await
            .map_err(|e| SpawnError::Redis(e.to_string()))?;

        println!(
            "[Population] Spawned inherited ({}) id={} norm={:.4} attractors={}",
            thronglet.name, thronglet.id, thronglet.stats.soul_norm, attractors.len(),
        );

        Ok(thronglet)
    }

    async fn register(&self, t: &Thronglet) -> redis::RedisResult<()> {
        let mut conn = self.redis.get_multiplexed_async_connection().await?;
        let id  = t.id.to_string();
        let now = unix_now();

        redis::cmd("HSET")
            .arg(format!("thronglet:{}:state", id))
            .arg("name").arg(&t.name)
            .arg("dominant_soul").arg(format!("{:?}", t.dominant_soul))
            .arg("data_dir").arg(t.data_dir.to_str().unwrap_or(""))
            .arg("epoch").arg(0u32)
            .arg("phase").arg("dark")
            .arg("zone").arg("forbidden")
            .arg("concept_count").arg(0u32)
            .arg("soul_norm").arg(t.stats.soul_norm.to_string())
            .arg("dream_count").arg(0u32)
            .arg("born_at").arg(now)
            .arg("pos_x").arg(t.position.x)
            .arg("pos_y").arg(t.position.y)
            .arg("pos_z").arg(t.position.z)
            .query_async::<()>(&mut conn)
            .await?;

        let _: () = conn.sadd("thronglets:active", &id).await?;
        let _: () = conn.set(
            format!("thronglet:{}:next_sleep", id),
            now + 3600,
        ).await?;

        Ok(())
    }

    pub async fn get_state(&self, id: &Uuid) -> Option<ThrongletState> {
        let mut conn = self.redis.get_multiplexed_async_connection().await.ok()?;
        let key = format!("thronglet:{}:state", id);

        let fields: Vec<Option<String>> = redis::cmd("HMGET")
            .arg(&key)
            .arg("name")
            .arg("dominant_soul")
            .arg("phase")
            .arg("zone")
            .arg("epoch")
            .arg("concept_count")
            .arg("soul_norm")
            .arg("dream_count")
            .arg("data_dir")
            .query_async(&mut conn)
            .await
            .ok()?;

        Some(ThrongletState {
            id:            *id,
            name:          fields[0].clone().unwrap_or_default(),
            dominant_soul: fields[1].clone().unwrap_or_default(),
            phase:         fields[2].clone().unwrap_or_else(|| "dark".into()),
            zone:          fields[3].clone().unwrap_or_else(|| "forbidden".into()),
            epoch:         fields[4].as_deref().and_then(|v| v.parse().ok()).unwrap_or(0),
            concept_count: fields[5].as_deref().and_then(|v| v.parse().ok()).unwrap_or(0),
            soul_norm:     fields[6].as_deref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            dream_count:   fields[7].as_deref().and_then(|v| v.parse().ok()).unwrap_or(0),
            data_dir:      fields.get(8).cloned().flatten().unwrap_or_default(),
        })
    }

    pub async fn update_after_feed(
        &self,
        id:            &Uuid,
        soul_norm:     f64,
        concept_count: usize,
    ) -> redis::RedisResult<()> {
        let mut conn = self.redis.get_multiplexed_async_connection().await?;
        let key = format!("thronglet:{}:state", id);

        let phase = match (soul_norm, concept_count) {
            (n, c) if n < 0.05 || c < 5  => "dark",
            (n, c) if n < 0.25 || c < 20 => "aware",
            (n, c) if n < 0.45 || c < 50 => "engaged",
            (n, c) if n < 0.60 || c < 90 => "understanding",
            _                             => "transcendent",
        };

        let zone = match soul_norm {
            n if n < 0.05 => "forbidden",
            n if n < 0.25 => "core",
            n if n < 0.60 => "working",
            _             => "frontier",
        };

        redis::cmd("HSET")
            .arg(&key)
            .arg("soul_norm").arg(soul_norm.to_string())
            .arg("phase").arg(phase)
            .arg("zone").arg(zone)
            .arg("concept_count").arg(concept_count.to_string())
            .query_async::<()>(&mut conn)
            .await?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct ThrongletState {
    pub id:            Uuid,
    pub name:          String,
    pub dominant_soul: String,
    pub phase:         String,
    pub zone:          String,
    pub epoch:         u32,
    pub concept_count: usize,
    pub soul_norm:     f64,
    pub dream_count:   u32,
    pub data_dir:      String,
}

fn archetype_filename(archetype: &SoulArchetype) -> &'static str {
    match archetype {
        SoulArchetype::Khaos       => "khaos_soul.bin",
        SoulArchetype::Gaia        => "gaia_soul.bin",
        SoulArchetype::Tartaros    => "tartaros_soul.bin",
        SoulArchetype::Eros        => "eros_soul.bin",
        SoulArchetype::UnifiedOmni => "unified_omni_soul.bin",
    }
}

fn newborn_soul(archetype: &SoulArchetype) -> DVector<f64> {
    let seed = match archetype {
        SoulArchetype::Khaos       => 0.11,
        SoulArchetype::Gaia        => 0.23,
        SoulArchetype::Tartaros    => 0.37,
        SoulArchetype::Eros        => 0.51,
        SoulArchetype::UnifiedOmni => 0.67,
    };

    let raw = DVector::from_fn(SOUL_DIM, |i, _| {
        ((i as f64 * 0.031 + seed) * 2.7).sin() * 0.01
    });

    let norm = raw.norm().max(1e-10);
    raw * (NEWBORN_NORM / norm)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug)]
pub enum SpawnError {
    Io(std::io::Error),
    SoulSave(String),
    Redis(String),
}

impl std::fmt::Display for SpawnError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Io(e)        => write!(f, "IO error: {}", e),
            Self::SoulSave(e)  => write!(f, "Soul save error: {}", e),
            Self::Redis(e)     => write!(f, "Redis error: {}", e),
        }
    }
}
