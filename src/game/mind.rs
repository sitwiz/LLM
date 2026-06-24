use crate::embedding::Embedder;
use crate::game::thronglet::{SoulArchetype, Thronglet};
use crate::soul::geometry::update_soul;
use crate::soul::persistence::{load_soul, save_soul};
use crate::unified_omni_agi::vfe::minimise_vfe;
use nalgebra::DVector;
use redis::Client;

pub struct FeedResult {
    pub soul_norm_before: f64,
    pub soul_norm_after:  f64,
    pub vfe:              f64,
    pub confidence:       f64,
    pub phase:            String,
    pub zone:             String,
}

pub struct ThrongletMind {
    embedder: Embedder,
    redis:    Client,
}

impl ThrongletMind {
    pub fn new(redis: Client) -> Self {
        Self {
            embedder: Embedder::new(),
            redis,
        }
    }

    pub async fn feed(
        &self,
        thronglet: &mut Thronglet,
        knowledge: &str,
    ) -> Result<FeedResult, String> {
        let soul_path = thronglet.data_dir
            .join(archetype_filename(&thronglet.dominant_soul));

        let mut soul = load_soul(&soul_path)
            .map_err(|e| format!("Soul load failed: {}", e))?;

        let soul_norm_before = soul.norm();

        let obs = tokio::task::block_in_place(|| {
            self.embedder
                .embed_to_soul(knowledge)
                .unwrap_or_else(|_| DVector::zeros(256))
        });

        let (belief, _) = minimise_vfe(&soul, &obs, &obs, 0.15);

        soul = update_soul(&soul, &belief.position);

        save_soul(&soul, &soul_path)
            .map_err(|e| format!("Soul save failed: {}", e))?;

        thronglet.stats.soul_norm = soul.norm();
        thronglet.stats.avg_vfe   = belief.vfe;
        thronglet.update_phase();

        let phase_str = format!("{:?}", thronglet.phase).to_lowercase();
        let zone_str  = format!("{:?}", thronglet.zone).to_lowercase();

        let id = thronglet.id.to_string();
        if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
            redis::cmd("HSET")
                .arg(format!("thronglet:{}:state", id))
                .arg("soul_norm").arg(soul.norm().to_string())
                .arg("phase").arg(&phase_str)
                .arg("zone").arg(&zone_str)
                .query_async::<()>(&mut conn)
                .await
                .unwrap_or(());
        }

        Ok(FeedResult {
            soul_norm_before,
            soul_norm_after: soul.norm(),
            vfe:             belief.vfe,
            confidence:      belief.confidence,
            phase:           phase_str,
            zone:            zone_str,
        })
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::population::PopulationManager;
    use crate::game::thronglet::{SoulArchetype, WorldPosition};

    #[test]
    fn test_feed() {
        let handle = std::thread::spawn(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();

            let redis = Client::open("redis://127.0.0.1/").unwrap();
            let mind  = std::sync::Arc::new(ThrongletMind::new(redis.clone()));
            let mind2 = mind.clone();

            rt.block_on(async move {
                let manager = PopulationManager::new(
                    redis,
                    std::env::temp_dir().join("trong_feed_test"),
                );

                let mut trong = manager.spawn(
                    "TestKhaos".to_string(),
                    SoulArchetype::Khaos,
                    WorldPosition { x: 0, y: 0, z: 0 },
                ).await.unwrap();

                let result = mind2.feed(
                    &mut trong,
                    "Void. Entropy. The absence before existence.",
                ).await.unwrap();

                println!("[Test] norm {:.4} -> {:.4}  vfe={:.4}  conf={:.3}  phase={}",
                    result.soul_norm_before, result.soul_norm_after,
                    result.vfe, result.confidence, result.phase);

                assert!(result.soul_norm_after > 0.0);
                std::fs::remove_dir_all(&trong.data_dir).ok();
            });

            drop(mind);
            drop(rt);
        });

        handle.join().unwrap();
    }
}
