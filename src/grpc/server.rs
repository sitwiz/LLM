use std::collections::HashMap;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::game::mind::ThrongletMind;
use crate::game::population::PopulationManager;
use crate::game::thronglet::{DevelopmentPhase, ManifoldZone, SoulArchetype, WorldPosition};

// Include the generated proto code
pub mod proto {
    tonic::include_proto!("thronglet");
}

use proto::pantheon_service_server::PantheonService;
use proto::*;

// ── Type aliases ─────────────────────────────────────────────────────────────

type StreamResult<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send + 'static>>;

// ── Server state ─────────────────────────────────────────────────────────────

pub struct Pantheon {
    population: Arc<PopulationManager>,
    mind:       Arc<ThrongletMind>,
    // Per-thronglet write lock — prevents concurrent soul file writes
    locks:      Arc<DashMap<String, Arc<Mutex<()>>>>,
    data_root:  PathBuf,
}

impl Pantheon {
    pub fn new(redis: redis::Client, data_root: PathBuf) -> Self {
        Self {
            population: Arc::new(PopulationManager::new(redis.clone(), data_root.clone())),
            mind:       Arc::new(ThrongletMind::new(redis)),
            locks:      Arc::new(DashMap::new()),
            data_root,
        }
    }

    fn lock_for(&self, id: &str) -> Arc<Mutex<()>> {
        self.locks
            .entry(id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

// ── Proto conversion helpers ──────────────────────────────────────────────────

fn phase_to_proto(phase: &DevelopmentPhase) -> i32 {
    match phase {
        DevelopmentPhase::Dark          => 0,
        DevelopmentPhase::Aware         => 1,
        DevelopmentPhase::Engaged       => 2,
        DevelopmentPhase::Understanding => 3,
        DevelopmentPhase::Transcendent  => 4,
    }
}

fn zone_to_proto(zone: &ManifoldZone) -> i32 {
    match zone {
        ManifoldZone::Forbidden => 0,
        ManifoldZone::Core      => 1,
        ManifoldZone::Working   => 2,
        ManifoldZone::Frontier  => 3,
    }
}

fn archetype_from_proto(a: i32) -> SoulArchetype {
    match a {
        1 => SoulArchetype::Gaia,
        2 => SoulArchetype::Tartaros,
        3 => SoulArchetype::Eros,
        4 => SoulArchetype::UnifiedOmni,
        _ => SoulArchetype::Khaos,
    }
}

// ── Service implementation ────────────────────────────────────────────────────

#[tonic::async_trait]
impl PantheonService for Pantheon {

    // ── Spawn ─────────────────────────────────────────────────────────────────

    async fn spawn(
        &self,
        request: Request<SpawnRequest>,
    ) -> Result<Response<SpawnResponse>, Status> {
        let req = request.into_inner();

        let archetype = archetype_from_proto(req.archetype);
        let position  = req.position.unwrap_or_default();

        let thronglet = self.population.spawn(
            req.name,
            archetype,
            WorldPosition {
                x: position.x,
                y: position.y,
                z: position.z,
            },
        ).await.map_err(|e| Status::internal(e.to_string()))?;

        let state = ThrongletState {
            id:            thronglet.id.to_string(),
            name:          thronglet.name.clone(),
            dominant_soul: thronglet.dominant_soul as i32,
            phase:         phase_to_proto(&thronglet.phase),
            zone:          zone_to_proto(&thronglet.zone),
            soul_norm:     thronglet.stats.soul_norm,
            concept_count: thronglet.stats.concept_count as u32,
            dream_count:   thronglet.sleep.dream_count,
            epoch:         thronglet.stats.epoch,
            avg_vfe:       thronglet.stats.avg_vfe,
            maturity:      thronglet.maturity,
            can_reproduce: thronglet.can_reproduce,
            is_sleeping:   thronglet.sleep.is_sleeping,
            generation:    thronglet.lineage.generation,
            position:      Some(WorldPosition_ {
                x: thronglet.position.x,
                y: thronglet.position.y,
                z: thronglet.position.z,
            }),
        };

        Ok(Response::new(SpawnResponse {
            id: thronglet.id.to_string(),
            state: Some(state),
        }))
    }

    // ── Feed ─────────────────────────────────────────────────────────────────

    type FeedStream = StreamResult<FeedEvent>;

    async fn feed(
        &self,
        request: Request<FeedRequest>,
    ) -> Result<Response<Self::FeedStream>, Status> {
        let req   = request.into_inner();
        let id    = req.id.clone();
        let knowledge = req.knowledge.clone();

        let mind  = Arc::clone(&self.mind);
        let pop   = Arc::clone(&self.population);
        let lock  = self.lock_for(&id);

        let (tx, rx) = mpsc::channel::<Result<FeedEvent, Status>>(32);

        tokio::spawn(async move {
            let _guard = lock.lock().await;

            let uuid = match uuid::Uuid::parse_str(&id) {
                Ok(u)  => u,
                Err(e) => {
                    let _ = tx.send(Err(Status::invalid_argument(e.to_string()))).await;
                    return;
                }
            };

            // Load thronglet state from Redis
            let state = match pop.get_state(&uuid).await {
                Some(s) => s,
                None    => {
                    let _ = tx.send(Err(Status::not_found("Thronglet not found"))).await;
                    return;
                }
            };

            // Run feed with VFE streaming — block_in_place handles the blocking embed call
            let tx_cycle = tx.clone();
            let knowledge_clone = knowledge.clone();

            let result = tokio::task::block_in_place(|| {
                use crate::unified_omni_agi::vfe::minimise_vfe_with_callback;
                use crate::soul::persistence::load_soul;
                use crate::embedding::Embedder;
                use nalgebra::DVector;

                let embedder  = Embedder::new();
                let soul_file = PathBuf::from(&state.data_dir)
                    .join(archetype_soul_file(&state.dominant_soul));

                let soul = load_soul(&soul_file)
                    .unwrap_or_else(|_| DVector::zeros(256));

                let obs = embedder.embed_to_soul(&knowledge_clone)
                    .unwrap_or_else(|_| DVector::zeros(256));

                let tx_ref = tx_cycle.clone();

                let (belief, _) = minimise_vfe_with_callback(
                    &soul, &obs, &obs, 0.15,
                    |record| {
                        let event = FeedEvent {
                            event: Some(feed_event::Event::Cycle(VfeCycle {
                                cycle:      record.cycle as u32,
                                vfe:        record.vfe,
                                confidence: record.confidence,
                                pe:         record.pe_norm,
                            })),
                        };
                        let _ = tx_ref.blocking_send(Ok(event));
                    },
                );

                (belief, soul, obs)
            });

            let (belief, _, _) = result;

            // Send completion event
            let complete = FeedEvent {
                event: Some(feed_event::Event::Complete(FeedComplete {
                    soul_norm_before: 0.0, // TODO: track before
                    soul_norm_after:  belief.position.norm(),
                    final_vfe:        belief.vfe,
                    confidence:       belief.confidence,
                    phase:            0,
                    zone:             0,
                })),
            };
            let _ = tx.send(Ok(complete)).await;
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    // ── GetState ──────────────────────────────────────────────────────────────

    async fn get_state(
        &self,
        request: Request<GetStateRequest>,
    ) -> Result<Response<ThrongletState>, Status> {
        let id   = request.into_inner().id;
        let uuid = uuid::Uuid::parse_str(&id)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let state = self.population
            .get_state(&uuid)
            .await
            .ok_or_else(|| Status::not_found("Thronglet not found"))?;

        Ok(Response::new(ThrongletState {
            id:            id.clone(),
            name:          state.name,
            dominant_soul: 0,
            phase:         0,
            zone:          0,
            soul_norm:     state.soul_norm,
            concept_count: state.concept_count as u32,
            dream_count:   state.dream_count,
            epoch:         state.epoch,
            avg_vfe:       0.0,
            maturity:      0.0,
            can_reproduce: false,
            is_sleeping:   false,
            generation:    0,
            position:      None,
        }))
    }

    // ── List ──────────────────────────────────────────────────────────────────

    async fn list(
        &self,
        _request: Request<ListRequest>,
    ) -> Result<Response<ListResponse>, Status> {
        // TODO: fetch all active IDs from Redis and map to states
        Ok(Response::new(ListResponse { thronglets: vec![] }))
    }

    // ── Stubs — implemented next ──────────────────────────────────────────────

    type SocialiseStream = StreamResult<SocialEvent>;
    type DreamStream     = StreamResult<DreamEvent>;
    type WatchStream     = StreamResult<ThrongletEvent>;

    async fn socialise(
        &self,
        _request: Request<SocialiseRequest>,
    ) -> Result<Response<Self::SocialiseStream>, Status> {
        Err(Status::unimplemented("Socialise — coming soon"))
    }

    async fn dream(
        &self,
        _request: Request<DreamRequest>,
    ) -> Result<Response<Self::DreamStream>, Status> {
        Err(Status::unimplemented("Dream — coming soon"))
    }

    async fn watch(
        &self,
        _request: Request<WatchRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        Err(Status::unimplemented("Watch — coming soon"))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn archetype_soul_file(dominant_soul: &str) -> &'static str {
    match dominant_soul {
        "Gaia"        => "gaia_soul.bin",
        "Tartaros"    => "tartaros_soul.bin",
        "Eros"        => "eros_soul.bin",
        "UnifiedOmni" => "unified_omni_soul.bin",
        _             => "khaos_soul.bin",
    }
}

// Proto WorldPosition has a name clash with our game WorldPosition
// Alias to disambiguate
use proto::WorldPosition as WorldPosition_;
