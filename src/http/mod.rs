use axum::{
    extract::{Path, State},
    response::{
        sse::{Event, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::{Any, CorsLayer};
use axum::extract::ws::{WebSocket, WebSocketUpgrade, Message};
use tokio::sync::broadcast;

use crate::game::mind::ThrongletMind;
use crate::game::population::PopulationManager;
use crate::game::thronglet::{SoulArchetype, WorldPosition};

pub struct AppState {
    pub population: Arc<PopulationManager>,
    pub mind:       Arc<ThrongletMind>,
    pub broadcast:  broadcast::Sender<String>,
}

#[derive(Deserialize)]
pub struct SpawnBody {
    name:      String,
    archetype: Option<String>,
    x:         Option<i32>,
    y:         Option<i32>,
    z:         Option<i32>,
}

#[derive(Deserialize)]
pub struct FeedBody {
    id:        String,
    knowledge: String,
}

async fn spawn_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<SpawnBody>,
) -> Json<Value> {
    let archetype = match body.archetype.as_deref() {
        Some("Gaia")        => SoulArchetype::Gaia,
        Some("Tartaros")    => SoulArchetype::Tartaros,
        Some("Eros")        => SoulArchetype::Eros,
        Some("UnifiedOmni") => SoulArchetype::UnifiedOmni,
        _                   => SoulArchetype::Khaos,
    };

    match state.population.spawn(
        body.name,
        archetype,
        WorldPosition {
            x: body.x.unwrap_or(0),
            y: body.y.unwrap_or(0),
            z: body.z.unwrap_or(0),
        },
    ).await {
        Ok(t) => Json(json!({
            "id":            t.id.to_string(),
            "name":          t.name,
            "soul":          format!("{:?}", t.dominant_soul),
            "phase":         format!("{:?}", t.phase),
            "zone":          format!("{:?}", t.zone),
            "soul_norm":     t.stats.soul_norm,
            "concept_count": t.stats.concept_count,
            "dream_count":   t.sleep.dream_count,
            "generation":    t.lineage.generation,
        })),
        Err(e) => Json(json!({ "error": e.to_string() })),
    }
}

async fn feed_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<FeedBody>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    let population = Arc::clone(&state.population);
    let id         = body.id.clone();
    let knowledge  = body.knowledge.clone();

    tokio::spawn(async move {
        let uuid = match uuid::Uuid::parse_str(&id) {
            Ok(u) => u,
            Err(_) => {
                let _ = tx.send(Ok(Event::default().data(
                    json!({"error": "invalid id"}).to_string()
                ))).await;
                return;
            }
        };

        let thronglet_state = match population.get_state(&uuid).await {
            Some(s) => s,
            None => {
                let _ = tx.send(Ok(Event::default().data(
                    json!({"error": "not found"}).to_string()
                ))).await;
                return;
            }
        };

        let tx_inner   = tx.clone();
        let knowledge2 = knowledge.clone();
        let data_dir   = thronglet_state.data_dir.clone();
        let soul_name  = thronglet_state.dominant_soul.clone();

        tokio::task::block_in_place(move || {
            use crate::unified_omni_agi::vfe::minimise_vfe_with_callback;
            use crate::soul::persistence::load_soul;
            use crate::embedding::Embedder;
            use crate::memory::spatial::{SpatialIndex, ConceptPoint, uor_address};
            use nalgebra::DVector;
            use std::path::Path;

            let soul_file = format!("{}/{}", data_dir, soul_filename(&soul_name));
            let soul = load_soul(Path::new(&soul_file))
                .unwrap_or_else(|_| DVector::zeros(256));

            let embedder = Embedder::new();
            let obs = embedder.embed_to_soul(&knowledge2)
                .unwrap_or_else(|_| DVector::zeros(256));

            let soul_norm_before = soul.norm();

            let (belief, _) = minimise_vfe_with_callback(
                &soul, &obs, &obs, 0.15,
                |record| {
                    let event = Event::default().data(json!({
                        "type":       "cycle",
                        "cycle":      record.cycle,
                        "vfe":        record.vfe,
                        "confidence": record.confidence,
                        "pe":         record.pe_norm,
                    }).to_string());
                    let _ = tx_inner.blocking_send(Ok(event));
                },
            );

            // Compute UOR address for this concept before inserting
            let concept_uor = uor_address(&knowledge2);

            let complete = Event::default().data(json!({
                "type":             "complete",
                "soul_norm_before": soul_norm_before,
                "soul_norm_after":  belief.position.norm(),
                "final_vfe":        belief.vfe,
                "confidence":       belief.confidence,
                "uor_address":      concept_uor,
            }).to_string());
            let _ = tx_inner.blocking_send(Ok(complete));

            // Insert concept into spatial index
            let concept_path = format!("{}/memory_index.json", data_dir);
            let mut index = SpatialIndex::load(&concept_path, 1.0, 0);
            let concept = ConceptPoint::new(
                &knowledge2,
                &belief.position,
                0.6,
                &soul_name,
                1.2,
                1.5,
                0,
            );
            index.insert(concept);
            let _ = index.save(&concept_path);
            let concept_count = index.concepts.len();

            // Update Redis
            let norm_after = belief.position.norm();
            if let Ok(uuid) = uuid::Uuid::parse_str(&id) {
                let pop = population.clone();
                let _ = tokio::runtime::Handle::current()
                    .block_on(pop.update_after_feed(&uuid, norm_after, concept_count));
            }
        });
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15)))
}

async fn state_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<Value> {
    let uuid = match uuid::Uuid::parse_str(&id) {
        Ok(u) => u,
        Err(_) => return Json(json!({ "error": "invalid id" })),
    };

    match state.population.get_state(&uuid).await {
        Some(s) => {
            let concepts = tokio::task::block_in_place(|| {
                use crate::memory::spatial::SpatialIndex;
                let index_path = format!("{}/memory_index.json", s.data_dir);
                let index = SpatialIndex::load(&index_path, 1.0, 0);
                index.concepts.iter().map(|c| json!({
                    "name":        c.name,
                    "uor_address": c.uor_address,
                    "strength":    c.strength,
                    "zone":        c.zone.label(),
                })).collect::<Vec<_>>()
            });

            Json(json!({
                "id":            s.id.to_string(),
                "name":          s.name,
                "soul":          s.dominant_soul,
                "phase":         s.phase,
                "zone":          s.zone,
                "soul_norm":     s.soul_norm,
                "concept_count": s.concept_count,
                "dream_count":   s.dream_count,
                "epoch":         s.epoch,
                "concepts":      concepts,
            }))
        },
        None => Json(json!({ "error": "not found" })),
    }
}

async fn health_handler() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.broadcast.subscribe();
    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Ok(text) => {
                        if socket.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            msg = socket.recv() => {
                if msg.is_none() { break; }
            }
        }
    }
}

#[derive(Deserialize)]
struct ReproduceTestBody {
    trained_id: String,
}

async fn reproduce_test_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ReproduceTestBody>,
) -> Json<Value> {
    let uuid = match uuid::Uuid::parse_str(&body.trained_id) {
        Ok(u) => u,
        Err(e) => return Json(json!({ "error": e.to_string() })),
    };

    let trained_state = match state.population.get_state(&uuid).await {
        Some(s) => s,
        None => return Json(json!({ "error": "trained agent not found" })),
    };

    let result = tokio::task::block_in_place(|| {
        use crate::soul::persistence::load_soul;
        use crate::memory::spatial::SpatialIndex;
        use nalgebra::DVector;
        use std::path::Path;

        let soul_file = format!("{}/{}", trained_state.data_dir,
            match trained_state.dominant_soul.as_str() {
                "Gaia"        => "gaia_soul.bin",
                "Tartaros"    => "tartaros_soul.bin",
                "Eros"        => "eros_soul.bin",
                "UnifiedOmni" => "unified_omni_soul.bin",
                _             => "khaos_soul.bin",
            }
        );
        let trained_soul = load_soul(Path::new(&soul_file))
            .unwrap_or_else(|_| DVector::zeros(256));

        let index_path = format!("{}/memory_index.json", trained_state.data_dir);
        let index = SpatialIndex::load(&index_path, 1.0, 0);

        let mut concepts = index.concepts.clone();
        concepts.sort_by(|a, b| b.strength.partial_cmp(&a.strength)
            .unwrap_or(std::cmp::Ordering::Equal));

        let top_attractors: Vec<(String, DVector<f64>, f64)> = concepts.iter()
            .take(3)
            .map(|c| (c.name.clone(), c.position_vec(), c.strength))
            .collect();

        (trained_soul, top_attractors)
    });

    let (trained_soul, top_attractors) = result;
    let attractor_count = top_attractors.len();

    let inherited = match state.population.spawn_with_soul(
        format!("inherited-{}", &body.trained_id[..8]),
        crate::game::thronglet::SoulArchetype::Khaos,
        crate::game::thronglet::WorldPosition { x: 10, y: 0, z: 0 },
        trained_soul,
        top_attractors,
    ).await {
        Ok(t) => t,
        Err(e) => return Json(json!({ "error": format!("spawn inherited failed: {e}") })),
    };

    let naive = match state.population.spawn(
        "naive-baseline".to_string(),
        crate::game::thronglet::SoulArchetype::Khaos,
        crate::game::thronglet::WorldPosition { x: -10, y: 0, z: 0 },
    ).await {
        Ok(t) => t,
        Err(e) => return Json(json!({ "error": format!("spawn naive failed: {e}") })),
    };

    Json(json!({
        "inherited_id":         inherited.id.to_string(),
        "naive_id":             naive.id.to_string(),
        "attractors_inherited": attractor_count,
        "trained_soul_norm":    inherited.stats.soul_norm,
        "message":              "Feed both the same knowledge and compare VFE cycle counts"
    }))
}

pub fn router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/api/spawn",          post(spawn_handler))
        .route("/api/feed",           post(feed_handler))
        .route("/api/state/:id",      get(state_handler))
        .route("/health",             get(health_handler))
        .route("/ws",                 get(ws_handler))
        .route("/api/reproduce-test", post(reproduce_test_handler))
        .with_state(state)
        .layer(cors)
}

fn soul_filename(name: &str) -> &'static str {
    match name {
        "Gaia"        => "gaia_soul.bin",
        "Tartaros"    => "tartaros_soul.bin",
        "Eros"        => "eros_soul.bin",
        "UnifiedOmni" => "unified_omni_soul.bin",
        _             => "khaos_soul.bin",
    }
}
