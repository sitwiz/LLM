mod soul;
mod ollama;
mod pantheon;
mod quorum;
mod daemon;
mod embedding;
mod cortical_mesh;
mod brain;
mod memory;
mod neo_cortical_mesh;
mod unified_omni_agi;
mod session;
mod socialisation;
mod benchmarks;
mod game;
mod grpc;
mod http;

use quorum::Quorum;
use daemon::{DaemonState, dream_cycle};
use socialisation::SocialisedSession;
use crate::memory::episodic::Episode;
use benchmarks::compression::CompressionBenchmark;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

fn main() {
    let mut quorum = Quorum::new();

    // Start gRPC server in background
    let redis_grpc = redis::Client::open("redis://127.0.0.1/").unwrap();
    let pantheon   = grpc::server::Pantheon::new(
        redis_grpc,
        std::path::PathBuf::from("game_data"),
    );
    let svc = grpc::server::proto::pantheon_service_server::PantheonServiceServer::new(pantheon);

    let rt_grpc = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt_grpc.spawn(async move {
        tonic::transport::Server::builder()
            .accept_http1(true)
            .add_service(tonic_web::enable(svc))
            .serve("0.0.0.0:50051".parse().unwrap())
            .await
            .unwrap();
    });

    println!("[gRPC] PantheonService listening on 0.0.0.0:50051");

    // Start HTTP bridge on 8080
    let (broadcast_tx, _) = tokio::sync::broadcast::channel::<String>(256);

    let http_state = Arc::new(crate::http::AppState {
        population: Arc::new(crate::game::population::PopulationManager::new(
            redis::Client::open("redis://127.0.0.1/").unwrap(),
            std::path::PathBuf::from("game_data"),
        )),
        mind: Arc::new(crate::game::mind::ThrongletMind::new(
            redis::Client::open("redis://127.0.0.1/").unwrap(),
        )),
        broadcast: broadcast_tx,
    });

    let http_router = crate::http::router(http_state);
    rt_grpc.spawn(async move {
        axum::serve(
            tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap(),
            http_router,
        ).await.unwrap();
    });

    println!("[HTTP] Bridge listening on 0.0.0.0:8080");

    let daemon_state = Arc::new(Mutex::new(DaemonState::new()));

    let state_ref = Arc::clone(&daemon_state);
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(10));
            let should = state_ref.lock().unwrap().should_dream();
            if should {
                println!("\n[Daemon] Idle conditions met — triggering dream cycle...");
                state_ref.lock().unwrap().is_dreaming = true;

                use crate::soul::persistence::load_soul;
                use std::path::Path;
                let k = load_soul(Path::new("khaos_soul.bin"))
                    .unwrap_or_else(|_| nalgebra::DVector::zeros(256));
                let g = load_soul(Path::new("gaia_soul.bin"))
                    .unwrap_or_else(|_| nalgebra::DVector::zeros(256));
                let t = load_soul(Path::new("tartaros_soul.bin"))
                    .unwrap_or_else(|_| nalgebra::DVector::zeros(256));
                let e = load_soul(Path::new("eros_soul.bin"))
                    .unwrap_or_else(|_| nalgebra::DVector::zeros(256));
                let o = load_soul(Path::new("unified_omni_soul.bin"))
                    .unwrap_or_else(|_| nalgebra::DVector::zeros(256));

                let epoch = crate::memory::expanding::ExpandingManifold::load("manifold.json").epoch;
                let (_, _, _, _, _, records) = dream_cycle(&k, &g, &t, &e, &o, epoch);
                println!("[Daemon] Dream complete. {} records.", records.len());

                let mut s = state_ref.lock().unwrap();
                s.is_dreaming = false;
                s.last_dream  = Some(std::time::Instant::now());
                s.query_count = 0;
            }
        }
    });

    let queries = vec![
        "What is the origin of consciousness?",
        "How do I fix a memory leak in Rust?",
        "What is the connection between evolution and economics?",
        "Why does my distributed system keep failing under load?",
        "What is the weather today?",
        "What lies at the foundation of mathematics?",
    ];

    let session_id = format!("session_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let mut episode = Episode::new(&session_id);

    for query in &queries {
        let result = quorum.ask(query);
        daemon_state.lock().unwrap().record_query();

        let approved = !result.response.starts_with("[Neo Cortical Mesh blocked");
        let primary  = result.activated.first()
            .cloned()
            .unwrap_or_else(|| "none".to_string());

        episode.record_turn(
            result.session.turn_count,
            query,
            &primary,
            result.session.velocity,
            &result.phase,
            approved,
        );

        println!("[Session] Turn={} Velocity={:.4} Hull={:.4} Arc={} Severity={:.3}",
            result.session.turn_count,
            result.session.velocity,
            result.session.hull_area,
            result.session.arc_detected,
            result.session.arc_severity);

        println!();
    }

    // ── Compression benchmark ─────────────────────────────────────────────
    println!("\n[Benchmark] Running compression benchmark...");
    let benchmark   = CompressionBenchmark::new();
    let session_num = quorum.memory().episodic.len() as u64;
    if let Some(result) = benchmark.run(
        &quorum.memory().spatial,
        &quorum.memory().manifold,
        session_num,
    ) {
        result.print();
        benchmark.save_result(&result);
    }

    // ── Socialisation ─────────────────────────────────────────────────────
    println!("\n[Social] Initialising Instance B...");
    let mut instance_b = Quorum::new_instance_b();
    instance_b.reset_session();

    let mut social_session = SocialisedSession::new();
    social_session.run(&mut quorum, &mut instance_b, 5);

    for turn in &social_session.history {
        episode.record_turn(
            turn.turn,
            &turn.question,
            &turn.asker,
            turn.soul_distance,
            "social",
            turn.approved,
        );
    }

    if quorum.memory().manifold.epoch > 0 {
        episode.mark_dream();
    }

    quorum.memory_mut().episodic.commit(episode).unwrap_or_else(|e| {
        eprintln!("[Episodic] Failed to commit session: {}", e);
    });

    println!("\n[Episodic] Session committed. Total episodes: {}",
        quorum.memory().episodic.len());

    let state = daemon_state.lock().unwrap();
    println!("\n[Daemon] Queries: {} Should dream: {}",
        state.query_count, state.should_dream());
    drop(state);

    println!("[Daemon] Waiting for background thread...");
    thread::sleep(Duration::from_secs(35));
}
