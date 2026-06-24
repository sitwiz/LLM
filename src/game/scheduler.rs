use redis::{AsyncCommands, Client};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{interval, Duration};
use uuid::Uuid;

const SLEEP_INTERVAL_SECS: u64 = 3600;
const SLEEP_DURATION_SECS: u64 = 60;
const SCHEDULER_TICK_MS:   u64 = 10_000;

fn sleep_key(id: &Uuid)     -> String { format!("thronglet:{}:sleeping", id) }
fn state_key(id: &Uuid)     -> String { format!("thronglet:{}:state", id) }
fn event_channel(id: &Uuid) -> String { format!("thronglet:{}:events", id) }

pub struct SleepScheduler {
    redis: Client,
}

impl SleepScheduler {
    pub fn new(redis: Client) -> Self {
        Self { redis }
    }

    pub async fn run(&self) {
        let mut ticker = interval(Duration::from_millis(SCHEDULER_TICK_MS));
        loop {
            ticker.tick().await;
            let now = unix_now();

            if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
                let ids: Vec<String> = conn.smembers("thronglets:active").await.unwrap_or_default();

                for id_str in ids {
                    let Ok(id) = Uuid::parse_str(&id_str) else { continue };

                    let next_sleep: Option<u64> = conn
                        .get(format!("thronglet:{}:next_sleep", id))
                        .await
                        .ok();

                    let Some(next_sleep) = next_sleep else { continue };

                    if now >= next_sleep {
                        let redis = self.redis.clone();
                        tokio::spawn(async move {
                            run_sleep_cycle(redis, id, now).await;
                        });
                    }
                }
            }
        }
    }
}

async fn run_sleep_cycle(redis: Client, id: Uuid, now: u64) {
    let Ok(mut conn) = redis.get_multiplexed_async_connection().await else { return };

    let _: () = conn.set(sleep_key(&id), 1).await.unwrap_or(());

    let _: () = conn.publish(
        event_channel(&id),
        serde_json::json!({
            "event": "sleep_start",
            "id": id,
            "duration": SLEEP_DURATION_SECS
        }).to_string()
    ).await.unwrap_or(());

    let data_dir: Option<String> = conn.hget(state_key(&id), "data_dir").await.ok();
    let epoch: Option<u32>       = conn.hget(state_key(&id), "epoch").await.ok();

    if let (Some(dir), Some(ep)) = (data_dir, epoch) {
        run_dream_cycle_for(&id, std::path::Path::new(&dir), ep).await;
    }

    let _: () = conn.del(sleep_key(&id)).await.unwrap_or(());

    let _: () = conn
        .set(format!("thronglet:{}:next_sleep", id), now + SLEEP_INTERVAL_SECS)
        .await
        .unwrap_or(());

    let _: () = conn.hincr(state_key(&id), "dream_count", 1).await.unwrap_or(());

    let _: () = conn.publish(
        event_channel(&id),
        serde_json::json!({ "event": "wake", "id": id }).to_string()
    ).await.unwrap_or(());
}

async fn run_dream_cycle_for(_id: &Uuid, data_dir: &std::path::Path, epoch: u32) {
    let data_dir = data_dir.to_path_buf();
    tokio::task::spawn_blocking(move || {
        crate::daemon::dream_cycle_scoped(&data_dir, epoch);
    }).await.ok();
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
