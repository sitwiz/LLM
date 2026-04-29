//! Layer 7 — Episodic memory.
//! Records complete sessions as compressed episode summaries.
//! Each episode stores: session id, turn count, personalities activated,
//! queries asked, VFE trajectory, and a compressed text summary.
//! Persisted to disk as newline-delimited JSON.

use serde::{Serialize, Deserialize};
use std::io::{BufRead, Write};

const MAX_EPISODES:      usize = 500;
const MAX_QUERY_PREVIEW: usize = 60;   // chars per query stored in episode

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeTurn {
    pub turn:        usize,
    pub query:       String,       // truncated to MAX_QUERY_PREVIEW chars
    pub personality: String,
    pub vfe_final:   f64,
    pub phase:       String,
    pub approved:    bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub session_id:    String,
    pub timestamp:     u64,           // unix seconds
    pub turn_count:    usize,
    pub turns:         Vec<EpisodeTurn>,
    pub personalities: Vec<String>,   // unique personalities activated
    pub avg_vfe:       f64,
    pub dream_fired:   bool,
    pub spark_count:   u32,
}

impl Episode {
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id:    session_id.to_string(),
            timestamp:     unix_now(),
            turn_count:    0,
            turns:         Vec::new(),
            personalities: Vec::new(),
            avg_vfe:       0.0,
            dream_fired:   false,
            spark_count:   0,
        }
    }

    pub fn record_turn(
        &mut self,
        turn:        usize,
        query:       &str,
        personality: &str,
        vfe_final:   f64,
        phase:       &str,
        approved:    bool,
    ) {
        // Truncate query for storage
        let query_preview = if query.len() > MAX_QUERY_PREVIEW {
            format!("{}…", &query[..MAX_QUERY_PREVIEW])
        } else {
            query.to_string()
        };

        self.turns.push(EpisodeTurn {
            turn,
            query: query_preview,
            personality: personality.to_string(),
            vfe_final,
            phase: phase.to_string(),
            approved,
        });

        // Update personality list
        if !self.personalities.contains(&personality.to_string()) {
            self.personalities.push(personality.to_string());
        }

        self.turn_count += 1;
        self.avg_vfe = self.turns.iter().map(|t| t.vfe_final).sum::<f64>()
            / self.turn_count as f64;
    }

    pub fn mark_dream(&mut self)  { self.dream_fired = true; }
    pub fn add_spark(&mut self)   { self.spark_count += 1; }

    /// One-line summary for display.
    pub fn summary(&self) -> String {
        format!(
            "[{}] turns={} personalities={} avg_vfe={:.3} dream={} sparks={}",
            self.session_id,
            self.turn_count,
            self.personalities.join("+"),
            self.avg_vfe,
            self.dream_fired,
            self.spark_count,
        )
    }
}

pub struct EpisodicMemory {
    pub episodes: Vec<Episode>,
    path:         String,
}

impl EpisodicMemory {
    pub fn new(path: &str) -> Self {
        Self {
            episodes: Vec::new(),
            path:     path.to_string(),
        }
    }

    pub fn load(path: &str) -> Self {
        let mut episodes = Vec::new();

        if let Ok(file) = std::fs::File::open(path) {
            let reader = std::io::BufReader::new(file);
            for line in reader.lines().flatten() {
                if let Ok(ep) = serde_json::from_str::<Episode>(&line) {
                    episodes.push(ep);
                }
            }
        }

        println!("  [Episodic] Loaded {} episodes from disk.", episodes.len());

        Self {
            episodes,
            path: path.to_string(),
        }
    }

    /// Append the current episode to disk without rewriting the full file.
    pub fn commit(&mut self, episode: Episode) -> anyhow::Result<()> {
        let line = serde_json::to_string(&episode)?;

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        writeln!(file, "{}", line)?;

        self.episodes.push(episode);

        // Trim in-memory buffer if over limit
        if self.episodes.len() > MAX_EPISODES {
            let drain = self.episodes.len() - MAX_EPISODES;
            self.episodes.drain(0..drain);
        }

        Ok(())
    }

    /// Most recent N episodes.
    pub fn recent(&self, n: usize) -> Vec<&Episode> {
        self.episodes.iter().rev().take(n).collect()
    }

    /// Search episodes that activated a specific personality.
    pub fn by_personality(&self, name: &str) -> Vec<&Episode> {
        self.episodes.iter()
            .filter(|e| e.personalities.iter().any(|p| p == name))
            .collect()
    }

    /// Episodes where a query keyword appears.
    pub fn search(&self, keyword: &str) -> Vec<&Episode> {
        let kw = keyword.to_lowercase();
        self.episodes.iter()
            .filter(|e| e.turns.iter().any(|t| t.query.to_lowercase().contains(&kw)))
            .collect()
    }

    pub fn len(&self)      -> usize { self.episodes.len() }
    pub fn is_empty(&self) -> bool  { self.episodes.is_empty() }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_episode() -> Episode {
        let mut ep = Episode::new("test-session");
        ep.record_turn(0, "What is Rust?", "Gaia", 0.04, "understanding", true);
        ep.record_turn(1, "What is consciousness?", "Khaos", 0.03, "transcendent", true);
        ep
    }

    #[test]
    fn test_episode_turn_count() {
        let ep = make_episode();
        assert_eq!(ep.turn_count, 2);
    }

    #[test]
    fn test_unique_personalities() {
        let ep = make_episode();
        assert_eq!(ep.personalities.len(), 2);
        assert!(ep.personalities.contains(&"Gaia".to_string()));
        assert!(ep.personalities.contains(&"Khaos".to_string()));
    }

    #[test]
    fn test_avg_vfe() {
        let ep = make_episode();
        let expected = (0.04 + 0.03) / 2.0;
        assert!((ep.avg_vfe - expected).abs() < 1e-10);
    }

    #[test]
    fn test_query_truncation() {
        let mut ep = Episode::new("s");
        let long_query = "x".repeat(200);
        ep.record_turn(0, &long_query, "Gaia", 0.04, "understanding", true);
        assert!(ep.turns[0].query.len() <= MAX_QUERY_PREVIEW + 4);
    }

    #[test]
    fn test_commit_and_load() {
        let path = "/tmp/test_episodic.jsonl";
        let mut mem = EpisodicMemory::new(path);
        mem.commit(make_episode()).unwrap();
        let loaded = EpisodicMemory::load(path);
        assert_eq!(loaded.episodes.len(), 1);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_search() {
        let mut mem = EpisodicMemory::new("/tmp/unused");
        mem.episodes.push(make_episode());
        let results = mem.search("rust");
        assert_eq!(results.len(), 1);
    }
}
