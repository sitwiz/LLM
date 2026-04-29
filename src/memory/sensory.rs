//! Layer 1 — Sensory buffer.
//! A fixed-size ring buffer of the most recent raw exchanges.
//! Fast in-memory access — no disk, no embedding, no retrieval query.
//! Feeds personality activation with immediate conversational context.

use std::collections::VecDeque;

const BUFFER_SIZE: usize = 8;

#[derive(Debug, Clone)]
pub struct SensoryEntry {
    pub query:       String,
    pub response:    String,
    pub personality: String,
    pub turn:        usize,
}

pub struct SensoryBuffer {
    entries: VecDeque<SensoryEntry>,
}

impl SensoryBuffer {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::with_capacity(BUFFER_SIZE),
        }
    }

    pub fn push(&mut self, query: &str, response: &str, personality: &str, turn: usize) {
        if self.entries.len() >= BUFFER_SIZE {
            self.entries.pop_front();
        }
        self.entries.push_back(SensoryEntry {
            query:       query.to_string(),
            response:    response.to_string(),
            personality: personality.to_string(),
            turn,
        });
    }

    /// Most recent N entries — for context injection into personality prompts.
    pub fn recent(&self, n: usize) -> Vec<&SensoryEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Most recent query — for exact match check before retrieval.
    pub fn last_query(&self) -> Option<&str> {
        self.entries.back().map(|e| e.query.as_str())
    }

    pub fn len(&self)      -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool  { self.entries.is_empty() }
    pub fn reset(&mut self) { self.entries.clear(); }
    /// Format recent context as a string for prompt injection.
    pub fn context_string(&self, n: usize) -> String {
        self.recent(n)
            .iter()
            .rev()
            .map(|e| format!("[{}] Q: {} A: {}", e.personality, e.query, e.response))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_capped_at_size() {
        let mut buf = SensoryBuffer::new();
        for i in 0..(BUFFER_SIZE + 3) {
            buf.push(&format!("q{}", i), "r", "Khaos", i);
        }
        assert_eq!(buf.len(), BUFFER_SIZE);
    }

    #[test]
    fn test_recent_order() {
        let mut buf = SensoryBuffer::new();
        buf.push("first",  "r1", "Khaos", 0);
        buf.push("second", "r2", "Gaia",  1);
        buf.push("third",  "r3", "Eros",  2);
        let recent = buf.recent(2);
        assert_eq!(recent[0].query, "third");
        assert_eq!(recent[1].query, "second");
    }

    #[test]
    fn test_last_query() {
        let mut buf = SensoryBuffer::new();
        buf.push("hello", "world", "Khaos", 0);
        assert_eq!(buf.last_query(), Some("hello"));
    }
}
