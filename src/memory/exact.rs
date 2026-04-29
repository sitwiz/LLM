//! Layer 3 — Exact match query hash.
//! HashMap keyed on normalised query string.
//! Returns instantly on repeated queries without touching the manifold
//! or running VFE minimisation.
//! Entries expire after TTL queries to prevent stale hits.

use std::collections::HashMap;

const TTL_QUERIES: u32 = 100;   // entries expire after this many total queries
const MAX_ENTRIES: usize = 512; // cap to prevent unbounded growth

#[derive(Debug, Clone)]
pub struct ExactEntry {
    pub response:    String,
    pub personality: String,
    pub hit_count:   u32,
    pub stored_at:   u32,  // total query count when stored
}

pub struct ExactMatchIndex {
    entries:      HashMap<String, ExactEntry>,
    query_count:  u32,
}

impl ExactMatchIndex {
    pub fn new() -> Self {
        Self {
            entries:     HashMap::new(),
            query_count: 0,
        }
    }

    /// Normalise query for matching — lowercase, trim, collapse whitespace.
    fn normalise(query: &str) -> String {
        query.trim().to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Look up a query. Returns the cached response if found and not expired.
    pub fn lookup(&mut self, query: &str) -> Option<&ExactEntry> {
        self.query_count += 1;
        let key = Self::normalise(query);

        // Check expiry first
        if let Some(entry) = self.entries.get(&key) {
            if self.query_count - entry.stored_at > TTL_QUERIES {
                self.entries.remove(&key);
                return None;
            }
        }

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.hit_count += 1;
            Some(entry)
        } else {
            None
        }
    }

    /// Store a query-response pair.
    pub fn store(&mut self, query: &str, response: &str, personality: &str) {
        if self.entries.len() >= MAX_ENTRIES {
            self.evict_oldest();
        }
        let key = Self::normalise(query);
        self.entries.insert(key, ExactEntry {
            response:    response.to_string(),
            personality: personality.to_string(),
            hit_count:   0,
            stored_at:   self.query_count,
        });
    }

    /// Evict the oldest entry by stored_at.
    fn evict_oldest(&mut self) {
        if let Some(key) = self.entries.iter()
            .min_by_key(|(_, v)| v.stored_at)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&key);
        }
    }

    /// Purge all expired entries.
    pub fn purge_expired(&mut self) {
        let qc = self.query_count;
        self.entries.retain(|_, v| qc - v.stored_at <= TTL_QUERIES);
    }

    pub fn len(&self)      -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool  { self.entries.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_hit() {
        let mut idx = ExactMatchIndex::new();
        idx.store("What is Rust?", "A systems language.", "Gaia");
        let hit = idx.lookup("What is Rust?");
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().personality, "Gaia");
    }

    #[test]
    fn test_normalisation() {
        let mut idx = ExactMatchIndex::new();
        idx.store("What  is  Rust?", "A systems language.", "Gaia");
        let hit = idx.lookup("what is rust?");
        assert!(hit.is_some());
    }

    #[test]
    fn test_ttl_expiry() {
        let mut idx = ExactMatchIndex::new();
        idx.store("q", "r", "Gaia");
        // Advance query count past TTL
        for i in 0..(TTL_QUERIES + 1) {
            idx.lookup(&format!("other_{}", i));
        }
        let hit = idx.lookup("q");
        assert!(hit.is_none());
    }

    #[test]
    fn test_hit_count_increments() {
        let mut idx = ExactMatchIndex::new();
        idx.store("q", "r", "Gaia");
        idx.lookup("q");
        idx.lookup("q");
        let entry = idx.entries.get(&ExactMatchIndex::normalise("q")).unwrap();
        assert_eq!(entry.hit_count, 2);
    }
}
