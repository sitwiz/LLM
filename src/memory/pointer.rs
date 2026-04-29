//! Layer 5 — Pointer index with 150-character summaries.
//! Older concepts that have drifted from working memory compress
//! into a summary plus a pointer to the full concept in the spatial index.
//! Retrieval is two-phase: match summary first, load full concept on demand.

use serde::{Serialize, Deserialize};
use std::path::Path;

const SUMMARY_MAX_CHARS: usize = 150;
const COMPRESSION_AGE:   u32   = 10;  // compress concepts older than this many epochs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerEntry {
    pub concept_name: String,
    pub personality:  String,
    pub summary:      String,       // ≤150 chars
    pub epoch_stored: u32,
    pub visit_count:  u32,
    pub strength:     f64,
    pub zone_label:   String,
}

impl PointerEntry {
    pub fn new(
        concept_name: &str,
        personality:  &str,
        full_text:    &str,
        epoch_stored: u32,
        visit_count:  u32,
        strength:     f64,
        zone_label:   &str,
    ) -> Self {
        // Truncate to SUMMARY_MAX_CHARS at a word boundary
        let summary = if full_text.len() <= SUMMARY_MAX_CHARS {
            full_text.to_string()
        } else {
            let truncated = &full_text[..SUMMARY_MAX_CHARS];
            match truncated.rfind(' ') {
                Some(pos) => format!("{}…", &truncated[..pos]),
                None      => format!("{}…", truncated),
            }
        };

        Self {
            concept_name: concept_name.to_string(),
            personality:  personality.to_string(),
            summary,
            epoch_stored,
            visit_count,
            strength,
            zone_label: zone_label.to_string(),
        }
    }

    /// Simple keyword match against summary — used for fast pre-filter.
    pub fn matches(&self, query: &str) -> bool {
        let q = query.to_lowercase();
        self.summary.to_lowercase().contains(&q) ||
        self.concept_name.to_lowercase().contains(&q)
    }
}

pub struct PointerIndex {
    pub entries: Vec<PointerEntry>,
}

impl PointerIndex {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn load(path: &Path) -> Self {
        let entries = std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { entries }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, serde_json::to_string(&self.entries)?)?;
        Ok(())
    }

    /// Add or update a pointer entry.
    pub fn upsert(&mut self, entry: PointerEntry) {
        if let Some(existing) = self.entries.iter_mut()
            .find(|e| e.concept_name == entry.concept_name && e.personality == entry.personality)
        {
            existing.visit_count  = entry.visit_count;
            existing.strength     = entry.strength;
            existing.zone_label   = entry.zone_label;
        } else {
            self.entries.push(entry);
        }
    }

    /// Compress eligible concepts from the spatial index into pointer entries.
    /// Called after dream cycle or on startup.
    pub fn compress_from_spatial(
        &mut self,
        concepts: &[crate::memory::spatial::ConceptPoint],
        current_epoch: u32,
    ) {
        for concept in concepts {
            let age = current_epoch.saturating_sub(concept.epoch);
            if age >= COMPRESSION_AGE {
                let entry = PointerEntry::new(
                    &concept.name,
                    &concept.personality,
                    &concept.name,  // use concept name as summary text
                    concept.epoch,
                    concept.visit_count,
                    concept.strength,
                    concept.zone.label(),
                );
                self.upsert(entry);
            }
        }
    }

    /// Search summaries for keyword match — fast pre-filter before spatial lookup.
    pub fn search(&self, query: &str, max: usize) -> Vec<&PointerEntry> {
        let mut results: Vec<&PointerEntry> = self.entries.iter()
            .filter(|e| e.matches(query))
            .collect();
        // Sort by strength descending
        results.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        results.truncate(max);
        results
    }

    pub fn len(&self)      -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool  { self.entries.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_truncation() {
        let long = "a".repeat(200);
        let entry = PointerEntry::new("test", "Khaos", &long, 0, 1, 1.2, "frontier");
        assert!(entry.summary.len() <= SUMMARY_MAX_CHARS + 4);
    }

    #[test]
    fn test_keyword_match() {
        let entry = PointerEntry::new(
            "Rust memory leak", "Gaia",
            "How to fix a memory leak in Rust using RAII",
            0, 1, 1.2, "frontier"
        );
        assert!(entry.matches("memory"));
        assert!(!entry.matches("python"));
    }

    #[test]
    fn test_upsert_updates_existing() {
        let mut idx = PointerIndex::new();
        let e1 = PointerEntry::new("x", "Gaia", "text", 0, 1, 1.2, "frontier");
        let e2 = PointerEntry::new("x", "Gaia", "text", 0, 5, 1.5, "working");
        idx.upsert(e1);
        idx.upsert(e2);
        assert_eq!(idx.entries.len(), 1);
        assert_eq!(idx.entries[0].visit_count, 5);
    }

    #[test]
    fn test_search_returns_by_strength() {
        let mut idx = PointerIndex::new();
        idx.upsert(PointerEntry::new("rust leak",    "Gaia", "rust memory leak fix", 0, 1, 1.2, "frontier"));
        idx.upsert(PointerEntry::new("rust borrow",  "Gaia", "rust borrow checker",  0, 5, 1.8, "working"));
        let results = idx.search("rust", 2);
        assert_eq!(results[0].strength, 1.8);
    }
}
