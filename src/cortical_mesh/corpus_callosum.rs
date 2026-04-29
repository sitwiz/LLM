use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Message between nodes
#[derive(Debug, Clone)]
pub struct NodeMessage {
    pub from:    String,
    pub to:      String,
    pub content: String,
    pub round:   usize,
}

/// The CorpusCallosum — global message bus between nodes
/// Messages are queued by recipient and flushed on read
pub struct CorpusCallosum {
    queues: Arc<Mutex<HashMap<String, Vec<NodeMessage>>>>,
}

impl CorpusCallosum {
    pub fn new() -> Self {
        Self {
            queues: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Send a message to a specific node
    pub fn send(&self, msg: NodeMessage) {
        let mut queues = self.queues.lock().unwrap();
        queues.entry(msg.to.clone()).or_default().push(msg);
    }

    /// Broadcast a message to all nodes except sender
    pub fn broadcast(&self, from: &str, content: &str,
                     round: usize, all_nodes: &[String]) {
        let mut queues = self.queues.lock().unwrap();
        for node in all_nodes {
            if node != from {
                queues.entry(node.clone()).or_default().push(NodeMessage {
                    from:    from.to_string(),
                    to:      node.clone(),
                    content: content.to_string(),
                    round,
                });
            }
        }
    }

    /// Read and flush all messages for a node
    pub fn flush(&self, node: &str) -> Vec<NodeMessage> {
        let mut queues = self.queues.lock().unwrap();
        queues.remove(node).unwrap_or_default()
    }

    /// Check if a node has pending messages
    pub fn has_messages(&self, node: &str) -> bool {
        let queues = self.queues.lock().unwrap();
        queues.get(node).map(|q| !q.is_empty()).unwrap_or(false)
    }

    pub fn clone_ref(&self) -> Self {
        Self {
            queues: Arc::clone(&self.queues),
        }
    }
}
