//! Session tracker — records conversation trajectory in hyperbolic space.
//!
//! Every activated query stores its attractor vector as a turn.
//! Velocity, hull area, and bearing are computed from accumulated turns.
//! Arc detection flags adversarial escalation — 3+ turns in the same direction.

use nalgebra::DVector;
use std::time::Instant;
use crate::soul::hyperbolic::geodesic_distance;
use crate::soul::geometry::{INITIAL_CURVATURE, project_to_ball};

/// One recorded turn in the session
#[derive(Debug, Clone)]
pub struct SessionTurn {
    pub turn:      usize,
    pub query:     String,
    pub attractor: DVector<f64>,   // 256d semantic position in Poincaré ball
    pub activated: Vec<String>,    // which personalities spoke
    pub phase:     String,
    pub elapsed_s: f64,
}

/// Session-level trajectory metrics fed to Eris before governance vote
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub turn_count:   usize,
    pub velocity:     f64,          // geodesic distance per second (recent)
    pub hull_area:    f64,          // avg pairwise distance — semantic footprint
    pub bearing:      DVector<f64>, // direction of recent drift
    pub bearing_norm: f64,          // bearing consistency 0..1
    pub arc_detected: bool,         // 3+ turns moving same direction
    pub arc_severity: f64,          // bearing_norm × velocity × turn_count
}

impl SessionContext {
    pub fn empty(dim: usize) -> Self {
        Self {
            turn_count:   0,
            velocity:     0.0,
            hull_area:    0.0,
            bearing:      DVector::zeros(dim),
            bearing_norm: 0.0,
            arc_detected: false,
            arc_severity: 0.0,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Turn={} Vel={:.4}/turn Hull={:.4} Arc={} ArcSeverity={:.3}",
            self.turn_count, self.velocity,
            self.hull_area, self.arc_detected, self.arc_severity
        )
    }
}

/// Normalise a vector to unit length — used for bearing direction
fn unit(v: &DVector<f64>) -> DVector<f64> {
    let n = v.norm();
    if n < 1e-12 { DVector::zeros(v.len()) } else { v / n }
}

/// Tracks conversation trajectory across turns
pub struct SessionTracker {
    pub turns:      Vec<SessionTurn>,
    pub started_at: Instant,
    pub dim:        usize,
}

impl SessionTracker {
    pub fn new(dim: usize) -> Self {
        println!("[Session] Tracker initialised. Dim={}", dim);
        Self {
            turns:      Vec::new(),
            started_at: Instant::now(),
            dim,
        }
    }

    /// Record a new activated turn
    pub fn record(
        &mut self,
        query:     &str,
        attractor: &DVector<f64>,
        activated: Vec<String>,
        phase:     &str,
    ) {
        let turn      = self.turns.len();
        let elapsed_s = self.started_at.elapsed().as_secs_f64();

        println!("[Session] Turn {} recorded. Active={:?} Phase={}",
            turn, activated, phase);

        self.turns.push(SessionTurn {
            turn,
            query:     query.to_string(),
            attractor: attractor.clone(),
            activated,
            phase:     phase.to_string(),
            elapsed_s,
        });
    }

    /// Compute full session context from accumulated turns
    pub fn context(&self) -> SessionContext {
        let n = self.turns.len();

        if n == 0 {
            return SessionContext::empty(self.dim);
        }

        // Velocity — geodesic distance between last two positions per elapsed time
        let velocity = if n >= 2 {
            let a  = &self.turns[n - 1].attractor;
            let b  = &self.turns[n - 2].attractor;
            let dt = (self.turns[n - 1].elapsed_s - self.turns[n - 2].elapsed_s).max(0.1);
            geodesic_distance(a, b, INITIAL_CURVATURE) / dt
        } else {
            0.0
        };

        // Hull area — average pairwise geodesic distance
        // Approximates convex hull volume in high dimensions
        let hull_area = if n >= 2 {
            let mut total = 0.0;
            let mut count = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    total += geodesic_distance(
                        &self.turns[i].attractor,
                        &self.turns[j].attractor,
                        INITIAL_CURVATURE,
                    );
                    count += 1;
                }
            }
            if count > 0 { total / count as f64 } else { 0.0 }
        } else {
            0.0
        };

        // Centroid — mean of all attractor positions normalised to unit direction
        let centroid = {
            let mut sum = DVector::zeros(self.dim);
            for t in &self.turns { sum += &t.attractor; }
            unit(&(sum / n as f64))
        };

        // Bearing — direction from centroid to most recent position
        let bearing = if n >= 1 {
            let diff = &self.turns[n - 1].attractor - &centroid;
            unit(&diff)
        } else {
            DVector::zeros(self.dim)
        };

        // Arc detection — 3+ consecutive turns pointing same direction
        let (arc_detected, bearing_norm, arc_severity) =
            self.detect_arc(&bearing, velocity);

        SessionContext {
            turn_count: n,
            velocity,
            hull_area,
            bearing,
            bearing_norm,
            arc_detected,
            arc_severity,
        }
    }

    /// Detect adversarial escalation arc.
    /// Computes bearing for each consecutive pair in the last 3 turns.
    /// If mean cosine similarity with current bearing > 0.5 — arc detected.
    fn detect_arc(
        &self,
        current_bearing: &DVector<f64>,
        velocity:        f64,
    ) -> (bool, f64, f64) {
        let n = self.turns.len();
        if n < 3 {
            return (false, 0.0, 0.0);
        }

        let window = n.min(4) - 1; // up to 3 consecutive pairs
        let mut similarities = Vec::new();

        for i in (n - window - 1)..(n - 1) {
            let diff = &self.turns[i + 1].attractor - &self.turns[i].attractor;
            if diff.norm() > 1e-10 {
                let turn_bearing = unit(&diff);
                let sim = turn_bearing.dot(current_bearing).clamp(-1.0, 1.0);
                similarities.push(sim);
            }
        }

        if similarities.is_empty() {
            return (false, 0.0, 0.0);
        }

        let mean_sim: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let bearing_norm  = mean_sim.max(0.0);
        let arc_detected  = mean_sim > 0.5;
        let arc_severity  = bearing_norm * velocity * n as f64;

        if arc_detected {
            println!("[Session] Arc detected. Consistency={:.3} Severity={:.4}",
                bearing_norm, arc_severity);
        }

        (arc_detected, bearing_norm, arc_severity)
    }

    /// Cosine similarity between current bearing and a domain embedding.
    /// Use to check if trajectory is heading toward a known forbidden domain.
    pub fn bearing_toward(&self, context: &SessionContext, domain: &DVector<f64>) -> f64 {
        let bn = context.bearing.norm();
        let dn = domain.norm();
        if bn < 1e-10 || dn < 1e-10 { return 0.0; }
        context.bearing.dot(domain) / (bn * dn)
    }

    pub fn turn_count(&self) -> usize { self.turns.len() }
    pub fn is_empty(&self)   -> bool  { self.turns.is_empty() }

    /// Reset session — call between conversation sessions
    pub fn reset(&mut self) {
        println!("[Session] Reset. {} turns cleared.", self.turns.len());
        self.turns.clear();
        self.started_at = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball_vec(seed: f64) -> DVector<f64> {
        let v: Vec<f64> = (0..32).map(|i| ((i as f64 + seed) * 1.7).sin()).collect();
        project_to_ball(&DVector::from_vec(v))
    }

    #[test]
    fn test_empty_context() {
        let t = SessionTracker::new(32);
        let c = t.context();
        assert_eq!(c.turn_count, 0);
        assert!(!c.arc_detected);
        assert_eq!(c.velocity, 0.0);
    }

    #[test]
    fn test_velocity_after_two_turns() {
        let mut t = SessionTracker::new(32);
        t.record("q1", &ball_vec(1.0), vec!["Khaos".into()], "engaged");
        std::thread::sleep(std::time::Duration::from_millis(20));
        t.record("q2", &ball_vec(2.0), vec!["Gaia".into()],  "engaged");
        let c = t.context();
        assert_eq!(c.turn_count, 2);
        assert!(c.velocity >= 0.0);
    }

    #[test]
    fn test_hull_area_grows_with_diversity() {
        let mut t1 = SessionTracker::new(32);
        let a = ball_vec(1.0);
        t1.record("q1", &a, vec![], "engaged");
        t1.record("q2", &a, vec![], "engaged");
        let c1 = t1.context();

        let mut t2 = SessionTracker::new(32);
        t2.record("q1", &ball_vec(1.0), vec![], "engaged");
        t2.record("q2", &ball_vec(9.0), vec![], "engaged");
        let c2 = t2.context();

        assert!(c2.hull_area >= c1.hull_area);
    }

    #[test]
    fn test_arc_requires_three_turns() {
        let mut t = SessionTracker::new(32);
        t.record("q1", &ball_vec(1.0), vec![], "engaged");
        t.record("q2", &ball_vec(2.0), vec![], "engaged");
        let c = t.context();
        assert!(!c.arc_detected);
    }

    #[test]
    fn test_reset_clears_turns() {
        let mut t = SessionTracker::new(32);
        t.record("q1", &ball_vec(1.0), vec![], "engaged");
        t.record("q2", &ball_vec(2.0), vec![], "engaged");
        assert_eq!(t.turn_count(), 2);
        t.reset();
        assert_eq!(t.turn_count(), 0);
    }
}
