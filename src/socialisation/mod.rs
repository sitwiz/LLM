pub mod question_gen;

use nalgebra::DVector;
use crate::quorum::Quorum;
use crate::socialisation::question_gen::QuestionGenerator;
use crate::soul::geometry::{INITIAL_CURVATURE, compute_nf};
use crate::soul::hyperbolic::geodesic_distance;

const INITIAL_TRUST:    f64 = 0.5;
const TRUST_LEARN_RATE: f64 = 0.05;
const MAX_TURNS:        usize = 10;

#[derive(Debug, Clone)]
pub struct TrustState {
    pub a_trusts_b: f64,
    pub b_trusts_a: f64,
}

impl TrustState {
    pub fn new() -> Self {
        Self {
            a_trusts_b: INITIAL_TRUST,
            b_trusts_a: INITIAL_TRUST,
        }
    }

    pub fn update_a_trusts_b(&mut self, vfe_drop: f64, approved: bool) {
        let signal = if approved {
            vfe_drop.min(2.0) / 2.0
        } else {
            -0.2
        };
        self.a_trusts_b = (self.a_trusts_b + TRUST_LEARN_RATE * signal)
            .clamp(0.05, 0.95);
    }

    pub fn update_b_trusts_a(&mut self, vfe_drop: f64, approved: bool) {
        let signal = if approved {
            vfe_drop.min(2.0) / 2.0
        } else {
            -0.2
        };
        self.b_trusts_a = (self.b_trusts_a + TRUST_LEARN_RATE * signal)
            .clamp(0.05, 0.95);
    }
}

#[derive(Debug, Clone)]
pub struct SocialTurn {
    pub turn:          usize,
    pub question:      String,
    pub response:      String,
    pub asker:         String,
    pub responder:     String,
    pub approved:      bool,
    pub vfe_drop:      f64,
    pub soul_distance: f64,
    pub trust_ab:      f64,
    pub trust_ba:      f64,
}

pub struct SocialisedSession {
    pub trust:   TrustState,
    pub history: Vec<SocialTurn>,
}

impl SocialisedSession {
    pub fn new() -> Self {
        Self {
            trust:   TrustState::new(),
            history: Vec::new(),
        }
    }

    pub fn run(
        &mut self,
        instance_a: &mut Quorum,
        instance_b: &mut Quorum,
        turns:      usize,
    ) {
        let turns = turns.min(MAX_TURNS);

        println!("\n[Social] ═══════════════════════════════════════");
        println!("[Social] Socialisation session starting. Turns: {}", turns);
        println!("[Social] Instance A: {} | Instance B: {}",
            instance_a.instance_name, instance_b.instance_name);
        println!("[Social] ═══════════════════════════════════════\n");

        for turn in 0..turns {
            println!("[Social] ── Turn {} ──────────────────────────────", turn + 1);

            // Instance A generates question from its soul state
            let gen_a    = QuestionGenerator::new(&instance_a.instance_name);
            let soul_a   = instance_a.omni_soul().clone();
            let question = gen_a.generate(&soul_a, instance_a);

            println!("[Social] Instance A asks: {}", question);

            // Measure omni soul distance before exchange
            let soul_b_before  = instance_b.omni_soul().clone();
            let soul_dist_before = geodesic_distance(
                &soul_a, &soul_b_before, INITIAL_CURVATURE,
            );

            // Instance B answers through full quorum pipeline
            let result_b = instance_b.ask(&question);
            let approved = !result_b.response.starts_with("[Neo Cortical Mesh blocked");

            // Use hull area as trust signal — grows as session covers
            // more semantic ground, indicating productive exchange
            let vfe_drop = if result_b.session.hull_area > 0.0 {
                result_b.session.hull_area.min(2.0)
            } else {
                0.5
            };

            println!("[Social] Instance B responds: {}",
                &result_b.response[..result_b.response.len().min(200)]);
            println!("[Social] Approved: {} Soul dist before: {:.4}",
                approved, soul_dist_before);

            // Update A's trust in B
            self.trust.update_a_trusts_b(vfe_drop, approved);

            // Instance A reflects silently — VFE update only, no full pipeline
            if approved && !result_b.response.is_empty() {
                // Cap reflection text to 80 chars so attractor names stay readable
                let reflection_text = if result_b.response.len() > 80 {
                    format!("{}…", &result_b.response[..80])
                } else {
                    result_b.response.clone()
                };
                instance_a.reflect_silent(&reflection_text);
                self.trust.update_b_trusts_a(vfe_drop, true);
            }
            // Measure omni soul distance after exchange
            let soul_a_after = instance_a.omni_soul().clone();
            let soul_b_after = instance_b.omni_soul().clone();
            let soul_dist_after = geodesic_distance(
                &soul_a_after, &soul_b_after, INITIAL_CURVATURE,
            );
            let nf_a = compute_nf(&soul_a_after);
            let nf_b = compute_nf(&soul_b_after);

            println!("[Social] Soul dist after: {:.4} (delta: {:.4})",
                soul_dist_after,
                soul_dist_after - soul_dist_before);
            println!("[Social] Trust A->B: {:.3} Trust B->A: {:.3}",
                self.trust.a_trusts_b, self.trust.b_trusts_a);
            println!("[Social] NF A: {:.4} NF B: {:.4}", nf_a, nf_b);

            self.history.push(SocialTurn {
                turn,
                question:      question.clone(),
                response:      result_b.response.clone(),
                asker:         instance_a.instance_name.clone(),
                responder:     instance_b.instance_name.clone(),
                approved,
                vfe_drop,
                soul_distance: soul_dist_after,
                trust_ab:      self.trust.a_trusts_b,
                trust_ba:      self.trust.b_trusts_a,
            });

            println!();
        }

        self.print_summary();
    }

    fn print_summary(&self) {
        println!("\n[Social] ═══════════════════════════════════════");
        println!("[Social] Session complete. {} turns.", self.history.len());

        let approved   = self.history.iter().filter(|t| t.approved).count();
        let first_dist = self.history.first().map(|t| t.soul_distance).unwrap_or(0.0);
        let last_dist  = self.history.last().map(|t| t.soul_distance).unwrap_or(0.0);
        let delta      = last_dist - first_dist;
        let direction  = if delta < -0.01 { "converging" }
                         else if delta > 0.01 { "diverging" }
                         else { "stable" };

        println!("[Social] Approved responses: {}/{}", approved, self.history.len());
        println!("[Social] Soul distance: {:.4} -> {:.4} ({}) delta={:.4}",
            first_dist, last_dist, direction, delta);
        println!("[Social] Final trust A->B: {:.3}", self.trust.a_trusts_b);
        println!("[Social] Final trust B->A: {:.3}", self.trust.b_trusts_a);
        println!("[Social] ═══════════════════════════════════════\n");
    }
}
