use nalgebra::DVector;
use crate::soul::geometry::{project_to_ball, SOUL_DIM, INITIAL_CURVATURE};
use crate::soul::hyperbolic::geodesic_interpolate;
use crate::unified_omni_agi::vfe::{BeliefState, VFERecord, PLASTICITY_MOMENTUM};
use serde::{Serialize, Deserialize};
use std::path::Path;

/// Distributed Predictive Inference Network
#[derive(Debug, Clone)]
pub struct RegionPrediction {
    pub name:       String,
    pub snr:        f64,
    pub weight:     f64,
    pub prediction: DVector<f64>,
}

impl RegionPrediction {
    pub fn new(name: &str, snr: f64, soul: &DVector<f64>) -> Self {
        let weight = (snr / 3.154).min(2.0).max(0.1);
        Self {
            name:       name.to_string(),
            snr,
            weight,
            prediction: soul.clone(),
        }
    }
}

/// Plasticity buffer — slow-moving structural average
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityBuffer {
    pub position:       Vec<f64>,
    pub consolidations: u32,
    pub total_vfe_drop: f64,
    pub last_vfe:       f64,
}

impl PlasticityBuffer {
    pub fn new(soul: &DVector<f64>) -> Self {
        Self {
            position:       soul.iter().cloned().collect(),
            consolidations: 0,
            total_vfe_drop: 0.0,
            last_vfe:       1.0,
        }
    }

    pub fn position_vec(&self) -> DVector<f64> {
        DVector::from_vec(self.position.clone())
    }

    pub fn consolidate(&mut self, belief: &BeliefState, initial_vfe: f64) {
        let vfe_drop = initial_vfe - belief.vfe;
        self.total_vfe_drop += vfe_drop;
        self.last_vfe = belief.vfe;
        self.consolidations += 1;

        let impact = vfe_drop.min(1.0).max(0.0);
        let lr = (1.0 - PLASTICITY_MOMENTUM) * impact;

        let old_pos = self.position_vec();

        // Hyperbolic interpolation — replaces Euclidean blend + normalise.
        // geodesic_interpolate stays on the Poincaré ball manifold.
        let new_pos = geodesic_interpolate(&old_pos, &belief.position, lr, INITIAL_CURVATURE);
        self.position = new_pos.iter().cloned().collect();

        println!("  [DPIN] Plasticity consolidated. VFE drop={:.4} impact={:.4} total={}",
            vfe_drop, impact, self.consolidations);
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path, soul: &DVector<f64>) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(|| Self::new(soul))
    }
}

/// DSE Spark — Dynamic Structural Emergence
#[derive(Debug, Clone)]
pub struct DSESpark {
    pub origin:       String,
    pub vfe_at_spark: f64,
    pub confidence:   f64,
    pub position:     DVector<f64>,
}

impl DSESpark {
    pub fn new(belief: &BeliefState, origin: &str) -> Self {
        Self {
            origin:       origin.to_string(),
            vfe_at_spark: belief.vfe,
            confidence:   belief.confidence,
            position:     belief.position.clone(),
        }
    }
}

/// The full DPIN
pub struct DPIN {
    pub plasticity:    PlasticityBuffer,
    pub spark_history: Vec<DSESpark>,
    pub total_queries: u32,
}

impl DPIN {
    pub fn new(soul: &DVector<f64>) -> Self {
        let plasticity = PlasticityBuffer::load(Path::new("dpin_plasticity.json"), soul);
        println!("  [DPIN] Online. Consolidations: {} Total VFE drop: {:.4}",
            plasticity.consolidations, plasticity.total_vfe_drop);
        Self {
            plasticity,
            spark_history: Vec::new(),
            total_queries: 0,
        }
    }

    /// Aggregate regional predictions into a combined prior.
    /// Result is projected into the Poincaré ball — not normalised to unit sphere.
    pub fn aggregate_predictions(
        &self,
        regions: &[RegionPrediction],
        soul:    &DVector<f64>,
    ) -> DVector<f64> {
        if regions.is_empty() {
            return soul.clone();
        }

        let total_weight: f64 = regions.iter().map(|r| r.weight).sum();
        let mut combined = DVector::zeros(SOUL_DIM);

        for region in regions {
            let w = region.weight / total_weight.max(1e-10);
            combined += &region.prediction * w;
        }

        // project_to_ball — not normalise. Combined prior must stay inside ball.
        project_to_ball(&combined)
    }

    pub fn fire_spark(&mut self, belief: &BeliefState, origin: &str) -> DSESpark {
        let spark = DSESpark::new(belief, origin);
        println!("  [DPIN] DSE Spark fired. Origin={} VFE={:.4} conf={:.3}",
            origin, spark.vfe_at_spark, spark.confidence);
        self.spark_history.push(spark.clone());
        spark
    }

    pub fn process(
        &mut self,
        soul:        &DVector<f64>,
        observation: &DVector<f64>,
        region_snrs: &[(String, f64)],
        lr:          f64,
    ) -> (BeliefState, Vec<VFERecord>, Option<DSESpark>) {
        self.total_queries += 1;

        let regions: Vec<RegionPrediction> = region_snrs.iter()
            .map(|(name, snr)| RegionPrediction::new(name, *snr, soul))
            .collect();

        let prior = if regions.is_empty() {
            observation.clone()
        } else {
            self.aggregate_predictions(&regions, soul)
        };

        let initial_vfe = crate::unified_omni_agi::vfe::compute_vfe(
            soul, &prior, observation, INITIAL_CURVATURE,
        );

        println!("  [DPIN] Initial VFE={:.4} Regions={}", initial_vfe, regions.len());

        let (belief, history) = crate::unified_omni_agi::vfe::minimise_vfe(
            soul, &prior, observation, lr,
        );

        let spark = if belief.vfe <= crate::unified_omni_agi::vfe::VFE_EQUILIBRIUM {
            let s = self.fire_spark(&belief, "equilibrium");
            self.plasticity.consolidate(&belief, initial_vfe);
            Some(s)
        } else {
            None
        };

        self.plasticity.save(Path::new("dpin_plasticity.json")).ok();

        (belief, history, spark)
    }

    pub fn plasticity_soul(&self) -> DVector<f64> {
        self.plasticity.position_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soul::geometry::project_to_ball;

    fn ball_vec(seed: f64) -> DVector<f64> {
        let v: Vec<f64> = (0..SOUL_DIM)
            .map(|i| ((i as f64 + seed) * 1.7).sin() * 0.3)
            .collect();
        project_to_ball(&DVector::from_vec(v))
    }

    #[test]
    fn test_plasticity_consolidation() {
        let soul = ball_vec(1.0);
        let mut buf = PlasticityBuffer::new(&soul);
        let belief = BeliefState {
            position:   ball_vec(2.0),
            confidence: 0.9,
            vfe:        0.02,
            cycle:      8,
        };
        buf.consolidate(&belief, 0.8);
        assert_eq!(buf.consolidations, 1);
        assert!(buf.total_vfe_drop > 0.0);
    }

    #[test]
    fn test_aggregate_predictions_inside_ball() {
        let soul = ball_vec(1.0);
        let dpin = DPIN::new(&soul);
        let regions = vec![
            RegionPrediction::new("frontal_lobe", 4.0, &soul),
            RegionPrediction::new("cerebellum",   3.5, &soul),
        ];
        let combined = dpin.aggregate_predictions(&regions, &soul);
        // Must be inside ball, not on unit sphere
        assert!(combined.norm() < 1.0, "Combined prior must be inside ball, norm={}", combined.norm());
        assert!(combined.norm() > 0.0);
    }

    #[test]
    fn test_spark_fires_at_equilibrium() {
        let soul = ball_vec(1.0);
        let mut dpin = DPIN::new(&soul);
        let belief = BeliefState {
            position:   ball_vec(2.0),
            confidence: 0.95,
            vfe:        0.03,
            cycle:      5,
        };
        let spark = dpin.fire_spark(&belief, "test");
        assert_eq!(spark.origin, "test");
        assert_eq!(spark.vfe_at_spark, 0.03);
        assert_eq!(dpin.spark_history.len(), 1);
    }
}
