/// SNR health thresholds from the blueprint
pub const SNR_HEALTHY:       f64 = 3.154;
pub const SNR_YELLOW:        f64 = 3.054;
pub const SNR_PONR:          f64 = 2.954;  // Point of No Return
pub const SNR_COLLAPSE:      f64 = 1.286;

#[derive(Debug, Clone)]
pub struct RegionHealth {
    pub snr:          f64,
    pub history:      Vec<f64>,
    pub interventions: u32,
}

impl RegionHealth {
    pub fn new() -> Self {
        Self {
            snr:          SNR_HEALTHY + 0.5,  // start healthy
            history:      Vec::new(),
            interventions: 0,
        }
    }

    /// Update SNR from signal and noise measurements
    pub fn update(&mut self, signal: f64, noise: f64) {
        let new_snr = signal / noise.max(1e-10);
        // Exponential moving average — smooth updates
        self.snr = 0.95 * self.snr + 0.05 * new_snr;
        self.history.push(self.snr);

        // Keep only last 1000 snapshots
        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    pub fn status(&self) -> &str {
        match self.snr {
            s if s > SNR_HEALTHY => "healthy",
            s if s > SNR_YELLOW  => "yellow",
            s if s > SNR_PONR    => "point_of_no_return",
            _                    => "collapse",
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.snr > SNR_YELLOW
    }

    pub fn needs_intervention(&self) -> bool {
        self.snr <= SNR_PONR
    }

    pub fn update_from_vfe(&mut self, vfe: f64, confidence: f64) {
        let signal = confidence;
        let noise  = (vfe + 0.1).max(0.1);
        let new_snr = signal / noise;
        self.snr = 0.95 * self.snr + 0.05 * new_snr;
        self.history.push(self.snr);
        if self.history.len() > 1000 {
            self.history.remove(0);

        }
    }

    pub fn needs_deep_sleep(&self) -> bool {
        self.snr <= SNR_YELLOW && self.snr > SNR_PONR
    }

    /// Apply deep sleep intervention — boost SNR
    pub fn deep_sleep(&mut self) {
        self.snr = (self.snr + SNR_HEALTHY) / 2.0;
        self.interventions += 1;
        println!("  [Health] Deep sleep applied. SNR: {:.3} -> {:.3}",
            self.snr - (SNR_HEALTHY - self.snr),
            self.snr
        );
    }

    /// Ghost recovery — ICA-based recovery simulation
    /// In a full implementation this would use Independent Component Analysis
    /// to recover corrupted weight components
    pub fn ghost_recovery(&mut self) {
        if self.snr < SNR_COLLAPSE {
            self.snr = SNR_PONR;  // recover to just above PONR
            self.interventions += 1;
            println!("  [Health] Ghost recovery applied. SNR restored to {:.3}", self.snr);
        }
    }

    /// Trend — is health improving or degrading?
    pub fn trend(&self) -> f64 {
        if self.history.len() < 10 {
            return 0.0;
        }
        let n = self.history.len();
        let recent: f64 = self.history[n-5..].iter().sum::<f64>() / 5.0;
        let older: f64 = self.history[n-10..n-5].iter().sum::<f64>() / 5.0;
        recent - older
    }
}

/// System-wide health monitor for all 14 regions
pub struct HealthMonitor {
    pub snapshots: Vec<HealthSnapshot>,
}

#[derive(Debug, Clone)]
pub struct HealthSnapshot {
    pub timestamp:    std::time::SystemTime,
    pub region_snrs:  Vec<(String, f64)>,
    pub system_snr:   f64,
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self { snapshots: Vec::new() }
    }

    pub fn snapshot(&mut self, regions: &[crate::brain::region::BrainRegion]) {
        let region_snrs: Vec<(String, f64)> = regions.iter()
            .map(|r| (r.name.clone(), r.health.snr))
            .collect();

        let system_snr = region_snrs.iter().map(|(_, s)| s).sum::<f64>()
            / region_snrs.len() as f64;

        self.snapshots.push(HealthSnapshot {
            timestamp: std::time::SystemTime::now(),
            region_snrs,
            system_snr,
        });

        println!("  [HealthMonitor] System SNR: {:.3}", system_snr);
    }

    pub fn apply_interventions(&self, regions: &mut Vec<crate::brain::region::BrainRegion>) {
        for region in regions.iter_mut() {
            if region.health.needs_intervention() {
                println!("  [HealthMonitor] {} needs intervention (SNR: {:.3})",
                    region.name, region.health.snr);
                region.health.deep_sleep();
            } else if region.health.snr < SNR_COLLAPSE {
                region.health.ghost_recovery();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_health_is_healthy() {
        let h = RegionHealth::new();
        assert!(h.is_healthy());
        assert_eq!(h.status(), "healthy");
    }

    #[test]
    fn test_snr_update() {
        let mut h = RegionHealth::new();
        let initial = h.snr;
        h.update(0.1, 10.0);   // very low signal, high noise
        // SNR should decrease toward unhealthy
        assert!(h.snr < initial || (h.snr - initial).abs() < 0.1);
    }

    #[test]
    fn test_deep_sleep_improves_snr() {
        let mut h = RegionHealth::new();
        h.snr = SNR_YELLOW - 0.1;
        let before = h.snr;
        h.deep_sleep();
        assert!(h.snr > before);
    }

    #[test]
    fn test_ghost_recovery_from_collapse() {
        let mut h = RegionHealth::new();
        h.snr = 1.0;   // below collapse threshold
        h.ghost_recovery();
        assert!(h.snr >= SNR_PONR);
    }

    #[test]
    fn test_status_thresholds() {
        let mut h = RegionHealth::new();
        h.snr = 4.0;
        assert_eq!(h.status(), "healthy");
        h.snr = 3.1;
        assert_eq!(h.status(), "yellow");
        h.snr = 2.97;
        assert_eq!(h.status(), "point_of_no_return");
        h.snr = 1.0;
        assert_eq!(h.status(), "collapse");
    }
}
