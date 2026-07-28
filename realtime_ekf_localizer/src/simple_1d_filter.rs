//! Port of `Simple1DFilter` (`ekf_module.hpp`) — the scalar Kalman filter used for the z,
//! roll, and pitch channels that the 6-state EKF does not carry.

/// Scalar Kalman filter with lazy first-observation initialization.
#[derive(Clone, Copy, Debug)]
pub struct Simple1DFilter {
    initialized: bool,
    x: f64,
    var: f64,
    proc_var_x_c: f64,
}

impl Default for Simple1DFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Simple1DFilter {
    /// Construct uninitialized (`x = 0`, `var = 1e9`, no process noise).
    #[must_use]
    pub fn new() -> Self {
        Self {
            initialized: false,
            x: 0.0,
            var: 1e9,
            proc_var_x_c: 0.0,
        }
    }

    /// Initialize from an observation and its variance.
    pub fn init(&mut self, init_obs: f64, obs_var: f64) {
        self.x = init_obs;
        self.var = obs_var;
        self.initialized = true;
    }

    /// Predict (inflate variance by `proc_var * dt²`) and blend in an observation. The first
    /// call initializes instead.
    pub fn update(&mut self, obs: f64, obs_var: f64, dt: f64) {
        if !self.initialized {
            self.init(obs, obs_var);
            return;
        }

        // Prediction step (current variance).
        let proc_var_x_d = self.proc_var_x_c * dt * dt;
        self.var += proc_var_x_d;

        // Update step.
        let kalman_gain = self.var / (self.var + obs_var);
        self.x += kalman_gain * (obs - self.x);
        self.var *= 1.0 - kalman_gain;
    }

    /// Set the continuous-time process variance.
    pub fn set_proc_var(&mut self, proc_var: f64) {
        self.proc_var_x_c = proc_var;
    }

    /// Current state estimate.
    #[must_use]
    pub fn get_x(&self) -> f64 {
        self.x
    }

    /// Current variance.
    #[must_use]
    pub fn get_var(&self) -> f64 {
        self.var
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::arithmetic_side_effects,
    clippy::allow_attributes,
    reason = "test code"
)]
mod tests {
    use super::*;

    // Transcription of the Simple1DFilter tests in test_ekf_module.cpp.

    #[test]
    fn init_and_update() {
        let mut filter = Simple1DFilter::new();
        // Before init, update() must initialize from the first observation.
        filter.update(5.0, 2.0, 0.1);
        assert_eq!(filter.get_x(), 5.0);
        assert_eq!(filter.get_var(), 2.0);

        // kalman_gain = 2 / (2 + 2) = 0.5; x = 5 + 0.5 * (7 - 5) = 6; var = 0.5 * 2 = 1.
        filter.update(7.0, 2.0, 0.1);
        assert_eq!(filter.get_x(), 6.0);
        assert_eq!(filter.get_var(), 1.0);
    }

    #[test]
    fn process_variance_inflates_prediction() {
        let mut filter = Simple1DFilter::new();
        filter.set_proc_var(4.0);
        filter.init(0.0, 1.0);

        // proc_var_x_d = 4 * 0.25 = 1; var = 2; gain = 0.5; x = 5; var = 1.
        filter.update(10.0, 2.0, 0.5);
        assert_eq!(filter.get_x(), 5.0);
        assert_eq!(filter.get_var(), 1.0);
    }
}
