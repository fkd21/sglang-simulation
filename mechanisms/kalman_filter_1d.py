"""Lightweight 1D Kalman Filter for metric prediction.

This module provides a simple Kalman Filter implementation for tracking
scalar metrics with velocity estimation, using pure Python (no external dependencies).

State vector: [value, velocity]
- value: current metric estimate
- velocity: rate of change per time step

The filter uses a constant velocity model: value(t+1) = value(t) + velocity * dt
"""

from typing import Tuple


class KalmanFilter1D:
    """Lightweight 1D Kalman Filter for time series prediction.

    Tracks a scalar metric and its rate of change (velocity) using
    a constant velocity model. Suitable for predicting metrics like
    alpha values and memory usage.

    No external dependencies - uses pure Python with list-based 2x2 matrices.

    Example:
        >>> kf = KalmanFilter1D(initial_value=0.0, process_noise=0.01,
        ...                     measurement_noise=0.1, dt=5.0)
        >>> for measurement in [0.5, 0.6, 0.7, 0.8]:
        ...     kf.update(measurement)
        >>> predicted, velocity = kf.predict(steps_ahead=3)
        >>> print(f"Predicted: {predicted:.2f}, Velocity: {velocity:.3f}")
    """

    def __init__(self,
                 initial_value: float = 0.0,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 dt: float = 5.0):
        """Initialize Kalman Filter.

        Args:
            initial_value: Initial state estimate for the metric value
            process_noise: Process noise covariance (Q). Lower values make
                          predictions smoother but less responsive. Typical: 0.001-0.1
            measurement_noise: Measurement noise variance (R). Higher values
                              mean less trust in measurements. Typical: 0.01-1.0
            dt: Time step between measurements in seconds (default: 5.0)
        """
        # State vector: [value, velocity]
        self.state = [initial_value, 0.0]

        # State covariance matrix P (2x2)
        # Initial uncertainty is high
        self.P = [[1.0, 0.0],
                  [0.0, 1.0]]

        # Process noise covariance Q (2x2)
        # Velocity has lower process noise than value
        self.Q = [[process_noise, 0.0],
                  [0.0, process_noise * 0.1]]

        # Measurement noise variance R (scalar)
        self.R = measurement_noise

        # Time step (seconds)
        self.dt = dt

        # Tracking
        self.initialized = False
        self.measurement_count = 0

    def predict(self, steps_ahead: int = 1) -> Tuple[float, float]:
        """Predict metric value N steps into the future.

        Uses linear extrapolation based on current value and velocity:
        predicted_value = current_value + velocity * dt * steps_ahead

        Args:
            steps_ahead: Number of time steps to predict ahead (default: 1)

        Returns:
            Tuple of (predicted_value, predicted_velocity)

        Example:
            >>> kf.predict(steps_ahead=3)  # Predict 3 time steps ahead
            (0.95, 0.05)
        """
        if not self.initialized:
            return (0.0, 0.0)

        # Constant velocity model: x(t+k) = x(t) + v*dt*k
        value = self.state[0] + self.state[1] * self.dt * steps_ahead
        velocity = self.state[1]  # Velocity assumed constant

        return (value, velocity)

    def update(self, measurement: float) -> None:
        """Update filter with new measurement.

        Performs one iteration of Kalman filter prediction and update steps:
        1. Predict: Extrapolate state forward in time
        2. Update: Correct prediction using measurement

        Args:
            measurement: Observed metric value at current time step

        Example:
            >>> kf.update(0.85)  # Feed new alpha measurement
        """
        self.measurement_count += 1

        # First measurement: initialize state directly
        if not self.initialized:
            self.state[0] = measurement
            self.state[1] = 0.0
            self.initialized = True
            return

        # ===== PREDICTION STEP =====
        # State transition matrix F (constant velocity model)
        # [1  dt] [value]     [value + velocity*dt]
        # [0   1] [velocity] = [velocity          ]
        F = [[1.0, self.dt],
             [0.0, 1.0]]

        # Predicted state: x_pred = F * x
        x_pred = [
            F[0][0] * self.state[0] + F[0][1] * self.state[1],
            F[1][0] * self.state[0] + F[1][1] * self.state[1]
        ]

        # Predicted covariance: P_pred = F * P * F^T + Q
        P_pred = self._matrix_multiply_2x2(
            self._matrix_multiply_2x2(F, self.P),
            self._transpose_2x2(F)
        )
        P_pred = self._matrix_add_2x2(P_pred, self.Q)

        # ===== UPDATE STEP =====
        # Observation matrix H = [1, 0] (we only observe value, not velocity)
        # Innovation (measurement residual): y = z - H * x_pred
        innovation = measurement - x_pred[0]

        # Innovation covariance: S = H * P_pred * H^T + R
        # Since H = [1, 0], this simplifies to: S = P_pred[0][0] + R
        S = P_pred[0][0] + self.R

        # Kalman gain: K = P_pred * H^T * S^-1
        # K is a 2x1 vector: [K0, K1]
        K = [P_pred[0][0] / S, P_pred[1][0] / S]

        # State update: x = x_pred + K * y
        self.state[0] = x_pred[0] + K[0] * innovation
        self.state[1] = x_pred[1] + K[1] * innovation

        # Covariance update: P = (I - K*H) * P_pred
        # I - K*H = [[1-K[0], 0], [-K[1], 1]]
        self.P = [
            [(1 - K[0]) * P_pred[0][0], (1 - K[0]) * P_pred[0][1]],
            [-K[1] * P_pred[0][0] + P_pred[1][0], -K[1] * P_pred[0][1] + P_pred[1][1]]
        ]

    def get_state(self) -> Tuple[float, float]:
        """Get current state estimate.

        Returns:
            Tuple of (current_value, current_velocity)

        Example:
            >>> value, velocity = kf.get_state()
            >>> print(f"Value: {value:.2f}, Velocity: {velocity:.4f}")
        """
        return (self.state[0], self.state[1])

    def get_velocity_magnitude(self) -> float:
        """Get absolute magnitude of velocity.

        Useful for detecting rapid changes in metric regardless of direction.

        Returns:
            Absolute value of velocity (always non-negative)

        Example:
            >>> abs_velocity = kf.get_velocity_magnitude()
            >>> if abs_velocity > 0.05:
            ...     print("Rapid change detected!")
        """
        return abs(self.state[1])

    # ========== Helper Methods for 2x2 Matrix Operations ==========

    def _matrix_multiply_2x2(self, A, B):
        """Multiply two 2x2 matrices: C = A * B.

        Args:
            A: 2x2 matrix as list of lists [[a11, a12], [a21, a22]]
            B: 2x2 matrix as list of lists [[b11, b12], [b21, b22]]

        Returns:
            2x2 matrix C = A * B
        """
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]

    def _transpose_2x2(self, A):
        """Transpose a 2x2 matrix: A^T.

        Args:
            A: 2x2 matrix as list of lists [[a11, a12], [a21, a22]]

        Returns:
            2x2 matrix A^T = [[a11, a21], [a12, a22]]
        """
        return [[A[0][0], A[1][0]],
                [A[0][1], A[1][1]]]

    def _matrix_add_2x2(self, A, B):
        """Add two 2x2 matrices: C = A + B.

        Args:
            A: 2x2 matrix as list of lists
            B: 2x2 matrix as list of lists

        Returns:
            2x2 matrix C = A + B
        """
        return [
            [A[0][0] + B[0][0], A[0][1] + B[0][1]],
            [A[1][0] + B[1][0], A[1][1] + B[1][1]]
        ]
