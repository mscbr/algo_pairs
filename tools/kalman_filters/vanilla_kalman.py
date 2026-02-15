import numpy as np


class VanillaKalmanFilter:

    def __init__(self, delta=1e-4, R=1e-3):
        # measurement noise variance
        self.R = R

        # co-variance of process noise(2 dimensions)
        self.Q = delta / (1-delta) * np.eye(2)

        # current state (slope, intercept) — 2x1 column vector
        self.x = np.zeros((2, 1))

        # state covariance
        self.P = np.ones((2, 2))

    def step_forward(self, y1, y2):
        # Before entering the equations, let's define H as (1, 2) matrix
        H = np.array([y2, 1])[None]
        # and define z
        z = y1

        ## TIME UPDATE ##
        # first thing is to predict new state as the previous one (2x1)
        x_hat = self.x

        # then, the uncertainty or covariance prediction
        P_hat = self.P + self.Q

        ## MEASUREMENT UPDATE ##
        # calc the Kalman gain
        K = P_hat.dot(H.T)/(H.dot(P_hat.dot(H.T))+self.R)

        # state update part 1 (measurement estimation)
        z_hat = H.dot(x_hat)
        # state update part 2
        x = x_hat + K.dot(z-z_hat)

        # uncertainty update
        self.P = (np.eye(2)-K.dot(H)).dot(P_hat)

        # store current state only
        self.x = x

        return x

    def regression(self, series1, series2):
        n = series1.shape[0]
        state_means = np.empty((n, 2), dtype="float64")

        for t in range(n):
            x = self.step_forward(series1[t], series2[t])
            state_means[t, 0] = x[0, 0]
            state_means[t, 1] = x[1, 0]

        hedge_ratio = - state_means[:, 0]
        spread = series1 + (series2 * hedge_ratio)

        return spread, hedge_ratio
