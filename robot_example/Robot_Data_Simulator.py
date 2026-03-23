import numpy as np

# Default values
theta_true = np.array([0.1285, 0.0534])
prior_mean = np.zeros(2)
prior_cov = np.diag([1.2, 1.2])
Psi = np.diag([50, 50, 0.05, 0.05])
Lambda = np.diag([0.01, 0.01])
gamma = 0.99
KDX_FIXED = 0.0013
KDY_FIXED = 0.000125
PRIOR_VAR_KP = 1.2
EPS = 1e-5

# Simulation (deterministic, no noise)
def simulate_robot(theta):
    kpx, kpy = theta
    kdx, kdy = 0.0013, 0.000125
    t_total = 5.0
    dt = 0.01
    t = np.arange(0, t_total, dt)
    n = len(t)

    # The ideal trace should be a circle
    x_ref = 20 * np.cos(2 * np.pi * t / t_total)
    y_ref = 20 * np.sin(2 * np.pi * t / t_total) + 224.15
    dx_ref_dt = -20 * (2 * np.pi / t_total) * np.sin(2 * np.pi * t / t_total)
    dy_ref_dt = 20 * (2 * np.pi / t_total) * np.cos(2 * np.pi * t / t_total)

    rewards = []

    x_actual = np.zeros(n)
    y_actual = np.zeros(n)
    x_actual[0] = 20.0
    y_actual[0] = 224.15
    dx_actual_dt = np.zeros(n)
    dy_actual_dt = np.zeros(n)
    dx_actual_dt[0] = dx_ref_dt[0]
    dy_actual_dt[0] = dy_ref_dt[0]

    for i in range(1, n):
        ex = x_ref[i - 1] - x_actual[i - 1]
        ey = y_ref[i - 1] - y_actual[i - 1]
        dex = dx_ref_dt[i - 1] - dx_actual_dt[i - 1]
        dey = dy_ref_dt[i - 1] - dy_actual_dt[i - 1]
        s = np.array([ex, ey, dex, dey])

        Fx = ex * kpx + dex * kdx
        Fy = ey * kpy + dey * kdy
        a = np.array([Fx, Fy])

        r = -s @ Psi @ s - a @ Lambda @ a
        rewards.append(r)

        dx_actual_dt[i] = dx_actual_dt[i - 1] + Fx * dt
        dy_actual_dt[i] = dy_actual_dt[i - 1] + Fy * dt
        x_actual[i] = x_actual[i - 1] + dx_actual_dt[i] * dt
        y_actual[i] = y_actual[i - 1] + dy_actual_dt[i] * dt

    # Calculate discounted return R(theta), scaled to O(1)
    R = sum((gamma ** i) * r for i, r in enumerate(rewards))
    return R / 5_000_000

# Calculate log-prior (Gaussian, computed directly in log space)
def log_prior(theta):
    diff = theta - prior_mean
    return -0.5 * diff @ np.linalg.inv(prior_cov) @ diff

# log-posterior and its numerical gradient (for HMC leapfrog)
def log_posterior_and_grad(theta):
    theta = np.asarray(theta, dtype=np.float64)
    kpx, kpy = theta

    R = simulate_robot(theta)
    R_x = simulate_robot((kpx + EPS, kpy))
    R_y = simulate_robot((kpx, kpy + EPS))

    log_post = log_prior(theta) + R
    dlogP_dx = -kpx / PRIOR_VAR_KP
    dlogP_dy = -kpy / PRIOR_VAR_KP
    grad = np.array([
        dlogP_dx + (R_x - R) / EPS,
        dlogP_dy + (R_y - R) / EPS
    ])

    return log_post, grad
