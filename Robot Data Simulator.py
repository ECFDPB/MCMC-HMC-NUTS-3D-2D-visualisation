import numpy as np

# Default values
theta_true = np.array([0.1285, 0.0534, 0.0013, 0.000125])
prior_mean = np.zeros(4)
prior_cov = np.diag([1.2, 1.2, 0.006, 0.006])
Psi = np.diag([50, 50, 0.5, 0.5])
Lambda = np.diag([0.01, 0.01])
gamma = 0.99

# Simulation with errors
def simulate_robot_errors(theta):
    kpx, kpy, kdx, kdy = theta
    t_total = 5.0
    dt = 0.001
    t = np.arange(0, t_total, dt)
    n = len(t)

    # The ideal trace should be a circle
    x_ref = 20 * np.cos(2 * np.pi * t / 5)
    y_ref = 20 * np.sin(2 * np.pi * t / 5)
    dx_ref_dt = -20 * (2 * np.pi / 5) * np.sin(2 * np.pi * t / 5)
    dy_ref_dt = 20 * (2 * np.pi / 5) * np.cos(2 * np.pi * t / 5)

    # The observed data
    rewards = []

    x_actual = np.zeros(n)
    y_actual = np.zeros(n)
    x_actual[0] = 20.0
    y_actual[0] = 0.0
    dx_actual_dt = np.zeros(n)
    dy_actual_dt = np.zeros(n)
    dx_actual_dt[0] = dx_ref_dt[0]
    dy_actual_dt[0] = dy_ref_dt[0]

    for i in range(1, n):
        ex = x_ref[i - 1] - x_actual[i - 1] + np.random.normal(0, 0.5)
        ey = y_ref[i - 1] - y_actual[i - 1] + np.random.normal(0, 0.5)
        dex = dx_ref_dt[i - 1] - dx_actual_dt[i - 1] + np.random.normal(0, 0.1)
        dey = dy_ref_dt[i - 1] - dy_actual_dt[i - 1] + np.random.normal(0, 0.1)
        s = np.array([ex, ey, dex, dey])

        Fx = ex * kpx + dex * kdx
        Fy = ey * kpy + dey * kdy
        a = np.array([Fx, Fy])

        r = -s @ Psi @ s - a @ Lambda @ a
        rewards.append(r)

        dx_actual_dt[i] = dx_actual_dt[i - 1] + Fx * dt * 0.1
        dy_actual_dt[i] = dy_actual_dt[i - 1] + Fy * dt * 0.1
        x_actual[i] = x_actual[i - 1] + dx_actual_dt[i] * dt
        y_actual[i] = y_actual[i - 1] + dy_actual_dt[i] * dt

        if x_actual[i] < -110 or x_actual[i] > 110 or y_actual[i] < 170 or y_actual[i] > 330:
            return -1e8

    # Calculate J(theta)
    R = 0
    for i, r in enumerate(rewards):
        R += (gamma ** i) * r
    J = np.exp(R)
    return J

# Simulation without errors
def simulate_robot_accurate(theta):
    kpx, kpy, kdx, kdy = theta
    t_total = 5.0
    dt = 0.001
    t = np.arange(0, t_total, dt)
    n = len(t)

    # The ideal trace should be a circle
    x_ref = 20 * np.cos(2 * np.pi * t / 5)
    y_ref = 20 * np.sin(2 * np.pi * t / 5)
    dx_ref_dt = -20 * (2 * np.pi / 5) * np.sin(2 * np.pi * t / 5)
    dy_ref_dt = 20 * (2 * np.pi / 5) * np.cos(2 * np.pi * t / 5)

    # The observed data
    rewards = []

    x_actual = np.zeros(n)
    y_actual = np.zeros(n)
    x_actual[0] = 20.0
    y_actual[0] = 0.0
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

        dx_actual_dt[i] = dx_actual_dt[i - 1] + Fx * dt * 0.1
        dy_actual_dt[i] = dy_actual_dt[i - 1] + Fy * dt * 0.1
        x_actual[i] = x_actual[i - 1] + dx_actual_dt[i] * dt
        y_actual[i] = y_actual[i - 1] + dy_actual_dt[i] * dt

        if x_actual[i] < -110 or x_actual[i] > 110 or y_actual[i] < 170 or y_actual[i] > 330:
            return -1e8

    # Calculate J(theta)
    R = 0
    for i, r in enumerate(rewards):
        R += (gamma ** i) * r
    J = np.exp(R)
    return J

# Calculate prior distribution
def prior_prob(theta):
    diff = theta - prior_mean
    return np.exp(-0.5 * diff @ np.linalg.inv(prior_cov) @ diff)

# Calculate the relative value of the posterior function, with errors
def posterior_relative_errors(theta):
    J = simulate_robot_errors(theta)
    if J == -1e8:
        return 0.0
    return prior_prob(theta) * J

# Calculate the relative value of the posterior function, without errors
def posterior_relative_accurate(theta):
    J = simulate_robot_accurate(theta)
    if J == -1e8:
        return 0.0
    return prior_prob(theta) * J