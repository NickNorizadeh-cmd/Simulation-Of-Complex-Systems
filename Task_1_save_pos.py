import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------
# Parameters for the Lennard-Jones gas
# -----------------------------
m = 1       # Mass (units of m0)
sigma = 1   # Size (units of sigma0)
eps = 1     # Energy (unit of epsilon0)
v0 = 1      # Initial speed (units of v0 = sqrt((2 * epsilon0) / m0))

# Simulation parameters
N_particles = 100
dt = 0.001   # Time step (units of t0 = sigma * sqrt(m0 /(2 * epsilon0)))
cutoff_radius = 5 * sigma

# -----------------------------
# Neighbor list
# -----------------------------
def list_neighbours(x, y, N_particles, cutoff_radius):
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)
        neighbor_indices = np.where(distances <= cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number

# -----------------------------
# Force calculation
# -----------------------------
def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                r2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                r = np.sqrt(r2)
                ka2 = sigma ** 2 / r2
                F = 24 * epsilon / r * (2 * ka2 ** 6 - ka2 ** 3)
                Fx[i] += F * (x[i] - x[j]) / r
                Fy[i] += F * (y[i] - y[j]) / r
    return Fx, Fy

# -----------------------------
# Simulation runner
# -----------------------------
def run_simulation(L, steps=10000):
    x0, y0 = np.meshgrid(
        np.linspace(-L/2, L/2, int(np.sqrt(N_particles))),
        np.linspace(-L/2, L/2, int(np.sqrt(N_particles))),
    )
    x0 = x0.flatten()[:N_particles]
    y0 = y0.flatten()[:N_particles]
    phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi

    vx = v0 * np.cos(phi0)
    vy = v0 * np.sin(phi0)

    neighbours, neighbour_number = list_neighbours(x0, y0, N_particles, cutoff_radius)

    x, y = x0.copy(), y0.copy()
    step = 0
    position_x = []
    position_y = []

    x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

    running = True
    while running:
        x_half = x + 0.5 * vx * dt
        y_half = y + 0.5 * vy * dt

        fx, fy = total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)

        nvx = vx + fx / m * dt
        nvy = vy + fy / m * dt

        nx = x_half + 0.5 * nvx * dt
        ny = y_half + 0.5 * nvy * dt

        # Reflecting boundary conditions
        for j in range(N_particles):
            if nx[j] < x_min:
                nx[j] = x_min + (x_min - nx[j])
                nvx[j] = -nvx[j]
            if nx[j] > x_max:
                nx[j] = x_max - (nx[j] - x_max)
                nvx[j] = -nvx[j]
            if ny[j] < y_min:
                ny[j] = y_min + (y_min - ny[j])
                nvy[j] = -nvy[j]
            if ny[j] > y_max:
                ny[j] = y_max - (ny[j] - y_max)
                nvy[j] = -nvy[j]

        # Update neighbour list occasionally
        if step % 10 == 0:
            neighbours, neighbour_number = list_neighbours(nx, ny, N_particles, cutoff_radius)

        # Update variables
        x, y = nx, ny
        vx, vy = nvx, nvy

        # Save trajectory of particle 44
        position_x.append(x[44])
        position_y.append(y[44])

        if step >= steps:
            running = False
        step += 1

    # Save trajectory arrays with L in filename
    np.savez(f"trajectory_L{int(L/sigma)}sigma.npz",
             position_x=np.array(position_x),
             position_y=np.array(position_y))

    print(f"Saved trajectory for L={L/sigma}σ to trajectory_L{int(L/sigma)}sigma.npz")

    return position_x, position_y, x, y

# -----------------------------
# Run for multiple box sizes
# -----------------------------
for L in [10 * sigma, 16 * sigma]:
    run_simulation(L)
