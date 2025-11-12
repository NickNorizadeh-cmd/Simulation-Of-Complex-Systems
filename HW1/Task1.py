import numpy as np
import math
import matplotlib.pyplot as plt

# --- Parameters ---
m = 1
sigma = 1
eps = 1
v0 = 1
N_particles = 100
dt = 0.001
cutoff_radius = 5 * sigma

# --- Neighbour list ---
def list_neighbours(x, y, N_particles, cutoff_radius):
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j])**2 + (y - y[j])**2)
        neighbor_indices = np.where(distances <= cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number

# --- Force calculation ---
def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                r2 = dx*dx + dy*dy
                r = math.sqrt(r2)
                ka2 = (sigma*sigma) / r2
                F = 24 * epsilon / r * (2 * ka2**6 - ka2**3)
                Fx[i] += F * dx / r
                Fy[i] += F * dy / r
    return Fx, Fy

# --- Vectorized MSD ---
def calc_msd_vectorized(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N = X.size
    msd = np.empty(N-1, dtype=float)
    for n in range(1, N):
        dx = X[n:] - X[:-n]
        dy = Y[n:] - Y[:-n]
        msd[n-1] = np.mean(dx*dx + dy*dy)
    return msd

# --- Simulation function with boundaries set per run ---
def run_simulation(L, steps=10000, track_index=44):
    # Box boundaries specific to this run
    x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

    # Initial positions on grid
    grid_n = int(np.sqrt(N_particles))
    x0, y0 = np.meshgrid(
        np.linspace(-L/2, L/2, grid_n),
        np.linspace(-L/2, L/2, grid_n),
    )
    x0 = x0.flatten()[:N_particles]
    y0 = y0.flatten()[:N_particles]

    # Random orientations
    phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi

    # State variables
    x = x0.copy()
    y = y0.copy()
    vx = v0 * np.cos(phi0)
    vy = v0 * np.sin(phi0)
    phi = phi0.copy()

    neighbours, neighbour_number = list_neighbours(x, y, N_particles, cutoff_radius)

    position_x, position_y = [], []
    step = 0
    running = True

    while running:
        # Half-step positions
        x_half = x + 0.5 * vx * dt
        y_half = y + 0.5 * vy * dt

        # Forces
        fx, fy = total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)

        # Velocity update
        nvx = vx + fx / m * dt
        nvy = vy + fy / m * dt

        # Full positions
        nx = x_half + 0.5 * nvx * dt
        ny = y_half + 0.5 * nvy * dt

        # Reflecting boundaries
        for j in range(N_particles):
            if nx[j] < x_min:
                nx[j] = x_min + (x_min - nx[j]); nvx[j] = -nvx[j]
            if nx[j] > x_max:
                nx[j] = x_max - (nx[j] - x_max); nvx[j] = -nvx[j]
            if ny[j] < y_min:
                ny[j] = y_min + (y_min - ny[j]); nvy[j] = -nvy[j]
            if ny[j] > y_max:
                ny[j] = y_max - (ny[j] - y_max); nvy[j] = -nvy[j]

        # Speed and orientation (vectorized)
        nv = np.sqrt(nvx**2 + nvy**2)
        nphi = np.arctan2(nvy, nvx)

        # Update neighbour list occasionally
        if step % 10 == 0:
            neighbours, neighbour_number = list_neighbours(nx, ny, N_particles, cutoff_radius)

        # Commit new state
        x, y, vx, vy, v, phi = nx, ny, nvx, nvy, nv, nphi

        # Record tracked particle
        position_x.append(x[track_index])
        position_y.append(y[track_index])

        if step % 100 == 0:
            print(step, end="\r")

        if step >= steps:
            running = False
        step += 1

    # Save trajectory
    fname = f"trajectory_L{int(L/sigma)}sigma.npz"
    np.savez(fname, position_x=np.array(position_x), position_y=np.array(position_y))
    print(f"\nSaved trajectory for L={L/sigma}σ to {fname}")

# --- Run for multiple L values ---
L_values = [10*sigma, 16*sigma]
for L in L_values:
    run_simulation(L)

# --- Plotting trajectories and MSD ---
files = [f"trajectory_L{int(L/sigma)}sigma.npz" for L in L_values]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Trajectories
for fname in files:
    data = np.load(fname)
    px = data["position_x"]
    py = data["position_y"]
    axes[0].plot(px, py, label=fname.replace(".npz",""))
axes[0].set_title("Trajectories of particle 44")
axes[0].set_aspect("equal", adjustable="box")
axes[0].legend()

# MSD
downsample = 10
for fname in files:
    data = np.load(fname)
    px = data["position_x"]
    py = data["position_y"]
    msd = calc_msd_vectorized(px, py)
    time = np.arange(1, len(msd)+1) * dt
    axes[1].loglog(time[::downsample], msd[::downsample],
                   label=fname.replace(".npz",""))
axes[1].set_title("Mean Squared Displacement")
axes[1].legend()
axes[1].grid(True, which="both", ls="--")

plt.show()
