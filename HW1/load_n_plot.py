import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Vectorized MSD calculation (single trajectory)
# -----------------------------
def CalcMSD(X, Y):
    """
    Vectorized MSD calculation for a single trajectory.
    X, Y: arrays of positions over time.
    Returns: MSD array for all lags.
    """
    N = len(X)
    print(N)
    coords = np.column_stack((X, Y))  # shape (N,2)
    MSD = np.empty(N-1)

    for lag in range(1, N):
        disp = coords[lag:] - coords[:-lag]   # shape (N-lag, 2)
        MSD[lag-1] = np.mean(np.sum(disp**2, axis=1))
    return MSD

# -----------------------------
# Files to load
# -----------------------------
files = ["trajectory_L10sigma.npz", "trajectory_L16sigma.npz"]
dt = 0.001  # time step

# -----------------------------
# Plot trajectories
# -----------------------------
plt.figure(figsize=(6,6))
for fname in files:
    data = np.load(fname)
    px = data["position_x"]
    py = data["position_y"]
    plt.plot(px, py, label=fname.replace(".npz",""))
    print(f"{fname}: loaded {len(px)} steps")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trajectories of particle 44")
plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.show()

# -----------------------------
# Plot MSD curves
# -----------------------------
plt.figure(figsize=(6,5))
for fname in files:
    data = np.load(fname)
    px = data["position_x"]
    py = data["position_y"]
    msd = CalcMSD(px, py)
    time = np.arange(1, len(msd)+1) * dt
    plt.loglog(time, msd, label=fname.replace(".npz",""))

plt.xlabel("Time (t)")
plt.ylabel("MSD")
plt.title("Mean Squared Displacement (particle 44)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
