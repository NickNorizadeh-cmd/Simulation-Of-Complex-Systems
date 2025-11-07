# Parameters for the Lennard-Jones gas.
m = 1  # Mass (units of m0).
sigma = 1  # Size (units of sigma0).
eps = 1  # Energy (unit of epsilon0).
v0 = 1  # Initial speed (units of v0 = sqrt((2 * epsilon0) / m0)).

# Parameters for the simulation.
N_particles = 1  # Number of particles.

dt = 0.001   # Time step (units of t0 = sigma * sqrt(m0 /(2 * epsilon0))).

L = 10 * sigma  # Box size (units of sigma0).
x_min, x_max, y_min, y_max = -L/2, L/2, -L/2, L/2

cutoff_radius = 5 * sigma  # Cutoff_radius for neighbours list.

###################
# Generate initial positions on a grid and orientations at random.
import numpy as np
x0, y0 = np.meshgrid(
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
    np.linspace(- L / 2, L / 2, int(np.sqrt(N_particles))),
)
x0 = x0.flatten()[:N_particles]
y0 = y0.flatten()[:N_particles]
phi0 = (2 * np.random.rand(N_particles) - 1) * np.pi

# plt.scatter(x0,y0) # Plot of initial configuration

# Initialize the neighbour list.
def list_neighbours(x, y, N_particles, cutoff_radius):
    '''Prepare a neigbours list for each particle.'''
    neighbours = []
    neighbour_number = []
    for j in range(N_particles):
        distances = np.sqrt((x - x[j]) ** 2 + (y - y[j]) ** 2)
        neighbor_indices = np.where(distances <= cutoff_radius)
        neighbours.append(neighbor_indices)
        neighbour_number.append(len(neighbor_indices))
    return neighbours, neighbour_number

neighbours, neighbour_number = list_neighbours(x0, y0, N_particles, cutoff_radius)

# Initialize the variables for the leapfrog algorithm.
# Current time step.
x = x0
y = y0
x_half = np.zeros(N_particles)
y_half = np.zeros(N_particles)
v = v0
phi = phi0
vx = v0 * np.cos(phi0)
vy = v0 * np.sin(phi0)

# Next time step.
nx = np.zeros(N_particles)
ny = np.zeros(N_particles)
nv = np.zeros(N_particles)
nphi = np.zeros(N_particles)
nvx = np.zeros(N_particles)
nvy = np.zeros(N_particles)

###################

def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    '''
    Calculate the total force on each particle due to the interaction with a 
    neighbours list with the particles interacting through a Lennard-Jones 
    potential.
    '''
    Fx = np.zeros(N_particles)
    Fy = np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                r2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                r = np.sqrt(r2)
                ka2 = sigma ** 2 / r2
                
                # Force on i due to j.
                F = 24 * epsilon / r * (2 * ka2 ** 6 - ka2 ** 3)  # Modulus.
                
                Fx[i] += F * (x[i] - x[j]) / r
                Fy[i] += F * (y[i] - y[j]) / r
    return Fx, Fy

##############

import time
from scipy.constants import Boltzmann as kB 
from tkinter import *
import math 
import matplotlib.pyplot as plt
window_size = 600


step = 0

position_x = []
position_y = []

def stop_loop(event):
    global running
    running = False
running = True  # Flag to control the loop.
while running:
    x_half = x + 0.5 * vx * dt # r_n_1/2      
    y_half = y + 0.5 * vy * dt
    #print("Running")      

    fx, fy = \
        total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)
    
    nvx = vx + fx / m * dt # v_n+1  
    nvy = vy + fy / m * dt #
        
    nx = x_half + 0.5 * nvx * dt # r_n_1
    ny = y_half + 0.5 * nvy * dt       
    
    # Reflecting boundary conditions.
    for j in range(N_particles):
        if nx[j] < x_min:
            nx[j] = x_min + (x_min - nx[j])
            nvx[j] = - nvx[j]

        if nx[j] > x_max:
            nx[j] = x_max - (nx[j] - x_max)
            nvx[j] = - nvx[j]

        if ny[j] < y_min:
            ny[j] = y_min + (y_min - ny[j])
            nvy[j] = - nvy[j]
            
        if ny[j] > y_max:
            ny[j] = y_max - (ny[j] - y_max)
            nvy[j] = - nvy[j]
    
    nv = np.sqrt(nvx ** 2 + nvy ** 2)
    for i in range(N_particles):
        nphi[i] = math.atan2(nvy[i], nvx[i])
    
    # Update neighbour list.
    if step % 10 == 0:
        neighbours, neighbour_number = \
            list_neighbours(nx, ny, N_particles, cutoff_radius)

    # Update variables for next iteration.
    x = nx
    #print("Shape of x", x.shape)
    y = ny
    vx = nvx
    vy = nvy
    v = nv
    phi = nphi

    position_x.append(x) # Issue is with appending. We should figure out why this is not appending 100 points
    position_y.append(y)

    if step >= 10000:
        running = False

    step += 1

####################

def CalcMSD(X, Y):
    N = len(X)
    MSD = []
    for n in range(1, N):  # n is the lag
        sum_sq_disp = 0
        count = 0
        for i in range(N - n):
            dx = X[i + n] - X[i]
            dy = Y[i + n] - Y[i]
            sum_sq_disp += dx**2 + dy**2
            count += 1
        MSD.append(sum_sq_disp / count)
    return MSD

##################
position_x = np.array(position_x)
#print("Position_x", arr.shape)

#print(CalcMSD(position_x,position_y))
n_list = np.arange(1,10001)
plt.loglog(n_list* dt, CalcMSD(position_x, position_y))
#plt.xlim(1e0, 1e1)  # sets x-axis range from 10^0 to 10^1
#plt.autoscale(enable=True, axis='y', tight=True)  # still autoscale y-axis
plt.show()

