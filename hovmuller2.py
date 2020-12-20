import numpy as np
import poisson as ps
import rossby as rb
import matplotlib.pyplot as plt

L = 1.0
dx = 1/40.0
N = int(L/dx)

x_periodic = np.linspace(0.0, L, N + 1)[:-1]
x_Dirichlet = np.linspace(0.0, L, N + 1)[1:-1]

dt = 1/10.0
T = 150.0
M = int(T/dt)

t = np.linspace(0.0, T, M + 1)

sigma = 0.1


# Gaussian function, periodic

x = x_periodic
hovmuller = np.zeros((M + 1, len(x)))

psi0 = np.exp(-(x - x[0])**2/sigma)
hovmuller[-1, :] = psi0

psi = np.copy(psi0)
for i in range(1, M + 1):
	psi = rb.step_Rossby_central(psi, dx, dt, periodic=True)
	hovmuller[-i - 1, :] = psi

plt.contourf(x, t, hovmuller)
plt.title("Hovmuller diagram: $e^{-(x - x_0)^2/\\sigma}$, periodic")
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.show()


# Gaussian function, bounded

x = x_Dirichlet
hovmuller = np.zeros((M + 1, len(x)))

psi0 = np.exp(-(x - x[0])**2/sigma)
hovmuller[-1, :] = psi0

psi = np.copy(psi0)
for i in range(1, M + 1):
	psi = rb.step_Rossby_central(psi, dx, dt)
	hovmuller[-i - 1, :] = psi

plt.contourf(x, t, hovmuller)
plt.title("Hovmuller diagram: $e^{-(x - x_0)^2/\\sigma}$, bounded")
plt.xlabel("$x$")
plt.ylabel("$t$")
plt.show()