# This program generated stream functions evolved in time, computed 
# with two different time-derivative approximations, and plots 
# assorted solutions against each other.

import numpy as np
import poisson as ps
import rossby as rb
import matplotlib.pyplot as plt

L = 1.0
dx = 1/40.0
dt = 1/10.0
N = int(L/dx)

x_periodic = np.linspace(0.0, L, N + 1)[:-1]
x_Dirichlet = np.linspace(0.0, L, N + 2)[1:-1]

sigma = 0.1

times = [50.0, 100.0, 150.0]


# Periodic boundary conditions

x = x_periodic
psi0 = np.sin(4.0*np.pi*x)
plt.plot(x, psi0)
plt.title("Sine function, $t = 0$")
plt.xlabel("$x$")
plt.ylabel("$\\psi(x, t)$")
plt.show()

psi_forward = np.copy(psi0)
psi_central = np.copy(psi0)

i = 0
while (i + 1)*dt <= 150:
	psi_forward = rb.step_Rossby_forward(psi_forward, dx, dt, periodic=True)
	psi_central = rb.step_Rossby_central(psi_central, dx, dt, periodic=True)

	t = (i + 1)*dt
	if t in times:
		plt.plot(x, psi_forward)
		plt.plot(x, psi_central)
		plt.title("Evolved sine functions, $t = " + str(t) + "$")
		plt.xlabel("$x$")
		plt.ylabel("$\\psi(x, t)$")
		plt.legend(("Forward", "Central"))
		plt.show()

	i += 1


# Dirichlet boundary conditions

x = x_Dirichlet
psi_2 = np.exp(-(x - x[0])**2/sigma)
plt.plot(x, psi_2)
plt.title("Initial Gaussian function, $t = 0$")
plt.xlabel("$x$")
plt.ylabel("$\\psi(x, t)$")
plt.show()

psi_forward = np.copy(psi0)
psi_central = np.copy(psi0)

i = 0
while (i + 1)*dt <= 150:
	psi_forward = rb.step_Rossby_forward(psi_forward, dx, dt)
	psi_central = rb.step_Rossby_central(psi_central, dx, dt)

	t = (i + 1)*dt
	if t in times:
		plt.plot(x, psi_forward)
		plt.plot(x, psi_central)
		plt.title("Evolved Gaussian functions, $t = " + str(t) + "$")
		plt.xlabel("$x$")
		plt.ylabel("$\\psi(x, t)$")
		plt.legend(("Forward", "Central"))
		plt.show()

	i += 1