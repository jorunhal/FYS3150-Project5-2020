import numpy as np
import poisson as ps

def step_Rossby_forward(psi, dx, dt, periodic=False):
	# Takes an array of values at positions j*dx of a function which obeys 
	# the Rossby wave equation at a certain point in time, and uses functions 
	# which solve the Poisson equation to compute the values of the function 
	# a time dt later. The time-step approximation is that of the forward 
	# difference. If periodic is False, Dirichlet boundary conditions are 
	# employed.

	y = ps.doublediff(psi, periodic=periodic) - ps.diff_sym(psi, periodic=periodic)*dt*dx/2.0
	if periodic:
		return ps.solve_Poisson_periodic(y)
	else:
		return ps.solve_Poisson_Dirichlet(y)

def step_Rossby_central(psi, dx, dt, periodic=False):
	# Takes an array of values at positions j*dx of a function which obeys 
	# the Rossby wave equation at a certain point in time, and uses functions 
	# which solve the Poisson equation to compute the values of the function 
	# a time dt later. The time-step approximation is that of the central
	# difference. If periodic is False, Dirichlet boundary conditions are 
	# employed.

	psi_mid = step_Rossby_forward(psi, dx, dt/2.0, periodic=periodic)

	y = ps.doublediff(psi, periodic=periodic) - ps.diff_sym(psi_mid, periodic=periodic)*dx*dt/2.0
	if periodic:
		return ps.solve_Poisson_periodic(y)
	else:
		return ps.solve_Poisson_Dirichlet(y)