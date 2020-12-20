import numpy as np

def diff_sym(x, periodic=False):
	# Returns the symmetric difference between the elements of a 
	# vector x. If periodic is True, the first and last elements 
	# are treated as if they are adjacent. Otherwise, it is assumed 
	# that imagined elements outside the range of x are all equal 
	# to zero.

	y = np.zeros(len(x))

	y[1:-1] = x[2:] - x[:-2]
	if periodic:
		y[0] = x[1] - x[-1]
		y[-1] = x[0] - x[-2]
	else:
		y[0] = x[1]
		y[-1] = -x[-2]
	return y/2.0

def doublediff(x, periodic=False):
	# Returns the double difference between the elements of a 
	# vector x. If periodic is True, the first and last elements 
	# are treated as if they are adjacent. Otherwise, it is assumed 
	# that imagined elements outside the range of x are all equal 
	# to zero.

	y = np.zeros(len(x))

	y[1:-1] = x[2:] - 2.0*x[1:-1] + x[:-2]
	if periodic:
		y[0] = x[1] - 2.0*x[0] + x[-1]
		y[-1] = x[0] - 2.0*x[-1] + x[-2]
	else:
		y[0] = x[1] - 2.0*x[0]
		y[-1] = -2.0*x[-1] + x[-2]

	return y

def solve_Poisson_Dirichlet(b):
	# Computes the solution x to the matrix equation A*x = b where
	# A is the tridiagonal matrix with respective values 1, -2, 1 
	# along the lower, main, and upper diagonals. Dirichlet boundary 
	# conditions are assumed, and so any imagined elements beyond the 
	# range of the solution are zero by implication. The solution is
	# computed using a closed-form expression.

	n = len(b)

	x = np.zeros(n)
	for k in range(n):
		for l in range(k):
			x[k] -= b[l]*(l + 1)*(n - k)/(n + 1)
		for l in range(k, n):
			x[k] -= b[l]*(k + 1)*(n - l)/(n + 1)

	return x

def solve_Poisson_periodic(b):
	# Computes the solution x to the matrix equation A*x = b where
	# A is the tridiagonal matrix with respective values 1, -2, 1 
	# along the lower, main, and upper diagonals. Periodic boundary 
	# conditions are assumed, meaning that an imagined element after 
	# the last element of the solution is equal to the first. The 
	# solution is computed using a closed-form expression, after 
	# which an estimated average value is added.

	n = len(b)

	x = np.zeros(n)
	s = np.linspace(1.0, n - 1, n - 1)
	for k in range(n):
		coeffs = np.zeros(n)
		for l in range(n - 1):
			coeffs[l] = -np.average(np.cos(2.0*np.pi*(k - l)*s/n)/np.sin(np.pi*s/n)**2)/4.0
		x[k] = np.sum(coeffs*b)

	b_avg = np.average(b)
	i = np.linspace(0, n - 1, n)
	x_avg = np.average((n/2.0 - i)**2*b)/2.0
	x += x_avg - x/n

	return x