from fenics import *
from matplotlib.pyplot import show
import numpy as np

#Problem parameters
T = 2.0 #Total time
num_timesteps = 10 
dt = T/ num_timesteps
alpha = 3; beta = 1.2

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1]+beta*t', degree=2, alpha = alpha, beta = beta, t=0)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

#Define inital condition
u_n = interpolate(u_D,V) #n=0, u_D used as inital condition

# Define funtions
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta-2-2*alpha)

#Setup of variational problem
F = u*v*dx + dt*dot(grad(u),grad(v))*dx - (u_n+dt*f)*v*dx
a,l = lhs(F),rhs(F)

# Compute solution
u = Function(V)
t = 0


vtkfile = File('heat_equation_results/heat_equation_solution.pvd') #stores solution
    
for n in range(num_timesteps):
    t += dt
    u_D.t = t #Must be updated before solving!

    solve(a == l, u, bc) #Solve timestep

    plot(u)   
    # Save solution to file in VTK format
    vtkfile << (u,t)

    
    #Calculate the error
    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))

    u_n.assign(u) #Update previous solution


# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Print errornorm
print('error_L2 =', error_L2)
 
show()