from fenics import *
from matplotlib.pyplot import show
import numpy as np

# Constants
T = 5
num_Tsteps = 500
dt = T / num_Tsteps
eps = 0.01
K = 10

mesh = Mesh("navier_stokes/cylinder/cylinder.xml.gz")

# Create mixed element for the species
P1 = FiniteElement("P", triangle, 1)  # Standard first order
element = MixedElement([P1, P1, P1])

# Define function spaces
W = VectorFunctionSpace(mesh, "P", 2)  # Velocity function space
V = FunctionSpace(mesh, element)  # Species function space

# Initilize timeseries object for the velocity
timeseries_w = TimeSeries("navier_stokes/cylinder/velocity_series")

# Create test functions
v_1, v_2, v_3 = TestFunctions(V)

w = Function(W)
u = Function(V)
u_n = Function(V)

# Initial concentration
u_0 = Expression(("sin(x[0])", "cos(x[0]*x[1])", "exp(x[1])"), degree=1)
u_n = project(u_0, V)


# Note that split(x) creates symbolic notations used for the variational formulation.
# Not actual vector components.
u_1, u_2, u_3 = split(u)  # Solution functions at t = n+1
u_n1, u_n2, u_n3 = split(u_n)  # Storage functions at t = n

# Define expressions for variational formulation
k = Constant(dt)
e = Constant(eps)
K = Constant(K)  # Rate of reactions

f1 = Expression(
    "pow(x[0]-0.1,2)+pow(x[1]-0.1,2)<0.05*0.05 ? 0.1 : 0", degree=1)
f2 = Expression(
    "pow(x[0]-0.1,2)+pow(x[1]-0.3,2)<0.05*0.05 ? 0.1 : 0", degree=1)
f3 = Constant(0)

# Define variational problem
F = (
    ((u_1 - u_n1) / k) * v_1 * dx
    + dot(w, grad(u_1)) * v_1 * dx
    + e * dot(grad(u_1), grad(v_1)) * dx
    + K * u_1 * u_2 * v_1 * dx
    + ((u_2 - u_n2) / k) * v_2 * dx
    + dot(w, grad(u_2)) * v_2 * dx
    + e * dot(grad(u_2), grad(v_2)) * dx
    + K * u_1 * u_2 * v_2 * dx
    + ((u_3 - u_n3) / k) * v_3 * dx
    + dot(w, grad(u_3)) * v_3 * dx
    + e * dot(grad(u_3), grad(v_3)) * dx
    - K * u_1 * u_2 * v_3 * dx
    + K * u_3 * v_3 * dx
    - f1 * v_1 * dx
    - f2 * v_2 * dx
    - f3 * v_3 * dx
)

# Add progress bar
progress = Progress("Time-stepping", num_Tsteps)
set_log_level(LogLevel.PROGRESS)

# Add xdmf file to write
vtkffile_u1 = File("ad_reactions/u_1.pvd")
vtkffile_u2 = File("ad_reactions/u_2.pvd")
vtkffile_u3 = File("ad_reactions/u_3.pvd")

t = 0

for n in range(num_Tsteps):
    t += dt
    timeseries_w.retrieve(w.vector(), t)
    solve(F == 0, u)
    u_n.assign(u)

    # Write to files
    _u1, _u2, _u3 = u.split()  # Creates objects of the sub components

    vtkffile_u1 << (_u1, t)
    vtkffile_u2 << (_u2, t)
    vtkffile_u3 << (_u3, t)

    # Update progress bar
    progress += 1

    plot(w)

show()
