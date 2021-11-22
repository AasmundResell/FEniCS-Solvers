from fenics import *
from mshr import *
from matplotlib.pyplot import show
import numpy as np

# Constants
mu = 0.001
T = 5
num_Tsteps = 5000
dt = T / num_Tsteps
rho = 1

channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
cylinder = Circle(Point(0.2, 0.2), 0.05)
domain = channel - cylinder
mesh = generate_mesh(domain, 64)

# Save mesh for later use
File('navier_stokes/cylinder/cylinder.xml.gz') << mesh

# Define function spaces
V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Creating boundary conditions by C strings
inflow = "near(x[0],0)"
outflow = "near(x[0],2.2)"
walls = "near(x[1],0.41) || near(x[1],0)"
cylinder_walls = "on_boundary && x[0] > 0.1 && x[0] < 0.3 && x[1] > 0.1 && x[1]  < 0.3"

# inlet boundary expression
u_d = Expression(('1.5 * 4*x[1]*(0.41-x[1])/pow(0.41,2)', 0), degree=2)

bcu_noSlip_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_noSlip_Cwalls = DirichletBC(V, Constant((0, 0)), cylinder_walls)
bcu_inFlow = DirichletBC(V, u_d, inflow)
bcp_outFlow = DirichletBC(Q, Constant(0), outflow)

bcu = [bcu_noSlip_walls, bcu_noSlip_Cwalls, bcu_inFlow]
bcp = [bcp_outFlow]

# intermediate variables
u_n = Function(V)
u_ = Function(V)
p_n = Function(Q)
p_ = Function(Q)

# Define expressions
U = 0.5 * (u_n + u)  # = u_(n+1/2)
n = FacetNormal(mesh)
f = Constant((0, 0))
k = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)


def epsilon(v): return 0.5 * (grad(v) + grad(v).T)


def sigma(u, p): return 2 * mu * epsilon(u) - p * Identity(len(u))


# Define variational problem
F1 = (
    rho * dot((u - u_n) / k, v) * dx
    + rho * dot(dot(u_n, nabla_grad(u_n)), v) * dx
    + inner(sigma(U, p_n), epsilon(v)) * dx
    + dot(p_n * n, v) * ds
    - mu * dot(nabla_grad(U) * n, v) * ds
    - dot(f, v) * dx
)
a1 = lhs(F1)
l1 = rhs(F1)

a2 = dot(grad(p), grad(q)) * dx
l2 = dot(grad(p_n), grad(q)) * dx - (1 / k) * div(u_) * q * dx

a3 = dot(u, v) * dx
l3 = dot(u_, v) * dx - k * dot(grad(p_ - p_n), v) * dx

# Assembly of matrices BEFORE starting time loop
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Add progress bar
progress = Progress("Time-stepping", num_Tsteps)
set_log_level(LogLevel.PROGRESS)

# Add xdmf file to write
xdmffile_u = XDMFFile("navier_stokes/cylinder/velocity.xdmf")
xdmffile_p = XDMFFile("navier_stokes/cylinder/pressure.xdmf")

# Included to obtain readable xdmf files
xdmffile_u.parameters["flush_output"] = True
xdmffile_p.parameters["flush_output"] = True


# Create time series
timeseries_u = TimeSeries("navier_stokes/cylinder/velocity_series")
timeseries_p = TimeSeries("navier_stokes/cylinder/pressure_series")

# Time-stepping
t = 0

for i in range(num_Tsteps):
    # Update current time
    t += dt
    # Step 1: Tentative velocity step
    b1 = assemble(l1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(l2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(l3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Write to files
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    u_n.assign(u_)
    p_n.assign(p_)

    progress += 1

    # Plot solution

    print('max u:', u_.vector().get_local().max())

plot(u_)
show()
