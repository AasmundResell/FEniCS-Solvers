from fenics import *
from matplotlib.pyplot import show
import numpy as np
# Constants
mu = 1
T = 5
num_Tsteps = 5000
dt = T / num_Tsteps
rho = 1

mesh = UnitSquareMesh(20, 20)

V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)

u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Creating boundary conditions by C strings
inflow = "near(x[0],0)"
outflow = "near(x[0],1)"
walls = "near(x[1],1) || near(x[1],0)"

bcu_noSlip = DirichletBC(V, Constant((0, 0)), walls)
bcp_inFlow = DirichletBC(Q, Constant(8), inflow)
bcp_outFlow = DirichletBC(Q, Constant(0), outflow)

bcu = [bcu_noSlip]
bcp = [bcp_inFlow, bcp_outFlow]

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
#xdmffile_u = XDMFFile("navier_stokes/channel/velocity.xdmf")
#xdmffile_p = XDMFFile("navier_stokes/channel/pressure.xdmf")

# Create time series
#timeseries_u = TimeSeries("navier_stokes/channel/velocity_series")
#timeseries_p = TimeSeries("navier_stokes/channel/pressure_series")

# Time-stepping
t = 0

for i in range(num_Tsteps):
    # Update current time
    t += dt
    # Step 1: Tentative velocity step
    b1 = assemble(l1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(l2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(l3)
    solve(A3, u_.vector(), b3)

    # Write to files
#   xdmffile_u.write(u_, t)
#   xdmffile_p.write(p_, t)

#   timeseries_u.store(u_.vector(), t)
#   timeseries_p.store(p_.vector(), t)

    u_n.assign(u_)
    p_n.assign(p_)

    progress += 1

    # Plot solution
    plot(u_)

    # Compute error
    u_e = Expression(('4*x[1]*(1.0 - x[1])', '0'), degree=2)
    u_e = interpolate(u_e, V)
    error = np.abs(u_e.vector().get_local() - u_.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    print('max u:', u_.vector().get_local().max())
show()
