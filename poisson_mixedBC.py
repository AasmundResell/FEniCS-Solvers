"""
Description: Mixed boundary conditions of Dirichlet,
Neumann and Robin.
"""
from ufl import *
from dolfin import *
from matplotlib.pyplot import show
import sympy as sym

# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, "P", 1)

# Define expressions

# Define manufactured solution in sympy and derive f, g, etc.
x, y = sym.symbols('x[0], x[1]')  # needed by UFL
u = 1 + x**2 + 2*y**2  # exact solution
u_e = u  # exact solution
u_00 = u.subs(x, 0)  # restrict to x = 0
u_01 = u.subs(x, 1)  # restrict to x = 1
u_10 = u.subs(y, 0)  # restrict to y = 0
u_11 = u.subs(y, 1)  # restrict to y = 1
f = -sym.diff(u, x, 2) - sym.diff(u, y, 2)  # -Laplace(u)
f = sym.simplify(f)  # simplify f
g = -sym.diff(u, y).subs(y, 1)  # compute g = -du/dn
r = 1000  # Robin data, arbitrary
s = u  # Robin data, u = s

# Collect variables
variables = [u_e, u_00, u_01, f, g, u_10, u_11, r, s]
# Turn into C/C++ code strings
variables = [sym.printing.ccode(var) for var in variables]
# Turn into FEniCS Expressions
variables = [Expression(var, degree=2) for var in variables]
# Extract variables
u_e, u_00, u_01, f, g, u_10, u_11, r, s = variables

# Materials parameters for kappa
k0 = 1
k1 = 1


# Kappa using direct c++ expression (more efficient)
tol = 1E-14
kappa = Expression("x[1] <= 0.5 + tol ? k0 : k1",
                   degree=0, tol=tol, k0=k0, k1=k1)


#######################################################
# MIXED BOUNDARY CONDITIONS

# Define outer boundaries as subdomain classes
class BoundaryX0(SubDomain):
    tol = 1E-14

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)


class BoundaryX1(SubDomain):
    tol = 1E-14

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1, tol)


class BoundaryY0(SubDomain):
    tol = 1E-14

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)


class BoundaryY1(SubDomain):
    tol = 1E-14

    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1, tol)


# marks the different boundaries

boundary_markers = MeshFunction("size_t", mesh, 1)

boundaryX0 = BoundaryX0()
boundaryX0.mark(boundary_markers, 0)
boundaryX1 = BoundaryX1()
boundaryX0.mark(boundary_markers, 1)
boundaryY0 = BoundaryY0()
boundaryX0.mark(boundary_markers, 2)
boundaryY1 = BoundaryY1()
boundaryX0.mark(boundary_markers, 3)

File("poisson_marked/facetfunc.pvd").write(boundary_markers)

subdomains = []

subdomains.append(boundaryX0)
subdomains.append(boundaryX1)
subdomains.append(boundaryY0)
subdomains.append(boundaryY1)

boundary_conditions = {
    0: {'Dirichlet': u_00},
    1: {'Dirichlet': u_01},
    2: {'Dirichlet': u_10},
    3: {'Dirichlet': u_11}}


# Define subdomains of surface integrals
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)

# Define funtions
u = TrialFunction(V)
v = TestFunction(V)

# Ordering BC for the variational formulation
bcs = []
integrals_N = []
integrals_R = []

print(boundary_conditions)

for n in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[n]:
        print(n)
        bc = DirichletBC(
            V, boundary_conditions[n]['Dirichlet'], subdomains[n])
        bcs.append(bc)
        print('Dirichlet')
    if 'Neumann' in boundary_conditions[n]:
        if boundary_conditions[n]['Neumann'] != 0:
            g = boundary_conditions[n]['Neumann']
            integrals_N.append(g * v * ds(n))
            print('Neumann')
    if 'Robin' in boundary_conditions[n]:
        r, s = boundary_conditions[n]['Robin']
        integrals_R.append(r * (u-s) * v * ds(n))
        print('Robin')

# Setup of variational problem
F = kappa*dot(grad(u), grad(v))*dx + \
    sum(integrals_R) - f*v*dx + sum(integrals_N)
a, b = lhs(F), rhs(F)

# Compute solution
u = Function(V)
solve(a == b, u, bcs)

vtkfile = File("poisson/solution.pvd")  # stores solution
vtkfile << u
U_E = Function(V)
U_E.assign(u_e)

plot(U_E)
show()
plot(u)
show()

###########################################################
# DEBUGGING BOUNDARY CONDITIONS
debug = True
if debug:
    for x in mesh.coordinates():
        if boundaryX0.inside(x, True):
            print('%s is on x = 0' % x)
        if boundaryX1.inside(x, True):
            print('%s is on x = 1' % x)
        if boundaryY0.inside(x, True):
            print('%s is on y = 0' % x)
        if boundaryY1.inside(x, True):
            print('%s is on y = 1' % x)

    print('Num of Dirichlet BC: ', len(bcs))
    if V.ufl_element().degree() == 1:
        d2v = dof_to_vertex_map(V)
        coor = mesh.coordinates()
        for i, bc in enumerate(bcs):
            print('Dirichlet condition %d' % i)
            boundary_values = bc.get_boundary_values()
            print('hei')
            for dof in boundary_values:
                print('   dof %2d: u = %g' % (dof, boundary_values[dof]))
                if V.ufl_element().degree() == 1:
                    print('   at point %s' %
                          (str(tuple(coor[d2v[dof]].tolist()))))
