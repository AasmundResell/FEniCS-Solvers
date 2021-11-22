from mshr import *
from fenics import *
from matplotlib.pyplot import show

R = 5
a = 1.2
b = 1.0
r = 0.1
n = 10
c_1 = 0.8
c_2 = 1.4

domain = Circle(Point(0, 0), R)

cylinder = Circle(Point(0, 0), a) - Circle(Point(0, 0), b)

# Define geometry for wires (N = North (up), S = South (down))
angles_N = [pi / n + i * 2 * pi / n for i in range(n)]
angles_S = [i * 2 * pi / n for i in range(n)]

copper_N = [Circle(Point(c_1 * sin(i), c_1 * cos(i)), r) for i in angles_N]
copper_S = [Circle(Point(c_2 * sin(i), c_2 * cos(i)), r) for i in angles_S]

# Set the different geometries as subdomains
domain.set_subdomain(1, cylinder)
for i in range(n):
    domain.set_subdomain(2 + i, copper_N[i])
    domain.set_subdomain(2 + n + i, copper_S[i])

mesh = generate_mesh(domain, 100)

V = FunctionSpace(mesh, "P", 1)


def boundary(x, on_boundary):
    tol = 1e-14
    return on_boundary & near(x[0] ** 2 + x[1] ** 2, R ** 2, tol)


bc = DirichletBC(V, Constant(0), boundary)

markers = MeshFunction("size_t", mesh, 2, mesh.domains())
dx = Measure("dx", domain=mesh, subdomain_data=markers)


class Permability(UserExpression):
    def __init__(self, markers, **kwargs):
        self.markers = markers

        # To remove attribute error:'Permability' object
        # has no attribute '_ufl_shape'
        super(Permability, self).__init__(**kwargs)

    def eval_cell(self, values, x, cell):
        if self.markers[cell.index] == 0:  # Vacum
            values[0] = 4 * pi * 1e-7
        if self.markers[cell.index] == 1:  # Iron
            values[0] = 1e-5
        else:  # Copper
            values[0] = 1.26e-6

    def value_shape(self):
        return ()


# Define variational formulation
J_N = Constant(1.0)
J_S = Constant(-1.0)
mu = Permability(markers, degree=1)
A_z = TrialFunction(V)
v = TestFunction(V)
a = (1 / mu) * dot(grad(A_z), grad(v)) * dx
L_N = sum(J_N * v * dx(i) for i in range(2, 2 + n))
L_S = sum(J_S * v * dx(i) for i in range(2 + n, 2 + 2 * n))
L = L_N + L_S

A_z = Function(V)
solve(a == L, A_z)

W = VectorFunctionSpace(mesh, "P", 1)
B = project(as_vector((A_z.dx(1), -A_z.dx(0))), W)

File("Maxwell/potential.pvd").write(A_z)
File("Maxwell/field.pvd").write(B)
