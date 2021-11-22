"""
Description: Standard poisson solver with DirichletBC
Code is refactored to differentiate reusable and non-reusable code
"""
from fenics import *
from matplotlib.pyplot import show
import numpy as np


def solver(f, u_D, Nx, Ny, Nz, degree=1):
    # Create mesh and define function space
    #  mesh = UnitCubeMesh(Nx, Ny, Nz)
    mesh = UnitSquareMesh(Nx, Ny)
    V = FunctionSpace(mesh, "P", degree)

    bc = DirichletBC(V, u_D, "on_boundary")

    # Define funtions
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Constant(-6)

    # Setup of variational problem
    a = dot(grad(u), grad(v)) * dx
    b = f * v * dx

    # Compute solution
    u = Function(V)
    solve(a == b, u, bc)

    return u


def runSolver():
    u_D = Expression("1+x[0]*x[0]+2*x[1]*x[1]", degree=2)
    f = Constant(-6.0)

    u = solver(f, u_D, 20, 20, 8)

    plot(u)
    show()
    vtkfile = File("poisson_refactored/solution.pvd")  # stores solution
    vtkfile << u


def test_solver():

    tol = 1e-10
    u_D = Expression("1+x[0]*x[0]+2*x[1]*x[1]", degree=2)
    f = Constant(-6.0)

    # Iterate over mesh sizes and degrees
    for Nx, Ny in [(3, 3), (3, 5), (5, 3), (20, 20)]:
        for degree in 1, 2, 3:
            print("Solving on a 2 x (%d x %d) with a P%d elements." % (Nx, Ny, degree))

            # Compute solution
            u = solver(f, u_D, Nx, Ny, 8, degree)
            # Extract the mesh
            mesh = u.function_space().mesh()

            # Compute maximum error at vertices
            vertex_values_u_D = u_D.compute_vertex_values(mesh)
            vertex_values_u = u.compute_vertex_values(mesh)
            error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

            # Check maximum error
            msg = "error_max = %g" % error_max
            assert error_max < tol, msg


runSolver()
