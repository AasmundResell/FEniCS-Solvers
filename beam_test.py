from fenics import *
from matplotlib import interactive
import numpy as np

import matplotlib.pyplot as plt


mesh = UnitIntervalMesh(10)

V = FunctionSpace(mesh, 'P',1)

ub = Expression('0.5+0.5*x[0]',degree=1)

tol = 1E-14
def boundary(x):
    return near(x[0], 0, tol) or near(x[0], 1, tol)

bc = DirichletBC(V,ub,boundary)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-10.0)
A = Constant(10.0)

a = A*inner(grad(u),grad(v))*dx
l = f*v*dx

u = Function(V)
solve(a == l, u,bc)

plot(u)
plot(mesh)

vtkfile = File('solution/solution.pvd')
vtkfile << u

plt.show()