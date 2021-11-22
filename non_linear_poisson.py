# Warning: from fenics import * will import both ‘sym‘ and
# ‘q‘ from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *
from matplotlib.pyplot import show

def q(u):
    return 1+u**2

import sympy as sym
import numpy as np

#Creating a manufactured solution, utilizing symbolic computing
x,y = sym.symbols('x[0],x[1]')
u = 1 + x + 2*y
f = -sym.diff(q(u)*sym.diff(u,x),x)-sym.diff(q(u)*sym.diff(u,y),y)
f = sym.simplify(f)
u_code = sym.printing.ccode(u) #ask for C/C++ code
f_code = sym.printing.ccode(f) #ask for C/C++ code
print('u =', u_code)
print('f =', f_code)


#W
# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition 
u_D = Expression(u_code, degree=1)
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)


# Define funtions
u = Function(V)     #Skips the definition of trial function for non-linear problem 
v = TestFunction(V)
f = Expression(f_code,degree=1)

#Setup of variational problem
F = q(u)*dot(grad(u),grad(v))*dx-f*v*dx

# Compute solution
solve(F==0,u,bc)

vtkfile = File('Non_linear_poisson/solution.pvd') #stores solution
vtkfile << u
plot(u)

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Print errornorm
print('error_L2 =', error_L2)

show()