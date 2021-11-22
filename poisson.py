"""
Description: Standard poisson solver with neumann and dirichlet BC
and multiple domains with different properties

"""
from fenics import *
from matplotlib.pyplot import show


# Create mesh and define function space
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, "P", 1)

# Define boundary condition
u_L = Expression("1+2*x[1]*x[1]", degree=2)  # DirichletBC
u_R = Expression("2+2*x[1]*x[1]", degree=2)  # DirichletBC
g = Expression("4*x[1]", degree=1)  # NeumannBC

# Materials parameters for kappa
k0 = 100
k1 = 0.01

"""
Following section illustrates differnt ways to implement
the kappa expression to apply different materials at
different sections of the domain.
"""

#####################################################
# Kappa using python class Expression


class kappaClass(UserExpression):
    def set_k_values(self, k0, k1):
        self.k0, self.k1 = k0, k1

    def eval(self, value, x):
        tol = 1e-14
        if x[1] >= 0.5 + tol:
            value[0] = self.k1
        else:
            value[0] = self.k0

    def value_shape(self):  # Gets rid of warning
        return ()


# Initialize kappa class
#kappaC = kappaClass(degree=1)
#kappaC.set_k_values(k0, k1)

######################################################
# Kappa creating subdomains using the SubDomain class
# and MeshFunction to define different materials


class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return x[1] <= 0.5 + tol


class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return x[1] >= 0.5 + tol


materials = MeshFunction("size_t", mesh, 2)  # size_t = unsigned int

# Mark the respective cells in subdomains with 0 and 1
subdomain0 = Omega_0()
subdomain1 = Omega_1()
subdomain0.mark(materials, 0)
subdomain1.mark(materials, 1)

# Illustrate that the materials have been applied correct
# plot(materials)
# show()

# Store the values of the mesh function for later use
#File("poisson/materials.xml.gz") << materials
# Read back againg as File('materials.xml.gz') >> materials

# To apply the respective kappa values a FEniCS
# expression is created


class K(UserExpression):
    def set_values(self, materials, k0, k1):
        self.material = materials
        self.k0 = k0
        self.k1 = k1

    def eval_cell(self, values, x, cell):
        if self.material[cell.index] == 0:
            values[0] = self.k0
        if self.material[cell.index] == 1:
            values[0] = self.k1

    def value_shape(self):  # Gets rid of warning
        return ()


#kappaSub = K(degree=0)
#kappaSub.set_values(materials, k0, k1)

#####################################################
# Applying kappa by the materials MeshFunction (created earlier)
# and creating a C++ class that is directly compiled
# which is more efficient than applying with python class

# cppcode = """
# #include <dolfin/function/Expression.h>

# class K : public Expression
# {
# public:

#   K() : Expression()
#   {
#   }

#   void eval(Array<double>& values,
#   const Array<double>& x,
#   const ufc::cell& cell) const
#   {
#     if ((*materials)[cell.index] == 0)
#       values[0] = k_0;
#     else
#       values[0] = k_1;
#   }
#   std::shared_ptr<MeshFunction<std::size_t>> materials;
#   double k_0;
#   double k_1;
# };
# }
# """
# materialsC = MeshFunction('size_t', mesh, 2)
# subdomain0.mark(materialsC, 0)
# subdomain1.mark(materialsC, 1)


# kappa = Expression(cppcode)
# kappa.materials = materialsC
# kappa.k_0 = k0
# kappa.k_1 = k1
######################################################
# Kappa using direct c++ expression (more efficient)
tol = 1E-14
kappaExp = Expression("x[1] <= 0.5 + tol ? k0 : k1",
                      degree=0, tol=tol, k0=k0, k1=k1)


######################################################


# Define outer boundaries as subdomain classes
class Boundary1(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[0], 0, tol)


class Boundary2(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1e-14
        return on_boundary and near(x[0], 1, tol)


boundary1 = Boundary1()
boundary2 = Boundary2()

# Applying BC

# DirichletBC ensures that v=0 on DirichletBC so that
# NeumannBC are not added on these boundaries.
# By python classes
#bc1 = DirichletBC(V, u_L, boundary1)
#bc2 = DirichletBC(V, u_R, boundary2)

# alternative (more efficient) way of adding BC
bc1 = DirichletBC(V, u_L, 'on_boundary && near(x[0],0,1e-14)')
bc2 = DirichletBC(V, u_R, 'on_boundary && near(x[0],1,1e-14)')
bcs = [bc1, bc2]

# Define funtions
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)

# Setup of variational problem
a1 = kappaExp * dot(grad(u), grad(v)) * dx
b1 = f * v * dx - g * v * ds

# Compute solution
u = Function(V)
solve(a1 == b1, u, bcs)

vtkfile = File("poisson/solution.pvd")  # stores solution
vtkfile << u

plot(u)
show()
