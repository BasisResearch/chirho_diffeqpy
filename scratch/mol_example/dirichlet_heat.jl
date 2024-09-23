using .OrdinaryDiffEq, .ModelingToolkit, .MethodOfLines, .DomainSets

# Parameters, variables, and derivatives
@parameters t x
@parameters Dn
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dn * Dxx(u(t, x))
bcs = [
    # Initial condition at t=0.
    u(0, x) ~ cos(10. * x),
    # Boundary condition on left side of rod = 1.
    u(t, 0) ~ 1.0,
    # Boundary condition at right side of rod, consistent with initial condition.
    u(t, 1) ~ cos(10. * -1.),
]

# Space and time domains
domains = [
    t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)
]

# PDE system with default diffusivity Dn of 0.1.
@named pdesys = PDESystem(
    eq, bcs, domains, [t, x], [u(t, x)], [Dn => 0.1];
)

# Method of lines discretization
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
ode_prob = discretize(pdesys, discretization)
