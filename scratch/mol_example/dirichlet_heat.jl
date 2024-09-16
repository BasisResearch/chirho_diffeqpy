using OrdinaryDiffEq, ModelingToolkit, MethodOfLines, DomainSets
# Method of Manufactured Solutions: exact solution
u_exact = (x, t) -> exp.(-t) * cos.(x)

# Parameters, variables, and derivatives
@parameters t x
@parameters Dn,
@variables u(..)
Dt = Differential(t)
Dxx = Differential(x)^2

# 1D PDE and boundary conditions
eq = Dt(u(t, x)) ~ Dn * Dxx(u(t, x))
bcs = [u(0, x) ~ cos(10. * x),
    u(t, 0) ~ 1.0, #  exp(-t),
    u(t, 1) ~ cos(10. * -1.), #exp(-t) * cos(1)]
]

# Space and time domains
domains = [t ∈ Interval(0.0, 1.0),
    x ∈ Interval(0.0, 1.0)]

# PDE system
@named pdesys = PDESystem(
    eq, bcs, domains, [t, x], [u(t, x)], [Dn => 0.1];
)

# Method of lines discretization
dx = 0.01
order = 2
discretization = MOLFiniteDifference([x => dx], t)

# Convert the PDE problem into an ODE problem
prob = discretize(pdesys, discretization)

# Solve ODE problem
using OrdinaryDiffEq
sol = solve(prob, Tsit5(), saveat = 0.2)

# Plot results and compare with exact solution
discrete_x = sol[x]
discrete_t = sol[t]
solu = sol[u(t, x)]

using Plots
plt = plot()

for i in eachindex(discrete_t)
    plot!(discrete_x, solu[i, :], label = "Numerical, t=$(discrete_t[i])")
#     scatter!(
#         discrete_x, u_exact(discrete_x, discrete_t[i]), label = "Exact, t=$(discrete_t[i])")
end
plt