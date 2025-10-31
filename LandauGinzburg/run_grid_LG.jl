using Distributed

@everywhere begin 
import Pkg; Pkg.activate("./aim1d")
    Pkg.instantiate();
end
@everywhere begin
    include("./LG_pde.jl")
    using ProgressMeter
    using Random
    using HDF5
end


# system size (length)
L= 4.0
# number of grid points = cells here
N = 128
dx = L/N

nsaves= 100

seed = 39 #35


seed0 = rand(1:99999)

# Parameters to sweep
Ds = 10 .^ (collect(LinRange(-3, 0, 30)))
gammas = 10 .^ (collect(LinRange(-2, 0, 10)))


for gamma = gammas
    for D = Ds
        println("D=$(D)")
        N_rep = 500 # number of replicates
        r = 1.0
        dt = 0.03*dx^2/D # time step size - set using CFL condition
        T = 100 /D # scaling of time with D
        x0 = zeros(N)
        params=(
            x0=x0,
            L = L,
            N = N,
            dx = L/N,
            r = r,
            dt = dt,
            T = T,
            D = D,
            gamma = gamma,
            n_step = Int(floor(T/ dt)),
            n_sub = 100,
            idx1 = Int(floor(N/4)),
            idx2 = Int(floor(3*N/4)),
            h0 = 10.0 #*gamma
        )
        # make folder before running!
        filename = "./pde_output/reps_D$(D)_g$(gamma).h5"
        run_replicates_disordered(filename, params, N_rep)
    end
end
println("~success~")