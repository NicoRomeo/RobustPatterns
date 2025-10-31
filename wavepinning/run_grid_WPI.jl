using Distributed

@everywhere begin 
import Pkg; Pkg.activate("./aim1d")
    Pkg.instantiate();
end
@everywhere begin
    include("./feedbackIsing.jl")
    using ProgressMeter
    using Random
    using HDF5
end

# System size
L = 4.0
# Cell size Δx
dx = 0.125
 

# Parameters to sweep
Ds = 10 .^ (collect(LinRange(-2, 1, 30)))
as = [1/8,1/(2*8),1/(3*8),1/(4*8),1/(5*8),1/(6*8),1/(7*8),1/(8*8),1/(10*8),1/(11*8),1/(12*8)]


for a = as
    println("dx/a=$(0.5/a)")
    for D = Ds
        println("D=$(D)")
        N0 = Int(L/a) # number of sites
        #target steady-state density (particle/site)
        rho_bar = 2
        N = Int(rho_bar) * 3 * N0 # total number of particles in systems

        gamma = 1.0
        k = 4.0/3.0
        N_rep = 1000
        h = 30 # source bias
        T = 5000 / D
        params = (
            a=a,
            dx = dx,
            L = L,
            D = Float64(D),
            N =Int64(N),
            gamma = Float64(gamma),
            k = Float64(k),
            h = Float64(h),
            T = T,
            idxp = Int(1/a)+1,  # position of positive source site
            idxm = Int(3/a)+1 
            )
        # make folder before running!
        filename = "./wp_output/reps_D$(D)_1oa$(Int(floor(1/a))).h5"
        run_sim_replicates_polarized(filename, params, N_rep)
    end
end
println("~success~")