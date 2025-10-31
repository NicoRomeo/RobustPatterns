using Distributed

@everywhere begin 
import Pkg; Pkg.activate("./aim1d")
    Pkg.instantiate();
end
@everywhere begin
    include("./DIM_1d.jl")
    using ProgressMeter
    using Random
    using HDF5
end



# system size
L= 4.0

# cell size Δx
dx = 0.125

    
# baseline reaction rate 𝛾
gamma = 1.0

# Parameter grid to sweep over
Ds = 10 .^ (collect(LinRange(-2, 1, 30)))
as = [1/8,1/(2*8),1/(3*8),1/(4*8),1/(5*8),1/(6*8),1/(7*8),1/(8*8),1/(12*8), 1/(24*8), 1/(48*8)]


for a = as
    println("dx/a=$(0.125/a)")
    for D = Ds
        println("D=$(D)")
        N_rep = 500
        beta = 1.5*log(1+sqrt(2)) 
        h = beta * 3
        T = 500 / D # scaling of time with D 
        idxp = Int(1/a+1) # position of positive source site
        idxm = Int(3/a+1) # position of negative source site
        params = (
            a=a,
            dx = dx,
            L = L,
            D = D,
            N = Int(L/a),
            gamma = gamma,
            beta = beta,
            h = h,
            T = T,
            idxp = idxp,
            idxm = idxm
            )
        # make folder before running! 
        filename = "./run_grid/reps_D$(D)_1oa$(Int(floor(1/a))).h5"
        run_sim_replicates(filename, params, N_rep)
    end
end
println("-- Run complete --")