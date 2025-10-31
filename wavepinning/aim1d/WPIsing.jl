using Distributed

import Pkg; Pkg.activate("aim1d")

begin
    using LinearAlgebra
    using Random

    #using SharedArray
    using Base.Threads
    using ProgressMeter
    using HDF5
end

rng = Random.default_rng()



function coarsegrain1d(state::Vector{Int32}, dx::Number)
    M = length(state)
    num_box = Int(ceil(M/dx))
    binned_rho = zeros(num_box)
    for i in 1:M
        idx = Int(ceil(i/dx))
        binned_rho[idx] += state[i]
    end
    #binned_rho[end] += state[M]
    return binned_rho
end


function brange(i, block_size)
    istart = (i - 1) * block_size + 1
    iend = istart + block_size - 1
    return istart:iend
end

function coarse_grain(s, block_size, L)
    #step = ceil(Int8, L*dx_over_a)
    num_blocks = Int(floor(L/block_size))
    sp = zeros(num_blocks)
    for i = 1:num_blocks
            sp[i] = sum(s[brange(i, block_size)])
    end
    return sp
end


function coarse_grain1d(s, block_size, L)
    #step = ceil(Int8, L*dx_over_a)
    num_blocks = Int(floor(L/block_size))
    sp = zeros(num_blocks)
    for i = 1:num_blocks
        sp[i] = sum(s[brange(i, block_size)])
    end
    return sp
end


function coarsegrain1d_times(states::Matrix{Int32}, dx::Number)
    ntimes  = size(states,1 )
    M = size(states,2)
    num_box = Int(ceil(M/dx))
    cg_arr = zeros(Int8, ntimes, num_box)
    for t = 1:ntimes
        cg_arr[t,:] = coarsegrain1d(states[t,:], dx)
    end
    return cg_arr
end

function make_particlelist(M::Int64, NA::Int64, NP::Int64)
    particles = zeros(Int32, NA+NP, 2)
    for m = 1:NA
        particles[m,1] = 0 #rand(0:M)
        particles[m,2] = 1 #rand([-1,1])
    end
    for m = 1:NP
        particles[m+NA,1] = 0 #rand(0:M)
        particles[m+NA,2] = -1 #rand([-1,1])
    end
    return particles
end


function make_particlelist_polarized(M::Int64, NA::Int64, NP::Int64)
    particles = zeros(Int32, NA+NP, 2)
    arr_P = [0; collect(Int(floor(M/2)):M)]
    for m = 1:NA
        particles[m,1] = rand(0:Int(floor(M/2)))
        particles[m,2] = 1 #rand([-1,1])
    end
    for m = 1:NP
        particles[m+NA,1] = rand(arr_P) #rand(0:M)
        particles[m+NA,2] = -1 #rand([-1,1])
    end
    return particles
end

function make_particlelist_homogeneous(M::Int64, ratio::Float64)
    
    M_incyto = Int(floor(ratio*M))
    particles = zeros(Int32, M+M_incyto, 2)
    m = 0
    for i=1:M
        m +=1
        particles[m,1] = i
        particles[m,2] = rand([-1,1])
    end
    for j = 1:M_incyto
        m +=1
        particles[m,1] = 0
        particles[m,2] = rand([-1,1])
    end
    return particles
end

function make_particlelist_equal(M::Int64, N::Int64)
    
    M_incyto = Int(floor((N-M)/2))
    particles = zeros(Int32, 2*(M+M_incyto), 2)
    m = 0
    for i=1:M
        m +=1
        particles[m,1] = i
        particles[m,2] = -1
        m +=1
        particles[m,1] = i
        particles[m,2] = 1
    end
    for j = 1:M_incyto
        m +=1
        particles[m,1] = 0
        particles[m,2] = -1
        m +=1
        particles[m,1] = 0
        particles[m,2] = 1
    end
    return particles
end

function make_states(particles::Matrix{Int32},M::Int64)
    states_A = zeros(Int32, M)
    states_P = zeros(Int32, M)
    NcytoA = Int32(0)
    NcytoP = Int32(0)
    for pos_spin = eachrow(particles)
        if pos_spin[1] == 0
            if pos_spin[2] > 0
                NcytoA += 1
            else
                NcytoP += 1
            end
        else
            if pos_spin[2] > 0
                states_A[pos_spin[1]] += 1
            else
                states_P[pos_spin[1]] += 1
            end
        end
    end
    return states_A, states_P, NcytoA, NcytoP
end


function run_step_periodic!(particles::Matrix{Int32},states_A::Vector{Int32}, states_P::Vector{Int32}, N_cytos::Vector{Int32},
        nsteps::Int64,Ddt::Float64, Ddt_2::Float64, gammadt::Float64, kdt::Float64,
        h::Float64, idxp::Int64, idxm::Int64)
    # adapted from David's tau-leaping code.

    N = size(particles,1)
    M = size(states_A, 1)
    posX = 0
    spin = 0
    m = 0
    prob_flip = 0.
    prob_pref = h / (M+h)
    
    rate_bind = gammadt *(M+h)
    # pick a particle
    for t = 1:nsteps
        i = rand(1:N)
        # get the position of particle number i - if posX = 0 the particle is in the cytoplasm.
        posX = particles[i,1]
        spin = particles[i,2]
        
        if posX == 0 # particle is in the cytoplasm
            if spin > 0
                if rand() < rate_bind # Acyto is bound
                    if rand() < prob_pref
                        newpos = idxp
                    else
                        newpos = rand(1:M)
                    end
                    particles[i,1] = newpos
                    states_A[newpos] +=1
                    N_cytos[1] -=1
                end
            else
                if rand() < rate_bind  # Pcyto is bound
                    if rand() < prob_pref
                        newpos = idxm
                    else
                        newpos = rand(1:M)
                    end
                    particles[i,1] = newpos
                    states_P[newpos] +=1
                    N_cytos[2] -=1 
                end
            end         
        else # particle is membrane-bound
            
            # generate random number
            rd = rand()
            if spin > 0 # species A
                if rd < Ddt # move to the left
                    newpos = ifelse(posX == 1, M, posX-1)
                    particles[i,1] = newpos
                    states_A[posX] -= 1
                    states_A[newpos] += 1
                elseif rd < Ddt_2 # move to the right
                    newpos = ifelse(posX == M, 1, posX+1)
                    particles[i,1] = newpos
                    states_A[posX] -= 1
                    states_A[newpos] += 1
                else
                    P = states_P[posX]
                    prob_unbind = gammadt + kdt*P*(P-1)
                    if rd < Ddt_2+prob_unbind # unbind
                        particles[i,1] = 0
                        states_A[posX] -= 1
                        N_cytos[1] +=1
                    end
                end
            else # if spin <0: species P
            # compute rate of unbinding
                if rd < Ddt # move to the left
                    left = ifelse(posX == 1, M, posX-1)
                    newpos = left
                    particles[i,1] = newpos
                    states_P[posX] -= 1
                    states_P[newpos] += 1
                elseif rd < Ddt_2 # move to the right
                    right = ifelse(posX == M, 1, posX+1)
                    newpos = right
                    particles[i,1] = newpos
                    states_P[posX] -= 1
                    states_P[newpos] += 1
                else
                    A = states_A[posX]
                    prob_unbind = gammadt + kdt*A*(A-1)
                    if rd < Ddt_2+prob_unbind # unbind
                        particles[i,1] = 0
                        states_P[posX] -= 1
                        N_cytos[2] +=1
                    end
                end 
            end # actions if membrane bund
        end # if particle bound or not
    end # for t loop
end #function


function run_sim(params; seed=42, debug=false)
    
    a = params.a
    dx = params.dx
    L = params.L
    M = Int(L/a)
    N = params.N

    # choose dt so that 2Ddt/a^2 < 0.5
    
    n0 = Int(floor(params.N/2))
     
    h = params.h
    k = params.k
    dt = 0.5/(2*params.D/a^2 + gamma + k * (0.25*n0/M)^2)
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt
    kdt = k * dt
    
    nsteps = Int(floor(params.T / (params.nsaves*dt)))
    nsaves = Int(params.nsaves)


    if debug
        nsteps = 1
    end
    println("Starting sim. nsteps = $(nsteps), dt=$(dt)")

    particles = make_particlelist_equal(M, N)

    states_A, states_P, N_cytoA, N_cytoP = make_states(particles, M)

    # Saving
    #floor(L/block_size)
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr_A = zeros(Int32, nsaves, num_box)
    saves_arr_P = zeros(Int32, nsaves, num_box)
    saves_arr_Ncyto = zeros(Int32, nsaves, 2)

    # warmup-compile
    #run_step_periodic!([1,1,2], [2,1,0], 5, 0.25, 0.5, 0., 0.)
     N_cytos = [Int32(N_cytoA), Int32(N_cytoP)]
    println("N_cytos =", N_cytos)
    println("size particles =", size(particles))


    Random.seed!(seed)
    @showprogress for t = 1:nsaves
        run_step_periodic!(particles, states_A, states_P, N_cytos, nsteps, Ddt, Ddt_2,
                           gammadt, kdt,
                            h, idxp, idxm)
        #println(size(coarse_grain(states_rho, block_size, M)))
        #println(block_size)
        #println(num_box)
        saves_arr_A[t,:] .= coarse_grain1d(states_A, block_size, M)
        saves_arr_P[t,:] .= coarse_grain1d(states_P, block_size, M)
        saves_arr_Ncyto[t,:] = copy(N_cytos)
    end
    # returns current state for checkpointing
    return particles, states_A, states_P, N_cytos, saves_arr_A, saves_arr_P, saves_arr_Ncyto
end




function replicate(input_tuple)
    
    params = input_tuple[1]
    seed = input_tuple[2]
    
    a = params.a
    dx = params.dx
    L = params.L
    M = Int(L/a)
    N = params.N

    idxp = params.idxp
    idxm = params.idxm

    # choose dt so that the total proba per step <1
    gamma = params.gamma#/4 #rescaling by average number of neighbors
    h = params.h
    k = params.k
    
    # typical density is bounded by n0/2
    n0 = Int(floor(params.N/2))
    
    dt = 0.5/maximum([2*params.D/a^2 + gamma + k * (0.25*n0/M)^2, gamma*(M+h)])
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt
    kdt = k*dt

    nsteps = Int(floor(params.T / dt))

    #particles = make_particlelist_polarized(M, Int(N/2), Int(N/2))
    particles = make_particlelist_equal(M, N)
    
    states_A, states_P, N_cytoA, N_cytoP = make_states(particles, M)
        
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr = zeros(Int32, num_box, 2)

    # Seed and run
    Random.seed!(seed)
    N_cytos = [Int32(N_cytoA), Int32(N_cytoP)]
    run_step_periodic!(particles, states_A, states_P, N_cytos, nsteps, Ddt, Ddt_2, gammadt, kdt, h, idxp, idxm)

    saves_arr[:,1] .= coarse_grain1d(states_A + states_P, block_size, M)
    saves_arr[:,2] .= coarse_grain1d(states_A - states_P, block_size, M)
    return saves_arr
end


function replicate_polarized(input_tuple)
    
    params = input_tuple[1]
    seed = input_tuple[2]
    
    a = params.a
    dx = params.dx
    L = params.L
    M = Int(L/a)
    N = params.N

    idxp = params.idxp
    idxm = params.idxm

    # choose dt so that the total proba per step <1
    gamma = params.gamma#/4 #rescaling by average number of neighbors
    h = params.h
    k = params.k
    
    # typical density is bounded by n0/2
    n0 = Int(floor(params.N/2))
    
    dt = 0.5/maximum([2*params.D/a^2 + gamma + k * (0.25*n0/M)^2, gamma*(M+h)])
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt
    kdt = k*dt

    nsteps = Int(floor(params.T / dt))

    particles = make_particlelist_polarized(M, Int(N/2), Int(N/2))
    #particles = make_particlelist_equal(M, N)
    
    states_A, states_P, N_cytoA, N_cytoP = make_states(particles, M)
        
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr = zeros(Int32, num_box, 2)

    # Seed and run
    Random.seed!(seed)
    N_cytos = [Int32(N_cytoA), Int32(N_cytoP)]
    run_step_periodic!(particles, states_A, states_P, N_cytos, nsteps, Ddt, Ddt_2, gammadt, kdt, h, idxp, idxm)

    saves_arr[:,1] .= coarse_grain1d(states_A + states_P, block_size, M)
    saves_arr[:,2] .= coarse_grain1d(states_A - states_P, block_size, M)
    return saves_arr
end


function run_sim_replicates(fname::String, params, N_reps)
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps ]
        
    results = @showprogress pmap(replicate, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.gamma, params.k, params.h, params.T, params.idxp, params.idxm]
    for i = 1:N_reps
        g_id["$(i)"] = results[i]
    end
    close(fid)
    return nothing
end


function run_sim_replicates_polarized(fname::String, params, N_reps)
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps ]
        
    results = @showprogress pmap(replicate_polarized, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.gamma, params.k, params.h, params.T, params.idxp, params.idxm]
    for i = 1:N_reps
        g_id["$(i)"] = results[i]
    end
    close(fid)
    return nothing
end



