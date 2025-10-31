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


function coarsegrain2d(state::Vector{Int32}, dx::Number)
    M = length(state)
    num_box = Int(ceil(M/dx))
    binned = zeros(Int32, num_box, num_box)
    for i in 1:M
        idx_i = Int(ceil(i/dx))
        for j in 1:M
            idx_j = Int(ceil(j/dx))
            binned[idx_i, idx_j] += state[i,j]
        end
    end
    #binned_rho[end] += state[M]
    return binned
end

function brange(i, block_size)
    istart = (i - 1) * block_size + 1
    iend = istart + block_size - 1
    return istart:iend
end

function coarse_grain(s, block_size, L)
    #step = ceil(Int8, L*dx_over_a)
    num_blocks = Int(floor(L/block_size))
    sp = zeros((num_blocks,num_blocks))
    for i = 1:num_blocks
        for j = 1:num_blocks
            sp[i,j] = sum(s[brange(i, block_size), brange(j, block_size)])
        end
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

function make_particlelist(M::Int64, N::Int64)
    particles = zeros(Int32, N, 2)
    for m = 1:N
        particles[m,1] = rand(1:M)
        particles[m,2] = rand([-1,1])
    end
    return particles
end


function make_particlelist_homogeneous(M::Int64)
    particles = zeros(Int32, M, 2)
    m = 0
    for i=1:M
        m +=1
        particles[m,1] = i
        particles[m,2] = rand([-1,1])
    end
    return particles
end


function make_particlelist_polarized(M::Int64)
    particles = zeros(Int32, M, 2)
    m = 0
    M2 = Int(floor(M/2))
    for i=1:M2
        m +=1
        particles[m,1] = i
        particles[m,2] = 1
    end
    
    for i=(M2+1):M
        m +=1
        particles[m,1] = i
        particles[m,2] = -1
    end
    return particles
end

function make_states(particles::Matrix{Int32},M::Int64)
    states_rho = zeros(Int32, M)
    states_m = zeros(Int32, M)
    for pos_spin = eachrow(particles)
        states_rho[pos_spin[1]] += 1
        states_m[pos_spin[1]] += pos_spin[2]
    end
    return states_rho, states_m
end

function run_step_periodic!(particles::Matrix{Int32}, states_rho::Vector{Int32}, states_m::Vector{Int32}, nsteps::Int64,
        Ddt::Float64, Ddt_2::Float64, gammadt::Float64, beta::Float64, h::Float64, idxp::Int64, idxm::Int64)
    # adapt David's tau-leaping code.

    N = size(particles,1)
    M = size(states_rho, 1)
    posX = 0
    spin = 0
    m = 0
    prob_flip = 0.

    rates = [gammadt*exp(-beta*m) for m = -10:10]
    # pick a particle
    for t = 1:nsteps
        i = rand(1:N)
        # get the position of particle number i
        posX = particles[i,1]
        spin = particles[i,2]
        m = states_m[posX]
        # compute rate of flip
        left = ifelse(posX == 1, M, posX-1)
        right = ifelse(posX == M, 1, posX+1)
        
        if posX == idxp
            prob_flip = gammadt*(exp(-beta*m*spin - h*spin))
        elseif posX == idxm
            prob_flip = gammadt*(exp(-beta*m*spin + h*spin))
        else
            if abs(m) < 10
                prob_flip = rates[m*spin+11] #  pre-computed values for small values of |m|
            else
                prob_flip = gammadt*(exp(-beta*m*spin))
            end
        end
        # generate random number
            
        rd = rand()
        if rd < Ddt # move to the left
            newpos = left
            particles[i,1] = newpos
            states_rho[posX] -= 1
            states_rho[newpos] += 1
            states_m[posX] -= spin
            states_m[newpos] += spin
        elseif rd < Ddt_2 # move to the right
            newpos = right
            particles[i,1] = newpos
            states_rho[posX] -= 1
            states_rho[newpos] += 1
            states_m[posX] -= spin
            states_m[newpos] += spin
        elseif rd < Ddt_2+prob_flip # flip
            states_m[posX] -= 2*spin
            particles[i,2] = -spin
        end
    end
end


function run_sim(params; seed=42, debug=false)
    
    a = params.a
    dx = params.dx
    L = params.L
    M = Int(L/a)
    N = params.N

    # choose dt so that 2Ddt/a^2 < 0.5
    beta = params.beta 
    h = params.h
    dt = 0.5/(2*params.D/a^2 + params.gamma*exp(5*beta+h))
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt
    
    nsteps = Int(floor(params.T / (params.nsaves*dt)))
    nsaves = Int(params.nsaves)


    if debug
        nsteps = 1
    end
    println("Starting sim. nsteps = $(nsteps), dt=$(dt)")

    particles = make_particlelist(M,N)
    states_rho, states_m = make_states(particles, M)

    # Saving
    #floor(L/block_size)
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr_rho = zeros(Int32, nsaves, num_box, num_box)
    saves_arr_m = zeros(Int32, nsaves, num_box, num_box)

    # warmup-compile
    #run_step_periodic!([1,1,2], [2,1,0], 5, 0.25, 0.5, 0., 0.)
 


    Random.seed!(seed)
    @showprogress for t = 1:nsaves
        run_step_periodic!(particles, states_rho, states_m, nsteps, Ddt, Ddt_2, gammadt, beta)
        #println(size(coarse_grain(states_rho, block_size, M)))
        #println(block_size)
        #println(num_box)
        saves_arr_rho[t,:,:] .= coarse_grain(states_rho, block_size, M)
        saves_arr_m[t,:,:] .= coarse_grain(states_m, block_size, M)
    end
    # returns current state for checkpointing
    return particles, states_rho, states_m, saves_arr_rho, saves_arr_m
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
    beta = params.beta#/4 #rescaling by average number of neighbors
    h = params.h
    dt = 0.5/(2*params.D/a^2 + params.gamma*exp(5*beta+h))
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt

    nsteps = Int(floor(params.T / dt))

    particles = make_particlelist_homogeneous(M)
    N = M
    
    states_rho, states_m = make_states(particles, M)
        
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr = zeros(Int32, num_box, 2)

    # Seed and run
    Random.seed!(seed)
    
    run_step_periodic!(particles, states_rho, states_m, nsteps, Ddt, Ddt_2, gammadt, beta, h, idxp, idxm)

    saves_arr[:,1] .= coarse_grain1d(states_rho, block_size, M)
    saves_arr[:,2] .= coarse_grain1d(states_m, block_size, M)
    return saves_arr
end



function replicate_time(input_tuple)
    
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
    beta = params.beta#/4 #rescaling by average number of neighbors
    h = params.h
    dt = 0.5/(2*params.D/a^2 + params.gamma*exp(5*beta+h))
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt

    nsaves = 100
    nsteps = Int(floor(params.T / (nsaves*dt)))
    

    particles = make_particlelist_homogeneous(M)
    N = M
    
    states_rho, states_m = make_states(particles, M)
        
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr = zeros(Int32, nsaves, num_box, 2)

    # Seed and run
    Random.seed!(seed)
    for t = 1:nsaves
    run_step_periodic!(particles, states_rho, states_m, nsteps, Ddt, Ddt_2, gammadt, beta, h, idxp, idxm)
        saves_arr[t,:,1] .= states_rho[:] #coarse_grain1d(states_rho, block_size, M)
        saves_arr[t,:,2] .= states_m[:] #coarse_grain1d(states_m, block_size, M)
    end
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
    beta = params.beta#/4 #rescaling by average number of neighbors
    h = params.h
    dt = 0.5/(2*params.D/a^2 + params.gamma*exp(5*beta+h))
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt

    nsteps = Int(floor(params.T / dt))

    particles = make_particlelist_polarized(M)
    N = M
    
    states_rho, states_m = make_states(particles, M)
        
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr = zeros(Int32, num_box, 2)

    # Seed and run
    Random.seed!(seed)
    
    run_step_periodic!(particles, states_rho, states_m, nsteps, Ddt, Ddt_2, gammadt, beta, h, idxp, idxm)

    saves_arr[:,1] .= coarse_grain1d(states_rho, block_size, M)
    saves_arr[:,2] .= coarse_grain1d(states_m, block_size, M)
    return saves_arr
end


function replicate_time_polarized(input_tuple)
    
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
    beta = params.beta#/4 #rescaling by average number of neighbors
    h = params.h
    dt = 0.5/(2*params.D/a^2 + params.gamma*exp(5*beta+h))
    Ddt = params.D*dt/a^2 
    Ddt_2 = 2*Ddt 
    gammadt = params.gamma * dt

    nsaves = 100
    nsteps = Int(floor(params.T / (nsaves*dt)))
    

    particles = make_particlelist_polarized(M)
    N = M
    
    states_rho, states_m = make_states(particles, M)
        
    block_size = Int(dx/a)
    num_box = Int(floor(M/block_size))
    saves_arr = zeros(Int32, nsaves, num_box, 2)

    # Seed and run
    Random.seed!(seed)
    for t = 1:nsaves
    run_step_periodic!(particles, states_rho, states_m, nsteps, Ddt, Ddt_2, gammadt, beta, h, idxp, idxm)
        saves_arr[t,:,1] .= coarse_grain1d(states_rho, block_size, M)
        saves_arr[t,:,2] .= coarse_grain1d(states_m, block_size, M)
    end
    return saves_arr
end

function run_sim_replicates(fname::String, params, N_reps)
    
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps ]
        
    results = @showprogress pmap(replicate, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.beta, params.gamma, params.h, params.T, params.idxp, params.idxm]
    for i = 1:N_reps
        g_id["$(i)"] = results[i]
    end
    close(fid)
    return nothing
end



function run_sim_replicates_time(fname::String, params, N_reps)
    
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps ]
        
    results = @showprogress pmap(replicate_time, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.beta, params.gamma, params.h, params.T, params.idxp, params.idxm]
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
    attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.beta, params.gamma, params.h, params.T, params.idxp, params.idxm]
    for i = 1:N_reps
        g_id["$(i)"] = results[i]
    end
    close(fid)
    return nothing
end

function run_sim_replicates_time_polarized(fname::String, params, N_reps)
    
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps ]
        
    results = @showprogress pmap(replicate_time_polarized, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.beta, params.gamma, params.h, params.T, params.idxp, params.idxm]
    for i = 1:N_reps
        g_id["$(i)"] = results[i]
    end
    close(fid)
    return nothing
end

function run_sim_replicates_draft(fname::String, params, N_reps; seed=42, debug=false, homogeneous=true)
    
    fid = h5open(fname, "w")
    
    status = @showprogress pmap(1:N_reps) do idx
        a = params.a
        dx = params.dx
        L = params.L
        M = Int(L/a)
        N = params.N
        
        idxp = params.idxp
        idxm = params.idxm

        # choose dt so that the total proba per step <1
        beta = params.beta#/4 #rescaling by average number of neighbors
        h = params.h
        dt = 0.5/(2*params.D/a^2 + params.gamma*exp(5*beta+h))
        Ddt = params.D*dt/a^2 
        Ddt_2 = 2*Ddt 
        gammadt = params.gamma * dt

        nsteps = Int(floor(params.T / (params.nsaves*dt)))
        nsaves = Int(params.nsaves)


        if debug
            nsteps = 1
        end
        println("Starting sim: D=$(params.D), 1/a=$(1/params.a), beta=$(params.beta), nsteps = $(nsteps), dt=$(dt)")

        if homogeneous
            particles = make_particlelist_homogeneous(M)
            N = M
        else
            particles = make_particlelist(M,N)
        end
        states_rho, states_m = make_states(particles, M)
        
        block_size = Int(dx/a)
        num_box = Int(floor(M/block_size))
        saves_arr = zeros(Int32, nsaves, num_box, 2)

        # Saving
        create_group(fid, "$(idx)")
        g_id =  fid["$(idx)"]
        attributes(g_id)["params"] = [params.a, params.L, params.dx, params.N, params.D, params.beta, params.gamma, params.h, dt, params.T, params.idxp, params.idxm]
        Random.seed!(seed+i*Int(floor(time())))
        for t = 1:nsaves
            run_step_periodic!(particles, states_rho, states_m, nsteps, Ddt, Ddt_2, gammadt, beta, h, idxp, idxm)

            saves_arr[t,:,1] .= coarse_grain1d(states_rho, block_size, M)
            saves_arr[t,:,2] .= coarse_grain1d(states_m, block_size, M)
        end
        # save final state for checkpointing
    end
    close(fid)
    return nothing
end



function warmup()
    L = 3.0
    a= 1/4
    gamma=1.0
    params = (
        a=a,
    dx = 0.5,
    L = L,
    D = 1.0,
    N = Int(L/a),
    gamma = gamma,
    beta = 0. *log(1+sqrt(2)),
    T = 1,
    J=1.0,
    nsaves= 0.01
    )
    run_sim(params; seed=42, debug=false)
end




