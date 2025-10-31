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

function ∇²(f)
    ∇²f = zero(f) # initialize to zero since we don't touch the boundaries
    for y = 2:size(f,2)-1, x = 2:size(f,1)-1 # compute ∇²f in interior of f
        ∇²f[x,y] = f[x-1,y] + f[x+1,y] + f[x,y-1] + f[x,y+1] - 4f[x,y]
    end
    return ∇²f
 end

 function lap1d(f)
    ∇²f = zero(f) # initialize to zero since we don't touch the boundaries
    for x = 2:size(f,1)-1 # compute ∇²f in interior of f
        ∇²f[x] = f[x-1] + f[x+1] - 2f[x]
    end
    ∇²f[1] = f[end] + f[2] - 2*f[1]
    ∇²f[end] = f[end-1] + f[1] - 2*f[end]
    return ∇²f
 end

 function div1d(f)
    divf = zero(f) # initialize to zero since we don't touch the boundaries
    for x = 1:size(f,2)-1 # compute ∇²f in interior of f
        divf[x] = f[x+1] - f[x]
    end
    divf[end] = f[1] - f[end]
    return divf
 end


function LandauGinzburg(n_step, n_sub, x0, D, gamma, dx, dt, N, r, h0, idx1, idx2)
    
    n_check = Int(ceil(n_step / n_sub))
    x = zeros(n_sub, N)
    x[1,:] = x0[:]
    noise = sqrt(2*gamma*dt)
    diff_coef = D/dx^2*dt
    x_c = x0[:]
    x_n = x0[:]
    rdt = r*dt
    h0dt = h0 *dt
    i_count = 1
    for n = 2:(n_step)
        lap = diff_coef * lap1d(x_c)
        x_c = x_c + rdt * x_c + lap - dt* (x_c .^3) + noise * randn(N)
        x_c[idx1] += h0dt
        x_c[idx2] -= h0dt
        if n % n_check == 0
            i_count += 1
            x[i_count,:] = x_c[:]
            if i_count == n_sub
                break
            end
        end
    end
    return x
end


function replicate(input_tuple)
    
    params = input_tuple[1]
    seed = input_tuple[2]
    
    
    n_step = Int(params.n_step)
    n_sub = Int(params.n_sub)
    x0 = params.x0
    N =  params.N
    D = params.D
    r = params.r
    gamma =  params.gamma
    L = params.L
    dx = params.dx
    dt = params.dt
    h0 = params.h0
    idx1 = Int(params.idx1)
    idx2 = Int(params.idx2)
    #noiseamp = params.noiseamp
   

    # Seed and run
    Random.seed!(seed)
    
    
    x = LandauGinzburg(n_step, n_sub, x0, D, gamma, dx, dt, N, r, h0, idx1, idx2)

    return x
end

function run_replicates(fname::String, params, N_reps)
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps]
        
    results = @showprogress pmap(replicate, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.h0, params.D, params.N, params. params.L, params.dt, params.dx]
    for i = 1:N_reps
        create_group(g_id, "$(i)")
        g_id2 = g_id["$(i)"]
        g_id2["x"] = results[i]
    end
    close(fid)
    return nothing
end

function replicate_disordered(input_tuple)
    
    params = input_tuple[1]
    seed = input_tuple[2]
    
    
    n_step = Int(params.n_step)
    n_sub = Int(params.n_sub)
    N =  params.N
    D = params.D
    r = params.r
    gamma =  params.gamma
    L = params.L
    dx = params.dx
    dt = params.dt
    h0 = params.h0
    idx1 = Int(params.idx1)
    idx2 = Int(params.idx2)

    # Seed and run
    Random.seed!(seed)
    x0 = 0.1*randn(N)
    
    x = phi4(n_step, n_sub, x0, D, gamma, dx, dt, N, r, h0, idx1, idx2)

    return x
end

function run_replicates_disordered(fname::String, params, N_reps)
    
    
    jobs_arr = [(params, i*Int(floor(time()))) for i = 1:N_reps]
        
    results = @showprogress pmap(replicate_disordered, jobs_arr)
    
    fid = h5open(fname, "w")
    create_group(fid, "res")
    g_id =  fid["res"]
    attributes(g_id)["params"] = [params.h0, params.D, params.N, params.gamma, params.L, params.dt, params.dx]
    for i = 1:N_reps
        create_group(g_id, "$(i)")
        g_id2 = g_id["$(i)"]
        g_id2["x"] = results[i]
    end
    close(fid)
    return nothing
end
