
"""
Compute an MCMC chain of length `n` using the `mcmc` algorithm, starting at `x0`

    mcmc_sample(mcmc::AbstractMCMC_Alg, x0, n_samples, p; verbose, rng)

Arguments:

    mcmc: Algorithm set up to sample
    x0: Initial sample
    n: Length of chain
    p: Parameters for `mcmc` to calculate log likelihood and sample from proposal (default nothing)
    verbose: Include progress bar
    rng: Random number generator (from Random)

Returns:

    chain: Chain of `n+1` samples (including x0)
"""
function mcmc_sample(mcmc::AbstractMCMC_SymMH, x0::Vector{Float64}, n::Int, p = nothing; verbose = true, rng::AbstractRNG = Random.GLOBAL_RNG)
    x = x0
    chain = Matrix{Float64}(undef, length(x0), n+1)
    chain[:,1] = x0
    verbose && (prog = Progress(n, 1, "Computing MCMC Chain..."))
    for i in 1:n
        y = samp_proposal(rng, mcmc, x, p)
        l_alpha = log_alpha(mcmc, x, y, p)
        if log(rand(rng)) < l_alpha
            x = y
        end
        chain[:,i+1] .= x
        update!(mcmc, i, x)
        verbose && next!(prog)
    end
    chain
end


function mcmc_sample(mcmcs::MCMC_Componentwise_t, x0::Vector{Float64}, n::Int, p = nothing; verbose = true, rng::AbstractRNG = Random.GLOBAL_RNG)
    x = x0
    @assert length(x0) == sum(mcmcs.ds) "x0 must be of length $(sum(mcmcs.ds))"
    chain = Matrix{Float64}(undef, length(x0), n+1)
    slices = [0; cumsum(mcmcs.ds)]
    d = length(mcmcs.ds)
    chain[:,1] = x0
    verbose && (prog = Progress(n, 1, "Computing MCMC Chain..."))
    for i in 1:n
        for j in 1:d
            mcmc = mcmcs.mcmcs[j]
            slice = slices[j]+1:slices[j+1]
            x_view = @view x[slice]
            x_param = @view x[[1:slices[j]; (slices[j+1]+1):slices[end]]]
            params = isnothing(p) ? x_param : (p, x_param)
            y = samp_proposal(rng, mcmc, x_view, params)
            l_alpha = log_alpha(mcmc, x_view, y, params)
            if log(rand(rng)) < l_alpha
                x_view .= y
            end
            update!(mcmc, i, x_view)
        end
        chain[:,i+1] .= x
        verbose && next!(prog)
    end
    chain
end