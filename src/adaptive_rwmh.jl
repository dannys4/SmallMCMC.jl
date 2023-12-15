struct MCMC_Adaptive_RWMH{T} <: AbstractMCMC_SymMH where {T}
    x_bar::Vector{Float64}
    cov_unc::Matrix{Float64}
    log_p::T
    s_d::Float64
    eps::Float64
    n_0::Int
    C_0::Matrix{Float64}
    exceed_n_0::Ref{Bool}
    n_stop_adapt::Int
end

function reset!(mcmc::MCMC_Adaptive_RWMH)
    mcmc.exceed_n_0[] = false
    mcmc.x_bar .= 0
    mcmc.cov_unc .= 0
end

"""
Constructor for the adaptive MCMC sampler
    
    MCMC_Adaptive_RWMH(log_p, C_0; n_0, n_stop_adapt, s_d, eps)

Arguments:

    log_p(x,p): log target density R^n->R sampled from with argument x and parameter p
    C_0: Initial covariance, R^{n x n}
    n_0: When to start using adaptive covariance matrix (default 1000)
    n_stop_adapt: When to stop using adaptive covariance matrix (default 10,000)
    s_d, eps: scaling, inflation factors in adaptive MCMC (see Haario et al)
"""
function MCMC_Adaptive_RWMH(log_p::T, C_0; s_d=2.4^2/size(C_0,2), eps=0.001, n_0=1000, n_stop_adapt=10_000) where {T}
    M,N = size(C_0)
    @assert M == N "C_0 must be square"
    x_bar = zeros(N)
    cov_unc = zeros(N,N)
    exceed_n_0 = Ref(false)
    MCMC_Adaptive_RWMH{T}(x_bar, cov_unc, log_p, s_d, eps, n_0, C_0, exceed_n_0, n_stop_adapt)
end

function getcov(mcmc::MCMC_Adaptive_RWMH)
    if mcmc.exceed_n_0[]
        return mcmc.s_d*(mcmc.cov_unc .- (mcmc.x_bar*mcmc.x_bar') + mcmc.eps*I)
    end
    mcmc.C_0
end

function update!(mcmc::MCMC_Adaptive_RWMH, n::Int, samp)
    n == mcmc.n_0 && (mcmc.exceed_n_0[] = true)
    n > mcmc.n_stop_adapt && return
    if n == 2
        mcmc.cov_unc .= mcmc.x_bar * mcmc.x_bar' + samp*samp'
        mcmc.x_bar .= (mcmc.x_bar + samp)/2
        return
    end
    mcmc.x_bar .= mcmc.x_bar*(n-1)/n + samp/n
    mcmc.cov_unc .= mcmc.cov_unc*(n-2)/(n-1) .+ (samp*samp')/(n-1)
end

# Proposal sample for adaptive kernel
function samp_proposal(rng::AbstractRNG, mcmc::MCMC_Adaptive_RWMH, x_prev, _)
    cov_cen = getcov(mcmc)
    rand(rng, MvNormal(x_prev, cov_cen))
end