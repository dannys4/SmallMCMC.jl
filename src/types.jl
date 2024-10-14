abstract type AbstractMCMC_Alg end
abstract type AbstractMCMC_SymMH end

# A symmetric proposal doesn't need proposal density evals
log_proposal(mcmc::AbstractMCMC_SymMH, x, given_y) = 0.
function log_alpha(mcmc::AbstractMCMC_SymMH, x, y, p)
    mcmc.log_p(y, p) - mcmc.log_p(x, p)
end

# Generic MH kernel
struct MCMC_MH{D,T} <: AbstractMCMC_SymMH where {D<:Distribution, T}
    g::D
    log_p::T
    function MCMC_MH(g::D1, log_p::T1) where {D1, T1}
        new{D1,T1}(g,log_p)
    end
end

update!(_::MCMC_MH,_::Int,_) = nothing
samp_proposal(rng::AbstractRNG, mcmc::MCMC_MH, _, _) = rand(rng,mcmc.g)

struct MCMC_Global{T}
    g::T
    function MCMC_Global(g::T1) where {T1}
        new{T1}(g)
    end
end

update!(_::MCMC_Global, _::Int, _) = nothing
samp_proposal(rng::AbstractRNG, mcmc::MCMC_Global, _, p) = rand(rng, mcmc.g(p))
log_alpha(mcmc::MCMC_Global, _, _, _) = 0.

struct MCMC_Componentwise_t
    mcmcs::Vector{<:AbstractMCMC_SymMH}
    ds::Vector{Int}
end
