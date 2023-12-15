module SmallMCMC
using Distributions, ProgressMeter, LinearAlgebra, Random

include("types.jl")
include("adaptive_rwmh.jl")
include("impl.jl")
include("utils.jl")

export mcmc_sample, MCMC_Adaptive_RWMH, MCMC_MH, MCMC_Global

end
