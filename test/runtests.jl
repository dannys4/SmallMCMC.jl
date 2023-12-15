using SmallMCMC, Test, Aqua, JET, Random, Distributions, LinearAlgebra

include("gmm.jl")

@testset "SmallMCMC.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SmallMCMC, ambiguities=false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SmallMCMC; target_defined_modules = true)
    end
    # Write your tests here.
    rng = Xoshiro(284028)
    @testset "MCMC Tests" begin
        GMMTest(rng)
    end
end
