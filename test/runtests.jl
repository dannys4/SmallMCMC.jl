using SmallMCMC
using Test
using Aqua
using JET

@testset "SmallMCMC.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(SmallMCMC)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(SmallMCMC; target_defined_modules = true)
    end
    # Write your tests here.
end
