using FieldEncoders
using CUDA
using Distributions
using Test

@testset "NeRFEncoder" begin

    T = Float32
    inputdim = 3
    bz = 5
    xs = randn(T, inputdim, bz)
    CUDA.functional() && (cuda_xs = CUDA.randn(T, inputdim, bz))

    @test FieldEncoders.get_out_dim(NeRFEncoder(inputdim, 1, T(0), T(10))) == 2 * inputdim
    @test FieldEncoders.get_out_dim(NeRFEncoder(inputdim, 2, T(0), T(10))) == 2 * 2 * inputdim
    @test FieldEncoders.get_out_dim(NeRFEncoder(inputdim, 3, T(0), T(10))) == 3 * 2 * inputdim
    @test FieldEncoders.get_out_dim(NeRFEncoder(inputdim, 3, T(0), T(10), true)) == 3 * 2 * inputdim + inputdim

    @test encode(NeRFEncoder(inputdim, 1, T(0), T(10)), xs) ≈ vcat(sin.(2π .* xs), cos.(2π .* xs))
    @test encode(NeRFEncoder(inputdim, 1, T(0), T(10), true), xs) ≈ vcat(xs, sin.(2π .* xs), cos.(2π .* xs))
    @inferred encode(NeRFEncoder(inputdim, 1, T(0), T(10)), xs)
    @inferred encode(NeRFEncoder(inputdim, 1, T(0), T(10), true), xs)
    if CUDA.functional()
        @test encode(NeRFEncoder(inputdim, 1, T(0), T(10)), cuda_xs) ≈ vcat(sin.(2π .* cuda_xs), cos.(2π .* cuda_xs))
        @test encode(NeRFEncoder(inputdim, 1, T(0), T(10), true), cuda_xs) ≈ vcat(cuda_xs, sin.(2π .* cuda_xs), cos.(2π .* cuda_xs))
        @inferred encode(NeRFEncoder(inputdim, 1, T(0), T(10)), cuda_xs)
        @inferred encode(NeRFEncoder(inputdim, 1, T(0), T(10), true), cuda_xs)
    end

    numsamples = 200000
    #μ = randn(T, inputdim)
    Σ = stack(1:bz) do i
        half_cov = randn(T, inputdim, inputdim)
        half_cov * half_cov'
    end
    #half_cov = randn(T, inputdim, inputdim)
    #Σ = half_cov * half_cov'
    @inferred encode(NeRFEncoder(inputdim, 1, T(0), T(10)), xs, Σ)
    if CUDA.functional()
        cuda_Σ = CuArray(Σ)
        @inferred encode(NeRFEncoder(inputdim, 1, T(0), T(10)), cuda_xs, cuda_Σ)
    end
    # encoder1 = NeRFEncoder(inputdim, 1, T(0), T(10))
    # enc_x = encode(encoder1, μ, Σ)
    # s = rand(MvNormal(μ, Σ), numsamples)
    # enc_s = encode(encoder1, s)
    # enc_x == sum(enc_s; dims = 2) ./ numsamples

end