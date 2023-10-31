using FieldEncoders
using CUDA
using Distributions
using Test
using Zygote

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

    Σ = stack(1:bz) do i
        half_cov = randn(T, inputdim, inputdim)
        half_cov * half_cov'
    end
    encoder = NeRFEncoder(inputdim, 1, T(0), T(10))
    @inferred encode(encoder, xs, Σ)
    
    Zygote.pullback(encoder, xs, Σ) do encoder1, xs1, Σ1
        encode(encoder1, xs1, Σ1)
    end

    if CUDA.functional()
        cuda_Σ = CuArray(Σ)
        @inferred encode(encoder, cuda_xs, cuda_Σ)

        Zygote.pullback(encoder, cuda_xs, cuda_Σ) do encoder1, cuda_xs1, cuda_Σ1
            encode(encoder1, cuda_xs1, cuda_Σ1)
        end
    end

end

@testset "HashEncoder" begin

    resolution = 128
    xs = range(0, 1, resolution)
    ys = range(0, 1, resolution)
    zs = range(0, 1, resolution)
    inputs = FieldEncoders.meshgrid(xs, ys, zs)

    numlevels = 8
    minres = 2
    maxres = 128
    log2_hashmap_size = 4
    features_per_level = 3
    hash_init_scale = 0.001f0
    encoder = HashEncoder(numlevels, minres, maxres, log2_hashmap_size, features_per_level, hash_init_scale)
    encoded = encode(encoder, inputs)
    @test size(encoded) == (numlevels * features_per_level, length(xs), length(ys), length(zs))
    @inferred encode(encoder, inputs)
    
    Zygote.pullback(encoder, inputs) do encoder1, inputs1
        encode(encoder1, inputs1)
    end
    
    if CUDA.functional()
        cuda_encoder = HashEncoder(numlevels, minres, maxres, log2_hashmap_size, features_per_level, hash_init_scale, CuArray)
        cuda_inputs = CuArray(inputs)
        cuda_encoded = encode(cuda_encoder, cuda_inputs)
        @test size(cuda_encoded) == (numlevels * features_per_level, length(xs), length(ys), length(zs))
        @inferred encode(cuda_encoder, cuda_inputs)

        Zygote.pullback(cuda_encoder, cuda_inputs) do cuda_encoder1, cuda_inputs1
            encode(cuda_encoder1, cuda_inputs1)
        end
    end

end
