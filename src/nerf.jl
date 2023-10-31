export NeRFEncoder

"""
$(TYPEDFIELDS)

NeRF encoder.
"""
struct NeRFEncoder{T<:Real} <: AbstractFieldEncoder 
    "Input dimension"
    inputdim::Int 
    "Number of encoded frequencies"
    numfrequencies::Int
    "Minimum frequency"
    minfreq::T
    "Maximum frequency"
    maxfreq::T
    "Append the position input to the encoding"
    includeinput::Bool
    function NeRFEncoder{T}(inputdim, numfrequencies, minfreq, maxfreq, includeinput) where {T<:Real}
        numfrequencies ≤ 0 && throw(ArgumentError("`numfrequencies` has to be a nonzero positive integer"))
        minfreq ≥ maxfreq && throw(ArgumentError("`minfreq` has to be less than `maxfreq`"))
        new(inputdim, numfrequencies, minfreq, maxfreq, includeinput)
    end
end
NeRFEncoder(inputdim, numfrequencies, minfreq::T, maxfreq::T, includeinput::Bool) where {T} = 
    NeRFEncoder{T}(inputdim, numfrequencies, minfreq, maxfreq, includeinput)
NeRFEncoder(inputdim, numfrequencies, minfreq::T, maxfreq::T) where {T} = 
    NeRFEncoder{T}(inputdim, numfrequencies, minfreq, maxfreq, false)

"""
$(TYPEDSIGNATURES)

Get output dimension.
"""
function get_out_dim(encoder::NeRFEncoder)
    outdim = encoder.inputdim * encoder.numfrequencies * 2
    encoder.includeinput && (outdim += encoder.inputdim)
    return outdim
end

"""
$(TYPEDSIGNATURES)

Transform positions `xs` using NeRF positional encoding or integrated positional encodings if `covs` is provided.

# Reference
- NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis, Mildenhall et al., ECCV 2020
- Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields, Barron et al., Arxiv 2021
"""
function encode(encoder::NeRFEncoder{T}, 
                xs::AbstractArray{T},
                covs::Union{Nothing,AbstractArray{T}}=nothing) where {T}
    check_inputdim(encoder, xs)
    a = reshape(T(2π) .* xs, 1, encoder.inputdim, :) # [inputdim,1,B]
    rs = isone(encoder.numfrequencies) ? (encoder.minfreq:encoder.minfreq) : 
        range(encoder.minfreq, encoder.maxfreq, encoder.numfrequencies)
    freqs = convert(typeof(xs).name.wrapper, 2 .^ rs) # [1,numfrequencies]
    scaled_xs = a .* freqs # [inputdim,numfrequencies,B]
    scaled_xs = reshape(scaled_xs, encoder.inputdim*encoder.numfrequencies, size(xs)[2:end]...)  # [inputdim*numfrequencies,B]
    encoded_xs = if isnothing(covs)
        sin.(vcat(scaled_xs, scaled_xs .+ T(π/2)))
    else
        freqspow2 = freqs .^ 2
        xs_var = stack(eachslice(covs, dims=tuple(3:ndims(covs)...))) do covs1
            vec(freqspow2 .* transpose(diag(covs1)))
        end 
        expected_sin(
            vcat(scaled_xs, scaled_xs .+ T(π/2)),
            vcat(xs_var, xs_var),
        )
    end
    encoder.includeinput ? vcat(xs, encoded_xs) : encoded_xs
end

(encoder::NeRFEncoder)(xs, args...) = encode(encoder, xs, args...)
