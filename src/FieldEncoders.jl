module FieldEncoders

using ChainRulesCore: @ignore_derivatives
using DocStringExtensions
using EllipsisNotation
using Flux
using LinearAlgebra: diag
using MLUtils

export get_out_dim, encode 

abstract type AbstractFieldEncoder end

include("utils.jl")
include("nerf.jl")
include("hash.jl")
include("sphericalharmonic.jl")

(encoder::AbstractFieldEncoder)(xs, args...) = encode(encoder, xs, args...)

end # module FieldEncoders
