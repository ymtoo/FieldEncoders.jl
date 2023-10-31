module FieldEncoders

using ChainRulesCore: @ignore_derivatives
using DocStringExtensions
using Flux
using LinearAlgebra: diag
using MLUtils

export get_out_dim, encode 

abstract type AbstractFieldEncoder end

include("utils.jl")
include("nerf.jl")
include("hash.jl")

end # module FieldEncoders
