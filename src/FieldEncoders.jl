module FieldEncoders

using DocStringExtensions
using LinearAlgebra
using MLUtils

export get_out_dim, encode 

abstract type AbstractFieldEncoder end

include("utils.jl")
include("nerf.jl")

end # module FieldEncoders
