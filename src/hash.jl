export HashEncoder

"""
$(TYPEDFIELDS)

Hash encoder.
"""
struct HashEncoder{ST,HT,ET} <: AbstractFieldEncoder
    "Number of feature grids"
    numlevels::Int64
    "Resolution of smallest feature grid"
    minres::Int64
    "Resolution of largest feature grid"
    maxres::Int64
    "Size of hash map is 2 ^ log2_hashmap_size"
    log2_hashmap_size::Int64
    "Number of features per level"
    features_per_level::Int64
    "Value to initialize hash grid"
    hash_init_scale::Float32
    "Number of columns of the hash table"
    hash_table_size::Int64
    "Growth factor"
    growth_factor::Float32
    "Scaling factors"
    scalings::ST
    "offsets of the hash table"
    hash_offset::HT
    "Hash table"
    hash_table::Embedding{ET}
end

Flux.@functor HashEncoder
Flux.trainable(m::HashEncoder) = (; m.hash_table)

function HashEncoder(numlevels::Int64, 
                     minres::Int64, 
                     maxres::Int64, 
                     log2_hashmap_size::Int64, 
                     features_per_level::Int64, 
                     hash_init_scale::Float32, 
                     ondevice=cpu)
    hash_table_size = Int64(2) ^ log2_hashmap_size
    levels = range(0, numlevels-1) |> collect |> ondevice
    growth_factor = (numlevels > 1 ? exp((log(maxres) - log(minres)) / (numlevels - 1)) : 1.0) |> Float32
    scalings = floor.(minres .* growth_factor .^ levels)
    hash_offset = levels .* hash_table_size
    x = randn(Float32, features_per_level, hash_table_size * numlevels) .* 2 .- 1 |> ondevice
    hash_table = Embedding(x)
    HashEncoder(numlevels, minres, maxres, log2_hashmap_size, features_per_level, hash_init_scale, 
        hash_table_size, growth_factor, scalings, hash_offset, hash_table)
end

HashEncoder(ondevice=cpu) = HashEncoder(Int64(16), 
                                        Int64(16), 
                                        Int64(1024), 
                                        Int64(19), 
                                        Int64(2), 
                                        Float32(0.001), 
                                        ondevice)

"""
$(TYPEDSIGNATURES)

Get output dimension.
"""
function get_out_dim(encoder::HashEncoder)
    return encoder.numlevels * encoder.features_per_level
end

function hash_fn(input::AbstractArray, hash_table_size, hash_offset)
    input = input .* convert(typeof(input).name.wrapper, [1,2654435761,805459861])
    x = selectdim(input, 1, 1) .⊻ selectdim(input, 1, 2)
    x .⊻= selectdim(input, 1, 3)
    x .%= hash_table_size
    x .+= hash_offset
    x
end

"""
$(TYPEDSIGNATURES)

Transform positions `xs` using hashing encoding.

# Reference
Instant Neural Graphics Primitives with a Multiresolution Hash Encoding, Müller et al., ACM Trans. Graph., 2022
"""
function encode(encoder::HashEncoder, xs::AbstractArray)
    @assert size(xs, 1) == 3
    xs = unsqueeze(xs; dims=2) # [3,1,...]
    scaled = xs .* unsqueeze(encoder.scalings; dims=1) .+ 1 # [3,L,...]
    scaled_c = ceil.(Int64, scaled) 
    scaled_f = floor.(Int64, scaled)

    offset = scaled - scaled_f

    hashed_0 = @ignore_derivatives hash_fn(scaled_c, encoder.hash_table_size, encoder.hash_offset)
    hashed_1 = @ignore_derivatives hash_fn(cat(unsqueeze(selectdim(scaled_c, 1, 1); dims=1),
                                               unsqueeze(selectdim(scaled_f, 1, 2); dims=1),
                                               unsqueeze(selectdim(scaled_c, 1, 3); dims=1); dims=1),
                                            encoder.hash_table_size, encoder.hash_offset)
    hashed_2 = @ignore_derivatives hash_fn(cat(unsqueeze(selectdim(scaled_f, 1, 1); dims=1),
                                               unsqueeze(selectdim(scaled_f, 1, 2); dims=1),
                                               unsqueeze(selectdim(scaled_c, 1, 3); dims=1); dims=1),
                                            encoder.hash_table_size, encoder.hash_offset)
    hashed_3 = @ignore_derivatives hash_fn(cat(unsqueeze(selectdim(scaled_f, 1, 1); dims=1),
                                               unsqueeze(selectdim(scaled_c, 1, 2); dims=1),
                                               unsqueeze(selectdim(scaled_c, 1, 3); dims=1); dims=1),
                                            encoder.hash_table_size, encoder.hash_offset)
    hashed_4 = @ignore_derivatives hash_fn(cat(unsqueeze(selectdim(scaled_c, 1, 1); dims=1),
                                               unsqueeze(selectdim(scaled_c, 1, 2); dims=1),
                                               unsqueeze(selectdim(scaled_f, 1, 3); dims=1); dims=1),
                                            encoder.hash_table_size, encoder.hash_offset)
    hashed_5 = @ignore_derivatives hash_fn(cat(unsqueeze(selectdim(scaled_c, 1, 1); dims=1),
                                               unsqueeze(selectdim(scaled_f, 1, 2); dims=1),
                                               unsqueeze(selectdim(scaled_f, 1, 3); dims=1); dims=1),
                                            encoder.hash_table_size, encoder.hash_offset)
    hashed_6 = @ignore_derivatives hash_fn(scaled_f, encoder.hash_table_size, encoder.hash_offset)
    hashed_7 = @ignore_derivatives hash_fn(cat(unsqueeze(selectdim(scaled_f, 1, 1); dims=1),
                                               unsqueeze(selectdim(scaled_c, 1, 2); dims=1),
                                               unsqueeze(selectdim(scaled_f, 1, 3); dims=1); dims=1),
                                            encoder.hash_table_size, encoder.hash_offset)

    f_0 = encoder.hash_table(hashed_0) # [features_per_level,numlevels,...]
    f_1 = encoder.hash_table(hashed_1)
    f_2 = encoder.hash_table(hashed_2)
    f_3 = encoder.hash_table(hashed_3)
    f_4 = encoder.hash_table(hashed_4)
    f_5 = encoder.hash_table(hashed_5)
    f_6 = encoder.hash_table(hashed_6)
    f_7 = encoder.hash_table(hashed_7)


    offset_1 = unsqueeze(selectdim(offset, 1, 1); dims=1)
    offset_2 = unsqueeze(selectdim(offset, 1, 2); dims=1)
    offset_3 = unsqueeze(selectdim(offset, 1, 3); dims=1)
    f_03 = f_0 .* offset_1 .+ f_3 .* (1 .- offset_1)
    f_12 = f_1 .* offset_1 .+ f_2 .* (1 .- offset_1)
    f_56 = f_5 .* offset_1 .+ f_6 .* (1 .- offset_1)
    f_47 = f_4 .* offset_1 .+ f_7 .* (1 .- offset_1)
    
    f0312 = f_03 .* offset_2 .+ f_12 .* (1 .- offset_2)
    f4756 = f_47 .* offset_2 .+ f_56 .* (1 .- offset_2)

    encoded_value = f0312 .* offset_3 .+ f4756 .* (1 .- offset_3)
    reshape(encoded_value, :, size(encoded_value)[3:end]...)
end
