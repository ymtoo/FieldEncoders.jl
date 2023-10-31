using Pkg
Pkg.activate("./scripts")

using FieldEncoders
using GLMakie
using ImageCore: RGB, N0f8, colorview

include("visualize_utils.jl")

resolution = 128
slice = 1

xs = range(0, 1, resolution)
ys = range(0, 1, resolution)
zs = range(0, 1, resolution)
grid = FieldEncoders.meshgrid(xs, ys, zs)

numlevels = 8
minres = 2
maxres = 128
log2_hashmap_size = 4
features_per_level = 3
hash_init_scale = 0.001f0

let 
    encoder = HashEncoder(numlevels, minres, maxres, log2_hashmap_size, features_per_level, hash_init_scale)
    encoded_values = encode(encoder, grid)

    grid_slice = grid[:,:,:,slice]
    encoded_values_slice = encoded_values[:,:,:,slice]

    fig = visualize_data(grid_slice[[3,1,2],:,:])
    save("./scripts/figures/hashencoder_input.png", fig)

    encoded_images = reshape(encoded_values_slice, 3, numlevels, resolution, resolution) |> collect
    encoded_images .-= minimum(encoded_images)
    encoded_images ./= maximum(encoded_images)
    rgb_encoded_images = [colorview(RGB, N0f8.(encoded_images[:,i,:,:])) for i ∈ 1:numlevels]
    fig = visualize_data(rgb_encoded_images)
    save("./scripts/figures/hashencoder_encoded.png", fig)
end