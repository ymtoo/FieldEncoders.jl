using Pkg
Pkg.activate("./scripts")

using FieldEncoders

include("visualize_utils.jl")

levels = 4

height = 100
width = 150

encoder = SphericalHarmonicEncoder(levels)

θ = range(-π, π, width)
ϕ = range(0, π, height)
grid = FieldEncoders.meshgrid(θ, ϕ)
grid_θ = grid[1,:,:]
grid_ϕ = grid[2,:,:]

directions = stack([cos.(grid_θ) .* sin.(grid_ϕ), 
                   sin.(grid_θ) .* sin.(grid_ϕ),
                   cos.(grid_ϕ)]; dims=1) 

encoded_values = encoder(directions)

for level ∈ 0:levels-1
    fig = visualize_data(encoded_values[(level^2+1):((level+1)^2),:,:])
    save("./scripts/figures/shencoder_encoded_level_$(level+1).png", fig)
end
