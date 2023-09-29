using FieldEncoders
using GLMakie

inputdim = 2
numfrequencies = 4
minfreq = 0.0
maxfreq = 6.0
includeinput = false
resolution = 128
covariance_magnitudes = [0.01, 0.1, 1.0]

encoder = NeRFEncoder(inputdim, numfrequencies, minfreq, maxfreq, includeinput)

x = range(0, 1, resolution)
grid = zeros(Float64, inputdim, resolution, resolution)
for (i, x) ∈ enumerate(range(0, 1, resolution))
    for (j, y) ∈ enumerate(range(0, 1, resolution))
        grid[1,i,j] = x
        grid[2,i,j] = y
    end
end

function visualize_data(x)
    n = size(x, 1)
    fig = Figure(resolution=(100*n,100))
    for i ∈ 1:n
        ax = Axis(fig[1,i])
        hidedecorations!(ax)
        heatmap!(ax, x[i,:,:]'; colormap = :plasma, colorrange = (-1.0, 1.0))
    end
    fig
end

let 
    fig = visualize_data(grid)
    save("./scripts/figures/nerfencoder_input.png", fig)
    encoded_x = encode(encoder, grid)
    fig = visualize_data(encoded_x)
    save("./scripts/figures/nerfencoder_encoded.png", fig)

    for covariance_magnitude ∈ covariance_magnitudes
        covs = zeros(Float64, inputdim, inputdim, 1, 1)
        covs[1,1,1,1] = 1 * covariance_magnitude
        covs[2,2,1,1] = 1 * covariance_magnitude
        encoded_x = encode(encoder, grid, covs)
        fig = visualize_data(encoded_x)
        save("./scripts/figures/nerfencoder_encoded_$(covariance_magnitude)_cov.png", fig)
    end
end


