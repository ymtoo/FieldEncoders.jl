using GLMakie

function visualize_data(x::AbstractArray{T,3}) where {T}
    n = size(x, 1)
    fig = Figure(resolution=(100*n,100))
    for i ∈ 1:n
        ax = Axis(fig[1,i])
        hidedecorations!(ax)
        heatmap!(ax, x[i,:,:]'; colormap = :plasma, colorrange = (-1.0, 1.0))
    end
    fig
end

function visualize_data(x::AbstractVector{T}) where {T<:AbstractMatrix}
    n = length(x)
    fig = Figure(resolution=(100*n,100))
    for i ∈ 1:n
        ax = Axis(fig[1,i])
        hidedecorations!(ax)
        image!(ax, rotr90(x[i]); colormap = :plasma)
    end
    fig
end
