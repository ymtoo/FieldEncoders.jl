"""
$(SIGNATURES)

Expected value of sin(x) where x ∼ N(μs, vars).
"""
function expected_sin(μs, vars)
    exp.(-vars ./ 2) .* sin.(μs)
end

"""
$(SIGNATURES)

Returns `true` if the first dimension of `xs` is equal to `inputdim` of `encoder`. 
"""
function check_inputdim(encoder::AbstractFieldEncoder, xs::AbstractArray)
    encoder.inputdim != size(xs, 1) && throw(ArgumentError("The first dimension of `xs` is not equal to `inputdim`")) 
end

"""
$(SIGNATURES)

Creates grids of coordinates specified by `xs`, `ys` and `zs`.
"""
function meshgrid(xs::AbstractVector{T}, ys::AbstractVector{T}, zs::AbstractVector{T}) where {T}
    xyz = similar(xs, 3, length(xs), length(ys), length(zs))
    for i ∈ eachindex(xs)
        for j ∈ eachindex(ys)
            for k ∈ eachindex(zs)
                xyz[:,i,j,k] = [xs[i],ys[j],zs[k]]
            end
        end
    end
    xyz
end
