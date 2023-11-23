export SphericalHarmonicEncoder

"""
$(TYPEDFIELDS)

Spherical harmonic encoder.
"""
struct SphericalHarmonicEncoder <: AbstractFieldEncoder
    "Number of spherical harmonic levels to encode"
    levels::Int
end

"""
$(TYPEDSIGNATURES)

Get output dimension.
"""
function get_out_dim(encoder::SphericalHarmonicEncoder)
    return encoder.levels ^ 2
end

function component_from_spherical_harmonics(levels::Int, directions::AbstractArray{T}) where {T}
    (levels < 1 || levels > 5) && throw(ArgumentError("SH levels must be in [1,4], got $(levels)"))
    size(directions, 1) != 3 && throw(ArgumentError("Direction input should have three dimensions. Got $(size(directions, 1))"))
    
    x = directions[1:1, ..]
    y = directions[2:2, ..]
    z = directions[3:3, ..]

    xx = x .^ 2
    yy = y .^ 2
    zz = z .^ 2

    # l0
    components = T(0.28209479177387814) * ones_like(directions, (1, size(directions)[2:end]...))

    # l1
    if levels > 1
        components = vcat(components, 
                          T(0.4886025119029199) .* y,
                          T(0.4886025119029199) .* z,
                          T(0.4886025119029199) .* x)
    end

    # l2
    if levels > 2
        components = vcat(components, 
                          T(1.0925484305920792) .* x .* y,
                          T(1.0925484305920792) .* y .* z,
                          T(0.9461746957575601) .* zz .- T(0.31539156525251999),
                          T(1.0925484305920792) .* x .* z,
                          T(0.5462742152960396) .* (xx .- yy))
    end

    # l3
    if levels > 3
        components = vcat(components, 
                          T(0.5900435899266435) .* y .* (T(3) .* xx .- yy),
                          T(2.890611442640554) .* x .* y .* z,
                          T(0.4570457994644658) .* y .* (T(5) .* zz .- 1),
                          T(0.3731763325901154) .* z .* (T(5) .* zz .- 3),
                          T(0.4570457994644658) .* x .* (T(5) .* zz .- 1),
                          T(1.445305721320277) .* z .* (xx .- yy),
                          T(0.5900435899266435) .* x .* (xx .- 3 .* yy))
    end

    # l4
    if levels > 4
        components = vcat(components, 
                          T(2.5033429417967046) * x * y * (xx - yy),
                          T(1.7701307697799304) * y * z * (T(3) * xx - yy),
                          T(0.9461746957575601) * x * y * (T(7) * zz - 1),
                          T(0.6690465435572892) * y * z * (T(7) * zz - 3),
                          T(0.10578554691520431) * (T(35) * zz * zz - T(30) * zz + T(3)),
                          T(0.6690465435572892) * x * z * (T(7) * zz - T(3)),
                          T(0.47308734787878004) * (xx - yy) * (7 * zz - 1),
                          T(1.7701307697799304) * x * z * (xx - T(3) * yy),
                          T(0.6258357354491761) * (xx * (xx - T(3) * yy) - yy * (T(3) * xx - yy)))
    end
    
    return components
end

"""
$(TYPEDSIGNATURES)

Transform positions `xs` using spherical harmonic encoding.
"""
encode(encoder::SphericalHarmonicEncoder, xs::AbstractArray) = 
    component_from_spherical_harmonics(encoder.levels, xs)
