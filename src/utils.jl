"""
$(SIGNATURES)

Expected value of sin(x) where x ∼ N(μs, vars).
"""
function expected_sin(μs, vars)
    exp.(-vars ./ 2) .* sin.(μs)
end

"""
$(SIGNATURES)

Return `true` if the first dimension of `xs` is equal to `inputdim` of `encoder`. 
"""
function check_inputdim(encoder::AbstractFieldEncoder, xs::AbstractArray)
    encoder.inputdim != size(xs, 1) && throw(ArgumentError("The first dimension of `xs` is not equal to `inputdim`")) 
end
