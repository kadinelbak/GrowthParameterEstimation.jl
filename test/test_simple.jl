module TestMod
    function test_func(x::Vector{<:Real}, y::Vector{<:Real})
        return x + y
    end
end