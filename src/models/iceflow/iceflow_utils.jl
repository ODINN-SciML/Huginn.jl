export inn1

# Helper functions for the staggered grid

####  Non-allocating functions  ####

diff_x!(O, I, Δx) = @views @. O = (I[begin+1:end,:] - I[1:end-1,:]) / Δx

diff_y!(O, I, Δy) = @views @. O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

avg!(O, I) = @views @. O = (I[1:end-1,1:end-1] + I[2:end,1:end-1] + I[1:end-1,2:end] + I[2:end,2:end]) * 0.25

avg_x!(O, I) = @views @. O = (I[1:end-1,:] + I[2:end,:]) * 0.5

avg_y!(O, I) = @views @. O = (I[:,1:end-1] + I[:,2:end]) * 0.5

####  Allocating functions  ####

"""
    avg(A)

4-point average of a matrix
"""
@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )


"""
    avg_x(A)

2-point average of a matrix's X axis
"""
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

"""
    avg_y(A)

2-point average of a matrix's Y axis
"""
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

"""
    diff_x(A)

2-point differential of a matrix's X axis
"""
@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])

"""
    diff_y(A)

2-point differential of a matrix's Y axis
"""
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])

"""
    inn(A)

Access the inner part of the matrix (-2,-2)
"""
@views inn(A) = A[2:end-1,2:end-1]

"""
    inn1(A)

Access the inner part of the matrix (-1,-1)
"""
@views inn1(A) = A[1:end-1,1:end-1]


### Enzyme scalar implementation of util functions

function avg_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += 0.25 * (A[i,j] + A[i+1,j] + A[i,j+1] + A[i+1,j+1])
        end
    end
end

function avg_x_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += 0.50 * (A[i,j] + A[i+1,j])
        end
    end
end

function avg_y_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += 0.50 * (A[i,j] + A[i,j+1])
        end
    end
end

function diff_x_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += A[i+1,j] - A[i,j]
        end
    end
end

function diff_y_enzyme!(R, A)
    @inbounds @simd for i in eachindex(R)
        R[i] = 0
    end

    @inbounds for i in axes(R, 1)
        @inbounds @simd for j in axes(R,2)
            R[i,j] += A[i,j+1] - A[i,j]
        end
    end
end